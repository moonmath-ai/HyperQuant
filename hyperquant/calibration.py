"""
Calibration utilities for HyperQuant.

  ``calibrate_lattice_bps_to_snr``  — maps a target bits/scalar operating
      point to the calibration SNR by running synthetic Gaussian experiments
      with ``lattice_quant.run_experiment``.  Results are cached to
      ``~/.cache/hyperquant/`` so subsequent calls are instant.

  ``kmeans_gaussian_grid`` — Lloyd's algorithm in PyTorch to build optimal
      scalar quantization grids for Gaussian sources.

All operations are CPU/CUDA-agnostic.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass

import torch

import pathlib as _pathlib
from . import lattice as lattice_quant
_PKG_CACHE = _pathlib.Path.home() / ".cache" / "hyperquant"
LATTICE_BPS_CALIBRATION_PATH = _PKG_CACHE / "lattice_bps_calibration.json"
KMEANS_GRIDS_DIR = _PKG_CACHE / "kmeans_grids"


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Lattice SNR ↔ bps calibration (on IID Gaussian)
# ─────────────────────────────────────────────────────────────────────────────

def _build_snr_bps_table(lattice: str, snr_min: float, snr_max: float,
                         snr_step: float, n_vectors: int) -> list[tuple[float, float]]:
    """Dense SNR sweep; returns [(snr_db, gaussian_bps), ...]."""
    table: list[tuple[float, float]] = []
    snrs = []
    s = snr_min
    while s <= snr_max + 1e-9:
        snrs.append(round(s, 4))
        s += snr_step
    for snr in snrs:
        r = lattice_quant.run_experiment(snr, n_vectors, lattice, with_lz=False)
        table.append((snr, r["rice_bps"]))
    return table


def _invert_table(table: list[tuple[float, float]], target_bps: float) -> float:
    """Linear interpolation: find SNR that gives ``target_bps``.

    ``table`` is sorted ascending in SNR (bps is monotone ascending in SNR).
    """
    snrs = [s for s, _ in table]
    bpss = [b for _, b in table]
    if target_bps <= bpss[0]:
        return snrs[0]
    if target_bps >= bpss[-1]:
        return snrs[-1]
    for i in range(len(table) - 1):
        if bpss[i] <= target_bps <= bpss[i + 1]:
            t = (target_bps - bpss[i]) / (bpss[i + 1] - bpss[i])
            return snrs[i] + t * (snrs[i + 1] - snrs[i])
    return snrs[-1]   # unreachable


def calibrate_lattice_bps_to_snr(
    lattices: list[str],
    target_bps_list: list[float],
    *,
    cache_path: str = str(LATTICE_BPS_CALIBRATION_PATH),
    snr_min: float = 0.0,
    snr_max: float = 34.0,
    snr_step: float = 0.5,
    n_vectors: int = 200_000,
    force_recompute: bool = False,
) -> dict[str, dict[float, dict[str, float]]]:
    """Return ``{lattice: {target_bps: {"snr_db": …, "gaussian_bps": …}}}``.

    A dense SNR sweep on IID Gaussian is run once per lattice (cached on disk),
    then linearly inverted to find the SNR that achieves each ``target_bps``.
    """
    if os.path.exists(cache_path) and not force_recompute:
        with open(cache_path) as f:
            cache = json.load(f)
    else:
        cache = {}

    out: dict[str, dict[float, dict[str, float]]] = {}
    for lat in lattices:
        if lat not in cache:
            print(f"  calibrating {lat} @ SNR ∈ [{snr_min}, {snr_max}] dB step {snr_step}…")
            cache[lat] = {
                "table": _build_snr_bps_table(lat, snr_min, snr_max, snr_step, n_vectors),
            }
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            with open(cache_path, "w") as f:
                json.dump(cache, f, indent=2)

        # tuples got serialised as lists — normalise.
        table = [(float(s), float(b)) for (s, b) in cache[lat]["table"]]
        out[lat] = {}
        for tb in target_bps_list:
            snr = _invert_table(table, tb)
            r = lattice_quant.run_experiment(snr, 100_000, lat, with_lz=False)
            out[lat][tb] = {"snr_db": snr, "gaussian_bps": r["rice_bps"]}
    return out


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Lloyd k-means VQ on IID Gaussian (extended HIGGS-style grid)
# ─────────────────────────────────────────────────────────────────────────────

_KMEANS_CACHE_DIR = str(KMEANS_GRIDS_DIR)


def _kmeans_cache_path(p: int, n_codes: int, n_samples: int, n_inits: int, seed: int) -> str:
    return os.path.join(
        _KMEANS_CACHE_DIR,
        f"p{p}_n{n_codes}_s{n_samples}_i{n_inits}_seed{seed}.pt",
    )


def kmeans_gaussian_grid(p: int, n_codes: int, *,
                        n_samples: int = 1_000_000,
                        n_iters: int = 50,
                        n_inits: int = 3,
                        seed: int = 0,
                        force_recompute: bool = False) -> torch.Tensor:
    """Run Lloyd's algorithm on N(0, I_p) samples to obtain an n_codes-point
    vector quantizer. Returns ``(n_codes, p) float32`` centroids.

    Results are cached to disk under ``kmeans_grids/`` so each non-stock
    (p, n_codes) only computes once across the whole experiment.
    """
    cache_path = _kmeans_cache_path(p, n_codes, n_samples, n_inits, seed)
    if os.path.exists(cache_path) and not force_recompute:
        return torch.load(cache_path, weights_only=True)

    g = torch.Generator(device="cpu").manual_seed(seed)

    best_centroids = None
    best_distortion = float("inf")
    for init in range(n_inits):
        g.manual_seed(seed + init)
        x = torch.randn(n_samples, p, generator=g)
        # Initialize centroids by picking n_codes random points (kmeans++ would
        # be better but for the small p=1,2 setting this converges fine).
        idx = torch.randperm(n_samples, generator=g)[:n_codes]
        centroids = x[idx].clone()

        for it in range(n_iters):
            d2 = (x ** 2).sum(-1, keepdim=True) \
                 - 2 * x @ centroids.T \
                 + (centroids ** 2).sum(-1)
            assign = d2.argmin(-1)
            new_centroids = torch.zeros_like(centroids)
            counts = torch.zeros(n_codes, dtype=torch.long)
            new_centroids.index_add_(0, assign, x)
            counts.index_add_(0, assign, torch.ones_like(assign))
            empty = counts == 0
            new_centroids[~empty] /= counts[~empty].unsqueeze(-1)
            # Re-seed any empty centroid to a random sample (rare).
            if empty.any():
                new_centroids[empty] = x[torch.randint(0, n_samples, (int(empty.sum()),), generator=g)]
            shift = (new_centroids - centroids).pow(2).sum().sqrt().item()
            centroids = new_centroids
            if shift < 1e-5:
                break

        # Final distortion.
        d2 = (x ** 2).sum(-1, keepdim=True) \
             - 2 * x @ centroids.T \
             + (centroids ** 2).sum(-1)
        distortion = d2.min(-1).values.mean().item()
        if distortion < best_distortion:
            best_distortion = distortion
            best_centroids = centroids

    assert best_centroids is not None
    os.makedirs(_KMEANS_CACHE_DIR, exist_ok=True)
    torch.save(best_centroids, cache_path)
    return best_centroids


# ─────────────────────────────────────────────────────────────────────────────
# 3.  Per-layer weight SNR (params-weighted mean)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class LayerSnrStat:
    name: str
    n_scalars: int
    snr_db: float          # per-layer SNR (= ‖W‖² / ‖W − Ŵ‖² in dB)


def weight_snr_stats(per_layer: list[LayerSnrStat]) -> dict:
    """Aggregate per-layer SNR into model-wide summary statistics."""
    if not per_layer:
        return {"snr_db_mean_paramw": math.nan, "snr_db_median": math.nan,
                "snr_db_min": math.nan, "snr_db_max": math.nan,
                "n_layers": 0}
    total = sum(s.n_scalars for s in per_layer)
    # Convert to linear ratio, weight by params, convert back.
    weighted_linear = sum(
        s.n_scalars / total * (10 ** (s.snr_db / 10.0)) for s in per_layer
    )
    mean_db = 10.0 * math.log10(weighted_linear)
    snrs = sorted(s.snr_db for s in per_layer)
    return {
        "snr_db_mean_paramw": mean_db,
        "snr_db_median": snrs[len(snrs) // 2],
        "snr_db_min": snrs[0],
        "snr_db_max": snrs[-1],
        "n_layers": len(per_layer),
    }
