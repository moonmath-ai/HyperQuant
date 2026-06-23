"""
Lattice-grid pseudo-quantization scheme for HIGGS-style weight compression.

Replaces the precomputed Gaussian-optimized grids in
``transformers.integrations.higgs.get_higgs_grid`` with the lattice
quantisers from ``hyperquant.lattice``:

    E8int (8-D), D4int (4-D), A2int (2-D), Z1int (1-D).

Data flow per weight matrix (matches HIGGS):

    W ──pad-cols-to-hadamard_size──► W'
       ──Hadamard along last dim──► U
       ──/= per-row-chunk norm──► U_norm   (per-scalar variance ≈ 1)
       ──reshape (..., n_dim)──► chunks
       ──α · chunks──► scaled
       ──nearest-lattice-point──► codes
       ──codes / α (and ×√3 for A2int's y axis)──► reconstructed
       ──×scale──► U_hat
       ──Hadamard along last dim──► W_hat'/n
       ──unpad──► W_hat   (same shape as W)

For HIGGS we use the same flow but with a finite codebook (``get_higgs_grid``)
indexed by ``argmax(2·u·gᵀ − ‖g‖²)``.

Both quantizers return (W_hat, total_bits_used, n_scalars) so the average
bits/scalar can be aggregated over the whole model. Lattices report the
Rice-coded rate (the actual achievable compression rate of
``lattice_quant.py``); HIGGS reports its native fixed ``bits`` rate.

This is a *pseudo*-quantizer — it produces dequantized bf16/fp32 weights
suitable for forward passes, so we can measure perplexity impact without
the FLUTE kernel. It is NOT optimized for inference speed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .lattice import (
    MSE_VORONOI_A2_ANALYTIC,
    MSE_VORONOI_D4_ANALYTIC,
    MSE_VORONOI_E8_ANALYTIC,
    MSE_VORONOI_Z1_ANALYTIC,
    N_DIM_A2,
    N_DIM_D4,
    N_DIM_E8,
    N_DIM_Z1,
    SQRT3,
    quantize_a2int,
    quantize_d4int,
    quantize_e8int,
    quantize_z1,
    rice_optimal,
    structural_decompose,
    structural_decompose_a2int,
    structural_decompose_d4,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fast Walsh–Hadamard transform (matches ``fast_hadamard_transform`` convention:
# unnormalized — entries ±1 — so ``H @ H = n · I``).
# ─────────────────────────────────────────────────────────────────────────────

def hadamard_transform(x: torch.Tensor) -> torch.Tensor:
    """Apply the unnormalized Walsh–Hadamard transform along the last axis.

    The last dimension must be a power of two. ``H`` has entries in ``{±1}`` so
    ``hadamard_transform(hadamard_transform(x)) == n · x``.
    """
    n = x.shape[-1]
    if n & (n - 1):
        raise ValueError(f"hadamard last-dim must be power of two, got {n}")
    h = 1
    while h < n:
        x = x.reshape(*x.shape[:-1], n // (2 * h), 2, h)
        a = x[..., 0, :]
        b = x[..., 1, :]
        x = torch.stack([a + b, a - b], dim=-2).reshape(*x.shape[:-3], n)
        h *= 2
    return x


# ─────────────────────────────────────────────────────────────────────────────
# Padding helpers (mirror ``integrations.higgs.pad_to_block``)
# ─────────────────────────────────────────────────────────────────────────────

def _pad_last_dim_to(x: torch.Tensor, multiple: int) -> tuple[torch.Tensor, int]:
    n = x.shape[-1]
    pad = (-n) % multiple
    if pad == 0:
        return x, 0
    return torch.nn.functional.pad(x, (0, pad), "constant", 0.0), pad


# ─────────────────────────────────────────────────────────────────────────────
# Rice-coded bit accounting for a lattice-quantized vector stream
# ─────────────────────────────────────────────────────────────────────────────

def _zigzag(x: torch.Tensor) -> torch.Tensor:
    return torch.where(x >= 0, 2 * x, -2 * x - 1)


def _rice_bits(values: torch.Tensor, k: int) -> int:
    """Total bits for Rice-k on (signed) int tensor (no header overhead)."""
    return int(((_zigzag(values) >> k).long().sum() + values.numel() * (1 + k)).item())


def _best_rice_total_bits(values: torch.Tensor, k_range=range(0, 9)) -> tuple[int, int]:
    """Return (best_k, total_bits) over the candidate ``k_range``."""
    best = None
    best_k = 0
    for k in k_range:
        b = _rice_bits(values, k)
        if best is None or b < best:
            best = b
            best_k = k
    return best_k, int(best)


def _rice_bits_for_codes(codes: torch.Tensor, lattice: str) -> int:
    """Return total Rice-coded bits used to encode ``codes`` for the given lattice.

    Mirrors the structural decomposition + Rice strategy from ``lattice_quant.py``.
    ``codes`` has shape ``(..., n_dim)`` and dtype int (already on CPU/GPU).
    """
    codes = codes.reshape(-1, codes.shape[-1])
    if lattice == "z1int":
        flat = codes.reshape(-1)
        _, bits = _best_rice_total_bits(flat)
        return bits

    if lattice == "a2int":
        T_y, N_x = structural_decompose_a2int(codes)
        _, bits_y = _best_rice_total_bits(T_y)
        _, bits_x = _best_rice_total_bits(N_x)
        return bits_y + bits_x

    if lattice == "e8int":
        c, S_syms, T = structural_decompose(codes)
        k_s, bits_s = _best_rice_total_bits(S_syms.reshape(-1))
        # Combined = 2·zz(T) + c (LSB carries coset). Encode at k_s.
        combined = 2 * _zigzag(T) + c
        bits_combined = int(((combined >> k_s).long().sum() + combined.numel() * (1 + k_s)).item())
        return bits_s + bits_combined

    if lattice == "d4int":
        S3, T = structural_decompose_d4(codes)
        k_s, bits_s = _best_rice_total_bits(S3.reshape(-1))
        # combined = 2·zz(T) with c=0 always; the LSB is implicit, so use k_t = k_s − 1.
        k_t = max(0, k_s - 1)
        bits_t = int(((_zigzag(T) >> k_t).long().sum() + T.numel() * (1 + k_t)).item())
        return bits_s + bits_t

    raise ValueError(f"Unknown lattice {lattice!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Lattice / HIGGS dispatch tables
# ─────────────────────────────────────────────────────────────────────────────

_LATTICE_INFO = {
    "e8int": (N_DIM_E8, MSE_VORONOI_E8_ANALYTIC, quantize_e8int),
    "d4int": (N_DIM_D4, MSE_VORONOI_D4_ANALYTIC, quantize_d4int),
    "a2int": (N_DIM_A2, MSE_VORONOI_A2_ANALYTIC, quantize_a2int),
    "z1int": (N_DIM_Z1, MSE_VORONOI_Z1_ANALYTIC, quantize_z1),
}


def lattice_alpha(snr_db: float, lattice: str) -> float:
    """α calibrated to hit ``snr_db`` for unit-variance Gaussian inputs."""
    n_dim, mse_analytic, _ = _LATTICE_INFO[lattice]
    snr_lin = 10.0 ** (snr_db / 10.0)
    return math.sqrt(snr_lin * mse_analytic / n_dim)


# ─────────────────────────────────────────────────────────────────────────────
# Pseudo-quantizers — operate on a single (out_features, in_features) weight
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class QuantStats:
    n_scalars: int     # number of weight scalars (after padding stripped)
    total_bits: int    # estimated total bits to encode the quantized payload


def _hadamard_normalize(weight: torch.Tensor, hadamard_size: int):
    """Return (u, scale, pad) where u is per-scalar-variance-1 Hadamard'd weight.

    ``u`` shape:  (out_features, n_chunks, hadamard_size)
    ``scale`` shape: (out_features, n_chunks)  — the ‖chunk‖ used to normalize.
    """
    w, pad = _pad_last_dim_to(weight, hadamard_size)
    out_features, padded_in = w.shape
    n_chunks = padded_in // hadamard_size
    w = w.reshape(out_features, n_chunks, hadamard_size)
    scale = torch.linalg.norm(w, dim=-1)             # (out, n_chunks)
    u = hadamard_transform(w) / scale.unsqueeze(-1).clamp(min=1e-12)
    return u, scale, pad


def _hadamard_denormalize(u_hat: torch.Tensor, scale: torch.Tensor,
                          original_in_features: int) -> torch.Tensor:
    """Inverse of ``_hadamard_normalize``: returns weight in original shape."""
    out_features, n_chunks, hadamard_size = u_hat.shape
    w_chunks = hadamard_transform(u_hat * scale.unsqueeze(-1)) / hadamard_size
    w = w_chunks.reshape(out_features, n_chunks * hadamard_size)
    return w[:, :original_in_features]


def lattice_pseudo_quantize(
    weight: torch.Tensor,
    lattice: str,
    snr_db: float,
    hadamard_size: int = 1024,
) -> tuple[torch.Tensor, QuantStats]:
    """Pseudo-quantize a 2-D weight matrix using lattice ``lattice`` at ``snr_db``.

    Returns:
        (dequantized weight matching ``weight.shape`` and dtype, QuantStats)
    """
    assert weight.ndim == 2, f"expected 2D weight, got {tuple(weight.shape)}"
    orig_dtype = weight.dtype
    orig_in = weight.shape[1]

    n_dim, _, quantize_fn = _LATTICE_INFO[lattice]
    alpha = lattice_alpha(snr_db, lattice)

    u, scale, _ = _hadamard_normalize(weight.to(torch.float32), hadamard_size)
    # u: (out, n_chunks, hadamard_size).  Per-scalar variance ≈ 1.
    out, n_chunks, n_h = u.shape
    assert n_h % n_dim == 0, f"hadamard_size {n_h} must divide by lattice dim {n_dim}"

    flat = u.reshape(-1, n_dim)
    # bf16 cast mirrors the calibration regime in ``lattice_quant.py``.
    scaled = (alpha * flat).to(torch.bfloat16)
    codes = quantize_fn(scaled)                # (N, n_dim) int32

    # Reconstruct physical lattice point (A2int stores n_y, n_x; physical y is n_y·√3).
    if lattice == "a2int":
        recon = torch.stack(
            [codes[:, 0].float() * SQRT3, codes[:, 1].float()], dim=1
        )
    else:
        recon = codes.float()

    u_hat = (recon / alpha).reshape(out, n_chunks, n_h).to(torch.float32)
    w_hat = _hadamard_denormalize(u_hat, scale, orig_in).to(orig_dtype)

    n_scalars = out * orig_in
    bits = _rice_bits_for_codes(codes, lattice)
    # bits cover all scalars including padded zeros — scale to actual stored scalars.
    bits_actual = bits * (n_scalars / (out * n_chunks * n_h))
    return w_hat, QuantStats(n_scalars=n_scalars, total_bits=int(round(bits_actual)))


_HIGGS_GRID_CACHE: dict[tuple, torch.Tensor] = {}


def _higgs_grid(p: int, n_codes: int, device, dtype) -> torch.Tensor:
    """Lazy loader for ``transformers.integrations.higgs.get_higgs_grid`` —
    falls through to a Lloyd's-algorithm Gaussian VQ for non-stock sizes.
    Cached per (p, n_codes, device, dtype) to avoid recomputing inside the
    quantize_model_inplace loop (224 layers per Llama).
    """
    key = (p, n_codes, str(device), dtype)
    cached = _HIGGS_GRID_CACHE.get(key)
    if cached is not None:
        return cached
    from transformers.integrations.higgs import get_higgs_grid
    try:
        g = get_higgs_grid(p, n_codes).to(device=device, dtype=dtype)
    except NotImplementedError:
        from hyperquant.calibration import kmeans_gaussian_grid
        g = kmeans_gaussian_grid(p, n_codes).to(device=device, dtype=dtype)
    _HIGGS_GRID_CACHE[key] = g
    return g


def higgs_pseudo_quantize(
    weight: torch.Tensor,
    p: int,
    bits: float,
    hadamard_size: int = 1024,
    n_codes: int | None = None,
) -> tuple[torch.Tensor, QuantStats]:
    """Pseudo-quantize a 2-D weight matrix using a HIGGS-style precomputed grid.

    ``bits`` is the bits/scalar to record in the stats. The grid has
    ``n_codes`` entries — if not provided, defaults to ``2**(p · bits)``
    (the standard HIGGS recipe; only integer ``bits`` are supported by
    the stock grid). For non-integer ``bits`` pass ``n_codes`` directly,
    e.g. ``bits=4.5, n_codes=512`` for a 2-D 512-point Lloyd grid.
    """
    assert weight.ndim == 2
    orig_dtype = weight.dtype
    orig_in = weight.shape[1]

    if n_codes is None:
        n_codes_int = int(round(2 ** (p * bits)))
    else:
        n_codes_int = int(n_codes)
    grid = _higgs_grid(p, n_codes_int, weight.device, torch.float32)   # (n_codes, p)
    grid_norm_sq = (grid ** 2).sum(dim=-1)                              # (n_codes,)

    u, scale, _ = _hadamard_normalize(weight.to(torch.float32), hadamard_size)
    out, n_chunks, n_h = u.shape
    assert n_h % p == 0, f"hadamard_size {n_h} must divide by HIGGS dim {p}"

    flat = u.reshape(-1, p)                                      # (N, p)

    # Chunked argmax to bound peak memory. Target ≤ 1 GiB scratch
    # (= chunk_rows · n_codes · 4 bytes). For 256-pt grid this is 1M
    # rows/chunk; for 1024-pt grid 256k rows/chunk.
    target_bytes = 1 << 30
    chunk_rows = max(1, target_bytes // (grid.shape[0] * 4))
    codes = torch.empty(flat.shape[0], device=flat.device, dtype=torch.long)
    for i in range(0, flat.shape[0], chunk_rows):
        sl = slice(i, i + chunk_rows)
        scores = 2 * flat[sl] @ grid.T - grid_norm_sq
        codes[sl] = scores.argmax(dim=-1)

    recon = grid[codes]                                          # (N, p)
    u_hat = recon.reshape(out, n_chunks, n_h)
    w_hat = _hadamard_denormalize(u_hat, scale, orig_in).to(orig_dtype)

    n_scalars = out * orig_in
    # Fixed bit budget: log2(n_codes) bits per p-D vector → log2(n_codes)/p per scalar.
    bits_per_scalar = math.log2(n_codes_int) / p
    bits_total = int(round(bits_per_scalar * n_scalars))
    return w_hat, QuantStats(n_scalars=n_scalars, total_bits=bits_total)


# ─────────────────────────────────────────────────────────────────────────────
# Apply over an entire model
# ─────────────────────────────────────────────────────────────────────────────

def quantize_model_inplace(
    model: torch.nn.Module,
    quantize_fn,                       # weight -> (w_hat, QuantStats)
    skip_module_names: tuple[str, ...] = ("lm_head",),
    *,
    measure_snr: bool = False,
) -> dict:
    """Walk all nn.Linear layers; replace each weight with its pseudo-quantized
    version (skipping names containing any of ``skip_module_names``).

    Returns a stats dict with total bits / scalars / per-layer breakdown.
    If ``measure_snr`` is True, also computes per-layer weight SNR
    (``10·log10(‖W‖² / ‖W − Ŵ‖²)``) and a params-weighted model-wide mean.
    """
    import math as _math
    import torch.nn as nn

    total_bits = 0
    total_scalars = 0
    n_quantized = 0
    per_layer: list[dict] = []
    weighted_inverse_snr_linear = 0.0    # Σ (n_i / N) · (1 / SNR_lin_i)
    snr_db_list: list[float] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name for skip in skip_module_names):
            continue

        weight = module.weight.data
        with torch.no_grad():
            w_orig = weight.detach().clone() if measure_snr else None
            w_hat, stats = quantize_fn(weight)
            module.weight.data.copy_(w_hat)

            snr_db = _math.nan
            if measure_snr:
                sig = w_orig.float().pow(2).sum().item()
                noise = (w_orig.float() - w_hat.float()).pow(2).sum().item()
                snr_lin = sig / noise if noise > 0 else float("inf")
                snr_db = 10.0 * _math.log10(snr_lin) if snr_lin > 0 else float("-inf")
                snr_db_list.append(snr_db)
                del w_orig

        total_bits += stats.total_bits
        total_scalars += stats.n_scalars
        n_quantized += 1
        per_layer.append({
            "name": name,
            "n_scalars": stats.n_scalars,
            "total_bits": stats.total_bits,
            "snr_db": snr_db,
        })

    bits_per_scalar = total_bits / max(1, total_scalars)
    out = {
        "n_layers_quantized": n_quantized,
        "n_scalars": total_scalars,
        "total_bits": total_bits,
        "bits_per_scalar": bits_per_scalar,
        "per_layer": per_layer,
    }
    if measure_snr and per_layer:
        # Params-weighted mean is computed in *linear* power-ratio space
        # (apparent noise power averages linearly across layers), then
        # converted back to dB.
        N_total = sum(p["n_scalars"] for p in per_layer)
        wmean_lin = sum(
            (p["n_scalars"] / N_total) * (10 ** (p["snr_db"] / 10.0))
            for p in per_layer
        )
        snr_db_list.sort()
        out["snr_db_paramw"] = 10.0 * _math.log10(wmean_lin)
        out["snr_db_median"] = snr_db_list[len(snr_db_list) // 2]
        out["snr_db_min"] = snr_db_list[0]
        out["snr_db_max"] = snr_db_list[-1]
    return out
