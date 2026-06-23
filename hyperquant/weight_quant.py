"""
Lattice quantization + chunked Hadamard rotation + per-tile FP8 (E4M3)
activations/weights, in pseudo-quantization form.

Inference math:
    Y  =  X · Wᵀ
       =  (X · Hᵀ) · (W · Hᵀ)ᵀ / N           # H Hᵀ = N · I, applied per chunk
       =  X' · W'ᵀ / N

So as long as Hadamard is applied to *both* X and W (per 1024-chunk along
the input dim), the inverse Hadamard never appears — it cancels in the
matmul, leaving a single 1/N factor that we fold into the stored weight.

Weight (one-time quantization):
    1.  W' = chunked_hadamard(W, hadamard_size)                       # bf16
    2.  Lattice-quantize W' per Hadamard-chunk → integer codes + α
    3.  Dequant: W'_lattice = codes · eff_scale                       # bf16, H-domain
    4.  Per-tile FP8 cast (simulated):
            for each tile of size `weight_tile_size` along the last dim,
            scale = amax / FP8_MAX, then cast bf16 → e4m3 → bf16.
    5.  Store W'_fp8 / hadamard_size  inside  nn.Linear.weight.

Activation (every forward, via pre-hook):
    1.  X' = chunked_hadamard(X, hadamard_size)                       # bf16
    2.  Per-tile FP8 cast (simulated): tile = `act_tile_size` along last dim.

The original `nn.Linear.forward` then does `(X'_fp8) @ (W'_fp8 / N)ᵀ`,
giving the correct result up to FP8 quantization noise.

This is a *pseudo*-quantizer — both X' and W' are cast to FP8 *and back to
bf16* before the bf16 matmul, exactly modelling the quantization error a
real FP8 MMA would introduce, but with a bf16 accumulator (close enough
for SNR/PPL purposes; a true FP8 MMA accumulates in FP32, only slightly
better).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from .quant_utils import (
    QuantStats,
    _hadamard_normalize,
    _LATTICE_INFO,
    _rice_bits_for_codes,
    hadamard_transform,
    lattice_alpha,
)
from .lattice import SQRT3


# E4M3 numerical range. torch.finfo(torch.float8_e4m3fn).max is 448.0.
FP8_E4M3 = torch.float8_e4m3fn
FP8_E4M3_MAX = 448.0


# ─────────────────────────────────────────────────────────────────────────────
# MMA dtype registry and per-tile quantization helpers
# ─────────────────────────────────────────────────────────────────────────────

# Supported MMA dtypes for the FP8/INT8/FP4 inference paths.
MMA_DTYPES = ("fp8_e4m3", "int8", "nvfp4", "mxfp4")

def chunked_hadamard(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Apply the un-normalized Walsh-Hadamard transform to each chunk along
    the last dim of ``x``. ``H Hᵀ = chunk_size · I`` so the inverse is
    ``chunked_hadamard(.) / chunk_size``.
    """
    shape = x.shape
    if shape[-1] % chunk_size != 0:
        raise ValueError(f"last dim {shape[-1]} must be divisible by chunk_size {chunk_size}")
    n_chunks = shape[-1] // chunk_size
    x_chunked = x.reshape(*shape[:-1], n_chunks, chunk_size)
    x_h = hadamard_transform(x_chunked)
    return x_h.reshape(*shape)


def simulate_per_tile_fp8(x: torch.Tensor, tile_size: int) -> torch.Tensor:
    """Per-tile FP8 (E4M3) quant-dequant simulation along the last dim.

    For each tile of size ``tile_size`` along the last dim, compute a scalar
    scale = max(|tile|)/FP8_MAX, cast to FP8, cast back to ``x.dtype``,
    re-multiply by scale.

    This emulates exactly what a real FP8 MMA path would do with per-tile
    activation/weight scaling — *except* the matmul itself happens in bf16
    rather than the FP8 hardware path.
    """
    shape = x.shape
    last_dim = shape[-1]
    if last_dim % tile_size != 0:
        raise ValueError(f"last dim {last_dim} must be divisible by tile_size {tile_size}")
    n_tiles = last_dim // tile_size
    orig_dtype = x.dtype

    x_tiled = x.reshape(*shape[:-1], n_tiles, tile_size)
    # amax in float32 to avoid bf16 overflow on the absolute value.
    amax = x_tiled.float().abs().amax(dim=-1, keepdim=True).clamp(min=1e-12)
    scale = amax / FP8_E4M3_MAX                                   # broadcasted shape

    x_scaled = x_tiled.float() / scale
    x_fp8 = x_scaled.to(FP8_E4M3)
    x_dq = (x_fp8.float() * scale).to(orig_dtype)
    return x_dq.reshape(*shape)


def simulate_per_tile_quant(x: torch.Tensor, tile_size: int,
                            mma_dtype: str) -> torch.Tensor:
    """Per-tile quantization simulation for the specified MMA dtype.

    Dispatches to the appropriate simulate function based on ``mma_dtype``:
    ``"fp8_e4m3"`` → FP8 E4M3 per-tile cast;
    ``"int8"`` → symmetric INT8 per-tile cast;
    ``"nvfp4"`` / ``"mxfp4"`` → approximate FP4 E2M1 per-tile cast.
    """
    if mma_dtype == "fp8_e4m3":
        return simulate_per_tile_fp8(x, tile_size)
    elif mma_dtype == "int8":
        shape, orig_dtype = x.shape, x.dtype
        x_f = x.reshape(-1, tile_size).float()
        scale = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 127.0
        x_q = (x_f / scale).round().clamp(-127.0, 127.0) * scale
        return x_q.reshape(*shape).to(orig_dtype)
    elif mma_dtype in ("nvfp4", "mxfp4"):
        # FP4 E2M1 representable magnitudes: {0, 0.5, 1, 1.5, 2, 3, 4, 6}
        fp4_levels = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.],
                                   dtype=torch.float32, device=x.device)
        shape, orig_dtype = x.shape, x.dtype
        x_f = x.reshape(-1, tile_size).float()
        scale = x_f.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12) / 6.0
        x_s = x_f / scale
        signs = x_s.sign()
        idx = (x_s.abs().unsqueeze(-1) - fp4_levels).abs().argmin(dim=-1)
        x_q = signs * fp4_levels[idx] * scale
        return x_q.reshape(*shape).to(orig_dtype)
    else:
        raise ValueError(f"Unknown mma_dtype={mma_dtype!r}. Choose from {MMA_DTYPES}")


# ─────────────────────────────────────────────────────────────────────────────
# Lattice quantization, dequantized but kept in Hadamard domain
# ─────────────────────────────────────────────────────────────────────────────

def _lattice_dequant_hadamard_domain(
    weight: torch.Tensor,
    lattice: str,
    snr_db: float,
    hadamard_size: int,
) -> tuple[torch.Tensor, QuantStats]:
    """Lattice-quantize ``weight`` per Hadamard chunk and return the dequantized
    result **still in Hadamard domain** (i.e. no inverse Hadamard applied).

    Output shape is ``(out, n_chunks * hadamard_size)`` — the input dimension
    is padded up to the next multiple of ``hadamard_size`` if needed.
    """
    assert weight.ndim == 2
    n_dim, _, quantize_fn = _LATTICE_INFO[lattice]
    alpha = lattice_alpha(snr_db, lattice)

    u, scale, _pad = _hadamard_normalize(weight.to(torch.float32), hadamard_size)
    # u: (out, n_chunks, hadamard_size). per-scalar variance ≈ 1.
    out, n_chunks, n_h = u.shape

    flat = u.reshape(-1, n_dim)
    scaled = (alpha * flat).to(torch.bfloat16)
    codes = quantize_fn(scaled)                            # (N, n_dim) int

    if lattice == "a2int":
        recon = torch.stack(
            [codes[:, 0].float() * SQRT3, codes[:, 1].float()], dim=1
        )
    else:
        recon = codes.float()

    u_hat = (recon / alpha).reshape(out, n_chunks, n_h)
    # Stay in Hadamard domain. Reconstruction in H-domain is u_hat · scale (not /N).
    w_h_dq = (u_hat * scale.unsqueeze(-1)).reshape(out, n_chunks * n_h)

    n_scalars = out * weight.shape[1]
    bits = _rice_bits_for_codes(codes, lattice)
    bits_actual = bits * (n_scalars / (out * n_chunks * n_h))
    return w_h_dq, QuantStats(n_scalars=n_scalars, total_bits=int(round(bits_actual)))


# ─────────────────────────────────────────────────────────────────────────────
# Full weight-side packer: lattice → FP8 → ÷N (Hadamard cancellation factor)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Fp8LatticeConfig:
    method: str                    # "lattice" | "fp8_hadamard" | "fp8_only"
    lattice: str | None = None     # set when method == "lattice"
    snr_db: float | None = None    # set when method == "lattice"
    hadamard_size: int = 1024
    weight_tile_size: int = 128
    act_tile_size: int = 128
    apply_hadamard: bool = True    # if False, skip Hadamard on activations


def quantize_weight_for_fp8_path(
    weight: torch.Tensor,
    cfg: Fp8LatticeConfig,
) -> tuple[torch.Tensor, QuantStats, torch.Tensor]:
    """Return (weight-to-store, QuantStats, effective_weight_in_original_basis).

    ``weight-to-store`` is what gets placed into ``nn.Linear.weight`` so the
    standard ``Y = X · Wᵀ`` matmul, combined with the activation pre-hook,
    produces the desired pseudo-quantized output.

    ``effective_weight_in_original_basis`` is the bf16 reconstruction of W
    that an *FP32-accumulating* MMA pipe would produce (i.e. the inverse
    Hadamard of the FP8 weight, with the /N factor), used to compute
    per-layer weight SNR.
    """
    O, I = weight.shape
    if I % cfg.hadamard_size != 0:
        raise ValueError(
            f"input dim {I} must be divisible by hadamard_size {cfg.hadamard_size} "
            f"for this POC (no padding)."
        )
    orig_dtype = weight.dtype

    if cfg.method == "fp8_only":
        # No Hadamard, no lattice. Just per-tile FP8 on the original weight.
        w_fp8_dq = simulate_per_tile_fp8(weight.to(torch.float32),
                                          cfg.weight_tile_size).to(orig_dtype)
        n_scalars = O * I
        stats = QuantStats(n_scalars=n_scalars, total_bits=8 * n_scalars)
        return w_fp8_dq, stats, w_fp8_dq

    # Compute W' = chunked_hadamard(W) on the input dim.
    w_h = chunked_hadamard(weight.to(torch.float32), cfg.hadamard_size)
    # If we're using lattice, replace W' with its lattice-quantized dequant.
    if cfg.method == "lattice":
        assert cfg.lattice is not None and cfg.snr_db is not None
        w_h_lattice, stats = _lattice_dequant_hadamard_domain(
            weight, cfg.lattice, cfg.snr_db, cfg.hadamard_size,
        )
        w_h_to_fp8 = w_h_lattice
    elif cfg.method == "fp8_hadamard":
        # Hadamard + FP8 only, no lattice.
        w_h_to_fp8 = w_h
        n_scalars = O * I
        stats = QuantStats(n_scalars=n_scalars, total_bits=8 * n_scalars)
    else:
        raise ValueError(f"Unknown method {cfg.method!r}")

    # Per-tile FP8 simulation along the last dim (still Hadamard-domain).
    w_h_fp8 = simulate_per_tile_fp8(w_h_to_fp8, cfg.weight_tile_size)

    # Store W' / N so the unmodified nn.Linear forward gives X' · W'ᵀ / N.
    weight_to_store = (w_h_fp8 / cfg.hadamard_size).to(orig_dtype)

    # For SNR measurement: the bf16-equivalent of the pseudo-quantized W in
    # the original basis is inverse_hadamard(W'_fp8) / N (= H · W'_fp8 / N).
    w_effective = (chunked_hadamard(w_h_fp8, cfg.hadamard_size)
                   / cfg.hadamard_size).to(orig_dtype)

    return weight_to_store, stats, w_effective


# ─────────────────────────────────────────────────────────────────────────────
# Activation-side: pre-hook that applies Hadamard + per-tile FP8
# ─────────────────────────────────────────────────────────────────────────────

def make_activation_prehook(cfg: Fp8LatticeConfig):
    """Build a forward-pre-hook that applies (Hadamard?) + per-tile FP8 to the
    first positional input. For ``method == "fp8_only"`` only the FP8 cast is
    applied (no Hadamard).
    """
    hadamard_size = cfg.hadamard_size
    act_tile_size = cfg.act_tile_size
    do_hadamard = cfg.apply_hadamard and cfg.method != "fp8_only"

    def hook(_module: nn.Module, inputs: tuple):
        x = inputs[0]
        if do_hadamard:
            x = chunked_hadamard(x, hadamard_size)
        x = simulate_per_tile_fp8(x, act_tile_size)
        return (x,) + inputs[1:]

    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Model-wide installer
# ─────────────────────────────────────────────────────────────────────────────

def install_fp8_path(
    model: nn.Module,
    cfg: Fp8LatticeConfig,
    skip_module_names: tuple[str, ...] = ("lm_head",),
    *,
    measure_snr: bool = False,
) -> dict:
    """Apply the FP8 + (lattice + Hadamard) scheme in-place over every
    ``nn.Linear`` of ``model``. Returns a stats dict (bits/scalar, per-layer
    SNR) and the list of installed hook handles.

    Caller should keep the returned ``hook_handles`` and call ``.remove()``
    on each before installing a different config.
    """
    total_bits = 0
    total_scalars = 0
    n_quantized = 0
    per_layer: list[dict] = []
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []
    snr_db_list: list[float] = []

    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if any(skip in name for skip in skip_module_names):
            continue

        w_orig = module.weight.data
        if measure_snr:
            w_orig_keep = w_orig.detach().clone()

        with torch.no_grad():
            weight_to_store, stats, w_effective = quantize_weight_for_fp8_path(w_orig, cfg)
            module.weight.data = weight_to_store

        snr_db = math.nan
        if measure_snr:
            sig = w_orig_keep.float().pow(2).sum().item()
            noise = (w_orig_keep.float() - w_effective.float()).pow(2).sum().item()
            snr_db = 10.0 * math.log10(sig / noise) if noise > 0 else float("inf")
            snr_db_list.append(snr_db)
            del w_orig_keep

        handle = module.register_forward_pre_hook(make_activation_prehook(cfg))
        hook_handles.append(handle)

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
        "hook_handles": hook_handles,
    }
    if measure_snr and per_layer:
        N_total = sum(p["n_scalars"] for p in per_layer)
        wmean_lin = sum(
            (p["n_scalars"] / N_total) * (10 ** (p["snr_db"] / 10.0))
            for p in per_layer
        )
        snr_db_list_sorted = sorted(snr_db_list)
        out["snr_db_paramw"] = 10.0 * math.log10(wmean_lin)
        out["snr_db_median"] = snr_db_list_sorted[len(snr_db_list_sorted) // 2]
        out["snr_db_min"] = snr_db_list_sorted[0]
        out["snr_db_max"] = snr_db_list_sorted[-1]
    return out


def remove_hooks(hook_handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for h in hook_handles:
        h.remove()
