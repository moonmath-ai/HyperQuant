"""
Pseudo-FP8-MMA hooks for the attention path (Q, K, V projections).

Companion to ``weight_quant.install_fp8_path``, which already handles
every ``nn.Linear`` weight (lattice → in-Hadamard-domain → per-tile FP8) and
activation (RHT + per-tile FP8 via a forward pre-hook). What this module
adds: the *outputs* of ``q_proj`` / ``k_proj`` / ``v_proj`` get cast to FP8
too, so the two attention matmuls (``Q · K^T`` and ``softmax · V``) see
fp8-cast operands.

Per layer we install three forward hooks:

  • ``q_proj`` output:
        bf16 → per-tile FP8 (E4M3) cast back to bf16.

  • ``k_proj`` output (pre-RoPE):
        bf16
        → lattice + (optional) subtractive Voronoi dither pseudo-quantize
          along ``head_dim``  (see ``rotate_dither_lattice_pseudo_quantize``)
        → per-tile FP8 cast back to bf16.

  • ``v_proj`` output (pre-RoPE):
        bf16
        → lattice + (optional) subtractive Voronoi dither pseudo-quantize
          along ``head_dim``
        → per-tile FP8 cast back to bf16.

Why no Hadamard on Q and K, even though the user asked for "RHT and
per-tile scaling on the other input of the MMA":

  • Llama applies *RoPE* between ``q_proj`` / ``k_proj`` and the
    ``Q · K^T`` matmul. RoPE acts as a head-position-dependent rotation
    on each head's vector, and it does *not* commute with the
    head-local Hadamard. So even though
    ``(H_n Q) · (H_n K)^T = Q · K^T`` in isolation, the post-RoPE
    matmul becomes
    ``(R_i H_n Q_i) · (R_j H_n K_j)^T = Q_i · (H_n^T R_i^T R_j H_n) · K_j``
    ≠ ``Q_i · (R_i^T R_j) · K_j``                in general.
  • Empirically: applying H_n before RoPE on a Llama attention layer
    crashes the post-RoPE output SNR to ~10 dB (catastrophic) versus
    ~22 dB intrinsic — see ``_smoke_fp8_mma.TestLlamaAttentionFidelity``.

The right place to apply RHT to Q and K is *after* RoPE, immediately
before the ``Q · K^T`` matmul — which lives inside
``LlamaAttention.forward`` and is not reachable from a post-projection
forward hook. Achieving that requires monkey-patching the attention
function, which is intentionally out of scope here.

The same argument applies to V's "RHT" partner — the softmax output
is the other operand of the ``softmax · V`` MMA and is also produced
inside ``LlamaAttention.forward``, so V also stays Hadamard-free.

Net effect (hooks-only scope):

  • Q, K, V all get per-tile FP8 (the "per-tile scaling" half of
    "RHT + per-tile scaling"). The matmul Q · K^T (post-RoPE) and
    softmax · V are *exactly* invariant up to the per-tile FP8 noise
    on the indicated operands.
  • The Q, K side of the FP8 cast misses the noise-shaping benefit of
    a head-aligned RHT; in practice per-tile FP8 alone is already
    high-quality (FP8 has 4 mantissa bits and the per-128-tile scale
    saturates near max(|x|)).

Both attention matmuls happen inside ``LlamaAttention.forward`` in bf16
on our pseudo-quant pipeline; we just feed them bf16 tensors whose
*content* lives on the FP8 grid. The accumulator (bf16/fp32) and the
post-matmul softmax/temperature are kept at their original precision —
matching how a real FP8 tensor-core MMA writes a higher-precision
accumulator that the next op consumes.

This module is *additive*: install ``install_fp8_path`` first (for the
Linear MMAs), then ``install_fp8_kv_attn_path`` (for the attention
operand FP8 casts). Both return removable hook handles.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass

import torch
import torch.nn as nn

from .weight_quant import (
    MMA_DTYPES,
    chunked_hadamard,
    simulate_per_tile_quant,
)
from .kv_quant import (
    _generate_rotation_per_layer,
    rotate_dither_lattice_pseudo_quantize,
)


# ─────────────────────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Fp8KvAttnConfig:
    """Configuration for the 8-bit-MMA attention-operand hooks.

    Attributes:
        lattice: Lattice kind for KV pseudo-quantization (``"e8int"``,
            ``"d4int"``, ``"a2int"``, ``"z1int"``). Must divide ``head_dim``.
        snr_db: Calibration SNR for the KV lattice quantizer (use
            ``hyperquant.calibration.calibrate_lattice_bps_to_snr`` to convert from
            target bps).
        head_dim: Per-head vector length (= ``config.head_dim``).
        kv_rotation_kind: KV rotation applied *inside* the lattice
            quantizer. Same dispatch as :class:`KVQuantConfig.rotation_kind`:
              * ``"none"``  — no rotation (the "no bias correction" variant
                              if also ``kv_apply_dither=False``).
              * ``"signs"`` — random ±1 diagonal per ``(layer, k|v)``;
                              cheap cousin of QJL that empirically reduces
                              inner-product bias by ~10× without a full
                              orthogonal matmul.
              * ``"qjl"``   — Haar-uniform orthogonal rotation per
                              ``(layer, k|v)``.
        kv_apply_dither: Subtractive Voronoi dither inside the lattice
            quantizer (the *strictly-unbiased* variant; Schuchman /
            Zamir-Feder).
        kv_seed: Seed for the KV rotation generator (used only if
            ``kv_rotation_kind != "none"``).
        act_tile_size: Per-tile 8-bit scale tile along ``head_dim`` (must
            divide ``head_dim``). 128-D heads with 128 tile = one scale
            per head, which is the natural granularity.
        mma_dtype: Precision used for the per-tile cast of K, V, Q.
            One of :data:`weight_quant.MMA_DTYPES`:
              * ``"fp8_e4m3"`` — 8-bit FP4-mantissa, 3-exponent (default).
              * ``"int8"``    — 8-bit symmetric integer per-tile.
              * ``"nvfp4"``   — 4-bit FP E2M1 with FP8 E4M3 block scale
                                (block size = 16; Blackwell native).
              * ``"mxfp4"``   — 4-bit FP E2M1 with E8M0 block scale
                                (block size = 32; OCP MXFP4 spec).
            All four reach the same throughput floor on their target HW
            (FP8/INT8 ≈ 2× BF16 on H100; both FP4 variants ≈ 4× BF16 on
            Blackwell). PPL differs by quantization grid spacing.
        quantize_q: Cast q_proj output to 8-bit (per-tile).
        quantize_k: Lattice + 8-bit on the K projection output.
        quantize_v: Lattice + 8-bit on the V projection output.
    """

    lattice: str
    snr_db: float
    head_dim: int = 128
    kv_rotation_kind: str = "none"          # "none" | "signs" | "qjl"
    kv_apply_dither: bool = True
    kv_seed: int = 0
    act_tile_size: int = 128
    mma_dtype: str = "fp8_e4m3"             # "fp8_e4m3" | "int8"
    quantize_q: bool = True
    quantize_k: bool = True
    quantize_v: bool = True


# ─────────────────────────────────────────────────────────────────────────────
# Hadamard helpers (head-local, orthogonal, matmul-invariant)
# ─────────────────────────────────────────────────────────────────────────────


def _normalized_chunked_hadamard(x: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Apply ``H / sqrt(N)`` independently per ``chunk_size``-D block along
    the last dim. ``H_n^T H_n = I`` per block, so the matmul of two
    same-blocked tensors is invariant: ``(H_n a) · (H_n b)^T = a · b^T``.
    """
    return chunked_hadamard(x, chunk_size) / math.sqrt(chunk_size)


# ─────────────────────────────────────────────────────────────────────────────
# Per-projection hook factories
# ─────────────────────────────────────────────────────────────────────────────


def _record_time(accumulator: dict, role: str, elapsed_s: float) -> None:
    """Push (role, elapsed) into the shared accumulator."""
    accumulator["elapsed_s_total"] += elapsed_s
    accumulator["n_calls_total"] += 1
    role_dict = accumulator.setdefault("by_role", {})
    bucket = role_dict.setdefault(role, {"elapsed_s": 0.0, "n_calls": 0})
    bucket["elapsed_s"] += elapsed_s
    bucket["n_calls"] += 1


def _timed(x_is_cuda: bool):
    """Context-manager-like helper returning (start_fn, stop_fn). ``stop_fn``
    returns the elapsed seconds. On CUDA we use ``cuda.Event``; on CPU we
    use ``time.perf_counter``.
    """
    if x_is_cuda:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        def stop() -> float:
            end.record()
            torch.cuda.synchronize()
            return start.elapsed_time(end) * 1e-3
        return stop

    t0 = time.perf_counter()
    return lambda: time.perf_counter() - t0


def _make_q_proj_fp8_hook(
    *,
    n_heads: int,
    head_dim: int,
    act_tile_size: int,
    mma_dtype: str,
    accumulator: dict,
):
    """Hook on ``q_proj``: cast its output Q to 8-bit (``mma_dtype``) with
    per-head per-tile scaling. No lattice on Q and *no* head-local
    Hadamard — the latter would not commute with the RoPE that runs
    between this hook and the ``Q · K^T`` matmul.
    """
    def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
        B, T, D = output.shape
        if D != n_heads * head_dim:
            raise RuntimeError(
                f"q_proj output {tuple(output.shape)} doesn't match "
                f"n_heads*head_dim = {n_heads}*{head_dim} = {n_heads * head_dim}"
            )

        stop = _timed(output.is_cuda)

        x = output.view(B, T, n_heads, head_dim)
        x_q = simulate_per_tile_quant(x, act_tile_size, mma_dtype)
        out = x_q.view(B, T, D)

        _record_time(accumulator, "q", stop())
        return out

    return hook


def _make_k_proj_fp8_hook(
    *,
    n_kv_heads: int,
    head_dim: int,
    lattice: str,
    snr_db: float,
    kv_rotation_kind: str,
    kv_rotation_tensor: torch.Tensor | None,
    kv_apply_dither: bool,
    act_tile_size: int,
    mma_dtype: str,
    accumulator: dict,
):
    """Hook on ``k_proj``: lattice + (rotation/dither) → per-tile 8-bit
    cast (no head-local Hadamard; see module docstring for why).
    """
    def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
        B, T, D = output.shape
        if D != n_kv_heads * head_dim:
            raise RuntimeError(
                f"k_proj output {tuple(output.shape)} doesn't match "
                f"n_kv_heads*head_dim = {n_kv_heads}*{head_dim} = {n_kv_heads * head_dim}"
            )

        stop = _timed(output.is_cuda)

        x = output.view(B, T, n_kv_heads, head_dim)

        x_lq, stats = rotate_dither_lattice_pseudo_quantize(
            x,
            rotation_kind=kv_rotation_kind,
            rotation_tensor=kv_rotation_tensor,
            apply_dither=kv_apply_dither,
            lattice=lattice,
            snr_db=snr_db,
        )

        x_q = simulate_per_tile_quant(x_lq, act_tile_size, mma_dtype)

        accumulator["kv_total_bits"] += int(stats.total_bits)
        accumulator["kv_n_scalars"] += int(stats.n_scalars)

        out = x_q.view(B, T, D)
        _record_time(accumulator, "k", stop())
        return out

    return hook


def _make_v_proj_fp8_hook(
    *,
    n_kv_heads: int,
    head_dim: int,
    lattice: str,
    snr_db: float,
    kv_rotation_kind: str,
    kv_rotation_tensor: torch.Tensor | None,
    kv_apply_dither: bool,
    act_tile_size: int,
    mma_dtype: str,
    accumulator: dict,
):
    """Hook on ``v_proj``: lattice + (rotation/dither) → per-tile 8-bit cast.
    No RHT (the ``softmax · V`` MMA's softmax-side operand is not
    rotated, and hooks can't reach it).
    """
    def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
        B, T, D = output.shape
        if D != n_kv_heads * head_dim:
            raise RuntimeError(
                f"v_proj output {tuple(output.shape)} doesn't match "
                f"n_kv_heads*head_dim = {n_kv_heads}*{head_dim} = {n_kv_heads * head_dim}"
            )

        stop = _timed(output.is_cuda)

        x = output.view(B, T, n_kv_heads, head_dim)

        x_lq, stats = rotate_dither_lattice_pseudo_quantize(
            x,
            rotation_kind=kv_rotation_kind,
            rotation_tensor=kv_rotation_tensor,
            apply_dither=kv_apply_dither,
            lattice=lattice,
            snr_db=snr_db,
        )

        x_q = simulate_per_tile_quant(x_lq, act_tile_size, mma_dtype)

        accumulator["kv_total_bits"] += int(stats.total_bits)
        accumulator["kv_n_scalars"] += int(stats.n_scalars)

        out = x_q.view(B, T, D)
        _record_time(accumulator, "v", stop())
        return out

    return hook


# ─────────────────────────────────────────────────────────────────────────────
# Model-wide installer
# ─────────────────────────────────────────────────────────────────────────────


def install_fp8_kv_attn_path(
    model: nn.Module,
    cfg: Fp8KvAttnConfig,
) -> dict:
    """Install Q/K/V FP8-cast hooks on every self-attention module of
    ``model``. Designed to be installed *on top of* ``install_fp8_path``
    (which handles the nn.Linear weight+activation FP8 path).

    Returns a dict:

        {
          "n_attention_layers": int,
          "hook_handles": list[RemovableHandle],
          "rotations": {layer_name: {"k": Tensor|None, "v": Tensor|None}},
          "memory_bytes_rotations": int,
          "accumulator": {
              "kv_total_bits": int,
              "kv_n_scalars": int,
              "elapsed_s_total": float,
              "n_calls_total": int,
              "by_role": {"q": {...}, "k": {...}, "v": {...}},
          },
        }
    """
    if cfg.head_dim % cfg.act_tile_size != 0 and cfg.act_tile_size > cfg.head_dim:
        raise ValueError(
            f"act_tile_size={cfg.act_tile_size} must divide head_dim={cfg.head_dim}"
        )
    if cfg.act_tile_size > cfg.head_dim:
        raise ValueError(
            f"act_tile_size {cfg.act_tile_size} > head_dim {cfg.head_dim}"
        )
    if cfg.mma_dtype not in MMA_DTYPES:
        raise ValueError(
            f"mma_dtype={cfg.mma_dtype!r} not in {MMA_DTYPES}"
        )

    handles: list[torch.utils.hooks.RemovableHandle] = []
    rotations: dict[str, dict[str, torch.Tensor | None]] = {}
    accumulator: dict = {
        "kv_total_bits": 0,
        "kv_n_scalars": 0,
        "elapsed_s_total": 0.0,
        "n_calls_total": 0,
        "by_role": {},
    }
    memory_bytes_rotations = 0

    n_layers = 0
    for name, module in model.named_modules():
        if not (
            hasattr(module, "k_proj") and hasattr(module, "v_proj")
            and hasattr(module, "q_proj")
            and isinstance(module.k_proj, nn.Linear)
            and isinstance(module.v_proj, nn.Linear)
            and isinstance(module.q_proj, nn.Linear)
        ):
            continue

        n_layers += 1
        head_dim = getattr(module, "head_dim", cfg.head_dim) or cfg.head_dim
        n_heads = module.q_proj.out_features // head_dim
        n_kv_heads = module.k_proj.out_features // head_dim
        device = module.k_proj.weight.device
        dtype = module.k_proj.weight.dtype

        # Per-layer KV rotation (used inside the lattice quantizer, not for
        # the RHT — that's a separate normalized Hadamard with no random
        # state).
        rotations[name] = {"k": None, "v": None}
        if cfg.kv_rotation_kind != "none":
            for role, do_role, seed_offset in (
                ("k", cfg.quantize_k, 2 * (n_layers - 1)),
                ("v", cfg.quantize_v, 2 * (n_layers - 1) + 1),
            ):
                if not do_role:
                    continue
                S = _generate_rotation_per_layer(
                    cfg.kv_rotation_kind, head_dim, cfg.kv_seed + seed_offset,
                    device=device, dtype=dtype,
                )
                rotations[name][role] = S
                if S is not None:
                    memory_bytes_rotations += S.numel() * S.element_size()

        if cfg.quantize_q:
            h_q = module.q_proj.register_forward_hook(
                _make_q_proj_fp8_hook(
                    n_heads=n_heads,
                    head_dim=head_dim,
                    act_tile_size=cfg.act_tile_size,
                    mma_dtype=cfg.mma_dtype,
                    accumulator=accumulator,
                )
            )
            handles.append(h_q)

        if cfg.quantize_k:
            h_k = module.k_proj.register_forward_hook(
                _make_k_proj_fp8_hook(
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    lattice=cfg.lattice,
                    snr_db=cfg.snr_db,
                    kv_rotation_kind=cfg.kv_rotation_kind,
                    kv_rotation_tensor=rotations[name]["k"],
                    kv_apply_dither=cfg.kv_apply_dither,
                    act_tile_size=cfg.act_tile_size,
                    mma_dtype=cfg.mma_dtype,
                    accumulator=accumulator,
                )
            )
            handles.append(h_k)

        if cfg.quantize_v:
            h_v = module.v_proj.register_forward_hook(
                _make_v_proj_fp8_hook(
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    lattice=cfg.lattice,
                    snr_db=cfg.snr_db,
                    kv_rotation_kind=cfg.kv_rotation_kind,
                    kv_rotation_tensor=rotations[name]["v"],
                    kv_apply_dither=cfg.kv_apply_dither,
                    act_tile_size=cfg.act_tile_size,
                    mma_dtype=cfg.mma_dtype,
                    accumulator=accumulator,
                )
            )
            handles.append(h_v)

    return {
        "n_attention_layers": n_layers,
        "hook_handles": handles,
        "rotations": rotations,
        "memory_bytes_rotations": memory_bytes_rotations,
        "accumulator": accumulator,
    }


def remove_fp8_kv_attn_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Convenience helpers
# ─────────────────────────────────────────────────────────────────────────────


def kv_bits_per_scalar(install_stats: dict) -> float:
    """Live KV bps from the hook accumulator (uses the lattice Rice bit
    accountant for K and V only). The FP8 cast that follows lattice
    dequant does not change the *stored* KV form — the lattice codes
    remain the cached representation; we just materialize them as FP8
    for the MMA. So the headline KV bps is the lattice bps.
    """
    acc = install_stats.get("accumulator", {})
    n = acc.get("kv_n_scalars", 0)
    if n == 0:
        return 0.0
    return acc["kv_total_bits"] / n


def hook_overhead_ms_per_token(install_stats: dict, n_tokens: int) -> float:
    """Total wall-clock spent inside Q/K/V FP8 hooks, normalized per scored
    token. ``n_tokens`` should equal the number of tokens that went
    through the model since the accumulator was last reset (i.e. across
    the whole PPL eval).
    """
    acc = install_stats.get("accumulator", {})
    if n_tokens <= 0:
        return 0.0
    return acc.get("elapsed_s_total", 0.0) * 1e3 / n_tokens


def reset_accumulator(install_stats: dict) -> None:
    """Zero out the per-hook accumulator (use before a timed eval)."""
    acc = install_stats.get("accumulator")
    if acc is None:
        return
    acc["kv_total_bits"] = 0
    acc["kv_n_scalars"] = 0
    acc["elapsed_s_total"] = 0.0
    acc["n_calls_total"] = 0
    for role_bucket in acc.get("by_role", {}).values():
        role_bucket["elapsed_s"] = 0.0
        role_bucket["n_calls"] = 0
