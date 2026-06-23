"""
QJL-style random rotation + lattice quantization, installed as forward hooks
on Llama's per-layer ``k_proj`` / ``v_proj`` to simulate KV-cache
pseudo-quantization.

Scheme (per cached vector ``x ∈ ℝ^{head_dim}`` at one (head, token) position):

    x ──S──► S·x  ──lattice quantize per n_dim-D block──► Q(S·x)
                                                          │
    x_hat = Sᵀ · Q(S·x)  ◄──────────────────────────────  │

with ``S ∈ ℝ^{head_dim×head_dim}`` an orthogonal matrix drawn once per
(layer, k|v) by Gram–Schmidting a Gaussian (``torch.linalg.qr``).

Because ``S`` is orthogonal, ``Sᵀ S = I`` and:

  • the quantization error ``e = x_hat − x = Sᵀ (Q(Sx) − Sx)`` has the same
    norm as the lattice error in the rotated space, but its *direction* is
    randomized by ``S``;
  • for any query ``q``, ``⟨q, x_hat⟩ − ⟨q, x⟩ = ⟨q, e⟩`` is approximately
    zero-mean across realizations of ``S`` and across positions, which is
    the "unbiased-under-inner-product" property motivating QJL.

We hook on the *output* of ``k_proj`` / ``v_proj``, i.e. **pre-RoPE**. RoPE
is a rotation that preserves L2 norms, so it preserves the lattice-quant
error norm; pre-RoPE quantization is therefore SNR-equivalent to post-RoPE
quantization and is much simpler to install (no monkey-patching of
``LlamaAttention.forward``).

Both K and V can be quantized independently; the QJL rotation is the same
mechanism but the rotation matrices are independent per (layer, role) so
the K and V error distributions are uncorrelated.

This is a *pseudo*-quantizer: it always returns a dequantized bf16 tensor
in the original ``[B, T, n_kv_heads*head_dim]`` shape. No KV-cache memory
savings — it only models the quantization noise the real cached values
would carry.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn

from .quant_utils import (
    QuantStats,
    _LATTICE_INFO,
    _rice_bits_for_codes,
    lattice_alpha,
)
from .lattice import SQRT3


# ─────────────────────────────────────────────────────────────────────────────
# Config + JL rotation generator
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class KVQuantConfig:
    """Configuration for KV-cache pseudo-quantization with optional random
    rotation and/or subtractive dither.

    Attributes:
        lattice: One of ``"e8int"``, ``"d4int"``, ``"a2int"``, ``"z1int"``.
        snr_db: Per-block calibration SNR (dB) for the lattice quantizer.
            Use ``hyperquant.calibration.calibrate_lattice_bps_to_snr`` to map a
            target bits/scalar to a calibration SNR.
        head_dim: Per-head vector length (= ``config.head_dim``). Must be
            divisible by the lattice dimension (``8`` for E8, ``4`` for D4,
            ``2`` for A2, ``1`` for Z1).
        seed: Master seed for the per-layer rotation generator. Layer ``i``'s
            K rotation uses ``seed + 2*i``, V rotation ``seed + 2*i + 1``.
        quantize_k: Quantize K projection outputs.
        quantize_v: Quantize V projection outputs.
        rotation_kind: One of:
            * ``"none"`` — no rotation; lattice quantizes ``x`` directly.
            * ``"signs"`` — random ±1 diagonal per ``(layer, k|v)``; cheap
              cousin of QJL (O(n) instead of O(n²) per vector). Same
              statistical role as QJL on permutation-symmetric inputs.
            * ``"qjl"`` — Haar-uniform orthogonal rotation per ``(layer, k|v)``.
        apply_dither: If True, add a subtractive dither uniform on the
            integer-lattice Voronoi cell before quantization and subtract
            it back after. Gives strict per-vector ``E[⟨q, e⟩] = 0``
            (Schuchman / Zamir-Feder), at the cost of ~3 dB extra noise
            variance.
        dither_seed: Optional. Seed for a deterministic dither stream.
            ``None`` (default) draws from the live PyTorch RNG so each
            forward pass gets independent dither (correct for our per-
            position unbiasedness story).
        target_bps: Optional. Just metadata for logging — does not affect
            quantization (the SNR controls the rate).
    """

    lattice: str
    snr_db: float
    head_dim: int = 128
    seed: int = 0
    quantize_k: bool = True
    quantize_v: bool = True
    rotation_kind: str = "qjl"          # "none" | "signs" | "qjl"
    apply_dither: bool = False
    dither_seed: int | None = None
    target_bps: float | None = None


def generate_jl_rotation(
    dim: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a strict orthogonal matrix ``S ∈ ℝ^{dim×dim}``.

    Built as the ``Q`` factor of ``QR(randn(dim, dim))`` with sign-adjustment
    on the diagonal so the resulting distribution is the Haar measure on
    O(dim). Deterministic given ``seed``.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    a = torch.randn(dim, dim, generator=g, dtype=torch.float32)
    q, r = torch.linalg.qr(a)
    # Sign-correct so the distribution is Haar-uniform on O(n).
    d = torch.sign(torch.diagonal(r))
    d[d == 0] = 1.0
    q = q * d.unsqueeze(0)
    return q.to(device=device, dtype=dtype)


def generate_sign_rotation(
    dim: int,
    seed: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return a random ``±1`` diagonal of length ``dim``, as a 1-D tensor.

    Applying this "rotation" is an O(n) element-wise multiply rather than
    QJL's O(n²) matmul. It is *its own inverse*, so the same vector is
    used for forward and back rotation in
    :func:`rotate_dither_lattice_pseudo_quantize`.
    """
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    signs = (torch.rand(dim, generator=g) > 0.5).to(torch.float32) * 2.0 - 1.0
    return signs.to(device=device, dtype=dtype)


def _apply_rotation(
    x: torch.Tensor,
    rotation_kind: str,
    rotation_tensor: torch.Tensor | None,
    *,
    inverse: bool,
) -> torch.Tensor:
    """Apply (or invert) the rotation along ``x.shape[-1]``.

    Conventions:
    * ``"none"`` — identity, regardless of ``inverse``.
    * ``"signs"`` — element-wise multiply by ``rotation_tensor`` of shape
      ``(head_dim,)``. The sign diag is its own inverse, so forward and
      back use the same op.
    * ``"qjl"`` — forward rotates via ``x @ Sᵀ`` (so the lattice sees the
      rotated frame); inverse rotates back via ``x @ S``.
    """
    if rotation_kind == "none" or rotation_tensor is None:
        return x
    if rotation_kind == "signs":
        # rotation_tensor shape: (head_dim,)
        return x * rotation_tensor
    if rotation_kind == "qjl":
        # rotation_tensor shape: (head_dim, head_dim)
        rot = rotation_tensor.to(dtype=x.dtype, device=x.device)
        return x @ (rot if inverse else rot.transpose(0, 1))
    raise ValueError(f"unknown rotation_kind {rotation_kind!r}")


# ─────────────────────────────────────────────────────────────────────────────
# Voronoi-cell dither (for subtractive dithering of the lattice quantizer)
# ─────────────────────────────────────────────────────────────────────────────


def voronoi_uniform_dither(
    leading_shape: tuple[int, ...],
    lattice: str,
    *,
    device: torch.device | str,
    dtype: torch.dtype = torch.float32,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample dither uniformly on the Voronoi cell of the integer lattice.

    Output shape: ``leading_shape + (n_dim,)``. By symmetry of the Voronoi
    cell of any lattice (every cell is centrally symmetric), the resulting
    tensor satisfies ``E[u] = 0`` exactly, which is the property that
    Schuchman / Zamir-Feder subtractive dither needs for strict
    inner-product unbiasedness.

    Implementation — the standard "mod-Λ" recipe:

      1. Sample ``u' ~ Uniform([-2, 2)^n)`` (cube wider than the lattice's
         covering radius for all four supported lattices).
      2. Quantize: ``λ = Q_Λ(u')``.
      3. Return ``u = u' − λ``.

    Step 3 is the projection onto the Voronoi cell of ``Λ``; the wide
    starting cube guarantees the distribution after projection is uniform
    on the cell. The result lives in the *same coordinate system* as the
    lattice quantizer's input (i.e. the α-scaled space).
    """
    n_dim, _, quantize_fn = _LATTICE_INFO[lattice]
    shape = tuple(leading_shape) + (n_dim,)

    if generator is None:
        u_cube = (torch.rand(shape, device=device, dtype=torch.float32) - 0.5) * 4.0
    else:
        # Generators are CPU-only in many torch builds; sample on CPU then move.
        u_cube_cpu = (
            torch.rand(shape, dtype=torch.float32, generator=generator) - 0.5
        ) * 4.0
        u_cube = u_cube_cpu.to(device=device)

    # Mod-lattice projection.
    codes = quantize_fn(u_cube.to(torch.bfloat16))
    if lattice == "a2int":
        recon = torch.stack(
            [codes[..., 0].float() * SQRT3, codes[..., 1].float()], dim=-1
        )
    else:
        recon = codes.float()
    u = u_cube - recon
    return u.to(dtype)


# ─────────────────────────────────────────────────────────────────────────────
# Core: (optional rotation) + (optional dither) + lattice pseudo-quantization
# Applied along the last dim (= head_dim) of an arbitrary-shape tensor.
# ─────────────────────────────────────────────────────────────────────────────


def rotate_dither_lattice_pseudo_quantize(
    x: torch.Tensor,
    *,
    rotation_kind: str,
    rotation_tensor: torch.Tensor | None,
    apply_dither: bool,
    lattice: str,
    snr_db: float,
    dither_generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, QuantStats]:
    """Pseudo-quantize one ``[..., head_dim]`` tensor.

    Pipeline (steps marked ``[opt]`` are skipped under the corresponding flag):

      1. ``[opt]`` rotate last dim by ``rotation_tensor`` (kind dispatch).
      2. Per-vector L2 normalize, then multiply by ``√head_dim`` so per-
         scalar variance ≈ 1 (matches ``lattice_alpha``'s calibration).
      3. Multiply by ``α`` to bring into the lattice's integer scale.
      4. ``[opt]`` add subtractive dither ``u`` uniform on the integer-lattice
         Voronoi cell (mean 0).
      5. Round to nearest lattice point.
      6. ``[opt]`` subtract dither ``u``.
      7. Divide by ``α``, then re-multiply by ``norm / √head_dim``.
      8. ``[opt]`` inverse-rotate.

    Returns ``(x_hat, QuantStats)`` where ``x_hat`` matches ``x.shape`` and
    ``x.dtype`` and ``QuantStats.total_bits`` is the Rice-coded bit budget
    for the integer lattice codes (independent of rotation and dither).
    """
    head_dim = x.shape[-1]
    n_dim, _, quantize_fn = _LATTICE_INFO[lattice]
    if head_dim % n_dim != 0:
        raise ValueError(
            f"head_dim {head_dim} must be divisible by lattice dim {n_dim}"
        )
    if rotation_kind not in ("none", "signs", "qjl"):
        raise ValueError(f"unknown rotation_kind {rotation_kind!r}")
    if rotation_kind != "none" and rotation_tensor is None:
        raise ValueError(
            f"rotation_kind={rotation_kind!r} requires a rotation_tensor"
        )

    alpha = lattice_alpha(snr_db, lattice)
    orig_dtype = x.dtype

    # Promote to fp32 for rotation + norm + lattice math.
    x_fp32 = x.to(torch.float32)

    # 1. Rotate.
    x_rot = _apply_rotation(x_fp32, rotation_kind, rotation_tensor, inverse=False)

    # 2. L2-normalize per (..., head_dim); per-scalar variance ≈ 1.
    norms = torch.linalg.norm(x_rot, dim=-1, keepdim=True).clamp(min=1e-12)
    u_unit = (x_rot / norms) * math.sqrt(head_dim)

    # 3. Bring to the lattice's integer scale.
    flat = u_unit.reshape(-1, n_dim)
    scaled = alpha * flat                                          # (N, n_dim)

    # 4. Optional subtractive dither (uniform on V_Λ_int, mean 0).
    if apply_dither:
        dither = voronoi_uniform_dither(
            leading_shape=(scaled.shape[0],),
            lattice=lattice,
            device=scaled.device,
            dtype=torch.float32,
            generator=dither_generator,
        )
        scaled_dithered = scaled + dither
    else:
        dither = None
        scaled_dithered = scaled

    # 5. Lattice-quantize (cast through bf16 to match the calibration regime).
    codes = quantize_fn(scaled_dithered.to(torch.bfloat16))

    # 6. Reconstruct, then subtract dither in lattice-integer scale (A2 has
    #    a √3 stretch on the first axis).
    if lattice == "a2int":
        recon = torch.stack(
            [codes[:, 0].float() * SQRT3, codes[:, 1].float()], dim=1
        )
    else:
        recon = codes.float()
    if apply_dither:
        recon = recon - dither

    # 7. Undo α-scale, undo normalization.
    u_unit_hat = (recon / alpha).reshape_as(u_unit)
    x_rot_hat = (u_unit_hat / math.sqrt(head_dim)) * norms

    # 8. Inverse rotation (signs is its own inverse; qjl uses S; none = id).
    x_hat = _apply_rotation(x_rot_hat, rotation_kind, rotation_tensor, inverse=True)

    n_scalars = x.numel()
    bits = _rice_bits_for_codes(codes, lattice)
    return x_hat.to(orig_dtype), QuantStats(
        n_scalars=n_scalars, total_bits=int(bits)
    )


def qjl_lattice_pseudo_quantize(
    x: torch.Tensor,
    rotation: torch.Tensor,
    lattice: str,
    snr_db: float,
) -> tuple[torch.Tensor, QuantStats]:
    """Backward-compatible thin wrapper for QJL-only (no dither).

    Equivalent to::

        rotate_dither_lattice_pseudo_quantize(
            x, rotation_kind="qjl", rotation_tensor=rotation,
            apply_dither=False, lattice=lattice, snr_db=snr_db,
        )
    """
    if x.shape[-1] != rotation.shape[0] or rotation.shape[0] != rotation.shape[1]:
        raise ValueError(
            f"rotation shape {tuple(rotation.shape)} incompatible with x last dim {x.shape[-1]}"
        )
    return rotate_dither_lattice_pseudo_quantize(
        x,
        rotation_kind="qjl",
        rotation_tensor=rotation,
        apply_dither=False,
        lattice=lattice,
        snr_db=snr_db,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Hook factory + model-wide installer
# ─────────────────────────────────────────────────────────────────────────────


def _make_kv_projection_hook(
    rotation_kind: str,
    rotation_tensor: torch.Tensor | None,
    apply_dither: bool,
    n_kv_heads: int,
    head_dim: int,
    lattice: str,
    snr_db: float,
    accumulator: dict,
):
    """Return a ``register_forward_hook`` callable for a ``k_proj`` / ``v_proj``.

    The hook receives the projection output (shape
    ``[B, T, n_kv_heads * head_dim]``), reshapes to
    ``[B, T, n_kv_heads, head_dim]``, applies the configured
    rotation + (optional) dither + lattice pseudo-quantization along the
    head dim, then reshapes back. It accumulates per-call ``total_bits``,
    ``n_scalars``, and ``elapsed_s`` into ``accumulator`` so the installer
    can report measured bps and per-forward overhead.
    """

    def hook(_module: nn.Module, _inputs, output: torch.Tensor) -> torch.Tensor:
        B, T, D = output.shape
        if D != n_kv_heads * head_dim:
            raise RuntimeError(
                f"projection output {tuple(output.shape)} doesn't match "
                f"({n_kv_heads}, {head_dim})"
            )
        x = output.view(B, T, n_kv_heads, head_dim)

        # Accumulate GPU wall-time using CUDA events when available, else
        # CPU-side time.perf_counter (the bulk of work is on-device).
        if x.is_cuda:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
        else:
            import time
            t0 = time.perf_counter()

        x_hat, stats = rotate_dither_lattice_pseudo_quantize(
            x,
            rotation_kind=rotation_kind,
            rotation_tensor=rotation_tensor,
            apply_dither=apply_dither,
            lattice=lattice,
            snr_db=snr_db,
        )

        if x.is_cuda:
            end.record()
            torch.cuda.synchronize()
            elapsed_s = start.elapsed_time(end) * 1e-3
        else:
            elapsed_s = time.perf_counter() - t0

        accumulator["total_bits"] += int(stats.total_bits)
        accumulator["n_scalars"] += int(stats.n_scalars)
        accumulator["elapsed_s"] += elapsed_s
        accumulator["n_calls"] += 1
        return x_hat.view(B, T, D)

    return hook


def _generate_rotation_per_layer(
    rotation_kind: str,
    head_dim: int,
    seed: int,
    *,
    device,
    dtype,
) -> torch.Tensor | None:
    if rotation_kind == "none":
        return None
    if rotation_kind == "signs":
        return generate_sign_rotation(head_dim, seed, device=device, dtype=dtype)
    if rotation_kind == "qjl":
        return generate_jl_rotation(head_dim, seed, device=device, dtype=dtype)
    raise ValueError(f"unknown rotation_kind {rotation_kind!r}")


def install_kv_cache_quant_path(
    model: nn.Module,
    cfg: KVQuantConfig,
    *,
    measure_snr: bool = False,
) -> dict:
    """Install lattice pseudo-quantization hooks on every Llama
    self-attention's ``k_proj`` / ``v_proj``.

    For each attention layer we draw two rotations (one each for K and V),
    seeded from ``cfg.seed`` + layer index, of the kind requested by
    ``cfg.rotation_kind`` (``"none"`` ⇒ no rotation, ``"signs"`` ⇒ ±1
    diagonal, ``"qjl"`` ⇒ Haar-uniform orthogonal). If ``cfg.apply_dither``
    is True the hooks additionally Voronoi-cell dither each cached vector
    before quantization and subtract the dither after.

    Returns a dict:

        {
          "n_attention_layers": int,
          "n_hooks": int,
          "hook_handles": list[RemovableHandle],
          "rotations": {layer_name: {"k": Tensor|None, "v": Tensor|None}, ...},
          "accumulator": {"total_bits": int, "n_scalars": int,
                          "elapsed_s": float, "n_calls": int},
          "memory_bytes_rotations": int,
          "bits_per_scalar": float,    # convenience getter; see helper
          "snr_db_paramw": float,      # if measure_snr=True
        }
    """
    handles: list[torch.utils.hooks.RemovableHandle] = []
    rotations: dict[str, dict[str, torch.Tensor | None]] = {}
    accumulator: dict = {
        "total_bits": 0, "n_scalars": 0,
        "elapsed_s": 0.0, "n_calls": 0,
    }
    snr_db_list: list[float] = []
    memory_bytes_rotations = 0

    n_layers = 0
    n_hooks = 0
    for name, module in model.named_modules():
        # Heuristic: anything that exposes k_proj + v_proj + q_proj is a
        # self-attention module (works for LlamaAttention as well as any
        # GQA-shaped attention with the same projection names).
        if not (
            hasattr(module, "k_proj")
            and hasattr(module, "v_proj")
            and isinstance(module.k_proj, nn.Linear)
            and isinstance(module.v_proj, nn.Linear)
        ):
            continue

        n_layers += 1
        head_dim = getattr(module, "head_dim", cfg.head_dim) or cfg.head_dim
        n_kv_heads = module.k_proj.out_features // head_dim
        device = module.k_proj.weight.device
        dtype = module.k_proj.weight.dtype

        rotations[name] = {"k": None, "v": None}

        for role, do, seed_offset, proj in (
            ("k", cfg.quantize_k, 2 * (n_layers - 1),     module.k_proj),
            ("v", cfg.quantize_v, 2 * (n_layers - 1) + 1, module.v_proj),
        ):
            if not do:
                continue
            S = _generate_rotation_per_layer(
                cfg.rotation_kind, head_dim, cfg.seed + seed_offset,
                device=device, dtype=dtype,
            )
            rotations[name][role] = S
            if S is not None:
                memory_bytes_rotations += S.numel() * S.element_size()
            handle = proj.register_forward_hook(
                _make_kv_projection_hook(
                    rotation_kind=cfg.rotation_kind,
                    rotation_tensor=S,
                    apply_dither=cfg.apply_dither,
                    n_kv_heads=n_kv_heads,
                    head_dim=head_dim,
                    lattice=cfg.lattice,
                    snr_db=cfg.snr_db,
                    accumulator=accumulator,
                )
            )
            handles.append(handle)
            n_hooks += 1

        if measure_snr:
            # Synthetic-Gaussian intrinsic SNR estimate, using this layer's
            # actual rotation (whichever of K/V is installed).
            rot_for_snr = rotations[name]["k"]
            if rot_for_snr is None:
                rot_for_snr = rotations[name]["v"]
            snr_db_list.append(
                _measure_synthetic_snr(
                    head_dim=head_dim,
                    lattice=cfg.lattice,
                    snr_db=cfg.snr_db,
                    rotation_kind=cfg.rotation_kind,
                    rotation_tensor=rot_for_snr,
                    apply_dither=cfg.apply_dither,
                    device=device,
                )
            )

    out: dict = {
        "n_attention_layers": n_layers,
        "n_hooks": n_hooks,
        "hook_handles": handles,
        "rotations": rotations,
        "accumulator": accumulator,
        # Backward-compat alias used by old callers.
        "bps_accumulator": accumulator,
        "memory_bytes_rotations": memory_bytes_rotations,
        "bits_per_scalar": 0.0,
    }
    if measure_snr and snr_db_list:
        out["snr_db_paramw"] = sum(snr_db_list) / len(snr_db_list)
        out["snr_db_min"] = min(snr_db_list)
        out["snr_db_max"] = max(snr_db_list)
    return out


def measured_bits_per_scalar(install_stats: dict) -> float:
    """Compute live bits/scalar from the hook accumulator after some forwards."""
    acc = install_stats.get("accumulator") or install_stats.get("bps_accumulator")
    if acc is None or acc.get("n_scalars", 0) == 0:
        return 0.0
    return acc["total_bits"] / acc["n_scalars"]


def measured_overhead_ms_per_call(install_stats: dict) -> float:
    """Mean wall-clock spent inside the K/V projection hooks (ms per call).

    "Per call" means per `(layer, k|v)` projection invocation — i.e. per
    one of the ``n_hooks`` projections per forward.
    """
    acc = install_stats.get("accumulator") or install_stats.get("bps_accumulator")
    if acc is None or acc.get("n_calls", 0) == 0:
        return 0.0
    return acc["elapsed_s"] / acc["n_calls"] * 1e3


def remove_kv_hooks(handles: list[torch.utils.hooks.RemovableHandle]) -> None:
    for h in handles:
        h.remove()


# ─────────────────────────────────────────────────────────────────────────────
# Intrinsic SNR estimator (for ``measure_snr=True``)
# ─────────────────────────────────────────────────────────────────────────────


def _measure_synthetic_snr(
    *,
    head_dim: int,
    lattice: str,
    snr_db: float,
    rotation_kind: str = "qjl",
    rotation_tensor: torch.Tensor | None = None,
    apply_dither: bool = False,
    n_samples: int = 8192,
    device: torch.device | str = "cuda",
) -> float:
    """Empirical 10·log10(‖x‖² / ‖x − x_hat‖²) (dB) over ``n_samples``
    iid-Gaussian vectors of length ``head_dim``.

    Driven by the same RNG every call (deterministic), so the SNR returned
    is repeatable per (rotation_kind, lattice, snr_db, apply_dither) config.
    """
    g = torch.Generator(device="cpu").manual_seed(0xDEADBEEF)
    x = torch.randn(n_samples, head_dim, generator=g, dtype=torch.float32).to(device)
    if rotation_kind != "none" and rotation_tensor is None:
        rotation_tensor = _generate_rotation_per_layer(
            rotation_kind, head_dim, seed=0, device=device, dtype=torch.float32
        )
    # Deterministic dither generator for SNR repeatability across configs.
    dither_gen = torch.Generator(device="cpu").manual_seed(0xC0FFEE) if apply_dither else None
    x_hat, _ = rotate_dither_lattice_pseudo_quantize(
        x.unsqueeze(0).unsqueeze(2),     # [1, n_samples, 1, head_dim]
        rotation_kind=rotation_kind,
        rotation_tensor=rotation_tensor,
        apply_dither=apply_dither,
        lattice=lattice,
        snr_db=snr_db,
        dither_generator=dither_gen,
    )
    x_hat = x_hat.squeeze(2).squeeze(0)
    sig = x.float().pow(2).sum().item()
    noise = (x.float() - x_hat.float()).pow(2).sum().item()
    if noise <= 0:
        return float("inf")
    return 10.0 * math.log10(sig / noise)


# ─────────────────────────────────────────────────────────────────────────────
# Residual-window KV quantization via attention patching
# ─────────────────────────────────────────────────────────────────────────────

import types as _types


def _mixed_attention(q, ke, kq, ve, vq, *, W: int, scaling: float):
    """Causal attention where key/value j is exact for query i iff 0 ≤ i−j < W,
    else quantized.  Computed in float32 to match SDPA's softmax precision.

    q, ke, kq : [B, H, T, d]
    ve, vq    : [B, H, T, d]
    Returns   : [B, T, H, d]
    """
    B, H, T, d = q.shape
    orig_dtype = q.dtype
    q  = q.float();  ke = ke.float();  kq = kq.float()
    ve = ve.float(); vq = vq.float()

    idx  = torch.arange(T, device=q.device)
    diff = idx[:, None] - idx[None, :]          # i − j  [T, T]
    neg  = torch.finfo(torch.float32).min

    Se = torch.matmul(q, ke.transpose(-1, -2)) * scaling
    Sq = torch.matmul(q, kq.transpose(-1, -2)) * scaling

    if W > 0:
        recent = (diff >= 0) & (diff < W)
        old    = diff >= W
    else:
        recent = torch.zeros_like(diff, dtype=torch.bool)
        old    = diff >= 0

    S = torch.where(recent, Se, torch.where(old, Sq, torch.full_like(Se, neg)))
    attn = torch.softmax(S, dim=-1)

    rf = recent.to(attn.dtype)
    of = old.to(attn.dtype)
    out = torch.matmul(attn * rf, ve) + torch.matmul(attn * of, vq)
    return out.to(orig_dtype).transpose(1, 2)   # [B, T, H, d]


def _make_patched_forward(attn_module, layer_cfg: dict):
    """Return a replacement forward for the given attention module."""

    def patched_forward(self, hidden_states, position_embeddings,
                        attention_mask=None, past_key_values=None,
                        cache_position=None, **kwargs):
        cfg = self._kv_reswin_cfg
        B, T, _ = hidden_states.shape
        head_dim = self.head_dim

        # Determine output shape from q_proj
        n_heads    = self.q_proj.out_features // head_dim
        n_kv_heads = self.k_proj.out_features // head_dim
        groups     = n_heads // n_kv_heads

        def shape(x, nh):
            return x.view(B, T, nh, head_dim).transpose(1, 2)

        q = shape(self.q_proj(hidden_states), n_heads)
        k = shape(self.k_proj(hidden_states), n_kv_heads)
        v = shape(self.v_proj(hidden_states), n_kv_heads)

        if not cfg["enabled"]:
            # Unpatched path: standard attention (no quantization)
            cos, sin = position_embeddings
            q, k = cfg["rope_fn"](q, k, cos, sin)
            from torch.nn.functional import scaled_dot_product_attention
            k_rep = k.repeat_interleave(groups, dim=1)
            v_rep = v.repeat_interleave(groups, dim=1)
            out = scaled_dot_product_attention(q, k_rep, v_rep,
                                               attn_mask=attention_mask,
                                               scale=cfg["scaling"])
            return self.o_proj(out.transpose(1, 2).reshape(B, T, -1)), None

        snr        = cfg["snr"]
        rot_kind   = cfg["rot_kind"]
        rot        = cfg["rot"]
        protect_k  = cfg["protect_k"]
        W          = cfg["W"]
        acc        = cfg["acc"]

        # Quantize K and V (pre-RoPE)
        if protect_k:
            k_q  = k
            acc["total_bits"]   += 16.0 * k.numel()
            acc["kv_n_scalars"] += k.numel()
        else:
            k_q, ks = rotate_dither_lattice_pseudo_quantize(
                k, rotation_kind=rot_kind, rotation_tensor=rot,
                apply_dither=False, lattice="e8int", snr_db=snr)
            frac = W / T
            acc["total_bits"]   += int(ks.total_bits) * (1 - frac) + 16.0 * k.numel() * frac
            acc["kv_n_scalars"] += k.numel()

        v_q, vs = rotate_dither_lattice_pseudo_quantize(
            v, rotation_kind=rot_kind, rotation_tensor=rot,
            apply_dither=False, lattice="e8int", snr_db=snr)
        frac = W / T
        acc["total_bits"]   += int(vs.total_bits) * (1 - frac) + 16.0 * v.numel() * frac
        acc["kv_n_scalars"] += v.numel()

        # Apply RoPE to exact and quantized K
        cos, sin = position_embeddings
        rope_fn  = cfg["rope_fn"]
        q_r, ke_r  = rope_fn(q, k,   cos, sin)
        _,   kq_r  = rope_fn(q, k_q, cos, sin)

        ke_r = ke_r.repeat_interleave(groups, dim=1)
        kq_r = kq_r.repeat_interleave(groups, dim=1)
        ve   = v.repeat_interleave(groups, dim=1)
        vq   = v_q.repeat_interleave(groups, dim=1)

        out = _mixed_attention(q_r, ke_r, kq_r, ve, vq,
                               W=W, scaling=cfg["scaling"])
        return self.o_proj(out.reshape(B, T, -1)), None

    return _types.MethodType(patched_forward, attn_module)


def install_kv_residual_quant(
    model: nn.Module,
    cfg: KVQuantConfig,
    *,
    residual_window: int = 32,
    protect_k_each_end: int = 1,
) -> dict:
    """Install residual-window KV quantization by patching each attention module.

    This implements the evaluation protocol used in the paper's OCTOPUS comparison.
    Unlike ``install_kv_cache_quant_path`` (which hooks k_proj/v_proj and quantizes
    ALL tokens simultaneously), this patches each attention module's forward to
    implement true per-query residual-window semantics:

      - Query at position i attends to **exact** (bf16) K/V for the most recent W
        positions and to **quantized** K/V for all older positions.
      - The first and last ``protect_k_each_end`` transformer blocks keep K in bf16
        (matching the OCTOPUS comparison protocol, which keeps the outer blocks exact).

    This is the API to use for PPL evaluation following the paper's protocol, in
    particular for the low-bps (≤ 2 bps) regime where the residual window provides
    a significant quality benefit.  For fast sweeps at moderate bps (≥ 3 bps),
    ``install_kv_cache_quant_path`` is a simpler alternative.

    Parameters
    ----------
    model               : HuggingFace AutoModelForCausalLM
    cfg                 : KVQuantConfig with lattice/snr_db/rotation_kind
    residual_window     : W — most-recent tokens kept in bf16 (default 32)
    protect_k_each_end  : outer blocks whose K is kept exact (default 1)

    Returns
    -------
    dict with '_orig_forwards' and 'hook_handles' (pass to
    ``remove_kv_residual_quant``), 'n_attention_layers', 'accumulator'.
    """
    # Detect apply_rotary_pos_emb for this model family
    def _get_rope_fn(module_name: str):
        """Try to import apply_rotary_pos_emb from the model's module."""
        for pkg in ("transformers.models.qwen2.modeling_qwen2",
                    "transformers.models.llama.modeling_llama",
                    "transformers.models.mistral.modeling_mistral",
                    "transformers.models.gemma2.modeling_gemma2"):
            try:
                import importlib
                m = importlib.import_module(pkg)
                if hasattr(m, "apply_rotary_pos_emb"):
                    return m.apply_rotary_pos_emb
            except ImportError:
                pass
        # Generic fallback: assume (cos, sin) rotation on last two dims
        def _fallback(q, k, cos, sin):
            import torch
            def rot(x, c, s):
                x1, x2 = x[..., ::2], x[..., 1::2]
                return torch.stack([x1*c - x2*s, x1*s + x2*c], dim=-1).flatten(-2)
            return rot(q, cos, sin), rot(k, cos, sin)
        return _fallback

    rope_fn = _get_rope_fn("")

    accumulator = {"total_bits": 0.0, "kv_n_scalars": 0}
    orig_forwards = {}
    n_layers = 0

    attention_modules = [
        (name, mod) for name, mod in model.named_modules()
        if (hasattr(mod, "k_proj") and hasattr(mod, "v_proj")
            and hasattr(mod, "q_proj") and hasattr(mod, "head_dim")
            and isinstance(mod.k_proj, nn.Linear))
    ]

    n_total = len(attention_modules)
    prot = set(range(protect_k_each_end)) | set(range(n_total - protect_k_each_end, n_total))

    for li, (name, attn) in enumerate(attention_modules):
        head_dim = attn.head_dim
        device   = attn.k_proj.weight.device
        dtype    = attn.k_proj.weight.dtype

        rot = (_generate_rotation_per_layer(
                   cfg.rotation_kind, head_dim, cfg.seed + 2 * li,
                   device=device, dtype=dtype)
               if cfg.rotation_kind != "none" else None)

        scaling = getattr(attn, "scaling",
                  getattr(attn, "scale", 1.0 / math.sqrt(head_dim)))

        attn._kv_reswin_cfg = {
            "enabled":    True,
            "snr":        cfg.snr_db,
            "rot_kind":   cfg.rotation_kind,
            "rot":        rot,
            "protect_k":  li in prot,
            "W":          residual_window,
            "acc":        accumulator,
            "rope_fn":    rope_fn,
            "scaling":    scaling,
        }
        orig_forwards[name] = attn.forward
        attn.forward = _make_patched_forward(attn, attn._kv_reswin_cfg)
        n_layers += 1

    return {
        "n_attention_layers": n_layers,
        "hook_handles": list(attention_modules),   # used by remove_kv_residual_quant
        "accumulator": accumulator,
        "_orig_forwards": orig_forwards,
    }


def remove_kv_residual_quant(install_stats: dict) -> None:
    """Undo patching installed by ``install_kv_residual_quant``."""
    orig = install_stats.get("_orig_forwards", {})
    for (name, attn) in install_stats.get("hook_handles", []):
        if name in orig:
            attn.forward = orig[name]
        if hasattr(attn, "_kv_reswin_cfg"):
            del attn._kv_reswin_cfg


def measured_bps_residual(install_stats: dict) -> float:
    acc = install_stats.get("accumulator", {})
    n   = acc.get("kv_n_scalars", 0)
    return acc.get("total_bits", 0) / n if n > 0 else 0.0
