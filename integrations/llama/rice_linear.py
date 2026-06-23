"""
RiceLinear: an nn.Linear replacement whose weight is stored as a MetalRice
Stage-2 Rice bitstream (E8int lattice @ ~4 bits/scalar) and decoded to bf16 on
every forward via the MetalRice CUDA kernel.

Design (memory-correct + efficient):
  * Encode once at model load: an orthonormal randomized Hadamard transform
    (RHT) along the contraction dim + per-output-row std normalization, then the
    MetalRice encoder produces (words, offsets, k). Only these compressed buffers
    (~0.53 B/scalar) + a per-row sigma + the RHT signs are kept resident; the
    bf16 weight is freed.
  * Decode per forward into a SINGLE shared bf16 scratch buffer (sized to the
    largest layer, reused across all layers), so the decompressed weight never
    persists -- this is what preserves the memory win. Decode and the matmul
    that consumes it are issued on the same CUDA stream, so reuse is safe in
    eager mode.

RHT is essential: the codec stores symbols as uint8 (|z| <= 255 => |alpha*w| <=
~255), so raw Llama weight outliers (20-30 sigma in some rows) would saturate
and destroy the model. The orthonormal rotation R = (1/sqrt(H))*H_block*D spreads
each outlier across its chunk; it is applied to BOTH the weight (offline) and the
activation (online) along the contraction dim, and cancels in the matmul:
    (x R) (W R)^T = x R R^T W^T = x W^T.

Reconstruction: W = diag(sigma) @ W_norm, so
    y = (RHT(x) @ W_norm^T) * sigma     (sigma broadcast over the out dim).
"""

from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load

_HERE = os.path.dirname(os.path.abspath(__file__))
_BENCH = os.path.join(_HERE, "..", "..", "cuda")

from hyperquant.weight_quant import chunked_hadamard

# E8int Rice codec defaults for the bf16 GEMV path.
ALPHA = 13.5
RICE_K = 3
SYMBOLS_PER_STREAM = 512

# Default Hadamard block size for the per-tile RHT (power of 2, must divide
# in_features).  128 divides all common hidden sizes.  Users may increase this
# to 256/512/1024 for a marginally stronger incoherence rotation; the actual
# block used is always reduced to the largest power-of-2 ≤ the requested
# value that divides in_features.
DEFAULT_HADAMARD = 128
_signs_cache: dict[tuple, torch.Tensor] = {}


def _get_signs(in_features: int, device, seed: int = 1234) -> torch.Tensor:
    key = (in_features, seed)
    s = _signs_cache.get(key)
    if s is None:
        g = torch.Generator().manual_seed(seed + in_features)
        s = (torch.randint(0, 2, (in_features,), generator=g).float() * 2 - 1)
        _signs_cache[key] = s
    return s.to(device)


def _rht(z: torch.Tensor, signs: torch.Tensor, H: int) -> torch.Tensor:
    """Orthonormal randomized Hadamard along the last dim (float32 math)."""
    return chunked_hadamard(z.float() * signs, H) / math.sqrt(H)


_ext = None


def get_ext():
    global _ext
    if _ext is None:
        _ext = load(
            name="metalrice_ext",
            sources=[
                os.path.join(_HERE, "binding.cu"),
                os.path.join(_HERE, "fused_decode_gemv.cu"),
                os.path.join(_HERE, "rice_fused_attention.cu"),
                os.path.join(_BENCH, "stage2_cuda_encoder.cu"),
                os.path.join(_BENCH, "stage2_cuda_decoder.cu"),
            ],
            extra_include_paths=[_BENCH],
            extra_cuda_cflags=["-O3", "-arch=sm_90a"],
            verbose=True,
        )
    return _ext


# One decode scratch per device, grown on demand.
_scratch: dict[torch.device, torch.Tensor] = {}


def _get_scratch(numel: int, device: torch.device) -> torch.Tensor:
    t = _scratch.get(device)
    if t is None or t.numel() < numel:
        t = torch.empty(numel, dtype=torch.bfloat16, device=device)
        _scratch[device] = t
    return t


class RiceLinear(nn.Module):
    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None,
                 *, alpha: float = ALPHA, rice_k: int | None = None,
                 sps: int = SYMBOLS_PER_STREAM,
                 hadamard: int = DEFAULT_HADAMARD):
        super().__init__()
        ext = get_ext()
        out_f, in_f = weight.shape
        self.out_features = int(out_f)
        self.in_features = int(in_f)
        self.alpha = float(alpha)
        self.sps = int(sps)

        w = weight.detach()
        dev = w.device
        # Choose the largest power-of-2 chunk that divides in_features and is
        # ≤ the requested hadamard size.
        H = hadamard
        while H > 1 and self.in_features % H != 0:
            H //= 2
        self.H = H
        signs = _get_signs(self.in_features, dev)
        # Rotate the weight into the incoherent basis, then per-row normalize.
        w_rot = _rht(w, signs, self.H)                         # [out, in], fp32
        sigma = w_rot.std(dim=1).clamp_min(1e-8)               # [out]
        w_norm = (w_rot / sigma[:, None]).to(torch.bfloat16).contiguous().cpu()
        # Rice k: pick the value (per layer) that minimizes the bitstream, so the
        # achieved rate matches the SNR/bps calibration (a fixed k is suboptimal
        # and inflates the rate by ~0.3-0.9 bits/scalar).
        if rice_k is None:
            best = None
            for k in (1, 2, 3, 4):
                enc = ext.rice_encode(w_norm, self.alpha, k, self.sps)
                if best is None or enc[0].numel() < best[1]:
                    best = (k, enc[0].numel(), enc)
            self.rice_k, _, enc = best
        else:
            self.rice_k = int(rice_k)
            enc = ext.rice_encode(w_norm, self.alpha, self.rice_k, self.sps)
        words, offsets, ks, num_streams, num_words, total = enc

        self.register_buffer("words", words)
        self.register_buffer("offsets", offsets)
        self.register_buffer("ks", ks)
        self.register_buffer("signs", signs.to(torch.bfloat16))
        self.register_buffer("sigma", sigma.to(torch.bfloat16).to(dev))
        self.num_streams = int(num_streams)
        self.num_words = int(num_words)
        self.total = int(total)
        self.bias = (None if bias is None
                     else nn.Parameter(bias.detach().clone(), requires_grad=False))

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_rot = _rht(x, self.signs.float(), self.H).to(torch.bfloat16)
        ext = get_ext()

        # Total number of token vectors in this call (may be > 1 during prefill
        # with a 3-D input [batch, seq_len, hidden_size]).
        n_tokens = x.numel() // self.in_features

        if n_tokens == 1:
            # Fused GEMV (batch = 1 decode token): decode Rice bitstream and
            # dot-product without materialising the full bf16 weight in global
            # memory.  Memory traffic: ~0.5 B/scalar (bitstream) + 2 B/scalar (x)
            # vs the two-kernel path's 4.5 B/scalar.
            x_flat = x_rot.reshape(self.in_features)  # always [in_features]
            y_flat = ext.rice_fused_gemv(
                self.words, self.offsets, self.ks,
                self.num_streams, self.sps, self.alpha,
                x_flat, self.sigma)
            y = y_flat.view(*x.shape[:-1], self.out_features)

        elif self.sps == SYMBOLS_PER_STREAM and n_tokens <= 8:
            # Warp-specialised GEMM: producer warp decodes, consumers accumulate.
            x2d = x_rot.reshape(n_tokens, self.in_features)
            y2d = ext.rice_fused_gemm_ws(
                self.words, self.offsets, self.ks,
                self.num_streams, self.sps, self.alpha,
                x2d, self.sigma)
            y = y2d.view(*x.shape[:-1], self.out_features)

        else:
            # Fallback: two-kernel path (prefill or unusual sps).
            n = self.out_features * self.in_features
            scratch = _get_scratch(n, x.device)
            ext.rice_decode_into(self.words, self.offsets, self.ks,
                                 self.num_streams, self.sps, self.num_words,
                                 self.alpha, scratch)
            W = scratch[:n].view(self.out_features, self.in_features)
            y = F.linear(x_rot, W) * self.sigma

        if self.bias is not None:
            y = y + self.bias
        return y

    def compressed_bytes(self) -> int:
        return (self.words.numel() * 4 + self.offsets.numel() * 4
                + self.ks.numel() + self.sigma.numel() * 2)


def convert_linears_to_rice(model: nn.Module, *, skip=("lm_head",),
                            sps: int = SYMBOLS_PER_STREAM, alpha: float = ALPHA,
                            rice_k: int | None = None,
                            hadamard: int = DEFAULT_HADAMARD,
                            verbose: bool = False) -> dict:
    """Replace every nn.Linear in `model` (except names containing `skip`) with a
    RiceLinear, freeing the bf16 weight. Returns stats.

    hadamard: Hadamard block size for the incoherence RHT.  The actual block
        used per layer is the largest power-of-2 ≤ hadamard that divides the
        layer's in_features, so passing hadamard=128 (default) works for any
        architecture including those with non-power-of-2 hidden sizes like
        Qwen2 (hidden_size=3584=28×128).
    """
    n_conv = 0
    comp_bytes = 0
    orig_bytes = 0
    for name, module in model.named_modules():
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(s in full for s in skip):
                continue
            orig_bytes += child.weight.numel() * 2
            rice = RiceLinear(child.weight.data,
                              child.bias.data if child.bias is not None else None,
                              alpha=alpha, rice_k=rice_k, sps=sps,
                              hadamard=hadamard)
            setattr(module, child_name, rice)
            comp_bytes += rice.compressed_bytes()
            n_conv += 1
            if verbose:
                print(f"  rice {full}: {child.weight.shape}", flush=True)
            del child
    torch.cuda.empty_cache()
    return {
        "n_converted": n_conv,
        "orig_weight_bytes": orig_bytes,
        "compressed_bytes": comp_bytes,
        "compression_x": (orig_bytes / comp_bytes) if comp_bytes else 0.0,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Unified entry point — user picks the MMA path
# ─────────────────────────────────────────────────────────────────────────────

def convert_linears(
    model: nn.Module,
    *,
    skip: tuple = ("lm_head",),
    alpha: float,
    hadamard: int | None = None,
    mma: str = "int8",
    rice_k: int | None = None,
    sps: int = SYMBOLS_PER_STREAM,
    verbose: bool = False,
) -> dict:
    """Replace every nn.Linear in ``model`` with a compressed variant.

    Parameters
    ----------
    alpha    : E8 lattice scale — obtain from
               ``hyperquant.quant_utils.lattice_alpha(snr_db, "e8int")`` after
               calibrating the target bps with ``calibrate_lattice_bps_to_snr``.
    hadamard : Hadamard block size for the incoherence RHT.  ``None`` (default)
               uses 1024 for ``mma="int8"`` and 128 for ``mma="bf16"``.
               The int8 RHT kernel requires a value in {256,512,1024,2048}
               (auto-reduced per layer to the largest valid divisor of in_features).
    mma      : MMA path to use:

               ``"int8"``  (default) — int8 IMMA via cublasLt.
                   Stores weights as E8int/Rice ~4.3 bpw, decodes to int8 each
                   forward, then runs a native INT8 tensor-core GEMM.
                   Per-row weight normalisation ensures every output channel
                   sits at the same quantisation operating point.
                   Best choice for batch-size ≥ 128 or prefill workloads where
                   the decode cost amortises over many token rows.

               ``"fp8"``   — Rice-coded weights, FP8 E4M3 GEMM (FP8Linear).
                   Weights decoded to FP8 E4M3 per forward; activations cast
                   to FP8 E4M3 with per-row scale; ``torch._scaled_mm`` runs
                   the FP8 GEMM.  Requires SM 89+ (H100 / Ada Lovelace).

               ``"bf16"``  — Rice-coded weights, bf16 GEMM (RiceLinear).
                   Same ~4 bps storage with the fused Rice-decode GEMV kernel.
                   Weights are decoded on-the-fly into bf16 partial dot products;
                   activations remain bf16 throughout.  Better for single-token
                   autoregressive decode (the fused GEMV eliminates the bf16
                   scratch-buffer write/read cycle).

    rice_k   : Rice parameter (``None`` = auto, ``"int8"`` path only uses k=2).
    sps      : Symbols per Rice stream (default 512).
    """
    mma = mma.lower()
    if mma in ("int8", "fp8"):
        from integrations.llama.int8_linear import (
            convert_linears_to_int8, convert_linears_to_fp8,
            DEFAULT_HADAMARD as _INT8_HAD)
        # The RHT kernel for int8/fp8 needs had_size ∈ {256,512,1024,2048}.
        h = hadamard if hadamard is not None else _INT8_HAD
        fn = convert_linears_to_int8 if mma == "int8" else convert_linears_to_fp8
        return fn(model, skip=skip, alpha=alpha,
                  hadamard=h, sps=sps, verbose=verbose)
    elif mma == "bf16":
        h = hadamard if hadamard is not None else DEFAULT_HADAMARD
        return convert_linears_to_rice(
            model, skip=skip, alpha=alpha,
            rice_k=rice_k, hadamard=h,
            sps=sps, verbose=verbose)
    else:
        raise ValueError(
            f"Unknown mma={mma!r}. Choose 'int8' (default), 'fp8', or 'bf16'.")
