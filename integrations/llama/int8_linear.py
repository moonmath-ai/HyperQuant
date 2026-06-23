"""
LatticeLinear — drop-in nn.Linear replacement using int8 IMMA inference.

Stores the weight as a PackedWeight (RHT-rotated + E8int/Rice-encoded).
On each forward:
  1. x → RHT → x'                       (orthonormal rotation)
  2. x' → per-token row-absmax → int8   (per-token dynamic quantization)
  3. decode packed W → int8 lattice Y   (runtime decode into int8)
  4. int8 IMMA GEMM via cublasLt        (native tensor-core path)
  5. dequant → bf16                      (per-token rescale)
  6. × per-row wscale                    (undo per-row normalisation)

The bf16 weight is never resident in HBM — only the ~4.3 bpw packed form.

Usage (via the unified convert_linears entry point):
    from integrations.llama.rice_linear import convert_linears
    stats = convert_linears(model, alpha=alpha)          # mma='int8' default
    stats = convert_linears(model, alpha=alpha, mma='bf16')  # bf16 path
"""
from __future__ import annotations

import gc
import math
import os
from functools import lru_cache

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

_HERE  = os.path.dirname(os.path.abspath(__file__))
_CUDA  = os.path.join(_HERE, "..", "..", "cuda")   # cuda/ at repo root

# The RHT kernel used by the int8 path supports only {256, 512, 1024, 2048}.
# 1024 is a good default (divides 4096, 14336, …).  For architectures like
# Qwen2 (hidden=3584=7×512), the auto-select below falls back to 512.
DEFAULT_HADAMARD = 256
_RHT_SIZES       = (2048, 1024, 512, 256)   # checked in order, largest first
DEFAULT_RICE_K   = 2
DEFAULT_SPS      = 512   # symbols per stream (S)


@lru_cache(maxsize=1)
def _get_int8_ext():
    return load(
        name="lattice_int8_ext",
        sources=[
            os.path.join(_CUDA, "lattice_ext.cu"),
            os.path.join(_CUDA, "rht_cuda.cu"),
            os.path.join(_CUDA, "stage2_cuda_encoder.cu"),
            os.path.join(_CUDA, "stage2_cuda_decoder.cu"),
        ],
        extra_include_paths=[_CUDA],
        extra_cuda_cflags=["-O3", "-arch=sm_90a"],
        extra_ldflags=["-lcublas", "-lcublasLt"],
        verbose=True,
    )


class LatticeLinear(nn.Module):
    """Compressed nn.Linear using int8 IMMA GEMM (cublasLt).

    Keeps only the packed weight (~4.3 bpw) and a per-output-row bf16 scale.
    Inference: RHT → int8 cast → E8int decode → int8 GEMM → dequant.
    """

    def __init__(self, packed, bias: torch.Tensor | None = None,
                 wscale: torch.Tensor | None = None):
        super().__init__()
        self.packed = packed
        self.in_features  = packed.K
        self.out_features = packed.N
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        # Per-output-channel weight scale [N].  The weight was row-normalised
        # before packing (each row divided by its std), so the lattice sits at
        # the chosen operating point for every row.  wscale = orig row std.
        if wscale is not None:
            self.register_buffer("wscale", wscale.detach().clone())
        else:
            self.wscale = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops = _get_int8_ext()
        y   = ops.lattice_linear_forward(x.contiguous(), self.packed)
        if self.wscale is not None:
            y = y * self.wscale
        if self.bias is not None:
            y = y + self.bias
        return y

    def compressed_bytes(self) -> int:
        b = self.packed.compressed_bytes
        if self.wscale is not None:
            b += self.wscale.numel() * 2
        return b

    def extra_repr(self) -> str:
        ratio = self.packed.bf16_bytes / max(self.compressed_bytes(), 1)
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, compression={ratio:.2f}x")


def _split_parent(model: nn.Module, name: str):
    parent, parts = model, name.split(".")
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


@torch.no_grad()
def convert_linears_to_int8(
    model: nn.Module,
    *,
    skip: tuple = ("lm_head",),
    alpha: float,
    rice_k: int = DEFAULT_RICE_K,
    hadamard: int = DEFAULT_HADAMARD,
    sps: int = DEFAULT_SPS,
    seed: int = 1337,
    verbose: bool = False,
) -> dict:
    """Replace every nn.Linear (except names containing `skip`) with a
    LatticeLinear that stores the weight as a Rice-encoded E8int bitstream
    and runs the forward via int8 cublasLt IMMA GEMM.

    Each weight is row-normalised before packing (per-row scale stored as
    wscale), which places every output channel at the same operating point
    and removes the dominant quantisation error from channels with small norms.

    Parameters
    ----------
    alpha    : E8 lattice scale (from calibration — e.g. lattice_alpha(snr_db))
    rice_k   : Rice parameter (2 is near-optimal for 4 bps)
    hadamard : Hadamard block size (auto-reduced to largest pow2 ≤ hadamard
               that divides in_features — so 128 works for any hidden size)
    sps      : symbols per stream for the Rice codec
    """
    ops = _get_int8_ext()

    # Return any cached-but-idle HBM to the driver before raw cudaMalloc.
    torch.cuda.empty_cache()

    n_conv = 0; orig_bytes = 0; comp_bytes = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(s in full for s in skip):
                continue

            W = child.weight.data
            assert W.is_cuda and W.dtype == torch.bfloat16, \
                f"{full}: need CUDA bf16 weight"

            # Auto-select the largest RHT-supported size (∈ {256,512,1024,2048})
            # that is ≤ hadamard AND divides K.
            H = next((s for s in _RHT_SIZES if s <= hadamard and W.shape[1] % s == 0),
                     _RHT_SIZES[-1])   # fall back to 256

            # Per-row normalisation: divide each output channel by its std.
            s     = W.float().std(dim=1).clamp_min(1e-8)       # [N]
            W_norm = (W.float() / s[:, None]).to(torch.bfloat16)
            wscale = s.to(torch.bfloat16)

            # lattice_pack expects N*K % 8 == 0 and sps | N*K.
            N, K = W.shape
            actual_sps = sps
            while actual_sps > 8 and (N * K) % actual_sps != 0:
                actual_sps //= 2

            packed = ops.lattice_pack(W_norm, float(alpha), rice_k,
                                      H, seed, actual_sps)
            new_mod = LatticeLinear(
                packed,
                bias=child.bias.data if child.bias is not None else None,
                wscale=wscale,
            )
            setattr(module, child_name, new_mod)

            # Free the bf16 weight immediately so HBM shrinks as we go.
            child.weight.data = torch.empty(0, device=W.device, dtype=W.dtype)
            del W, W_norm, child
            torch.cuda.empty_cache()

            orig_bytes += packed.bf16_bytes
            comp_bytes  += new_mod.compressed_bytes()
            n_conv += 1
            if verbose:
                bpw = 8 * new_mod.compressed_bytes() / (N * K)
                print(f"  int8 {full}: [{N}x{K}]  H={H}  {bpw:.2f} bpw")

    gc.collect(); torch.cuda.empty_cache()
    cx = orig_bytes / max(comp_bytes, 1)
    if verbose:
        print(f"  {n_conv} layers: {orig_bytes/2**20:.0f} → "
              f"{comp_bytes/2**20:.0f} MiB ({cx:.2f}×)")
    return {
        "n_converted":        n_conv,
        "orig_weight_bytes":  orig_bytes,
        "compressed_bytes":   comp_bytes,
        "compression_x":      cx,
    }


# ─────────────────────────────────────────────────────────────────────────────
# FP8 E4M3 path (mma="fp8")
# ─────────────────────────────────────────────────────────────────────────────

class FP8Linear(nn.Module):
    """Compressed nn.Linear using FP8 E4M3 GEMM (torch._scaled_mm).

    Pipeline:
      1. Weights: Rice bitstream → E8int decode → FP8 E4M3 (Y/alpha, scale_b=1.0)
      2. Activations: bf16 → RHT → per-row amax → FP8 E4M3 (scale_a per row)
      3. torch._scaled_mm(x_fp8, w_fp8.T, scale_a, scale_b=1.0) → bf16
      4. × per-row wscale (restore original weight magnitude)

    Requires SM 89+ (H100 / Ada) for native FP8 tensor cores.
    """

    def __init__(self, packed, bias: torch.Tensor | None = None,
                 wscale: torch.Tensor | None = None):
        super().__init__()
        self.packed = packed
        self.in_features  = packed.K
        self.out_features = packed.N
        if bias is not None:
            self.bias = nn.Parameter(bias.detach().clone(), requires_grad=False)
        else:
            self.register_parameter("bias", None)
        if wscale is not None:
            self.register_buffer("wscale", wscale.detach().clone())
        else:
            self.wscale = None

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        ops = _get_int8_ext()

        # Flatten any leading batch/sequence dims to 2-D [M, K] for torch._scaled_mm,
        # then restore the original leading shape on the output.
        leading = x.shape[:-1]   # e.g. (B,) or (B, T)
        K       = self.in_features
        N       = self.out_features
        x2d     = x.contiguous().reshape(-1, K)   # [M, K]

        # RHT on activations (bf16), then per-tensor FP8 E4M3 cast.
        x_rot   = ops.rht_apply(x2d, self.packed)
        scale_a = (x_rot.float().abs().max() / 448.0).clamp_min_(1e-12)
        x_fp8   = (x_rot.float() / scale_a).clamp(-448.0, 448.0).to(torch.float8_e4m3fn)

        # Decode W → FP8 (Y/alpha values; unit-variance, well within [-448, 448])
        w_fp8_u8 = ops.lattice_decode_fp8(self.packed)
        w_fp8    = w_fp8_u8.view(torch.float8_e4m3fn)

        # FP8 GEMM — scalar scale tensors (universally supported per-tensor scaling)
        scale_b = torch.ones(1, dtype=torch.float32, device=x.device)
        y2d = torch._scaled_mm(x_fp8, w_fp8.T,
                                scale_a=scale_a,
                                scale_b=scale_b,
                                out_dtype=torch.bfloat16)  # [M, N]

        y = y2d.reshape(*leading, N)   # restore original batch/seq dims

        if self.wscale is not None:
            y = y * self.wscale
        if self.bias is not None:
            y = y + self.bias
        return y

    def compressed_bytes(self) -> int:
        b = self.packed.compressed_bytes
        if self.wscale is not None:
            b += self.wscale.numel() * 2
        return b

    def extra_repr(self) -> str:
        ratio = self.packed.bf16_bytes / max(self.compressed_bytes(), 1)
        return (f"in={self.in_features}, out={self.out_features}, "
                f"bias={self.bias is not None}, compression={ratio:.2f}x")


@torch.no_grad()
def convert_linears_to_fp8(
    model: nn.Module,
    *,
    skip: tuple = ("lm_head",),
    alpha: float,
    rice_k: int = DEFAULT_RICE_K,
    hadamard: int = DEFAULT_HADAMARD,
    sps: int = DEFAULT_SPS,
    seed: int = 1337,
    verbose: bool = False,
) -> dict:
    """Replace every nn.Linear (except names containing `skip`) with a
    FP8Linear that stores the weight as a Rice-encoded E8int bitstream and
    runs the forward via FP8 E4M3 GEMM (torch._scaled_mm).

    Requires SM 89+ (H100 / Ada Lovelace) for native FP8 tensor cores.
    Same compression (~3.9×) and per-row normalisation as the int8 path.
    """
    ops = _get_int8_ext()
    torch.cuda.empty_cache()

    n_conv = 0; orig_bytes = 0; comp_bytes = 0
    for name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            if any(s in full for s in skip):
                continue

            W = child.weight.data
            assert W.is_cuda and W.dtype == torch.bfloat16, \
                f"{full}: need CUDA bf16 weight"

            H = next((s for s in _RHT_SIZES if s <= hadamard and W.shape[1] % s == 0),
                     _RHT_SIZES[-1])

            s      = W.float().std(dim=1).clamp_min(1e-8)
            W_norm = (W.float() / s[:, None]).to(torch.bfloat16)
            wscale = s.to(torch.bfloat16)

            N, K = W.shape
            actual_sps = sps
            while actual_sps > 8 and (N * K) % actual_sps != 0:
                actual_sps //= 2

            packed = ops.lattice_pack(W_norm, float(alpha), rice_k,
                                      H, seed, actual_sps)
            new_mod = FP8Linear(
                packed,
                bias=child.bias.data if child.bias is not None else None,
                wscale=wscale,
            )
            setattr(module, child_name, new_mod)

            child.weight.data = torch.empty(0, device=W.device, dtype=W.dtype)
            del W, W_norm, child
            torch.cuda.empty_cache()

            orig_bytes += packed.bf16_bytes
            comp_bytes  += new_mod.compressed_bytes()
            n_conv += 1
            if verbose:
                bpw = 8 * new_mod.compressed_bytes() / (N * K)
                print(f"  fp8 {full}: [{N}x{K}]  H={H}  {bpw:.2f} bpw")

    gc.collect(); torch.cuda.empty_cache()
    cx = orig_bytes / max(comp_bytes, 1)
    if verbose:
        print(f"  {n_conv} layers: {orig_bytes/2**20:.0f} → "
              f"{comp_bytes/2**20:.0f} MiB ({cx:.2f}×)")
    return {
        "n_converted":        n_conv,
        "orig_weight_bytes":  orig_bytes,
        "compressed_bytes":   comp_bytes,
        "compression_x":      cx,
    }
