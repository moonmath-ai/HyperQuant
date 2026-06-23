"""
HyperQuant — Llama / HuggingFace Transformers integration.

Weight quantization
-------------------
``convert_linears`` — unified entry point, ``mma`` selects the inference path:

  • ``mma="int8"`` (default) — LatticeLinear with cublasLt INT8 IMMA GEMM.
    Stores weights as E8int/Rice ~4.3 bpw; forward runs an INT8 tensor-core
    GEMM without materialising a bf16 weight scratch tensor.

  • ``mma="fp8"``            — FP8Linear with FP8 E4M3 GEMM (SM 89+).
  • ``mma="bf16"``           — RiceLinear with fused Rice-decode GEMV (bf16 matmul).
    Maximally compressed (~4 bps); decode is serialised but fused with the
    GEMV, which is better suited to single-token autoregressive decode.

KV-cache quantization
---------------------
``build_rice_kv_cache`` / ``build_lattice_kv_cache`` — DynamicCache subclasses
that store past K/V as Rice bitstreams (~3.8× actual compression at 4 bps) or
int8 codes (~2×) rather than bf16.

Requires SM 80+ (A100 / H100) and CUDA Toolkit ≥ 11.8.
CUDA extensions are compiled on first use via torch.utils.cpp_extension.
"""
from integrations.llama.rice_linear import (
    RiceLinear,
    convert_linears_to_rice,
    convert_linears,
)
from integrations.llama.int8_linear import LatticeLinear, FP8Linear
from integrations.llama.lattice_kv_cache import (
    LatticeQuantizedCache,
    RiceQuantizedCache,
    build_lattice_kv_cache,
    build_rice_kv_cache,
)

__all__ = [
    # Weight quantization
    "convert_linears",          # unified, mma='int8' default
    "LatticeLinear",            # int8 IMMA GEMM variant
    "FP8Linear",                # FP8 E4M3 GEMM variant (SM 89+)
    "RiceLinear",               # Rice-coded + bf16 GEMV variant
    "convert_linears_to_rice",  # low-level Rice path
    # KV-cache
    "LatticeQuantizedCache", "RiceQuantizedCache",
    "build_lattice_kv_cache",  "build_rice_kv_cache",
]
