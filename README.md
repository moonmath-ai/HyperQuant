# HyperQuant

**HyperQuant** is a data-free post-training quantization library for transformer models. It uses optimal low-dimensional lattice vector quantizers (E₈, D₄, A₂, ℤ¹) combined with Rice entropy coding to compress both **linear weights** and the **KV cache** of LLMs and diffusion transformers.

📄 **Paper**: [arxiv.org/abs/2606.23406](https://arxiv.org/abs/2606.23406)
🌐 **Project page**: [moonmath.ai/hyperquant](https://moonmath.ai/hyperquant/)

---

## Highlights

|                    | Weight quantization                 | KV-cache quantization            |
| ------------------ | ----------------------------------- | -------------------------------- |
| **Rate**     | 3–5 bps                            | 1.7–4 bps                       |
| **Memory**   | ~3.9× at 4 bps                     | ~3.8× at 4 bps                  |
| **Quality**  | Δppl ≤ 0.3 @ 4 bps (Llama-3.1-8B) | LPIPS 0.20–0.21 @ 4 bps (LTX-2) |
| **Setup**    | No calibration data, no fine-tuning | Same                             |
| **Hardware** | H100 / A100 (SM 80+)                | Same                             |

Key algorithmic ingredients:

1. **Per-tile Randomized Hadamard Transform (RHT)** — makes weights/activations approximately Gaussian, enabling optimal lattice quantization and eliminating outlier coordinates.
2. **Optimal lattice VQ** — E₈ packs tighter than any scalar grid in 8D, giving 0.65 dB granular gain over scalar quantization.
3. **Lossless bit-stripping and Rice entropy coding** — removes deterministically pinned bits losslessly, then applies variable-length codes over the unbounded integer lattice, enabling the model to operate at any bps without a fixed codebook size.
4. **Bias correction** — Quantized Johnson-Lindenstrauss (QJL) rotation and/or subtractive Voronoi dither preserve KV-cache inner-product unbiasedness and attention semantics.

---

## Installation

**Requirements**: Python ≥ 3.10, PyTorch ≥ 2.1, a CUDA GPU (SM 80+), and the CUDA toolkit.

```bash
git clone --branch HyperQuant https://github.com/moonmath-ai/LatticeQuant.git hyperquant
cd hyperquant
pip install -e .
```

Two CUDA extensions are compiled automatically on first use via `torch.utils.cpp_extension`:
- `metalrice_ext` — Rice encoder/decoder + fused GEMV/attention kernels (bf16 and Rice paths)
- `lattice_int8_ext` — cublasLt INT8 IMMA GEMM + FP8 E4M3 cast kernels (int8 and fp8 paths)

Both require `nvcc` and the CUDA headers (CUDA Toolkit ≥ 11.8).

For faster recompilation, set the build cache explicitly:

```bash
export TORCH_EXTENSIONS_DIR=/tmp/hyperquant_ext
```

---

## Quick Start

### Weight quantization (3.9× memory, ~5 minutes to quantize)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from hyperquant import calibrate_lattice_bps_to_snr, lattice_alpha
from integrations.llama.rice_linear import convert_linears

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=torch.bfloat16, device_map="cuda"
)

# Calibrate target rate → lattice alpha
snr_db = calibrate_lattice_bps_to_snr(
    lattices=["e8int"], target_bps_list=[4.0]
)["e8int"][4.0]["snr_db"]
alpha = lattice_alpha(snr_db, "e8int")

# Replace nn.Linear weights — mma="int8" (default) uses cublasLt INT8 IMMA GEMM;
# pass mma="bf16" for the Rice-coded fused bf16 GEMV path instead.
stats = convert_linears(model, skip=("lm_head",), alpha=alpha)
print(f"Converted {stats['n_converted']} layers: {stats['compression_x']:.2f}× compression")

# The model is ready — use it normally
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")
out = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(out[0]))
```

### KV-cache quantization (~3.8× memory per cached token)

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from hyperquant import calibrate_lattice_bps_to_snr
from hyperquant.kv_quant import KVQuantConfig
from integrations.llama.lattice_kv_cache import build_rice_kv_cache

model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct",
    dtype=torch.bfloat16, device_map="cuda"
)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
inputs = tokenizer("Once upon a time", return_tensors="pt").to("cuda")

kv_cfg = KVQuantConfig(
    lattice="e8int",
    snr_db=calibrate_lattice_bps_to_snr(
        lattices=["e8int"], target_bps_list=[4.0]
    )["e8int"][4.0]["snr_db"],
    rotation_kind="qjl",   # QJL (Haar-uniform) rotation for unbiased attention
)

# Build a Rice-coded KV cache and pass it at generation time
cache = build_rice_kv_cache(model, kv_cfg)
out = model.generate(**inputs, max_new_tokens=200, past_key_values=cache)
print(f"KV cache stored: {cache.stored_bytes()/1024**2:.1f} MB "
      f"(~3.8× compression vs bf16)")
```

See [`examples/`](examples/) for complete runnable scripts.

---

## Repository Structure

```
hyperquant/          Core Python package
├── lattice.py       E₈/D₄/A₂/ℤ¹ lattice quantizers + Rice codec
├── weight_quant.py  Weight quantization (RHT + FP8/INT8 MMA paths)
├── quant_utils.py   Shared quantization primitives and calibration tables
├── kv_quant.py      KV-cache quantization (QJL rotation, dither, hooks)
├── kv_cache_hooks.py  Pseudo-FP8/INT8 attention hooks
├── calibration.py   SNR ↔ bps calibration utilities
└── __init__.py

cuda/                MetalRice CUDA kernels (Rice codec + Randomized Hadamard Transform)
├── stage2_cuda_encoder.cu/h   GPU Rice encoder
├── stage2_cuda_decoder.cu/h   GPU Rice decoder (Stage 2)
├── stage3_cuda_decoder.cu/h   GPU Rice decoder (Stage 3, higher-throughput)
├── rht_cuda.cu/h              Randomized Hadamard Transform kernel
└── lattice_ext.cu             INT8 IMMA GEMM extension (cublasLt; compiles to lattice_int8_ext)

integrations/llama/  HuggingFace Transformers integration
├── rice_linear.py       convert_linears() unified entry point; RiceLinear (Rice-coded weights)
├── int8_linear.py       LatticeLinear (INT8 IMMA) and FP8Linear (FP8 E4M3 GEMM)
├── lattice_kv_cache.py  build_lattice_kv_cache / build_rice_kv_cache (DynamicCache subclasses)
├── rice_attention.py    Fused Rice-decode + attention kernel
├── binding.cu           PyTorch CUDA extension binding (compiles to metalrice_ext)
├── fused_decode_gemv.cu Fused Rice-decode + GEMV (eliminates bf16 scratch)
├── rice_fused_attention.cu  Fused Flash-Rice attention kernel
└── metalrice_device.cuh     Shared device utilities

examples/
├── quantize_weights.py   Weight-only quantization demo
└── quantize_kv_cache.py  KV-cache quantization demo
```

---

## User Guide

### Choosing a rate

HyperQuant operates on a continuous bits-per-scalar (bps) axis:

| bps | Quality (Llama-3.1-8B WikiText-2 ppl) | Memory vs bf16           |
| --- | ------------------------------------- | ------------------------ |
| 5   | +0.9%                                 | 3.2× weights            |
| 4   | +3.8%                                 | 3.9× weights, ~3.8× KV |
| 3   | +22%                                  | 5.3× weights            |
| 2   | (KV only) +7.4%                       | ~8× KV                  |

Use `calibrate_lattice_bps_to_snr` to map any bps to the SNR target that the
calibration sweep found empirically:

```python
from hyperquant.calibration import calibrate_lattice_bps_to_snr
result = calibrate_lattice_bps_to_snr(["e8int"], [3.0, 4.0, 5.0])
```

### Weight quantization in detail

`convert_linears` is the unified entry point. The `mma` argument selects the
inference path; all three achieve the same ~3.9× weight compression at 4 bps.

```python
from integrations.llama.rice_linear import convert_linears

stats = convert_linears(
    model,
    skip=("lm_head",),   # layers to keep in bf16
    alpha=alpha,          # lattice scale from calibration
    mma="int8",           # "int8" (default), "fp8", or "bf16"
    verbose=True,
)
# stats: {'n_converted', 'orig_weight_bytes', 'compressed_bytes', 'compression_x'}
```

**`mma="int8"` (default) — LatticeLinear, cublasLt INT8 IMMA GEMM**

Each weight is row-normalised (per-output-channel std), encoded as an E8int/Rice
bitstream, and stored alongside a bf16 per-row scale.  On each forward: RHT →
per-token row-absmax cast → int8 → E8int decode into int8 → cublasLt INT8 TN
GEMM → dequant → bf16.  No bf16 weight is ever written to GPU memory.  Best
at batch sizes ≥ 128 or prefill where decode cost amortises.

**`mma="fp8"` — FP8Linear, FP8 E4M3 GEMM (SM 89+)**

Each weight is row-normalised and Rice-encoded identically to the int8 path.
On each forward: RHT → per-tensor amax → FP8 E4M3 activation cast; E8int decode
→ FP8 E4M3 weights; `torch._scaled_mm(x_fp8, w_fp8.T, scale_a, scale_b=1.0)` →
bf16.  Requires H100 or Ada Lovelace (SM 89+) for native FP8 tensor-core GEMM.

**`mma="bf16"` — RiceLinear, fused Rice-decode bf16 GEMV**

Stores the weight as a variable-length Rice bitstream at ~4 bps.  The fused
CUDA kernel decodes directly into partial dot products without a bf16 scratch
tensor.  Better for single-token autoregressive decode.

**Hadamard block size**: leave unset to use the path-appropriate default (256
for `mma="int8"`, 128 for `mma="bf16"`).  The int8 RHT kernel supports
`{256, 512, 1024, 2048}`; the actual size used per layer is the largest value
in that set ≤ `hadamard` that divides `in_features`, so it works automatically
for any architecture including non-power-of-2 hidden sizes like Qwen2
(3584 = 14 × 256).

### KV-cache quantization in detail

HyperQuant offers two KV-cache storage backends:

| Class                     | Storage                     | Actual compression |
| ------------------------- | --------------------------- | ------------------ |
| `LatticeQuantizedCache` | int8 codes + fp16 norms     | ~1.97× vs bf16    |
| `RiceQuantizedCache`    | Rice bitstream + fp16 norms | ~3.8× vs bf16     |

Both are `DynamicCache` subclasses and drop into any HuggingFace model via
`past_key_values=`:

```python
from integrations.llama.lattice_kv_cache import (
    build_lattice_kv_cache,   # int8 ~2× compression
    build_rice_kv_cache,      # Rice ~4× compression
)

cache = build_rice_kv_cache(model, kv_cfg)
output = model.generate(**inputs, past_key_values=cache, max_new_tokens=200)
print(f"KV compressed to {cache.stored_bytes()/1024**2:.1f} MB")
```

**Rotation options** (`rotation_kind` in `KVQuantConfig`):

- `"qjl"` — Quantized Johnson-Lindenstrauss (QJL) rotation; Haar-uniform; strictly unbiased inner products on average; recommended.
- `"signs"` — Random ±1 diagonal; O(n) cost; good for high-rate (≥ 3 bps) use.
- `"none"` — No rotation; deterministic bias accumulates; not recommended.

### KV quantization for PPL evaluation — residual window

The paper's OCTOPUS comparison uses a *residual-window* protocol: each query
attends to exact (bf16) K/V for the most recent W tokens and to quantized K/V
for older ones, and the outermost transformer block on each end keeps K in full
precision.  This is the correct API to use for reproducible PPL evaluation at
low bps (≤ 2 bps):

```python
from hyperquant.kv_quant import (
    install_kv_residual_quant,
    remove_kv_residual_quant,
    measured_bps_residual,
)

# Load model with SDPA
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct", dtype=torch.bfloat16,
    device_map="cuda", attn_implementation="sdpa"
)

kv_cfg = KVQuantConfig(lattice="e8int", snr_db=snr_db, rotation_kind="qjl")
st = install_kv_residual_quant(
    model, kv_cfg,
    residual_window=32,       # keep last 32 tokens' K/V exact (bf16)
    protect_k_each_end=1,     # keep first/last block's K exact (paper protocol)
)
# ... evaluate PPL with use_cache=False ...
bps = measured_bps_residual(st)   # effective bps including residual overhead
remove_kv_residual_quant(st)
```

This reproduces the paper's Table 1 OCTOPUS comparison (+7.4% Δppl at 2 bps,
6.4× KV compression on Qwen2.5-7B-Instruct).

### Using the pseudo-quantization hooks (for fast PPL sweeps at moderate bps)

At 3–5 bps the simpler hook approach is faster and gives accurate results:

```python
from hyperquant.kv_quant import install_kv_cache_quant_path, remove_kv_hooks

kv_stats = install_kv_cache_quant_path(model, kv_cfg, measure_snr=True)
# ... run model with use_cache=False to measure PPL ...
remove_kv_hooks(kv_stats["hook_handles"])
```

### Activation FP8 / INT8 hooks (separate from weight compression)

`install_fp8_path` in `weight_quant.py` is a *separate* feature from
`convert_linears`: it installs per-tile activation hooks that cast the
*inputs* to each linear layer to FP8/INT8 at inference time, without
changing the weight storage format.  This is complementary to weight
compression and can be combined with it.

```python
from hyperquant.weight_quant import install_fp8_path

install_fp8_path(model, mma_dtype="int8")  # or "fp8_e4m3", "nvfp4", "mxfp4"
```

INT8 consistently outperforms FP8 on post-RHT data by ~0.1 PPL on
Llama and ~0.7 dB PSNR on LTX-2 video — because after RHT the distribution
is near-Gaussian (light-tailed), and INT8's uniform grid is a better fit than
FP8's wider exponent range.

---

## Citation

If you use HyperQuant in your research, please cite:

```bibtex
@article{domb2026hyperquant,
  title   = {HyperQuant: A Rate-Distortion-Optimal Quantization Pipeline for Large Language and Diffusion Models},
  author  = {Domb, Yuval and Sackstein, Hadar and Solberg, Tomer},
  journal = {arXiv preprint arXiv:2606.23406},
  year    = {2026},
  url     = {https://arxiv.org/abs/2606.23406}
}
```

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The MetalRice CUDA kernels in `cuda/` are derived from the MetalRice project
(see `cuda/` directory for the original license).
