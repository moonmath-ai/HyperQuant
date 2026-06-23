# HyperQuant — Llama / HuggingFace Integration

This directory contains everything needed to run HyperQuant with any
HuggingFace `AutoModelForCausalLM` model (Llama, Mistral, Qwen, etc.).

## Files

| File | Purpose |
|------|---------|
| `rice_linear.py` | `RiceLinear` — drop-in `nn.Linear` replacement; stores weights as a Rice bitstream decoded on each forward pass via the fused CUDA kernel |
| `lattice_kv_cache.py` | `LatticeQuantizedCache` (int8, ~2×) and `RiceQuantizedCache` (Rice, ~3.8×) — `DynamicCache` subclasses compatible with `model.generate(past_key_values=...)` |
| `rice_attention.py` | Fused Rice-decode + attention (Flash-Rice); reads the bitstream directly inside the attention computation, eliminating the bf16 K/V scratch buffer |
| `binding.cu` | PyTorch CUDA extension binding that exposes `rice_encode`, `rice_decode_into`, and the fused GEMV / attention ops |
| `fused_decode_gemv.cu` | Warp-specialized fused kernel: decodes Rice → partial dot products without a global-memory scratch buffer (~1.5× decode speedup vs two-kernel path) |
| `rice_fused_attention.cu` | Flash-Rice attention: two-pass decode (K pass → scores, V pass → output) in a single kernel, no bf16 K/V materialization |
| `metalrice_device.cuh` | Shared device-side BitReader and E₈ inverse-transform helpers |

## First-time build

The CUDA extension is compiled automatically the first time you import
`rice_linear.py`. It links the kernels in this directory with the Rice
encoder/decoder in `../../cuda/`. You need:
- `nvcc` (CUDA Toolkit ≥ 11.8)
- PyTorch with CUDA support (`torch.version.cuda` should be set)
- SM 80+ GPU (A100 / H100)

To force a rebuild:
```bash
rm -rf ~/.cache/torch_extensions/*/metalrice_ext
python -c "from integrations.llama.rice_linear import get_ext; get_ext()"
```

## Memory breakdown at 4 bps, Llama-3.1-8B

| Configuration | Resident (GiB) | vs bf16 |
|---|---|---|
| bf16 baseline | 14.96 | 1× |
| Rice weights only | **5.29** | **2.8×** |
| Rice KV only | 14.99 + Rice cache | ~3.8× per cached token |
| Rice weights + KV | **5.69** + Rice cache | best of both |
