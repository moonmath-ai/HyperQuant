# CUDA Kernels

GPU-side Rice codec kernels for the HyperQuant weight and KV-cache compression.

## Files

| File | Description |
|------|-------------|
| `stage2_cuda_encoder.cu/h` | Rice Stage-2 encoder: bf16 → E₈ zigzag symbols → Rice bitstream. Two-kernel pipeline (compute lengths → prefix-scan → write bits). Used once at model-load time, not on the forward-pass hot path. |
| `stage2_cuda_decoder.cu/h` | Rice Stage-2 decoder: bitstream → bf16. One thread per stream; supports cooperative shared-memory loading and the interleaved layout for coalesced reads. Used on every forward pass by `RiceLinear`. |
| `stage3_cuda_decoder.cu/h` | Stage-3 decoder variant with higher occupancy for very large stream counts. |
| `rht_cuda.cu/h` | Randomized Hadamard Transform (RHT) kernel: warp-local fast Walsh–Hadamard with random sign diagonal, zero inter-warp synchronization. Used by `weight_quant.py` for outlier spreading. |

## How the codec is invoked

```
Python side (rice_linear.py)
      │
      ├── rice_encode(weight_bf16, alpha, rice_k, sps)
      │       → (words, offsets, ks) uploaded to GPU once at load time
      │
      └── rice_decode_into(words, offsets, ks, ..., out_bf16)
              → fused Rice-decode directly into bf16 scratch, every forward

CUDA side (stage2_cuda_decoder.cu)
      └── decode_fused_stage2_to_bf16_kernel
              one thread per Rice stream; decodes E₈ zigzag symbols;
              applies inverse E₈ lattice transform; writes bf16 output
```

The `integrations/llama/` directory adds a further fusion level:
`fused_decode_gemv.cu` reads the bitstream and computes the dot product
without writing the bf16 weight to global memory at all.

## Building standalone

The kernels are linked automatically by `torch.utils.cpp_extension`. If you
want to experiment with them directly you can compile with nvcc:

```bash
nvcc -O3 -arch=sm_90a -std=c++17 \
     -I. stage2_cuda_decoder.cu -o stage2_decoder_test
```
