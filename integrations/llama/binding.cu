// PyTorch CUDA-extension binding around the MetalRice Stage-2 Rice codec.
//
// Exposes two ops to Python:
//   rice_encode(weight_bf16_cpu, alpha, rice_k, symbols_per_stream)
//       -> (words_i32_cuda, offsets_i32_cuda, k_u8_cuda,
//           num_streams, num_words, total_symbols)
//     Encodes a bf16 weight (host) into the Stage-2 Rice bitstream and uploads
//     the compressed buffers to the GPU. Run once per weight at model load.
//
//   rice_decode_into(words, offsets, k, num_streams, symbols_per_stream,
//                    num_words, alpha, out_bf16_cuda)
//     Launches the fused decode kernel straight into a caller-provided bf16
//     scratch tensor (no allocation, no H2D/D2H), on the current CUDA stream.
//     This is the per-forward dequant.
//
// Links stage2_cuda_encoder.cu (the class) and stage2_cuda_decoder.cu (the
// fused kernel via metalrice_launch_fused_decode_bf16) into the same extension.

#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <tuple>
#include <vector>

#include "stage2_cuda_encoder.h"
#include "stage2_cuda_decoder.h"  // metalrice_launch_fused_decode_bf16

// Launcher from rice_fused_attention.cu
void launch_rice_fused_attn_decode(
    const uint32_t* words_k, int wc_k,
    const uint32_t* offsets_k, const uint8_t* ks_k, int n_streams_k,
    const uint16_t* norms_k,
    const uint32_t* words_v, int wc_v,
    const uint32_t* offsets_v, const uint8_t* ks_v, int n_streams_v,
    const uint16_t* norms_v,
    const uint16_t* pend_uk, const uint16_t* pend_nk,
    const uint16_t* pend_uv, const uint16_t* pend_nv,
    int n_pend,
    const uint16_t* Q_prerot, float* out_rot,
    int n_kv_heads, int n_q_per_kv, int head_dim,
    int n_complete, int chunk_tokens, int sps, float alpha,
    cudaStream_t stream);
int rice_fused_attn_t_max();

// Launchers from fused_decode_gemv.cu
void launch_rice_fused_gemv(
    const uint32_t* words, int word_count,
    const uint32_t* offsets, const uint8_t* ks,
    int num_streams, int sps, float alpha,
    const uint16_t* x_rot, const uint16_t* sigma, uint16_t* y_out,
    int out_features, int in_features, cudaStream_t stream);

void launch_rice_fused_gemm_ws(
    const uint32_t* words, int word_count,
    const uint32_t* offsets, const uint8_t* ks,
    int num_streams, int sps, float alpha,
    const uint16_t* x_rot, const uint16_t* sigma, uint16_t* y_out,
    int out_features, int in_features, int batch, cudaStream_t stream);

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int64_t, int64_t, int64_t>
rice_encode(torch::Tensor weight_bf16_cpu, double alpha, int64_t rice_k,
            int64_t symbols_per_stream) {
  TORCH_CHECK(weight_bf16_cpu.dtype() == torch::kBFloat16,
              "weight must be bfloat16");
  TORCH_CHECK(weight_bf16_cpu.device().is_cpu(), "weight must be on CPU for encode");
  auto w = weight_bf16_cpu.contiguous();
  const int64_t n = w.numel();
  TORCH_CHECK(n % 8 == 0, "numel must be a multiple of 8 (E8 words)");

  const uint16_t* p = reinterpret_cast<const uint16_t*>(w.data_ptr());
  std::vector<uint16_t> bf16_words(p, p + n);

  Stage2CudaEncoder enc(bf16_words, static_cast<float>(alpha),
                        static_cast<uint8_t>(rice_k), /*groups_per_thread=*/1,
                        Stage2CudaEncoder::CacheMode::kB2U8Cache,
                        /*threads_per_block=*/256,
                        /*decoder_symbols_per_stream=*/static_cast<int>(symbols_per_stream));
  TORCH_CHECK(enc.ok(), "encoder init failed: ", enc.error_message());

  std::vector<uint32_t> words, offsets;
  std::vector<uint8_t> ks;
  uint32_t total_bits = 0;
  bool okk = enc.encode(words, offsets, ks, &total_bits, /*copy_output_to_host=*/true);
  TORCH_CHECK(okk, "encode failed: ", enc.error_message());

  const int64_t num_streams = static_cast<int64_t>(offsets.size());

  auto opt32 = torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA);
  auto opt8 = torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCUDA);
  auto t_words = torch::empty({static_cast<int64_t>(words.size())}, opt32);
  auto t_off = torch::empty({num_streams}, opt32);
  auto t_k = torch::empty({num_streams}, opt8);

  cudaMemcpy(t_words.data_ptr(), words.data(), words.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(t_off.data_ptr(), offsets.data(), offsets.size() * sizeof(uint32_t),
             cudaMemcpyHostToDevice);
  cudaMemcpy(t_k.data_ptr(), ks.data(), ks.size() * sizeof(uint8_t),
             cudaMemcpyHostToDevice);

  return {t_words, t_off, t_k, num_streams, n / 8, n};
}

void rice_decode_into(torch::Tensor words, torch::Tensor offsets, torch::Tensor ks,
                      int64_t num_streams, int64_t symbols_per_stream,
                      int64_t num_words, double alpha, torch::Tensor out_bf16) {
  TORCH_CHECK(out_bf16.dtype() == torch::kBFloat16 && out_bf16.is_cuda(),
              "out_bf16 must be a CUDA bfloat16 tensor");
  TORCH_CHECK(out_bf16.numel() >= num_words * 8, "out_bf16 too small");
  const int word_count = static_cast<int>(words.numel());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  metalrice_launch_fused_decode_bf16(
      reinterpret_cast<const uint32_t*>(words.data_ptr()), word_count,
      reinterpret_cast<const uint32_t*>(offsets.data_ptr()),
      reinterpret_cast<const uint8_t*>(ks.data_ptr()),
      static_cast<int>(num_streams), static_cast<int>(symbols_per_stream),
      static_cast<int>(num_words), static_cast<float>(alpha),
      reinterpret_cast<uint16_t*>(out_bf16.data_ptr()), 256,
      static_cast<void*>(stream));
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused Rice decode + linear (GEMV / warp-specialised GEMM)
// ─────────────────────────────────────────────────────────────────────────────

// rice_fused_gemv(words, offsets, ks, num_streams, sps, alpha, x_rot, sigma)
//   -> y  [out_features] bfloat16
//
// x_rot must be 1-D [in_features] or 2-D [1, in_features] bfloat16 on CUDA.
// Requires in_features % sps == 0 and sps <= 256.
torch::Tensor rice_fused_gemv(
    torch::Tensor words, torch::Tensor offsets, torch::Tensor ks,
    int64_t num_streams, int64_t sps, double alpha,
    torch::Tensor x_rot, torch::Tensor sigma)
{
  TORCH_CHECK(x_rot.is_cuda() && x_rot.dtype() == torch::kBFloat16,
              "x_rot must be cuda bfloat16");
  TORCH_CHECK(sigma.is_cuda() && sigma.dtype() == torch::kBFloat16,
              "sigma must be cuda bfloat16");
  const int64_t in_features  = x_rot.numel();
  const int64_t out_features = sigma.numel();
  TORCH_CHECK(in_features % sps == 0,
              "in_features (", in_features, ") must be divisible by sps (", sps, ")");
  const int64_t spr = in_features / sps;
  TORCH_CHECK(spr <= 256,
      "streams_per_row (in_features/sps = ", spr, ") must be <= 256 (BLOCK_THREADS)");

  auto y = torch::empty({out_features},
      torch::TensorOptions().dtype(torch::kBFloat16).device(x_rot.device()));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_rice_fused_gemv(
      reinterpret_cast<const uint32_t*>(words.data_ptr()),
      static_cast<int>(words.numel()),
      reinterpret_cast<const uint32_t*>(offsets.data_ptr()),
      reinterpret_cast<const uint8_t*>(ks.data_ptr()),
      static_cast<int>(num_streams),
      static_cast<int>(sps),
      static_cast<float>(alpha),
      reinterpret_cast<const uint16_t*>(x_rot.contiguous().data_ptr()),
      reinterpret_cast<const uint16_t*>(sigma.data_ptr()),
      reinterpret_cast<uint16_t*>(y.data_ptr()),
      static_cast<int>(out_features),
      static_cast<int>(in_features),
      stream);
  return y;
}

// rice_fused_gemm_ws(words, offsets, ks, num_streams, sps, alpha, x_rot, sigma)
//   -> y  [batch, out_features] bfloat16
//
// x_rot: [batch, in_features] bfloat16 on CUDA.  sps must == 512 (WS_SPS).
torch::Tensor rice_fused_gemm_ws(
    torch::Tensor words, torch::Tensor offsets, torch::Tensor ks,
    int64_t num_streams, int64_t sps, double alpha,
    torch::Tensor x_rot, torch::Tensor sigma)
{
  TORCH_CHECK(x_rot.is_cuda() && x_rot.dtype() == torch::kBFloat16,
              "x_rot must be cuda bfloat16");
  TORCH_CHECK(sigma.is_cuda() && sigma.dtype() == torch::kBFloat16,
              "sigma must be cuda bfloat16");
  TORCH_CHECK(x_rot.dim() == 2, "x_rot must be 2D [batch, in_features]");
  TORCH_CHECK(sps == 512,
              "rice_fused_gemm_ws requires sps == 512 (WS_SPS constant)");
  const int64_t batch        = x_rot.size(0);
  const int64_t in_features  = x_rot.size(1);
  const int64_t out_features = sigma.numel();
  TORCH_CHECK(in_features % sps == 0,
              "in_features must be divisible by sps");

  auto y = torch::zeros({batch, out_features},
      torch::TensorOptions().dtype(torch::kBFloat16).device(x_rot.device()));
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();
  launch_rice_fused_gemm_ws(
      reinterpret_cast<const uint32_t*>(words.data_ptr()),
      static_cast<int>(words.numel()),
      reinterpret_cast<const uint32_t*>(offsets.data_ptr()),
      reinterpret_cast<const uint8_t*>(ks.data_ptr()),
      static_cast<int>(num_streams),
      static_cast<int>(sps),
      static_cast<float>(alpha),
      reinterpret_cast<const uint16_t*>(x_rot.contiguous().data_ptr()),
      reinterpret_cast<const uint16_t*>(sigma.data_ptr()),
      reinterpret_cast<uint16_t*>(y.data_ptr()),
      static_cast<int>(out_features),
      static_cast<int>(in_features),
      static_cast<int>(batch),
      stream);
  return y;
}

// ─────────────────────────────────────────────────────────────────────────────
// Fused Rice decode + scaled dot-product attention (single decode step)
// ─────────────────────────────────────────────────────────────────────────────

// rice_fused_attn_decode(words_k, offsets_k, ks_k, norms_k,
//                        words_v, offsets_v, ks_v, norms_v,
//                        pend_uk, pend_nk, pend_uv, pend_nv, n_pend,
//                        Q_prerot, n_kv_heads, n_q_per_kv, head_dim,
//                        n_complete, chunk_tokens, sps, alpha)
//   -> out_rot [n_q_heads, head_dim] float32
//
// Q_prerot: Q already rotated to K's frame (Q @ rotation_k, inverse=True).
// out_rot:  output in V's normalised rotated frame; caller applies
//           inv_rotation to get the final attention output.
// Returns None if T_total > T_MAX (caller uses fallback two-kernel path).
torch::Tensor rice_fused_attn_decode(
    torch::Tensor words_k, torch::Tensor offsets_k, torch::Tensor ks_k,
    torch::Tensor norms_k,
    torch::Tensor words_v, torch::Tensor offsets_v, torch::Tensor ks_v,
    torch::Tensor norms_v,
    // Pending (may be None if n_pend==0)
    torch::Tensor pend_uk, torch::Tensor pend_nk,
    torch::Tensor pend_uv, torch::Tensor pend_nv,
    int64_t n_pend,
    torch::Tensor Q_prerot,
    int64_t n_kv_heads, int64_t n_q_per_kv, int64_t head_dim,
    int64_t n_complete, int64_t chunk_tokens, int64_t sps, double alpha)
{
  const int T_total = static_cast<int>(n_complete + n_pend);
  const int T_MAX   = rice_fused_attn_t_max();
  if (T_total > T_MAX || T_total == 0) {
    // Fallback sentinel: return a 0-element tensor
    return torch::empty({0}, Q_prerot.options());
  }

  const int n_q_heads = static_cast<int>(n_kv_heads * n_q_per_kv);
  auto out = torch::zeros({n_q_heads, static_cast<int>(head_dim)},
      torch::TensorOptions().dtype(torch::kFloat32).device(words_k.device()));

  cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

  auto ptr_or_null = [](const torch::Tensor& t) -> const uint16_t* {
    return t.defined() ? reinterpret_cast<const uint16_t*>(t.data_ptr()) : nullptr;
  };

  launch_rice_fused_attn_decode(
      reinterpret_cast<const uint32_t*>(words_k.data_ptr()),
      static_cast<int>(words_k.numel()),
      reinterpret_cast<const uint32_t*>(offsets_k.data_ptr()),
      reinterpret_cast<const uint8_t*>(ks_k.data_ptr()),
      static_cast<int>(offsets_k.numel()),
      reinterpret_cast<const uint16_t*>(norms_k.data_ptr()),
      reinterpret_cast<const uint32_t*>(words_v.data_ptr()),
      static_cast<int>(words_v.numel()),
      reinterpret_cast<const uint32_t*>(offsets_v.data_ptr()),
      reinterpret_cast<const uint8_t*>(ks_v.data_ptr()),
      static_cast<int>(offsets_v.numel()),
      reinterpret_cast<const uint16_t*>(norms_v.data_ptr()),
      ptr_or_null(pend_uk), ptr_or_null(pend_nk),
      ptr_or_null(pend_uv), ptr_or_null(pend_nv),
      static_cast<int>(n_pend),
      reinterpret_cast<const uint16_t*>(Q_prerot.data_ptr()),
      out.data_ptr<float>(),
      static_cast<int>(n_kv_heads), static_cast<int>(n_q_per_kv),
      static_cast<int>(head_dim), static_cast<int>(n_complete),
      static_cast<int>(chunk_tokens), static_cast<int>(sps),
      static_cast<float>(alpha),
      stream);
  return out;
}

int64_t rice_fused_attn_t_max_py() { return rice_fused_attn_t_max(); }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rice_encode",           &rice_encode,           "Encode bf16 weight -> Rice bitstream (CUDA)");
  m.def("rice_decode_into",      &rice_decode_into,      "Fused Rice decode into a bf16 scratch");
  m.def("rice_fused_gemv",       &rice_fused_gemv,       "Fused Rice decode + GEMV (batch=1)");
  m.def("rice_fused_gemm_ws",    &rice_fused_gemm_ws,    "Fused Rice decode + GEMM warp-specialized (batch≥1)");
  m.def("rice_fused_attn_decode",&rice_fused_attn_decode,"Fused Rice decode + attention (single decode step)");
  m.def("rice_fused_attn_t_max", &rice_fused_attn_t_max_py, "Max T for fused attention kernel");
}
