// lattice_ext.cu — PyTorch CUDA extension for int8 IMMA inference.
//
// Bridges the MetalRice compression pipeline (RHT + E8int/Rice encode/decode)
// with a cublasLt int8 TN GEMM so an nn.Linear can keep its weight compressed
// in HBM and reconstruct-and-multiply per forward without a bf16 scratch tensor.
//
// New device code (all else is glue to existing kernels in cuda/):
//   rowmax_cast_kernel  — per-token row-absmax → int8 cast + dequant factor
//   dequant_kernel      — int32 col-major output → bf16 row-major
//
// Exposed ops (via PYBIND11_MODULE lattice_int8_ext):
//   PackedWeight  — opaque handle: encoder, decoder, rotation, metadata
//   lattice_pack(W, alpha, rice_k, had_size, seed, S)  — offline encode
//   rht_apply(x, packed)                               — x' = xR (last dim)
//   lattice_linear_forward(x, packed)                  — rht→int8→GEMM→deq

#include "rht_cuda.h"
#include "stage2_cuda_decoder.h"
#include "stage2_cuda_encoder.h"

#include <torch/extension.h>

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#define LX_CUDA(call)                                                          \
  do {                                                                         \
    cudaError_t e__ = (call);                                                  \
    TORCH_CHECK(e__ == cudaSuccess, "CUDA error: ", cudaGetErrorString(e__),   \
                " at ", __FILE__, ":", __LINE__);                              \
  } while (0)
#define LX_LT(call)                                                            \
  do {                                                                         \
    cublasStatus_t s__ = (call);                                               \
    TORCH_CHECK(s__ == CUBLAS_STATUS_SUCCESS, "cublasLt error ", (int)s__,     \
                " at ", __FILE__, ":", __LINE__);                              \
  } while (0)

constexpr int kGroupsPerThread = 1;
constexpr int kThreadsPerBlock = 256;

cublasLtHandle_t lt_handle() {
  static cublasLtHandle_t lt = [] {
    cublasLtHandle_t h; LX_LT(cublasLtCreate(&h)); return h;
  }();
  return lt;
}
void* lt_workspace(size_t bytes) {
  static void* ws = nullptr; static size_t cap = 0;
  if (bytes > cap) {
    if (ws) cudaFree(ws);
    LX_CUDA(cudaMalloc(&ws, bytes)); cap = bytes;
  }
  return ws;
}

// Per-token row-absmax → int8 cast.
// deq[m] = amax / (127 * alpha)  so that  y = C * deq  recovers the bf16 output.
__global__ void rowmax_cast_kernel(const __nv_bfloat16* __restrict__ xp,
                                   int8_t*  __restrict__ x8,
                                   float*   __restrict__ deq,
                                   int M, int K, float alpha) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const long base = static_cast<long>(m) * K;
  float local = 0.0f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    local = fmaxf(local, fabsf(__bfloat162float(xp[base + k])));
  __shared__ float red[kThreadsPerBlock];
  red[threadIdx.x] = local; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float amax = red[0];
  const float sc = amax > 0.0f ? 127.0f / amax : 0.0f;
  if (threadIdx.x == 0) deq[m] = amax / (127.0f * alpha);
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    int q = __float2int_rn(__bfloat162float(xp[base + k]) * sc);
    q = q < -127 ? -127 : (q > 127 ? 127 : q);
    x8[base + k] = static_cast<int8_t>(q);
  }
}

// int32 col-major [M,N] → bf16 row-major, scaled by per-row deq[m].
__global__ void dequant_kernel(const int32_t* __restrict__ C,
                               const float*   __restrict__ deq,
                               __nv_bfloat16* __restrict__ y,
                               int M, int N) {
  const long i = static_cast<long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= static_cast<long>(M) * N) return;
  const int m = static_cast<int>(i / N), n = static_cast<int>(i % N);
  y[i] = __float2bfloat16(static_cast<float>(C[static_cast<long>(n)*M + m]) * deq[m]);
}

// cublasLt int8 TN GEMM: A[M,K] × B[K,N] → C[M,N] (int32, col-major).
void int8_gemm_tn(const int8_t* A, const int8_t* B, int32_t* C, int M, int N, int K) {
  auto lt = lt_handle();
  const size_t ws = 32u << 20; void* wsp = lt_workspace(ws);
  cublasLtMatmulDesc_t desc;
  LX_LT(cublasLtMatmulDescCreate(&desc, CUBLAS_COMPUTE_32I, CUDA_R_32I));
  const cublasOperation_t opT = CUBLAS_OP_T, opN = CUBLAS_OP_N;
  LX_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSA, &opT, sizeof opT));
  LX_LT(cublasLtMatmulDescSetAttribute(desc, CUBLASLT_MATMUL_DESC_TRANSB, &opN, sizeof opN));
  cublasLtMatrixLayout_t la, lb, lc;
  LX_LT(cublasLtMatrixLayoutCreate(&la, CUDA_R_8I, K, M, K));
  LX_LT(cublasLtMatrixLayoutCreate(&lb, CUDA_R_8I, K, N, K));
  LX_LT(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32I, M, N, M));
  cublasLtMatmulPreference_t pref; LX_LT(cublasLtMatmulPreferenceCreate(&pref));
  LX_LT(cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &ws, sizeof ws));
  cublasLtMatmulHeuristicResult_t heur{}; int nh = 0;
  LX_LT(cublasLtMatmulAlgoGetHeuristic(lt, desc, la, lb, lc, lc, pref, 1, &heur, &nh));
  TORCH_CHECK(nh > 0, "no int8 GEMM algo for M=", M, " N=", N, " K=", K);
  const int32_t one = 1, zero = 0;
  LX_LT(cublasLtMatmul(lt, desc, &one, A, la, B, lb, &zero, C, lc, C, lc,
                       &heur.algo, wsp, ws, 0));
  cublasLtMatmulPreferenceDestroy(pref);
  cublasLtMatrixLayoutDestroy(la); cublasLtMatrixLayoutDestroy(lb);
  cublasLtMatrixLayoutDestroy(lc); cublasLtMatmulDescDestroy(desc);
}

}  // namespace

// ── PackedWeight ─────────────────────────────────────────────────────────────
struct PackedWeight {
  std::unique_ptr<RhtCuda>           rht;
  std::unique_ptr<Stage2CudaDecoder> dec;
  int N = 0, K = 0, S = 0, had_size = 0;
  double alpha = 0.0;
  int64_t compressed_bytes = 0;
  int64_t bf16_bytes = 0;
  std::vector<uint8_t> decode_scratch;
};

static std::shared_ptr<PackedWeight>
lattice_pack(torch::Tensor weight, double alpha, int64_t rice_k,
             int64_t had_size, int64_t seed, int64_t S) {
  TORCH_CHECK(weight.is_cuda() && weight.scalar_type() == at::kBFloat16
              && weight.dim() == 2, "weight must be 2-D CUDA bf16");
  weight = weight.contiguous();
  const int N = weight.size(0), K = weight.size(1);
  const long numW = static_cast<long>(N) * K;
  TORCH_CHECK(numW % 8 == 0 && S % 8 == 0 && numW % S == 0,
              "N*K must be multiple of 8 and S must divide N*K");

  auto pw = std::make_shared<PackedWeight>();
  pw->N = N; pw->K = K; pw->S = S; pw->had_size = had_size;
  pw->alpha = alpha; pw->bf16_bytes = 2 * numW;

  pw->rht = std::make_unique<RhtCuda>(had_size, K, static_cast<uint32_t>(seed));
  TORCH_CHECK(pw->rht->ok(), pw->rht->error_message());
  auto Wp = torch::empty_like(weight);
  TORCH_CHECK(pw->rht->apply_lastdim(weight.data_ptr(), Wp.data_ptr(), N),
              pw->rht->error_message());
  LX_CUDA(cudaDeviceSynchronize());

  std::vector<uint16_t> Wp_host(numW);
  LX_CUDA(cudaMemcpy(Wp_host.data(), Wp.data_ptr(), numW * 2, cudaMemcpyDeviceToHost));
  Stage2CudaEncoder enc(Wp_host, static_cast<float>(alpha),
                        static_cast<uint8_t>(rice_k), kGroupsPerThread,
                        Stage2CudaEncoder::CacheMode::kB2U8Cache,
                        kThreadsPerBlock, static_cast<int>(S));
  TORCH_CHECK(enc.ok(), enc.error_message());
  std::vector<uint32_t> words, offs; std::vector<uint8_t> sk; uint32_t tb = 0;
  TORCH_CHECK(enc.encode(words, offs, sk, &tb), enc.error_message());
  pw->compressed_bytes = (int64_t)words.size()*4 + (int64_t)offs.size()*4 + sk.size();
  pw->dec = std::make_unique<Stage2CudaDecoder>(words, offs, sk, S, numW, kThreadsPerBlock);
  TORCH_CHECK(pw->dec->ok(), pw->dec->error_message());
  return pw;
}

static torch::Tensor rht_apply(torch::Tensor x, std::shared_ptr<PackedWeight> pw) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16);
  x = x.contiguous();
  TORCH_CHECK(x.size(-1) == pw->K);
  const long rows = x.numel() / pw->K;
  auto out = torch::empty_like(x);
  TORCH_CHECK(pw->rht->apply_lastdim(x.data_ptr(), out.data_ptr(), rows),
              pw->rht->error_message());
  return out;
}

static torch::Tensor lattice_linear_forward(torch::Tensor x,
                                            std::shared_ptr<PackedWeight> pw) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16);
  x = x.contiguous();
  const int K = pw->K, N = pw->N;
  TORCH_CHECK(x.size(-1) == K);
  const int M = x.numel() / K;

  // x' = xR
  auto xp = torch::empty_like(x);
  TORCH_CHECK(pw->rht->apply_lastdim(x.data_ptr(), xp.data_ptr(), M),
              pw->rht->error_message());

  // per-token int8 cast
  auto x8  = torch::empty({M, K}, x.options().dtype(at::kChar));
  auto deq = torch::empty({M},    x.options().dtype(at::kFloat));
  rowmax_cast_kernel<<<M, kThreadsPerBlock>>>(
      reinterpret_cast<const __nv_bfloat16*>(xp.data_ptr()),
      reinterpret_cast<int8_t*>(x8.data_ptr()),
      deq.data_ptr<float>(), M, K, static_cast<float>(pw->alpha));
  LX_CUDA(cudaGetLastError());

  // decode W → int8 on device
  TORCH_CHECK(pw->dec->decode_fused_to(OutputDtype::kInt8, pw->decode_scratch,
                                       static_cast<float>(pw->alpha), false),
              pw->dec->error_message());
  const int8_t* W8 = reinterpret_cast<const int8_t*>(pw->dec->device_byte_output());

  // int8 GEMM → int32 col-major
  auto C = torch::empty({M, N}, x.options().dtype(at::kInt));
  int8_gemm_tn(reinterpret_cast<const int8_t*>(x8.data_ptr()), W8,
               C.data_ptr<int32_t>(), M, N, K);

  // dequant → bf16 row-major
  auto y = torch::empty({M, N}, x.options());
  const long tot = static_cast<long>(M) * N;
  dequant_kernel<<<(tot + 255) / 256, 256>>>(
      C.data_ptr<int32_t>(), deq.data_ptr<float>(),
      reinterpret_cast<__nv_bfloat16*>(y.data_ptr()), M, N);
  LX_CUDA(cudaGetLastError());

  auto shape = x.sizes().vec(); shape.back() = N;
  return y.reshape(shape);
}

// ── FP8 E4M3 path ───────────────────────────────────────────────────────────
#include <cuda_fp8.h>

// Per-row amax → FP8 E4M3 cast.  FP8 E4M3 max = 448.
// scale_a[m] = amax / 448  so that  X_float = X_fp8 * scale_a  (per-row).
__global__ void fp8_rowmax_cast_kernel(const __nv_bfloat16* __restrict__ xp,
                                        uint8_t*  __restrict__ x8,
                                        float*    __restrict__ scale_a,
                                        int M, int K) {
  const int m = blockIdx.x;
  if (m >= M) return;
  const long base = static_cast<long>(m) * K;
  float local = 0.f;
  for (int k = threadIdx.x; k < K; k += blockDim.x)
    local = fmaxf(local, fabsf(__bfloat162float(xp[base + k])));
  __shared__ float red[kThreadsPerBlock];
  red[threadIdx.x] = local; __syncthreads();
  for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
    if (threadIdx.x < s) red[threadIdx.x] = fmaxf(red[threadIdx.x], red[threadIdx.x + s]);
    __syncthreads();
  }
  const float amax     = red[0];
  const float fp8_max  = 448.f;
  const float inv_sc   = amax > 0.f ? fp8_max / amax : 0.f;
  if (threadIdx.x == 0) scale_a[m] = amax / fp8_max;
  for (int k = threadIdx.x; k < K; k += blockDim.x) {
    float v = __bfloat162float(xp[base + k]) * inv_sc;
    v = fmaxf(-fp8_max, fminf(fp8_max, v));
    x8[base + k] = __nv_fp8_e4m3(v).__x;
  }
}

// Returns (x_fp8_u8 [M,K] uint8, scale_a [M] float32).
// Caller views x_fp8_u8 as torch.float8_e4m3fn for use with torch._scaled_mm.
static std::pair<torch::Tensor, torch::Tensor>
lattice_rht_fp8_cast(torch::Tensor x, std::shared_ptr<PackedWeight> pw) {
  TORCH_CHECK(x.is_cuda() && x.scalar_type() == at::kBFloat16);
  x = x.contiguous();
  TORCH_CHECK(x.size(-1) == pw->K);
  const int K = pw->K, M = static_cast<int>(x.numel() / K);
  auto xp     = torch::empty_like(x);
  TORCH_CHECK(pw->rht->apply_lastdim(x.data_ptr(), xp.data_ptr(), M),
              pw->rht->error_message());
  auto x8      = torch::empty({M, K}, x.options().dtype(at::kByte));
  auto scale_a = torch::empty({M},    x.options().dtype(at::kFloat));
  fp8_rowmax_cast_kernel<<<M, kThreadsPerBlock>>>(
      reinterpret_cast<const __nv_bfloat16*>(xp.data_ptr()),
      x8.data_ptr<uint8_t>(), scale_a.data_ptr<float>(), M, K);
  LX_CUDA(cudaGetLastError());
  auto shape = x.sizes().vec(); shape.back() = K;
  return {x8.reshape(shape), scale_a};
}

// Decodes Rice W → FP8 E4M3, returned as uint8 [N,K].
// Values represent Y/alpha (unit-variance weights); scale_b = 1.0.
// Caller views the result as torch.float8_e4m3fn.
static torch::Tensor lattice_decode_fp8(std::shared_ptr<PackedWeight> pw) {
  TORCH_CHECK(pw->dec->decode_fused_to(OutputDtype::kFp8E4M3, pw->decode_scratch,
                                       static_cast<float>(pw->alpha), false),
              pw->dec->error_message());
  const uint8_t* W8 = reinterpret_cast<const uint8_t*>(pw->dec->device_byte_output());
  auto out = torch::empty({pw->N, pw->K},
                          torch::TensorOptions().dtype(at::kByte).device(at::kCUDA));
  LX_CUDA(cudaMemcpyAsync(out.data_ptr(), W8,
                          static_cast<size_t>(pw->N) * pw->K,
                          cudaMemcpyDeviceToDevice, 0));
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  py::class_<PackedWeight, std::shared_ptr<PackedWeight>>(m, "PackedWeight")
      .def_readonly("N",                &PackedWeight::N)
      .def_readonly("K",                &PackedWeight::K)
      .def_readonly("S",                &PackedWeight::S)
      .def_readonly("had_size",         &PackedWeight::had_size)
      .def_readonly("alpha",            &PackedWeight::alpha)
      .def_readonly("compressed_bytes", &PackedWeight::compressed_bytes)
      .def_readonly("bf16_bytes",       &PackedWeight::bf16_bytes);
  m.def("lattice_pack",            &lattice_pack,
        "Offline pack: W' = RᵀW → E8int/Rice encode → PackedWeight",
        py::arg("weight"), py::arg("alpha"), py::arg("rice_k") = 2,
        py::arg("had_size") = 1024, py::arg("seed") = 1337, py::arg("S") = 512);
  m.def("rht_apply",               &rht_apply,  "x' = xR along the last dim");
  m.def("lattice_linear_forward",  &lattice_linear_forward,
        "rht → int8 cast → E8int decode → int8 IMMA GEMM → dequant → bf16");
  m.def("lattice_rht_fp8_cast",   &lattice_rht_fp8_cast,
        "rht → per-row FP8 E4M3 cast; returns (x_fp8_u8 [M,K], scale_a [M])");
  m.def("lattice_decode_fp8",     &lattice_decode_fp8,
        "E8int/Rice decode → FP8 E4M3 uint8 [N,K]; scale_b = 1.0");
}
