#include "rht_cuda.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <random>

namespace {

inline bool check_cuda(cudaError_t err, std::string& msg, const char* ctx) {
  if (err == cudaSuccess) return true;
  msg = std::string(ctx) + ": " + cudaGetErrorString(err);
  return false;
}

constexpr int kWarpsPerBlock = 8;
constexpr int kTPB = kWarpsPerBlock * 32;

// One WARP transforms one had_size-block (a contiguous H-segment of one row),
// entirely in registers + warp shuffles — no shared memory, no __syncthreads.
// y = (1/sqrt(H)) · FWHT_H(x ⊙ d).
//
// Element e of the block is held by lane (e & 31) in register slot (e >> 5), i.e.
// e = slot*32 + lane, SLOTS = H/32 registers per lane. The FWHT factorizes into
// independent per-bit butterfly stages (they commute), split by this mapping into:
//   * 5 low stages (len 1,2,4,8,16): partner is lane^len, same slot -> __shfl_xor.
//   * log2(H)-5 high stages (len 32..H/2): partner is slot^(len/32), same lane ->
//     pure register swaps, no communication.
// Signs are indexed by the block's position within its row so the same d
// (length K) applies to every row.
template <int H>
__global__ void rht_lastdim_kernel(
    const __nv_bfloat16* __restrict__ X,
    const int8_t* __restrict__ signs,
    __nv_bfloat16* __restrict__ Y,
    int blocks_per_row,
    int total_blocks) {
  constexpr int SLOTS = H / 32;
  const int warp_global =
      (blockIdx.x * (blockDim.x >> 5)) + static_cast<int>(threadIdx.x >> 5);
  if (warp_global >= total_blocks) return;

  const int lane = static_cast<int>(threadIdx.x & 31U);
  const long base = static_cast<long>(warp_global) * H;
  const int sign_base = (warp_global % blocks_per_row) * H;

  float reg[SLOTS];
#pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    const int e = s * 32 + lane;
    reg[s] = __bfloat162float(X[base + e]) * static_cast<float>(signs[sign_base + e]);
  }

  // Low stages: partner lane = lane ^ len; the lane whose `len` bit is 0 keeps
  // the sum, the other the difference.
#pragma unroll
  for (int len = 1; len < 32; len <<= 1) {
    const bool low = (lane & len) == 0;
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
      const float partner = __shfl_xor_sync(0xFFFFFFFFu, reg[s], len);
      reg[s] = low ? (reg[s] + partner) : (partner - reg[s]);
    }
  }

  // High stages: butterfly between register slots s and s^ls (ls = len/32).
#pragma unroll
  for (int ls = 1; ls < SLOTS; ls <<= 1) {
#pragma unroll
    for (int s = 0; s < SLOTS; ++s) {
      if ((s & ls) == 0) {
        const float a = reg[s];
        const float b = reg[s + ls];
        reg[s] = a + b;
        reg[s + ls] = a - b;
      }
    }
  }

  const float inv = rsqrtf(static_cast<float>(H));
#pragma unroll
  for (int s = 0; s < SLOTS; ++s) {
    const int e = s * 32 + lane;
    Y[base + e] = __float2bfloat16(reg[s] * inv);
  }
}

}  // namespace

RhtCuda::RhtCuda(int had_size, int K, uint32_t seed)
    : had_size_(had_size), K_(K) {
  if (had_size != 256 && had_size != 512 && had_size != 1024 && had_size != 2048) {
    error_message_ = "had_size must be one of {256,512,1024,2048}.";
    return;
  }
  if (K <= 0 || (K % had_size) != 0) {
    error_message_ = "K must be a positive multiple of had_size.";
    return;
  }

  // Fixed random ±1 diagonal D (the "R" in RHT). Seeded so it's reproducible and
  // can be regenerated offline for the W rotation.
  host_signs_.resize(static_cast<size_t>(K));
  std::mt19937 rng(seed);
  std::uniform_int_distribution<int> coin(0, 1);
  for (int i = 0; i < K; ++i) host_signs_[i] = coin(rng) ? 1 : -1;

  if (!check_cuda(cudaMalloc(&d_signs_, static_cast<size_t>(K) * sizeof(int8_t)),
                  error_message_, "cudaMalloc d_signs")) {
    return;
  }
  if (!check_cuda(cudaMemcpy(d_signs_, host_signs_.data(), static_cast<size_t>(K) * sizeof(int8_t),
                             cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy signs H2D")) {
    return;
  }
  ok_ = true;
}

RhtCuda::~RhtCuda() {
  if (d_signs_) cudaFree(d_signs_);
}

bool RhtCuda::apply_lastdim(const void* d_x_bf16, void* d_out_bf16, int rows, void* cuda_stream) {
  if (!ok_) return false;
  if (d_x_bf16 == nullptr || d_out_bf16 == nullptr || rows <= 0) {
    error_message_ = "apply_lastdim: invalid arguments.";
    return false;
  }
  const auto* X = static_cast<const __nv_bfloat16*>(d_x_bf16);
  auto* Y = static_cast<__nv_bfloat16*>(d_out_bf16);
  cudaStream_t s = static_cast<cudaStream_t>(cuda_stream);

  const int blocks_per_row = K_ / had_size_;
  const int total_blocks = rows * blocks_per_row;            // one warp each
  const int grid = (total_blocks + kWarpsPerBlock - 1) / kWarpsPerBlock;
  switch (had_size_) {
    case 256:  rht_lastdim_kernel<256><<<grid, kTPB, 0, s>>>(X, d_signs_, Y, blocks_per_row, total_blocks); break;
    case 512:  rht_lastdim_kernel<512><<<grid, kTPB, 0, s>>>(X, d_signs_, Y, blocks_per_row, total_blocks); break;
    case 1024: rht_lastdim_kernel<1024><<<grid, kTPB, 0, s>>>(X, d_signs_, Y, blocks_per_row, total_blocks); break;
    case 2048: rht_lastdim_kernel<2048><<<grid, kTPB, 0, s>>>(X, d_signs_, Y, blocks_per_row, total_blocks); break;
  }
  return check_cuda(cudaGetLastError(), error_message_, "rht_lastdim_kernel launch");
}
