#include "stage2_cuda_encoder.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <cstdint>
#include <vector>

namespace {

inline bool check_cuda(cudaError_t err, std::string& error_message, const char* context) {
  if (err == cudaSuccess) return true;
  error_message = std::string(context) + ": " + cudaGetErrorString(err);
  return false;
}

__device__ __forceinline__ int zigzag_i32(int v) {
  return (v >= 0) ? (v << 1) : ((-v << 1) - 1);
}

__device__ __forceinline__ int sign_i32(int v) {
  return (v > 0) - (v < 0);
}

__device__ __forceinline__ int round_to_int(float x) {
  return static_cast<int>(nearbyintf(x));
}

__device__ __forceinline__ void fix_parity_mod4(int u[8], const float w[8]) {
  int sum = 0;
  for (int i = 0; i < 8; ++i) sum += u[i];
  if ((sum & 3) == 0) return;

  int best_j = 0;
  float best_abs = -1.0f;
  float best_delta = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const float delta = static_cast<float>(u[i]) - w[i];
    const float ad = fabsf(delta);
    if (ad > best_abs) {
      best_abs = ad;
      best_j = i;
      best_delta = delta;
    }
  }

  int s = sign_i32(static_cast<int>(best_delta > 0.0f) - static_cast<int>(best_delta < 0.0f));
  if (s == 0) s = 1;
  u[best_j] += (-2 * s);
}

__device__ __forceinline__ void encode_word_to_z(
    const uint16_t* bf16_words,
    int word_id,
    float alpha,
    int z_out[8]) {
  float x[8];
  float w[8];
  const int base = word_id * 8;
  for (int i = 0; i < 8; ++i) {
    const __nv_bfloat16 xb = *reinterpret_cast<const __nv_bfloat16*>(&bf16_words[base + i]);
    x[i] = __bfloat162float(xb);
    const float scaled = alpha * x[i];
    const __nv_bfloat16 wb = __float2bfloat16_rn(scaled);
    w[i] = __bfloat162float(wb);
  }

  int u0[8];
  int u1[8];
  for (int i = 0; i < 8; ++i) {
    u0[i] = round_to_int(w[i] * 0.5f) * 2;
    u1[i] = round_to_int((w[i] - 1.0f) * 0.5f) * 2 + 1;
  }
  fix_parity_mod4(u0, w);
  fix_parity_mod4(u1, w);

  float d0 = 0.0f;
  float d1 = 0.0f;
  for (int i = 0; i < 8; ++i) {
    const float e0 = static_cast<float>(u0[i]) - w[i];
    const float e1 = static_cast<float>(u1[i]) - w[i];
    d0 += e0 * e0;
    d1 += e1 * e1;
  }
  const int* Y = (d0 <= d1) ? u0 : u1;

  const int c = Y[0] & 1;
  int s[8];
  for (int i = 0; i < 8; ++i) {
    s[i] = (Y[i] - c) >> 1;
  }
  int known_p8 = 0;
  for (int i = 0; i < 7; ++i) known_p8 += s[i];
  known_p8 &= 1;
  const int T = (s[7] - known_p8) >> 1;

  for (int i = 0; i < 7; ++i) {
    z_out[i] = zigzag_i32(s[i]);
  }
  z_out[7] = 2 * zigzag_i32(T) + c;
}

__device__ __forceinline__ void write_bit_atomic(uint32_t* words, uint32_t bit_pos, uint32_t bit) {
  if (bit == 0U) return;
  const int word_idx = static_cast<int>(bit_pos >> 5U);
  const uint32_t bit_idx = bit_pos & 31U;
  atomicOr(&words[word_idx], 1U << (31U - bit_idx));
}

__device__ __forceinline__ void write_bit_atomic_shared(uint32_t* words, uint32_t bit_pos, uint32_t bit) {
  if (bit == 0U) return;
  const int word_idx = static_cast<int>(bit_pos >> 5U);
  const uint32_t bit_idx = bit_pos & 31U;
  atomicOr(&words[word_idx], 1U << (31U - bit_idx));
}

__global__ void compute_stream_lengths_kernel(
    const uint16_t* bf16_words,
    int num_words,
    int groups_per_thread,
    float alpha,
    uint32_t rice_k,
    int num_streams,
    uint32_t* lengths_bits) {
  uint32_t length = 0U;
  const int stream_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (stream_id >= num_streams) return;

  const int first_word = stream_id * groups_per_thread;
  for (int g = 0; g < groups_per_thread; ++g) {
    const int word_id = first_word + g;
    int z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (word_id < num_words) {
      encode_word_to_z(bf16_words, word_id, alpha, z);
    }
    for (int i = 0; i < 8; ++i) {
      const uint32_t sym = static_cast<uint32_t>(z[i]);
      length += (sym >> rice_k) + 1U + rice_k;
    }
  }
  lengths_bits[stream_id] = length;
}

__global__ void compute_stream_lengths_and_cache_u8_kernel(
    const uint16_t* bf16_words,
    int num_words,
    int groups_per_thread,
    float alpha,
    uint32_t rice_k,
    int num_streams,
    uint32_t* lengths_bits,
    uint64_t* z_cache_packed,
    uint32_t* overflow_flag) {
  __shared__ uint32_t block_overflow;
  if (threadIdx.x == 0) block_overflow = 0U;
  __syncthreads();

  uint32_t length = 0U;
  const int stream_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const bool active = (stream_id < num_streams);

  const int first_word = stream_id * groups_per_thread;
  bool local_overflow = false;
  if (active) {
    for (int g = 0; g < groups_per_thread; ++g) {
      const int word_id = first_word + g;
      int z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
      if (word_id < num_words) {
        encode_word_to_z(bf16_words, word_id, alpha, z);
      }

      if (word_id < num_words) {
        uint64_t packed = 0ULL;
        for (int i = 0; i < 8; ++i) {
          const uint32_t sym = static_cast<uint32_t>(z[i]);
          if (sym > 0xFFU) {
            local_overflow = true;
          } else {
            packed |= (static_cast<uint64_t>(sym) << (i * 8));
          }
        }
        z_cache_packed[word_id] = packed;
      }

      for (int i = 0; i < 8; ++i) {
        const uint32_t sym = static_cast<uint32_t>(z[i]);
        length += (sym >> rice_k) + 1U + rice_k;
      }
    }
  }
  if (local_overflow) {
    atomicOr(&block_overflow, 1U);
  }
  __syncthreads();
  if (threadIdx.x == 0 && block_overflow != 0U) {
    atomicExch(overflow_flag, 1U);
  }
  if (active) {
    lengths_bits[stream_id] = length;
  }
}

__global__ void write_streams_kernel(
    const uint16_t* bf16_words,
    int num_words,
    int groups_per_thread,
    float alpha,
    uint32_t rice_k,
    int num_streams,
    const uint32_t* stream_offsets_bits,
    uint32_t payload_bits,
    uint32_t* words) {
  constexpr int kSharedWordCapacity = 4096;
  __shared__ uint32_t shared_words[kSharedWordCapacity];
  __shared__ uint32_t cta_start_bit;
  __shared__ uint32_t cta_word_count;
  __shared__ int use_shared_path;

  const int block_start = static_cast<int>(blockIdx.x * blockDim.x);
  const int block_end = min(num_streams, block_start + static_cast<int>(blockDim.x));
  const int active_threads = block_end - block_start;
  const uint32_t global_start_bit = stream_offsets_bits[block_start];
  const uint32_t global_end_bit =
      (block_end < num_streams) ? stream_offsets_bits[block_end] : payload_bits;
  const uint32_t block_bits = global_end_bit - global_start_bit;
  const uint32_t block_words = (block_bits + 31U) >> 5U;

  if (threadIdx.x == 0) {
    cta_start_bit = global_start_bit;
    cta_word_count = block_words;
    use_shared_path = (block_words <= static_cast<uint32_t>(kSharedWordCapacity)) ? 1 : 0;
  }
  __syncthreads();

  const int stream_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const bool active = (threadIdx.x < active_threads);
  const int first_word = stream_id * groups_per_thread;

  if (use_shared_path) {
    for (uint32_t i = static_cast<uint32_t>(threadIdx.x); i < cta_word_count; i += blockDim.x) {
      shared_words[i] = 0U;
    }
    __syncthreads();

    if (active) {
      uint32_t bit_pos = stream_offsets_bits[stream_id] - cta_start_bit;
      for (int g = 0; g < groups_per_thread; ++g) {
        const int word_id = first_word + g;
        int z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        if (word_id < num_words) {
          encode_word_to_z(bf16_words, word_id, alpha, z);
        }
        for (int i = 0; i < 8; ++i) {
          const uint32_t sym = static_cast<uint32_t>(z[i]);
          const uint32_t q = sym >> rice_k;
          const uint32_t rem = sym & ((1U << rice_k) - 1U);

          for (uint32_t j = 0; j < q; ++j) {
            write_bit_atomic_shared(shared_words, bit_pos, 0U);
            ++bit_pos;
          }
          write_bit_atomic_shared(shared_words, bit_pos, 1U);
          ++bit_pos;
          for (int b = static_cast<int>(rice_k) - 1; b >= 0; --b) {
            write_bit_atomic_shared(shared_words, bit_pos, (rem >> b) & 1U);
            ++bit_pos;
          }
        }
      }
    }
    __syncthreads();

    const uint32_t word_base = cta_start_bit >> 5U;
    const uint32_t shift = cta_start_bit & 31U;
    for (uint32_t i = static_cast<uint32_t>(threadIdx.x); i < cta_word_count; i += blockDim.x) {
      const uint32_t v = shared_words[i];
      if (v == 0U) continue;
      if (shift == 0U) {
        atomicOr(&words[word_base + i], v);
      } else {
        atomicOr(&words[word_base + i], (v >> shift));
        atomicOr(&words[word_base + i + 1U], (v << (32U - shift)));
      }
    }
    return;
  }

  if (!active) return;

  uint32_t bit_pos = stream_offsets_bits[stream_id];
  for (int g = 0; g < groups_per_thread; ++g) {
    const int word_id = first_word + g;
    int z[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    if (word_id < num_words) {
      encode_word_to_z(bf16_words, word_id, alpha, z);
    }
    for (int i = 0; i < 8; ++i) {
      const uint32_t sym = static_cast<uint32_t>(z[i]);
      const uint32_t q = sym >> rice_k;
      const uint32_t rem = sym & ((1U << rice_k) - 1U);

      for (uint32_t j = 0; j < q; ++j) {
        write_bit_atomic(words, bit_pos, 0U);
        ++bit_pos;
      }
      write_bit_atomic(words, bit_pos, 1U);
      ++bit_pos;
      for (int b = static_cast<int>(rice_k) - 1; b >= 0; --b) {
        write_bit_atomic(words, bit_pos, (rem >> b) & 1U);
        ++bit_pos;
      }
    }
  }
}

__global__ void write_streams_from_cache_u8_kernel(
    const uint64_t* z_cache_packed,
    int num_words,
    int groups_per_thread,
    uint32_t rice_k,
    int num_streams,
    const uint32_t* stream_offsets_bits,
    uint32_t payload_bits,
    uint32_t* words) {
  constexpr int kSharedWordCapacity = 4096;
  __shared__ uint32_t shared_words[kSharedWordCapacity];
  __shared__ uint32_t cta_start_bit;
  __shared__ uint32_t cta_word_count;
  __shared__ int use_shared_path;

  const int block_start = static_cast<int>(blockIdx.x * blockDim.x);
  const int block_end = min(num_streams, block_start + static_cast<int>(blockDim.x));
  const int active_threads = block_end - block_start;
  const uint32_t global_start_bit = stream_offsets_bits[block_start];
  const uint32_t global_end_bit =
      (block_end < num_streams) ? stream_offsets_bits[block_end] : payload_bits;
  const uint32_t block_bits = global_end_bit - global_start_bit;
  const uint32_t block_words = (block_bits + 31U) >> 5U;

  if (threadIdx.x == 0) {
    cta_start_bit = global_start_bit;
    cta_word_count = block_words;
    use_shared_path = (block_words <= static_cast<uint32_t>(kSharedWordCapacity)) ? 1 : 0;
  }
  __syncthreads();

  const int stream_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  const bool active = (threadIdx.x < active_threads);
  const int first_word = stream_id * groups_per_thread;

  if (use_shared_path) {
    for (uint32_t i = static_cast<uint32_t>(threadIdx.x); i < cta_word_count; i += blockDim.x) {
      shared_words[i] = 0U;
    }
    __syncthreads();

    if (active) {
      uint32_t bit_pos = stream_offsets_bits[stream_id] - cta_start_bit;
      for (int g = 0; g < groups_per_thread; ++g) {
        const int word_id = first_word + g;
        const uint64_t packed = (word_id < num_words) ? z_cache_packed[word_id] : 0ULL;
        for (int i = 0; i < 8; ++i) {
          const uint32_t sym = static_cast<uint32_t>((packed >> (i * 8)) & 0xFFULL);
          const uint32_t q = sym >> rice_k;
          const uint32_t rem = sym & ((1U << rice_k) - 1U);

          for (uint32_t j = 0; j < q; ++j) {
            write_bit_atomic_shared(shared_words, bit_pos, 0U);
            ++bit_pos;
          }
          write_bit_atomic_shared(shared_words, bit_pos, 1U);
          ++bit_pos;
          for (int b = static_cast<int>(rice_k) - 1; b >= 0; --b) {
            write_bit_atomic_shared(shared_words, bit_pos, (rem >> b) & 1U);
            ++bit_pos;
          }
        }
      }
    }
    __syncthreads();

    const uint32_t word_base = cta_start_bit >> 5U;
    const uint32_t shift = cta_start_bit & 31U;
    for (uint32_t i = static_cast<uint32_t>(threadIdx.x); i < cta_word_count; i += blockDim.x) {
      const uint32_t v = shared_words[i];
      if (v == 0U) continue;
      if (shift == 0U) {
        atomicOr(&words[word_base + i], v);
      } else {
        atomicOr(&words[word_base + i], (v >> shift));
        atomicOr(&words[word_base + i + 1U], (v << (32U - shift)));
      }
    }
    return;
  }

  if (!active) return;

  uint32_t bit_pos = stream_offsets_bits[stream_id];
  for (int g = 0; g < groups_per_thread; ++g) {
    const int word_id = first_word + g;
    const uint64_t packed = (word_id < num_words) ? z_cache_packed[word_id] : 0ULL;
    for (int i = 0; i < 8; ++i) {
      const uint32_t sym = static_cast<uint32_t>((packed >> (i * 8)) & 0xFFULL);
      const uint32_t q = sym >> rice_k;
      const uint32_t rem = sym & ((1U << rice_k) - 1U);

      for (uint32_t j = 0; j < q; ++j) {
        write_bit_atomic(words, bit_pos, 0U);
        ++bit_pos;
      }
      write_bit_atomic(words, bit_pos, 1U);
      ++bit_pos;
      for (int b = static_cast<int>(rice_k) - 1; b >= 0; --b) {
        write_bit_atomic(words, bit_pos, (rem >> b) & 1U);
        ++bit_pos;
      }
    }
  }
}

}  // namespace

Stage2CudaEncoder::Stage2CudaEncoder(
    const std::vector<uint16_t>& bf16_words,
    float alpha,
    uint8_t rice_k,
    int groups_per_thread,
    CacheMode cache_mode,
    int threads_per_block,
    int decoder_symbols_per_stream)
    : alpha_(alpha),
      rice_k_(rice_k),
      num_words_(static_cast<int>(bf16_words.size() / 8)),
      groups_per_thread_(groups_per_thread),
      cache_mode_(cache_mode),
      num_streams_(0),
      threads_per_block_(threads_per_block),
      decoder_symbols_per_stream_(decoder_symbols_per_stream) {
  if (bf16_words.empty() || (bf16_words.size() % 8) != 0 || num_words_ <= 0) {
    error_message_ = "Invalid bf16 input shape (expected N x 8 packed as uint16).";
    return;
  }
  if (!(alpha_ > 0.0f)) {
    error_message_ = "Invalid alpha (must be > 0).";
    return;
  }
  if (groups_per_thread_ <= 0) {
    error_message_ = "Invalid groups_per_thread (must be >= 1).";
    return;
  }
  num_streams_ = (num_words_ + groups_per_thread_ - 1) / groups_per_thread_;
  if (num_streams_ <= 0) {
    error_message_ = "Invalid encoder dimensions.";
    return;
  }

  // Decoder stream layout. The emitted offset table is a subsample of the
  // per-group offsets: one entry every offset_stride_ groups, where one
  // decoder stream spans decoder_symbols_per_stream_ symbols. 0 => legacy
  // behavior (decoder stream == encoder group == 8 * groups_per_thread).
  const int group_symbols = 8 * groups_per_thread_;
  if (decoder_symbols_per_stream_ <= 0) {
    decoder_symbols_per_stream_ = group_symbols;
  }
  if (decoder_symbols_per_stream_ % group_symbols != 0) {
    error_message_ =
        "decoder_symbols_per_stream must be a positive multiple of "
        "8 * groups_per_thread.";
    return;
  }
  offset_stride_ = decoder_symbols_per_stream_ / group_symbols;
  const int total_symbols = num_words_ * 8;
  num_decoder_streams_ =
      (total_symbols + decoder_symbols_per_stream_ - 1) / decoder_symbols_per_stream_;
  if (rice_k_ > 7U) {
    error_message_ = "Invalid Rice k for Stage-2 encoder (expected [0, 7]).";
    return;
  }
  if (cache_mode_ != CacheMode::kRecompute && cache_mode_ != CacheMode::kB2U8Cache) {
    error_message_ = "Invalid cache mode.";
    return;
  }
  if (threads_per_block_ <= 0 || threads_per_block_ > 1024) {
    error_message_ = "Invalid threads_per_block.";
    return;
  }

  if (!check_cuda(cudaMalloc(&d_bf16_words_, bf16_words.size() * sizeof(uint16_t)), error_message_, "cudaMalloc d_bf16_words")) return;
  if (!check_cuda(cudaMalloc(&d_lengths_, static_cast<size_t>(num_streams_) * sizeof(uint32_t)), error_message_, "cudaMalloc d_lengths")) return;
  if (!check_cuda(cudaMalloc(&d_offsets_, static_cast<size_t>(num_streams_) * sizeof(uint32_t)), error_message_, "cudaMalloc d_offsets")) return;
  if (!check_cuda(cudaMalloc(&d_cache_overflow_flag_, sizeof(uint32_t)), error_message_, "cudaMalloc d_cache_overflow_flag")) return;

  if (cache_mode_ == CacheMode::kB2U8Cache) {
    if (!check_cuda(
            cudaMalloc(&d_z_cache_packed_, static_cast<size_t>(num_words_) * sizeof(uint64_t)),
            error_message_,
            "cudaMalloc d_z_cache_packed")) {
      return;
    }
  }

  if (!check_cuda(cudaMemcpy(
                      d_bf16_words_,
                      bf16_words.data(),
                      bf16_words.size() * sizeof(uint16_t),
                      cudaMemcpyHostToDevice),
                  error_message_,
                  "cudaMemcpy bf16_words H2D")) {
    return;
  }

  ok_ = true;
}

Stage2CudaEncoder::~Stage2CudaEncoder() {
  if (d_bf16_words_) cudaFree(d_bf16_words_);
  if (d_lengths_) cudaFree(d_lengths_);
  if (d_offsets_) cudaFree(d_offsets_);
  if (d_words_) cudaFree(d_words_);
  if (d_z_cache_packed_) cudaFree(d_z_cache_packed_);
  if (d_cache_overflow_flag_) cudaFree(d_cache_overflow_flag_);
}

bool Stage2CudaEncoder::encode(
    std::vector<uint32_t>& encoded_words,
    std::vector<uint32_t>& stream_offsets_bits,
    std::vector<uint8_t>& stream_k,
    uint32_t* total_bits,
    bool copy_output_to_host) {
  if (!ok_) return false;
  last_timing_stats_ = TimingStats{};

  const int blocks = (num_streams_ + threads_per_block_ - 1) / threads_per_block_;
  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_k1_done = nullptr;
  cudaEvent_t ev_scan_done = nullptr;
  cudaEvent_t ev_k3_done = nullptr;
  if (!check_cuda(cudaEventCreate(&ev_start), error_message_, "cudaEventCreate ev_start")) return false;
  if (!check_cuda(cudaEventCreate(&ev_k1_done), error_message_, "cudaEventCreate ev_k1_done")) return false;
  if (!check_cuda(cudaEventCreate(&ev_scan_done), error_message_, "cudaEventCreate ev_scan_done")) return false;
  if (!check_cuda(cudaEventCreate(&ev_k3_done), error_message_, "cudaEventCreate ev_k3_done")) return false;

  const bool try_b2_cache = (cache_mode_ == CacheMode::kB2U8Cache && d_z_cache_packed_ != nullptr);
  uint32_t cache_overflow = 0U;
  if (try_b2_cache) {
    if (!check_cuda(cudaMemset(d_cache_overflow_flag_, 0, sizeof(uint32_t)), error_message_, "cudaMemset d_cache_overflow_flag")) return false;
  }

  if (!check_cuda(cudaEventRecord(ev_start), error_message_, "cudaEventRecord ev_start")) return false;
  if (try_b2_cache) {
    compute_stream_lengths_and_cache_u8_kernel<<<blocks, threads_per_block_>>>(
        d_bf16_words_,
        num_words_,
        groups_per_thread_,
        alpha_,
        static_cast<uint32_t>(rice_k_),
        num_streams_,
        d_lengths_,
        d_z_cache_packed_,
        d_cache_overflow_flag_);
    if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch compute_stream_lengths_and_cache_u8_kernel")) return false;
  } else {
    compute_stream_lengths_kernel<<<blocks, threads_per_block_>>>(
        d_bf16_words_,
        num_words_,
        groups_per_thread_,
        alpha_,
        static_cast<uint32_t>(rice_k_),
        num_streams_,
        d_lengths_);
    if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch compute_stream_lengths_kernel")) return false;
  }
  if (!check_cuda(cudaEventRecord(ev_k1_done), error_message_, "cudaEventRecord ev_k1_done")) return false;

  thrust::device_ptr<uint32_t> lengths_ptr(d_lengths_);
  thrust::device_ptr<uint32_t> offsets_ptr(d_offsets_);
  thrust::exclusive_scan(lengths_ptr, lengths_ptr + num_streams_, offsets_ptr);
  if (!check_cuda(cudaGetLastError(), error_message_, "thrust::exclusive_scan lengths->offsets")) return false;
  if (!check_cuda(cudaEventRecord(ev_scan_done), error_message_, "cudaEventRecord ev_scan_done")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize after scan")) return false;

  if (try_b2_cache) {
    if (!check_cuda(cudaMemcpy(
                        &cache_overflow,
                        d_cache_overflow_flag_,
                        sizeof(uint32_t),
                        cudaMemcpyDeviceToHost),
                    error_message_,
                    "cudaMemcpy cache_overflow D2H")) {
      return false;
    }
  }

  uint32_t last_len = 0U;
  uint32_t last_off = 0U;
  if (!check_cuda(cudaMemcpy(
                      &last_len,
                      d_lengths_ + (num_streams_ - 1),
                      sizeof(uint32_t),
                      cudaMemcpyDeviceToHost),
                  error_message_,
                  "cudaMemcpy last_len D2H")) {
    return false;
  }
  if (!check_cuda(cudaMemcpy(
                      &last_off,
                      d_offsets_ + (num_streams_ - 1),
                      sizeof(uint32_t),
                      cudaMemcpyDeviceToHost),
                  error_message_,
                  "cudaMemcpy last_off D2H")) {
    return false;
  }

  constexpr uint32_t kReadAheadPadBits = 32U;
  const uint32_t total_bits_local = last_off + last_len + kReadAheadPadBits;
  const uint32_t payload_bits_local = last_off + last_len;
  const int word_count = static_cast<int>((total_bits_local + 31U) >> 5U);

  if (word_count > words_capacity_) {
    if (d_words_) {
      cudaFree(d_words_);
      d_words_ = nullptr;
      words_capacity_ = 0;
    }
    if (!check_cuda(cudaMalloc(&d_words_, static_cast<size_t>(word_count) * sizeof(uint32_t)), error_message_, "cudaMalloc d_words")) return false;
    words_capacity_ = word_count;
  }
  if (!check_cuda(cudaMemset(d_words_, 0, static_cast<size_t>(word_count) * sizeof(uint32_t)), error_message_, "cudaMemset d_words")) return false;

  // Keep B2 cache mode bit-exact even when rare out-of-range symbols appear:
  // fallback to recompute write path if K1 observed any z > 255.
  const bool use_b2_cached_write = try_b2_cache && cache_overflow == 0U;
  if (use_b2_cached_write) {
    write_streams_from_cache_u8_kernel<<<blocks, threads_per_block_>>>(
        d_z_cache_packed_,
        num_words_,
        groups_per_thread_,
        static_cast<uint32_t>(rice_k_),
        num_streams_,
        d_offsets_,
        payload_bits_local,
        d_words_);
    if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch write_streams_from_cache_u8_kernel")) return false;
  } else {
    write_streams_kernel<<<blocks, threads_per_block_>>>(
        d_bf16_words_,
        num_words_,
        groups_per_thread_,
        alpha_,
        static_cast<uint32_t>(rice_k_),
        num_streams_,
        d_offsets_,
        payload_bits_local,
        d_words_);
    if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch write_streams_kernel")) return false;
  }
  if (!check_cuda(cudaEventRecord(ev_k3_done), error_message_, "cudaEventRecord ev_k3_done")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize after write")) return false;

  float ms_k1 = 0.0f;
  float ms_scan = 0.0f;
  float ms_k3 = 0.0f;
  float ms_total = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&ms_k1, ev_start, ev_k1_done), error_message_, "cudaEventElapsedTime k1")) return false;
  if (!check_cuda(cudaEventElapsedTime(&ms_scan, ev_k1_done, ev_scan_done), error_message_, "cudaEventElapsedTime scan")) return false;
  if (!check_cuda(cudaEventElapsedTime(&ms_k3, ev_scan_done, ev_k3_done), error_message_, "cudaEventElapsedTime k3")) return false;
  if (!check_cuda(cudaEventElapsedTime(&ms_total, ev_start, ev_k3_done), error_message_, "cudaEventElapsedTime total")) return false;
  last_timing_stats_.kernel1_ms = ms_k1;
  last_timing_stats_.scan_ms = ms_scan;
  last_timing_stats_.kernel3_ms = ms_k3;
  last_timing_stats_.total_gpu_ms = ms_total;

  if (copy_output_to_host) {
    encoded_words.resize(static_cast<size_t>(word_count));
    if (!check_cuda(cudaMemcpy(
                        encoded_words.data(),
                        d_words_,
                        static_cast<size_t>(word_count) * sizeof(uint32_t),
                        cudaMemcpyDeviceToHost),
                    error_message_,
                    "cudaMemcpy encoded_words D2H")) {
      return false;
    }

    // Emit one offset per decoder stream, not per encoder thread-group. When
    // offset_stride_ == 1 this is the full per-group table (legacy behavior);
    // otherwise it is a strided subsample at the decoder-stream boundaries.
    stream_offsets_bits.resize(static_cast<size_t>(num_decoder_streams_));
    if (offset_stride_ == 1) {
      if (!check_cuda(cudaMemcpy(
                          stream_offsets_bits.data(),
                          d_offsets_,
                          static_cast<size_t>(num_decoder_streams_) * sizeof(uint32_t),
                          cudaMemcpyDeviceToHost),
                      error_message_,
                      "cudaMemcpy offsets D2H")) {
        return false;
      }
    } else {
      // Strided D2H: pick every offset_stride_-th group offset. spitch walks
      // offset_stride_ source elements per destination element.
      if (!check_cuda(cudaMemcpy2D(
                          stream_offsets_bits.data(),
                          sizeof(uint32_t),
                          d_offsets_,
                          static_cast<size_t>(offset_stride_) * sizeof(uint32_t),
                          sizeof(uint32_t),
                          static_cast<size_t>(num_decoder_streams_),
                          cudaMemcpyDeviceToHost),
                      error_message_,
                      "cudaMemcpy2D strided offsets D2H")) {
        return false;
      }
    }

    stream_k.assign(static_cast<size_t>(num_decoder_streams_), rice_k_);
  } else {
    encoded_words.clear();
    stream_offsets_bits.clear();
    stream_k.clear();
  }

  if (total_bits != nullptr) {
    *total_bits = total_bits_local;
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_k1_done);
  cudaEventDestroy(ev_scan_done);
  cudaEventDestroy(ev_k3_done);
  return true;
}
