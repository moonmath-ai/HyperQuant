#include "stage3_cuda_decoder.h"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <string>
#include <vector>

namespace {

inline bool check_cuda(cudaError_t err, std::string& error_message, const char* context) {
  if (err == cudaSuccess) return true;
  error_message = std::string(context) + ": " + cudaGetErrorString(err);
  return false;
}

inline uint32_t read_bit_host(const std::vector<uint32_t>& words, uint32_t bit_pos) {
  const uint32_t word_idx = bit_pos >> 5U;
  const uint32_t bit_idx = bit_pos & 31U;
  if (word_idx >= words.size()) return 0U;
  return (words[word_idx] >> (31U - bit_idx)) & 1U;
}

inline uint32_t read_bits_host(const std::vector<uint32_t>& words, uint32_t& bit_pos, int nbits) {
  uint32_t v = 0U;
  for (int i = 0; i < nbits; ++i) {
    v = (v << 1) | read_bit_host(words, bit_pos);
    ++bit_pos;
  }
  return v;
}

inline bool consume_symbol_host_bounded(
    const std::vector<uint32_t>& words, uint32_t& bit_pos, int rice_k, uint32_t end_bit_exclusive) {
  while (bit_pos < end_bit_exclusive && read_bit_host(words, bit_pos) == 0U) {
    ++bit_pos;
  }
  if (bit_pos >= end_bit_exclusive) {
    return false;
  }
  ++bit_pos;  // unary terminator
  if (bit_pos + static_cast<uint32_t>(rice_k) > end_bit_exclusive) {
    return false;
  }
  (void)read_bits_host(words, bit_pos, rice_k);
  return true;
}

void build_chunk_plan_host(
    const std::vector<uint32_t>& words,
    int rice_k,
    int total_symbols,
    int subsequence_words,
    std::vector<uint32_t>& chunk_start_bits,
    std::vector<uint32_t>& chunk_output_offsets,
    uint32_t& total_payload_bits,
    std::string& error_message) {
  const uint32_t total_bits = static_cast<uint32_t>(words.size() * 32U);
  const int num_subsequences = static_cast<int>((words.size() + static_cast<size_t>(subsequence_words) - 1U) /
                                                static_cast<size_t>(subsequence_words));
  chunk_start_bits.assign(num_subsequences, 0U);
  std::vector<uint32_t> chunk_counts(num_subsequences, 0U);

  uint32_t bit_pos = 0U;
  int symbols_decoded = 0;
  for (int s = 0; s < num_subsequences; ++s) {
    chunk_start_bits[s] = bit_pos;
    const uint32_t chunk_word_end =
        static_cast<uint32_t>(std::min<size_t>(words.size(), static_cast<size_t>(s + 1) * static_cast<size_t>(subsequence_words)));
    const uint32_t chunk_hard_end_bit = chunk_word_end * 32U;

    uint32_t count = 0U;
    while (symbols_decoded < total_symbols && bit_pos < total_bits) {
      const uint32_t symbol_start = bit_pos;
      if (symbol_start >= chunk_hard_end_bit) break;
      if (!consume_symbol_host_bounded(words, bit_pos, rice_k, total_bits)) {
        error_message = "Stage-3 chunk-plan decode exceeded bitstream bounds.";
        return;
      }
      ++count;
      ++symbols_decoded;
    }
    chunk_counts[s] = count;
  }

  // Exhausted all subsequences before total_symbols: continue on the last chunk.
  while (symbols_decoded < total_symbols) {
    if (!consume_symbol_host_bounded(words, bit_pos, rice_k, total_bits)) {
      error_message = "Stage-3 chunk-plan decode could not reach total_symbols within bitstream bounds.";
      return;
    }
    ++chunk_counts.back();
    ++symbols_decoded;
  }
  total_payload_bits = bit_pos;

  chunk_output_offsets.assign(num_subsequences, 0U);
  uint32_t running = 0U;
  for (int s = 0; s < num_subsequences; ++s) {
    chunk_output_offsets[s] = running;
    running += chunk_counts[s];
  }
}

__device__ __forceinline__ uint32_t read_bit_device(const uint32_t* words, int word_count, uint32_t bit_pos) {
  const uint32_t word_idx = bit_pos >> 5U;
  const uint32_t bit_idx = bit_pos & 31U;
  if (word_idx >= static_cast<uint32_t>(word_count)) return 0U;
  return (words[word_idx] >> (31U - bit_idx)) & 1U;
}

__device__ __forceinline__ uint32_t read_bits_device(const uint32_t* words, int word_count, uint32_t& bit_pos, int nbits) {
  uint32_t v = 0U;
  for (int i = 0; i < nbits; ++i) {
    v = (v << 1) | read_bit_device(words, word_count, bit_pos);
    ++bit_pos;
  }
  return v;
}

__device__ __forceinline__ bool read_symbol_device_bounded(
    const uint32_t* words, int word_count, uint32_t& bit_pos, int rice_k, uint32_t end_bit_exclusive, uint32_t& sym_out) {
  uint32_t q = 0U;
  while (bit_pos < end_bit_exclusive && read_bit_device(words, word_count, bit_pos) == 0U) {
    ++bit_pos;
    ++q;
  }
  if (bit_pos >= end_bit_exclusive) {
    return false;
  }
  ++bit_pos;  // unary terminator
  if (bit_pos + static_cast<uint32_t>(rice_k) > end_bit_exclusive) {
    return false;
  }
  const uint32_t rem = read_bits_device(words, word_count, bit_pos, rice_k);
  sym_out = (q << rice_k) | rem;
  return true;
}

__global__ void phase1_provisional_counts_kernel(
    const uint32_t* words,
    int word_count,
    int num_subsequences,
    int subsequence_words,
    int rice_k,
    uint32_t* provisional_counts) {
  const int sid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (sid >= num_subsequences) return;

  const uint32_t start_bit = static_cast<uint32_t>(sid * subsequence_words * 32);
  const uint32_t end_word = static_cast<uint32_t>(min(word_count, (sid + 1) * subsequence_words));
  const uint32_t end_bit = end_word * 32U;

  uint32_t bit_pos = start_bit;
  uint32_t count = 0U;
  while (bit_pos < end_bit) {
    const uint32_t symbol_start = bit_pos;
    if (symbol_start >= end_bit) break;
    uint32_t sym_unused = 0U;
    if (!read_symbol_device_bounded(words, word_count, bit_pos, rice_k, end_bit, sym_unused)) {
      break;
    }
    ++count;
  }
  provisional_counts[sid] = count;
}

__global__ void phase1_populate_sync_points_kernel(
    const uint32_t* words,
    int word_count,
    const uint32_t* chunk_start_bits,
    int num_subsequences,
    int subsequence_words,
    int rice_k,
    uint32_t* sync_end_bits,
    uint32_t* sync_symbol_counts,
    uint8_t* sync_stable) {
  const int sid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (sid >= num_subsequences) return;

  uint32_t bit_pos = chunk_start_bits[sid];
  const uint32_t hard_end_word =
      static_cast<uint32_t>(min(word_count, (sid + 1) * subsequence_words));
  const uint32_t hard_end_bit = hard_end_word * 32U;

  uint32_t count = 0U;
  while (bit_pos < hard_end_bit) {
    uint32_t sym_unused = 0U;
    if (!read_symbol_device_bounded(words, word_count, bit_pos, rice_k, hard_end_bit, sym_unused)) {
      break;
    }
    ++count;
  }

  sync_end_bits[sid] = bit_pos;
  sync_symbol_counts[sid] = count;
  sync_stable[sid] = (bit_pos <= hard_end_bit) ? 1U : 0U;
}

__global__ void phase2_refine_sync_starts_kernel(
    const uint32_t* sync_start_bits_in,
    const uint32_t* sync_end_bits,
    int num_subsequences,
    uint32_t* sync_start_bits_out,
    int* changed_flag) {
  const int sid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (sid >= num_subsequences) return;

  uint32_t next_start = 0U;
  if (sid > 0) {
    next_start = sync_end_bits[sid - 1];
  }
  sync_start_bits_out[sid] = next_start;
  if (next_start != sync_start_bits_in[sid]) {
    atomicExch(changed_flag, 1);
  }
}

__global__ void phase3_count_from_sync_starts_kernel(
    const uint32_t* words,
    int word_count,
    const uint32_t* sync_start_bits,
    int num_subsequences,
    int rice_k,
    uint32_t total_payload_bits,
    uint32_t* sync_end_bits,
    uint32_t* sync_symbol_counts,
    uint8_t* sync_stable) {
  const int sid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (sid >= num_subsequences) return;

  uint32_t start_bit = sync_start_bits[sid];
  uint32_t end_bit = (sid + 1 < num_subsequences) ? sync_start_bits[sid + 1] : total_payload_bits;
  if (start_bit > end_bit) {
    start_bit = end_bit;
  }
  if (end_bit > total_payload_bits) {
    end_bit = total_payload_bits;
  }

  uint32_t bit_pos = start_bit;
  uint32_t count = 0U;
  while (bit_pos < end_bit) {
    uint32_t sym_unused = 0U;
    if (!read_symbol_device_bounded(words, word_count, bit_pos, rice_k, end_bit, sym_unused)) {
      break;
    }
    ++count;
  }

  sync_end_bits[sid] = bit_pos;
  sync_symbol_counts[sid] = count;
  sync_stable[sid] = (bit_pos <= end_bit) ? 1U : 0U;
}

__global__ void phase4_decode_write_kernel(
    const uint32_t* words,
    int word_count,
    const uint32_t* chunk_start_bits,
    const uint32_t* chunk_end_bits,
    const uint32_t* chunk_output_offsets,
    const uint32_t* chunk_symbol_counts,
    int num_subsequences,
    int rice_k,
    uint8_t* out) {
  const int sid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (sid >= num_subsequences) return;

  uint32_t bit_pos = chunk_start_bits[sid];
  const uint32_t end_bit = chunk_end_bits[sid];
  uint32_t out_pos = chunk_output_offsets[sid];
  const uint32_t next_out_pos = out_pos + chunk_symbol_counts[sid];

  while (bit_pos < end_bit && out_pos < next_out_pos) {
    uint32_t sym = 0U;
    if (!read_symbol_device_bounded(words, word_count, bit_pos, rice_k, end_bit, sym)) {
      break;
    }
    out[out_pos] = static_cast<uint8_t>(sym & 0xFFU);
    ++out_pos;
  }
}

__global__ void phase4_decode_write_validated_kernel(
    const uint32_t* words,
    int word_count,
    const uint32_t* chunk_start_bits,
    const uint32_t* chunk_output_offsets,
    int num_subsequences,
    int rice_k,
    uint32_t total_payload_bits,
    int total_symbols,
    uint8_t* out) {
  const int sid = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (sid >= num_subsequences) return;

  uint32_t bit_pos = chunk_start_bits[sid];
  const uint32_t end_bit = (sid + 1 < num_subsequences) ? chunk_start_bits[sid + 1] : total_payload_bits;
  uint32_t out_pos = chunk_output_offsets[sid];
  const uint32_t next_out_pos =
      (sid + 1 < num_subsequences) ? chunk_output_offsets[sid + 1] : static_cast<uint32_t>(total_symbols);

  while (bit_pos < end_bit && out_pos < next_out_pos) {
    uint32_t sym = 0U;
    if (!read_symbol_device_bounded(words, word_count, bit_pos, rice_k, end_bit, sym)) {
      break;
    }
    out[out_pos] = static_cast<uint8_t>(sym & 0xFFU);
    ++out_pos;
  }
}

}  // namespace

Stage3CudaDecoder::Stage3CudaDecoder(
    const std::vector<uint32_t>& encoded_words,
    uint8_t rice_k,
    int total_symbols,
    int subsequence_words,
    int threads_per_block)
    : total_symbols_(total_symbols),
      threads_per_block_(threads_per_block),
      word_count_(static_cast<int>(encoded_words.size())),
      subsequence_words_(subsequence_words),
      rice_k_(rice_k) {
  if (total_symbols_ <= 0 || threads_per_block_ <= 0 || word_count_ <= 0 || subsequence_words_ <= 0) {
    error_message_ = "Invalid Stage-3 decoder dimensions.";
    return;
  }
  num_subsequences_ = (word_count_ + subsequence_words_ - 1) / subsequence_words_;
  if (num_subsequences_ <= 0) {
    error_message_ = "No subsequences to decode.";
    return;
  }

  std::vector<uint32_t> chunk_start_bits;
  std::vector<uint32_t> chunk_output_offsets;
  build_chunk_plan_host(
      encoded_words,
      static_cast<int>(rice_k_),
      total_symbols_,
      subsequence_words_,
      chunk_start_bits,
      chunk_output_offsets,
      total_payload_bits_,
      error_message_);
  if (!error_message_.empty()) {
    return;
  }

  if (!chunk_output_offsets.empty() && chunk_output_offsets.back() > static_cast<uint32_t>(total_symbols_)) {
    error_message_ = "Invalid Stage-3 chunk output plan.";
    return;
  }

  if (!check_cuda(cudaMalloc(&d_words_, encoded_words.size() * sizeof(uint32_t)), error_message_, "cudaMalloc d_words")) return;
  if (!check_cuda(cudaMalloc(&d_chunk_start_bits_, chunk_start_bits.size() * sizeof(uint32_t)),
                  error_message_, "cudaMalloc d_chunk_start_bits")) return;
  if (!check_cuda(cudaMalloc(&d_chunk_output_offsets_, chunk_output_offsets.size() * sizeof(uint32_t)),
                  error_message_, "cudaMalloc d_chunk_output_offsets")) return;
  if (!check_cuda(cudaMalloc(&d_chunk_counts_, static_cast<size_t>(num_subsequences_) * sizeof(uint32_t)),
                  error_message_, "cudaMalloc d_chunk_counts")) return;
  if (!check_cuda(cudaMalloc(&d_phase1_counts_, static_cast<size_t>(num_subsequences_) * sizeof(uint32_t)),
                  error_message_, "cudaMalloc d_phase1_counts")) return;
  if (!check_cuda(cudaMalloc(&d_sync_end_bits_, static_cast<size_t>(num_subsequences_) * sizeof(uint32_t)),
                  error_message_, "cudaMalloc d_sync_end_bits")) return;
  if (!check_cuda(cudaMalloc(&d_sync_symbol_counts_, static_cast<size_t>(num_subsequences_) * sizeof(uint32_t)),
                  error_message_, "cudaMalloc d_sync_symbol_counts")) return;
  if (!check_cuda(cudaMalloc(&d_sync_stable_, static_cast<size_t>(num_subsequences_) * sizeof(uint8_t)),
                  error_message_, "cudaMalloc d_sync_stable")) return;
  if (!check_cuda(cudaMalloc(&d_sync_start_bits_a_, static_cast<size_t>(num_subsequences_) * sizeof(uint32_t)),
                  error_message_, "cudaMalloc d_sync_start_bits_a")) return;
  if (!check_cuda(cudaMalloc(&d_sync_start_bits_b_, static_cast<size_t>(num_subsequences_) * sizeof(uint32_t)),
                  error_message_, "cudaMalloc d_sync_start_bits_b")) return;
  if (!check_cuda(cudaMalloc(&d_sync_changed_, sizeof(int)),
                  error_message_, "cudaMalloc d_sync_changed")) return;
  if (!check_cuda(cudaMalloc(&d_out_, static_cast<size_t>(total_symbols_) * sizeof(uint8_t)),
                  error_message_, "cudaMalloc d_out")) return;

  if (!check_cuda(cudaMemcpy(d_words_, encoded_words.data(), encoded_words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy words H2D")) return;
  if (!check_cuda(cudaMemcpy(d_chunk_start_bits_, chunk_start_bits.data(), chunk_start_bits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy chunk_start_bits H2D")) return;
  if (!check_cuda(cudaMemcpy(d_sync_start_bits_a_, chunk_start_bits.data(), chunk_start_bits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy sync_start_bits_a H2D")) return;
  if (!check_cuda(cudaMemcpy(d_sync_start_bits_b_, chunk_start_bits.data(), chunk_start_bits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy sync_start_bits_b H2D")) return;
  if (!check_cuda(cudaMemcpy(d_chunk_output_offsets_, chunk_output_offsets.data(), chunk_output_offsets.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy chunk_output_offsets H2D")) return;
  std::vector<uint32_t> chunk_counts(static_cast<size_t>(num_subsequences_), 0U);
  for (int s = 0; s < num_subsequences_; ++s) {
    const uint32_t start = chunk_output_offsets[static_cast<size_t>(s)];
    const uint32_t end = (s + 1 < num_subsequences_)
        ? chunk_output_offsets[static_cast<size_t>(s + 1)]
        : static_cast<uint32_t>(total_symbols_);
    chunk_counts[static_cast<size_t>(s)] = end - start;
  }
  if (!check_cuda(cudaMemcpy(d_chunk_counts_, chunk_counts.data(), chunk_counts.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy chunk_counts H2D")) return;

  ok_ = true;
}

Stage3CudaDecoder::~Stage3CudaDecoder() {
  if (d_words_) cudaFree(d_words_);
  if (d_chunk_start_bits_) cudaFree(d_chunk_start_bits_);
  if (d_chunk_output_offsets_) cudaFree(d_chunk_output_offsets_);
  if (d_chunk_counts_) cudaFree(d_chunk_counts_);
  if (d_phase1_counts_) cudaFree(d_phase1_counts_);
  if (d_sync_end_bits_) cudaFree(d_sync_end_bits_);
  if (d_sync_symbol_counts_) cudaFree(d_sync_symbol_counts_);
  if (d_sync_stable_) cudaFree(d_sync_stable_);
  if (d_sync_start_bits_a_) cudaFree(d_sync_start_bits_a_);
  if (d_sync_start_bits_b_) cudaFree(d_sync_start_bits_b_);
  if (d_sync_changed_) cudaFree(d_sync_changed_);
  if (d_out_) cudaFree(d_out_);
}

bool Stage3CudaDecoder::decode(std::vector<uint8_t>& out) {
  if (!ok_) return false;

  const int blocks = (num_subsequences_ + threads_per_block_ - 1) / threads_per_block_;
  last_sync_iters_ = 0;

  if (force_validated_mode_) {
    thrust::device_ptr<uint32_t> fallback_counts_ptr(d_chunk_counts_);
    thrust::device_ptr<uint32_t> offsets_ptr(d_chunk_output_offsets_);
    thrust::exclusive_scan(fallback_counts_ptr, fallback_counts_ptr + num_subsequences_, offsets_ptr);
    if (!check_cuda(cudaGetLastError(), error_message_, "thrust::exclusive_scan validated mode")) return false;

    phase4_decode_write_validated_kernel<<<blocks, threads_per_block_>>>(
        d_words_,
        word_count_,
        d_chunk_start_bits_,
        d_chunk_output_offsets_,
        num_subsequences_,
        static_cast<int>(rice_k_),
        total_payload_bits_,
        total_symbols_,
        d_out_);
    if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch phase4_decode_write_validated_kernel(validated mode)")) return false;
    if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize validated mode")) return false;

    out.resize(static_cast<size_t>(total_symbols_));
    if (!check_cuda(cudaMemcpy(out.data(), d_out_, static_cast<size_t>(total_symbols_) * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                    error_message_, "cudaMemcpy out D2H validated mode")) return false;
    return true;
  }

  if (!check_cuda(cudaMemcpy(d_sync_start_bits_a_, d_chunk_start_bits_, static_cast<size_t>(num_subsequences_) * sizeof(uint32_t), cudaMemcpyDeviceToDevice),
                  error_message_, "cudaMemcpy sync_start_bits_a reset D2D")) return false;
  if (!check_cuda(cudaMemcpy(d_sync_start_bits_b_, d_chunk_start_bits_, static_cast<size_t>(num_subsequences_) * sizeof(uint32_t), cudaMemcpyDeviceToDevice),
                  error_message_, "cudaMemcpy sync_start_bits_b reset D2D")) return false;

  uint32_t* sync_starts_in = d_sync_start_bits_a_;
  uint32_t* sync_starts_out = d_sync_start_bits_b_;

  // Phase 1: local provisional per-chunk symbol counts from fixed chunk starts.
  phase1_provisional_counts_kernel<<<blocks, threads_per_block_>>>(
      d_words_,
      word_count_,
      num_subsequences_,
      subsequence_words_,
      static_cast<int>(rice_k_),
      d_phase1_counts_);
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch phase1_provisional_counts_kernel")) return false;

  // Phase 1 sync-point materialization (infrastructure for CUHD-style refinement passes).
  phase1_populate_sync_points_kernel<<<blocks, threads_per_block_>>>(
      d_words_,
      word_count_,
      d_chunk_start_bits_,
      num_subsequences_,
      subsequence_words_,
      static_cast<int>(rice_k_),
      d_sync_end_bits_,
      d_sync_symbol_counts_,
      d_sync_stable_);
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch phase1_populate_sync_points_kernel")) return false;

  // Phase 2 groundwork: iterative seam refinement over sync starts.
  for (int iter = 0; iter < max_refine_iters_; ++iter) {
    last_sync_iters_ = static_cast<uint32_t>(iter + 1);
    if (!check_cuda(cudaMemset(d_sync_changed_, 0, sizeof(int)), error_message_, "cudaMemset d_sync_changed")) return false;
    phase2_refine_sync_starts_kernel<<<blocks, threads_per_block_>>>(
        sync_starts_in,
        d_sync_end_bits_,
        num_subsequences_,
        sync_starts_out,
        d_sync_changed_);
    if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch phase2_refine_sync_starts_kernel")) return false;

    int changed = 0;
    if (!check_cuda(cudaMemcpy(&changed, d_sync_changed_, sizeof(int), cudaMemcpyDeviceToHost),
                    error_message_, "cudaMemcpy sync_changed D2H")) return false;
    std::swap(sync_starts_in, sync_starts_out);
    if (changed == 0) {
      break;
    }

    // Recompute sync points from refined starts before next refinement iteration.
    phase1_populate_sync_points_kernel<<<blocks, threads_per_block_>>>(
        d_words_,
        word_count_,
        sync_starts_in,
        num_subsequences_,
        subsequence_words_,
        static_cast<int>(rice_k_),
        d_sync_end_bits_,
        d_sync_symbol_counts_,
        d_sync_stable_);
    if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch phase1_populate_sync_points_kernel(iter)")) return false;
  }

  // Phase 3: count symbols from synchronized device starts, then prefix-sum on device.
  phase3_count_from_sync_starts_kernel<<<blocks, threads_per_block_>>>(
      d_words_,
      word_count_,
      sync_starts_in,
      num_subsequences_,
      static_cast<int>(rice_k_),
      total_payload_bits_,
      d_sync_end_bits_,
      d_sync_symbol_counts_,
      d_sync_stable_);
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch phase3_count_from_sync_starts_kernel")) return false;

  thrust::device_ptr<uint32_t> counts_ptr(d_sync_symbol_counts_);
  thrust::device_ptr<uint32_t> offsets_ptr(d_chunk_output_offsets_);
  thrust::exclusive_scan(counts_ptr, counts_ptr + num_subsequences_, offsets_ptr);
  if (!check_cuda(cudaGetLastError(), error_message_, "thrust::exclusive_scan")) return false;

  uint32_t last_offset = 0U;
  uint32_t last_count = 0U;
  if (!check_cuda(cudaMemcpy(&last_offset, d_chunk_output_offsets_ + (num_subsequences_ - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  error_message_, "cudaMemcpy last_offset D2H")) return false;
  if (!check_cuda(cudaMemcpy(&last_count, d_sync_symbol_counts_ + (num_subsequences_ - 1), sizeof(uint32_t), cudaMemcpyDeviceToHost),
                  error_message_, "cudaMemcpy last_count D2H")) return false;
  const uint32_t total_symbols_u32 = static_cast<uint32_t>(total_symbols_);
  if (last_offset + last_count != total_symbols_u32) {
    ++sync_fallback_count_;
    force_validated_mode_ = true;
    // Safety fallback: keep correctness with validated chunk plan path while
    // Step-4 full synchronized handoff is being integrated.
    thrust::device_ptr<uint32_t> fallback_counts_ptr(d_chunk_counts_);
    thrust::exclusive_scan(fallback_counts_ptr, fallback_counts_ptr + num_subsequences_, offsets_ptr);
    if (!check_cuda(cudaGetLastError(), error_message_, "thrust::exclusive_scan fallback")) return false;

    phase4_decode_write_validated_kernel<<<blocks, threads_per_block_>>>(
        d_words_,
        word_count_,
        d_chunk_start_bits_,
        d_chunk_output_offsets_,
        num_subsequences_,
        static_cast<int>(rice_k_),
        total_payload_bits_,
        total_symbols_,
        d_out_);
    if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch phase4_decode_write_validated_kernel(fallback)")) return false;
    if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize fallback")) return false;

    out.resize(static_cast<size_t>(total_symbols_));
    if (!check_cuda(cudaMemcpy(out.data(), d_out_, static_cast<size_t>(total_symbols_) * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                    error_message_, "cudaMemcpy out D2H fallback")) return false;
    return true;
  }

  // Phase 4: final decode/write from synchronized device state.
  phase4_decode_write_kernel<<<blocks, threads_per_block_>>>(
      d_words_,
      word_count_,
      sync_starts_in,
      d_sync_end_bits_,
      d_chunk_output_offsets_,
      d_sync_symbol_counts_,
      num_subsequences_,
      static_cast<int>(rice_k_),
      d_out_);
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch phase4_decode_write_kernel")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize")) return false;

  out.resize(static_cast<size_t>(total_symbols_));
  if (!check_cuda(cudaMemcpy(out.data(), d_out_, static_cast<size_t>(total_symbols_) * sizeof(uint8_t), cudaMemcpyDeviceToHost),
                  error_message_, "cudaMemcpy out D2H")) return false;
  return true;
}
