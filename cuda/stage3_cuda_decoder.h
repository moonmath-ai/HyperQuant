#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Stage3CudaDecoder {
 public:
  Stage3CudaDecoder(
      const std::vector<uint32_t>& encoded_words,
      uint8_t rice_k,
      int total_symbols,
      int subsequence_words,
      int threads_per_block);

  ~Stage3CudaDecoder();

  Stage3CudaDecoder(const Stage3CudaDecoder&) = delete;
  Stage3CudaDecoder& operator=(const Stage3CudaDecoder&) = delete;

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }

  // Decodes into host memory. Returns false on CUDA failure.
  bool decode(std::vector<uint8_t>& out);

 private:
  bool ok_ = false;
  std::string error_message_;

  int total_symbols_ = 0;
  int threads_per_block_ = 256;
  int word_count_ = 0;
  int subsequence_words_ = 0;
  int num_subsequences_ = 0;
  uint8_t rice_k_ = 0;
  uint32_t total_payload_bits_ = 0;
  int max_refine_iters_ = 12;
  bool force_validated_mode_ = false;
  uint64_t sync_fallback_count_ = 0;
  uint32_t last_sync_iters_ = 0;

  uint32_t* d_words_ = nullptr;
  uint32_t* d_chunk_start_bits_ = nullptr;
  uint32_t* d_chunk_output_offsets_ = nullptr;
  uint32_t* d_chunk_counts_ = nullptr;
  uint32_t* d_phase1_counts_ = nullptr;
  uint32_t* d_sync_end_bits_ = nullptr;
  uint32_t* d_sync_symbol_counts_ = nullptr;
  uint8_t* d_sync_stable_ = nullptr;
  uint32_t* d_sync_start_bits_a_ = nullptr;
  uint32_t* d_sync_start_bits_b_ = nullptr;
  int* d_sync_changed_ = nullptr;
  uint8_t* d_out_ = nullptr;
};
