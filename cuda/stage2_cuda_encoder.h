#pragma once

#include <cstdint>
#include <string>
#include <vector>

class Stage2CudaEncoder {
 public:
  enum class CacheMode : uint8_t {
    kRecompute = 0,
    // B2 path: K1 caches 8 pre-Rice symbols packed into one uint64 per 8D word.
    kB2U8Cache = 1,
  };

  struct TimingStats {
    float kernel1_ms = 0.0f;
    float scan_ms = 0.0f;
    float kernel3_ms = 0.0f;
    float total_gpu_ms = 0.0f;
  };

  // decoder_symbols_per_stream selects the granularity of the emitted offset
  // table, i.e. how many symbols the *decoder* treats as one independently
  // entered stream. It is decoupled from groups_per_thread (which is purely an
  // encoder thread-workload knob): the encoded bitstream is identical either
  // way, only which offsets get recorded changes. Pass 0 to keep the legacy
  // behavior (one offset per encoder thread-group == 8 * groups_per_thread).
  // When non-zero it must be a positive multiple of 8 * groups_per_thread so
  // that every decoder-stream boundary lands on a recorded group boundary.
  Stage2CudaEncoder(
      const std::vector<uint16_t>& bf16_words,
      float alpha,
      uint8_t rice_k,
      int groups_per_thread,
      CacheMode cache_mode,
      int threads_per_block,
      int decoder_symbols_per_stream = 0);

  ~Stage2CudaEncoder();

  Stage2CudaEncoder(const Stage2CudaEncoder&) = delete;
  Stage2CudaEncoder& operator=(const Stage2CudaEncoder&) = delete;

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }

  // Encodes bf16 vectors into Stage-2 stream format:
  // - encoded_words: packed bitstream (MSB-first per uint32 word)
  // - stream_offsets_bits: per-stream start bit offset
  // - stream_k: per-stream Rice k lookup
  // Returns false on CUDA failure.
  bool encode(
      std::vector<uint32_t>& encoded_words,
      std::vector<uint32_t>& stream_offsets_bits,
      std::vector<uint8_t>& stream_k,
      uint32_t* total_bits = nullptr,
      bool copy_output_to_host = true);

  const TimingStats& last_timing_stats() const { return last_timing_stats_; }

  // Effective decoder stream size (symbols) and number of decoder streams the
  // emitted offset table describes. The decoder must be constructed with these.
  int decoder_symbols_per_stream() const { return decoder_symbols_per_stream_; }
  int num_decoder_streams() const { return num_decoder_streams_; }

 private:
  bool ok_ = false;
  std::string error_message_;

  float alpha_ = 1.0f;
  uint8_t rice_k_ = 0;
  int num_words_ = 0;
  int groups_per_thread_ = 1;
  CacheMode cache_mode_ = CacheMode::kRecompute;
  int num_streams_ = 0;  // encoder thread-groups (one offset computed per group)
  int threads_per_block_ = 256;

  // Decoder stream layout (subsampled view of the per-group offsets emitted to
  // the caller). offset_stride_ = decoder_symbols_per_stream_ / (8 * gpt).
  int decoder_symbols_per_stream_ = 0;
  int num_decoder_streams_ = 0;
  int offset_stride_ = 1;

  uint16_t* d_bf16_words_ = nullptr;
  uint32_t* d_lengths_ = nullptr;
  uint32_t* d_offsets_ = nullptr;
  uint32_t* d_words_ = nullptr;
  uint64_t* d_z_cache_packed_ = nullptr;
  uint32_t* d_cache_overflow_flag_ = nullptr;
  int words_capacity_ = 0;

  TimingStats last_timing_stats_;
};
