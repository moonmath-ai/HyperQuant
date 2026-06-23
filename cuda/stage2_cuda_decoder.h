#pragma once

#include <cstdint>
#include <string>
#include <vector>

// Output element type for the inverse E8int / dequant step.
//   kBf16    : bf16(Y/alpha), 2 bytes/elem (default; bit-exact with the legacy path).
//   kFp8E4M3 : fp8 e4m3 of Y/alpha, 1 byte/elem.
//   kInt8    : the lattice integer Y saturated to int8, 1 byte/elem; reconstruct as
//              Y/alpha downstream (apply scale 1/alpha in the GEMM) -> lossless except
//              for rare |Y|>127 saturation.
enum class OutputDtype { kBf16 = 0, kFp8E4M3 = 1, kInt8 = 2 };

class Stage2CudaDecoder {
 public:
  struct TimingStats {
    float rice_ms = 0.0f;
    float inverse_ms = 0.0f;
    float fused_ms = 0.0f;
    float coop_ms = 0.0f;
    float fused_cast_ms = 0.0f;
    float total_gpu_ms = 0.0f;
  };

  Stage2CudaDecoder(
      const std::vector<uint32_t>& encoded_words,
      const std::vector<uint32_t>& stream_offsets_bits,
      const std::vector<uint8_t>& stream_k,
      int symbols_per_stream,
      int total_symbols,
      int threads_per_block);

  ~Stage2CudaDecoder();

  Stage2CudaDecoder(const Stage2CudaDecoder&) = delete;
  Stage2CudaDecoder& operator=(const Stage2CudaDecoder&) = delete;

  bool ok() const { return ok_; }
  const std::string& error_message() const { return error_message_; }

  // Decodes Rice symbols (zigzag-mapped z values) into host memory.
  bool decode(std::vector<uint8_t>& out, bool copy_output_to_host = true);

  // Full decode: Rice bitstream -> z symbols -> inverse E8int strip/dequant -> bf16 words.
  bool decode_to_bf16(std::vector<uint16_t>& out_bf16, float alpha, bool copy_output_to_host = true);

  // Same result as decode_to_bf16 but in a single fused kernel: z is kept in
  // registers and never written to global memory. last_timing_stats().fused_ms
  // / total_gpu_ms hold the single-kernel time.
  bool decode_fused_to_bf16(std::vector<uint16_t>& out_bf16, float alpha, bool copy_output_to_host = true);

  // Same result as decode_fused_to_bf16, but each warp first stages its 32
  // streams' contiguous bitstream region into shared memory with coalesced loads,
  // then decodes from smem. last_timing_stats().coop_ms /
  // total_gpu_ms hold the single-kernel time.
  bool decode_fused_coop_to_bf16(std::vector<uint16_t>& out_bf16, float alpha, bool copy_output_to_host = true);

  // Fused decode emitting the requested output dtype. Output is
  // returned as raw bytes (size = total_symbols * elem_size: 2 for bf16, 1 for
  // fp8/int8). last_timing_stats().fused_cast_ms / total_gpu_ms hold the kernel time.
  bool decode_fused_to(OutputDtype dtype, std::vector<uint8_t>& out_bytes, float alpha, bool copy_output_to_host = true);

  const TimingStats& last_timing_stats() const { return last_timing_stats_; }

  // Device pointers to the last decode output, for feeding a fused GEMM directly
  // (decode with copy_output_to_host=false, then read these). bf16 path writes
  // d_bf16_out_ (2 B/elem); fp8/int8 paths write d_byte_out_ (1 B/elem).
  const uint16_t* device_bf16_output() const { return d_bf16_out_; }
  const uint8_t* device_byte_output() const { return d_byte_out_; }

 private:
  bool ok_ = false;
  std::string error_message_;

  int num_streams_ = 0;
  int symbols_per_stream_ = 0;
  int total_symbols_ = 0;
  int threads_per_block_ = 256;

  uint32_t* d_words_ = nullptr;
  uint32_t* d_offsets_ = nullptr;
  uint8_t* d_k_ = nullptr;
  uint8_t* d_out_ = nullptr;
  uint16_t* d_bf16_out_ = nullptr;
  uint8_t* d_byte_out_ = nullptr;  // 1-byte outputs (fp8/int8), lazily allocated
  int word_count_ = 0;
  int num_words_ = 0;

  TimingStats last_timing_stats_;
};

// Decode an interleaved-layout bitstream: streams grouped 32
// to a warp, their 32-bit words stored round-robin so a warp's reads are
// coalesced with no shared memory. `group_base_words[g]` is the first word of
// group g; each stream is word-aligned in its column. Self-contained (allocates,
// runs one kernel, optionally copies bf16 back). For experiment/measurement.
bool decode_interleaved_layout(
    const std::vector<uint32_t>& interleaved_words,
    const std::vector<uint32_t>& group_base_words,
    const std::vector<uint8_t>& stream_k,
    int num_streams,
    int symbols_per_stream,
    int total_symbols,
    int threads_per_block,
    float alpha,
    std::vector<uint16_t>& out_bf16,
    float* out_ms,
    std::string* err,
    bool copy_output_to_host = true);

// Integration launcher (for the Llama RiceLinear torch extension). Decodes a
// Rice bitstream whose buffers are all already on device and owned by the
// caller, writing bf16 output into a caller-provided device buffer on the given
// stream. No allocations, no H2D/D2H. num_words = total_symbols / 8.
void metalrice_launch_fused_decode_bf16(
    const uint32_t* d_words, int word_count,
    const uint32_t* d_offsets, const uint8_t* d_k,
    int num_streams, int symbols_per_stream, int num_words,
    float alpha, uint16_t* d_bf16_out, int threads_per_block, void* cuda_stream);
