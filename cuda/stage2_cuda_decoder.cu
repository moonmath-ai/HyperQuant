#include "stage2_cuda_decoder.h"

#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace {

inline bool check_cuda(cudaError_t err, std::string& error_message, const char* context) {
  if (err == cudaSuccess) return true;
  error_message = std::string(context) + ": " + cudaGetErrorString(err);
  return false;
}

__device__ __forceinline__ uint32_t load_word_safe(const uint32_t* words, int word_count, int idx) {
  return (idx >= 0 && idx < word_count) ? words[idx] : 0U;
}

struct BitReader {
  const uint32_t* words;
  int word_count;
  int next_word_idx;
  int word_stride;  // 1 = contiguous; 32 = interleaved (lane stride within a group)
  uint32_t c1;
  uint32_t c2;
  int c1_bits;
  int c2_bits;
  uint64_t reg;
  int bits_in_reg;

  // Contiguous reader: bits packed sequentially from bit_offset (stride 1).
  __device__ BitReader(const uint32_t* in_words, int in_word_count, uint32_t bit_offset)
      : words(in_words),
        word_count(in_word_count),
        next_word_idx(0),
        word_stride(1),
        c1(0U),
        c2(0U),
        c1_bits(0),
        c2_bits(0),
        reg(0U),
        bits_in_reg(0) {
    init(static_cast<int>(bit_offset >> 5), static_cast<int>(bit_offset & 31U), 1);
  }

  // Strided reader: the stream's words are at start_word, start_word+stride,
  // start_word+2*stride, ... (used by the interleaved layout, stride=32). The
  // stream is word-aligned, so skip_bits is normally 0.
  __device__ BitReader(const uint32_t* in_words, int in_word_count, int start_word,
                       int skip_bits, int stride)
      : words(in_words),
        word_count(in_word_count),
        next_word_idx(0),
        word_stride(stride),
        c1(0U),
        c2(0U),
        c1_bits(0),
        c2_bits(0),
        reg(0U),
        bits_in_reg(0) {
    init(start_word, skip_bits, stride);
  }

  __device__ __forceinline__ void init(int start_word, int skip_bits, int stride) {
    word_stride = stride;
    c1 = load_word_safe(words, word_count, start_word);
    c2 = load_word_safe(words, word_count, start_word + stride);
    c1_bits = 32;
    c2_bits = 32;
    next_word_idx = start_word + 2 * stride;

    if (skip_bits > 0) {
      c1 = shift_left32_zeroed(c1, skip_bits);
      c1_bits -= skip_bits;
      if (c1_bits == 0) {
        promote_c2_to_c1();
      }
    }
  }

  __device__ __forceinline__ static uint32_t shift_left32_zeroed(uint32_t v, int shift) {
    return (shift >= 32) ? 0U : (v << shift);
  }

  __device__ __forceinline__ void promote_c2_to_c1() {
    c1 = c2;
    c1_bits = c2_bits;
    c2 = load_word_safe(words, word_count, next_word_idx);
    next_word_idx += word_stride;
    c2_bits = 32;
  }

  __device__ __forceinline__ void refill_reg(int needed_bits) {
    while (bits_in_reg < needed_bits) {
      if (c1_bits == 0) {
        promote_c2_to_c1();
      }

      const int room = 64 - bits_in_reg;
      const int take = (c1_bits < room) ? c1_bits : room;
      const uint32_t piece = c1 >> (32 - take);

      reg |= static_cast<uint64_t>(piece) << (64 - bits_in_reg - take);
      c1 = shift_left32_zeroed(c1, take);
      c1_bits -= take;
      bits_in_reg += take;
    }
  }

  __device__ __forceinline__ void refill(int needed_bits) {
    refill_reg(needed_bits);
  }

  __device__ __forceinline__ uint32_t read_bits(int nbits) {
    if (nbits <= 0) return 0U;
    refill(nbits);
    const uint32_t value = static_cast<uint32_t>(reg >> (64 - nbits));
    reg <<= nbits;
    bits_in_reg -= nbits;
    return value;
  }

  __device__ __forceinline__ uint32_t read_unary_quotient_clz() {
    uint32_t q = 0U;
    while (true) {
      refill(1);

      uint64_t probe = reg;
      if (bits_in_reg < 64) {
        // Force invalid tail bits to 1 so CLZ only scans valid bits.
        probe |= ((1ULL << (64 - bits_in_reg)) - 1ULL);
      }
      const uint32_t leading_zeros =
          (probe == 0ULL) ? 64U : static_cast<uint32_t>(__clzll(probe));
      q += leading_zeros;

      if (leading_zeros < static_cast<uint32_t>(bits_in_reg)) {
        const int consumed = static_cast<int>(leading_zeros) + 1;  // include unary terminator
        reg <<= consumed;
        bits_in_reg -= consumed;
        break;
      }

      reg = 0ULL;
      bits_in_reg = 0;
    }
    return q;
  }
};

__global__ void decode_stage2_kernel(
    const uint32_t* encoded_words,
    int word_count,
    const uint32_t* stream_offsets_bits,
    const uint8_t* stream_k,
    int num_streams,
    int symbols_per_stream,
    int total_symbols,
    uint8_t* out) {
  const int stream_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (stream_id >= num_streams) return;

  BitReader br(encoded_words, word_count, stream_offsets_bits[stream_id]);
  const uint32_t k = static_cast<uint32_t>(stream_k[stream_id]);
  const int base_out = stream_id * symbols_per_stream;

  int i = 0;
  for (; i + 3 < symbols_per_stream; i += 4) {
    const uint32_t q0 = br.read_unary_quotient_clz();
    const uint32_t r0 = br.read_bits(static_cast<int>(k));
    const uint32_t q1 = br.read_unary_quotient_clz();
    const uint32_t r1 = br.read_bits(static_cast<int>(k));
    const uint32_t q2 = br.read_unary_quotient_clz();
    const uint32_t r2 = br.read_bits(static_cast<int>(k));
    const uint32_t q3 = br.read_unary_quotient_clz();
    const uint32_t r3 = br.read_bits(static_cast<int>(k));

    const uint32_t sym0 = (q0 << k) | r0;
    const uint32_t sym1 = (q1 << k) | r1;
    const uint32_t sym2 = (q2 << k) | r2;
    const uint32_t sym3 = (q3 << k) | r3;

    const int out0 = base_out + i;
    const int out1 = out0 + 1;
    const int out2 = out0 + 2;
    const int out3 = out0 + 3;
    if (out0 < total_symbols) out[out0] = static_cast<uint8_t>(sym0 & 0xFFU);
    if (out1 < total_symbols) out[out1] = static_cast<uint8_t>(sym1 & 0xFFU);
    if (out2 < total_symbols) out[out2] = static_cast<uint8_t>(sym2 & 0xFFU);
    if (out3 < total_symbols) out[out3] = static_cast<uint8_t>(sym3 & 0xFFU);
  }

  for (; i < symbols_per_stream; ++i) {
    const uint32_t q = br.read_unary_quotient_clz();
    const uint32_t rem = br.read_bits(static_cast<int>(k));
    const uint32_t sym = (q << k) | rem;
    const int out_idx = base_out + i;
    if (out_idx < total_symbols) {
      out[out_idx] = static_cast<uint8_t>(sym & 0xFFU);
    }
  }
}

__device__ __forceinline__ int inv_zigzag_i32(uint32_t zz) {
  if ((zz & 1U) == 0U) return static_cast<int>(zz >> 1U);
  return -static_cast<int>((zz + 1U) >> 1U);
}

// Inverse E8int strip/dequant for a single 8-D word from its 8 z symbols (held
// in registers/local memory), templated on the output dtype. Shared by every
// decode path. `out8` points at this word's 8 output elements:
//   kBf16    -> uint16_t[8], value bf16(Y/alpha)            (2 bytes/elem)
//   kFp8E4M3 -> uint8_t[8],  value fp8_e4m3(Y/alpha)         (1 byte/elem)
//   kInt8    -> int8_t[8],   value sat(Y) (lattice integer)  (1 byte/elem),
//               reconstruct as Y/alpha downstream (GEMM scale = 1/alpha) -> lossless
//               except for rare saturation at |Y|>127.
template <OutputDtype OD>
__device__ __forceinline__ void inverse_e8int_word(
    const uint8_t z[8],
    float alpha,
    void* out8) {
  int s[8];
  for (int i = 0; i < 7; ++i) {
    s[i] = inv_zigzag_i32(static_cast<uint32_t>(z[i]));
  }
  const uint32_t z7 = static_cast<uint32_t>(z[7]);
  const int c = static_cast<int>(z7 & 1U);
  const int t = inv_zigzag_i32(z7 >> 1U);
  int known_p8 = 0;
  for (int i = 0; i < 7; ++i) known_p8 += s[i];
  known_p8 &= 1;
  s[7] = (t << 1) + known_p8;

  for (int i = 0; i < 8; ++i) {
    const int Y = (s[i] << 1) + c;
    if constexpr (OD == OutputDtype::kBf16) {
      const float x_hat = static_cast<float>(Y) / alpha;
      const __nv_bfloat16 xb = __float2bfloat16_rn(x_hat);
      static_cast<uint16_t*>(out8)[i] = *reinterpret_cast<const uint16_t*>(&xb);
    } else if constexpr (OD == OutputDtype::kFp8E4M3) {
      const float x_hat = static_cast<float>(Y) / alpha;
      const __nv_fp8_e4m3 v(x_hat);
      static_cast<uint8_t*>(out8)[i] = v.__x;
    } else {  // kInt8: emit the lattice integer Y, saturating to int8.
      const int q = (Y < -127) ? -127 : ((Y > 127) ? 127 : Y);
      static_cast<int8_t*>(out8)[i] = static_cast<int8_t>(q);
    }
  }
}

// bf16 wrapper preserving the existing call sites (two-kernel / fused / coop /
// interleaved bf16 paths).
__device__ __forceinline__ void inverse_e8int_word_to_bf16(
    const uint8_t z[8],
    float alpha,
    uint16_t* bf16_out8) {
  inverse_e8int_word<OutputDtype::kBf16>(z, alpha, bf16_out8);
}

__device__ __forceinline__ void decode_word_from_z_to_bf16(
    const uint8_t* z_symbols,
    int word_id,
    float alpha,
    uint16_t* bf16_out) {
  const int base = word_id * 8;
  uint8_t z[8];
  for (int i = 0; i < 8; ++i) z[i] = z_symbols[base + i];
  inverse_e8int_word_to_bf16(z, alpha, &bf16_out[base]);
}

__global__ void inverse_e8int_z_to_bf16_kernel(
    const uint8_t* z_symbols,
    int num_words,
    float alpha,
    uint16_t* bf16_out) {
  const int word_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (word_id >= num_words) return;
  decode_word_from_z_to_bf16(z_symbols, word_id, alpha, bf16_out);
}

// Fused Rice decode + inverse E8int. One thread per stream Rice-decodes its
// symbols 8 at a time into registers, immediately applies the inverse E8int
// transform, and writes 8 bf16 outputs directly. The z intermediate never hits
// global memory (saves a num_words*8-byte write + read and the worst-coalesced
// z store). Relies on symbols_per_stream being a multiple of 8, so every stream
// holds a whole number of 8-D words (enforced by the encoder grouping rule).
__global__ void decode_fused_stage2_to_bf16_kernel(
    const uint32_t* encoded_words,
    int word_count,
    const uint32_t* stream_offsets_bits,
    const uint8_t* stream_k,
    int num_streams,
    int symbols_per_stream,
    int num_words,
    float alpha,
    uint16_t* bf16_out) {
  const int stream_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (stream_id >= num_streams) return;

  BitReader br(encoded_words, word_count, stream_offsets_bits[stream_id]);
  const uint32_t k = static_cast<uint32_t>(stream_k[stream_id]);
  const int base_word = (stream_id * symbols_per_stream) / 8;  // S multiple of 8
  const int words_per_stream = symbols_per_stream / 8;

  for (int w = 0; w < words_per_stream; ++w) {
    uint8_t z[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t q = br.read_unary_quotient_clz();
      const uint32_t rem = br.read_bits(static_cast<int>(k));
      z[p] = static_cast<uint8_t>(((q << k) | rem) & 0xFFU);
    }
    const int word_id = base_word + w;
    if (word_id < num_words) {
      inverse_e8int_word_to_bf16(z, alpha, &bf16_out[word_id * 8]);
    }
  }
}

// Fused decode templated on output dtype. Same structure as
// decode_fused_stage2_to_bf16_kernel but the inverse writes bf16 / fp8 / int8.
// `out_base` points at element 0; element size is 2 bytes for bf16, 1 otherwise.
template <OutputDtype OD>
__global__ void decode_fused_out_kernel(
    const uint32_t* __restrict__ encoded_words,
    int word_count,
    const uint32_t* __restrict__ stream_offsets_bits,
    const uint8_t* __restrict__ stream_k,
    int num_streams,
    int symbols_per_stream,
    int num_words,
    float alpha,
    void* __restrict__ out_base) {
  constexpr int kElemBytes = (OD == OutputDtype::kBf16) ? 2 : 1;
  const int stream_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (stream_id >= num_streams) return;

  BitReader br(encoded_words, word_count, stream_offsets_bits[stream_id]);
  const uint32_t k = static_cast<uint32_t>(stream_k[stream_id]);
  const int base_word = (stream_id * symbols_per_stream) / 8;
  const int words_per_stream = symbols_per_stream / 8;

  for (int w = 0; w < words_per_stream; ++w) {
    uint8_t z[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t q = br.read_unary_quotient_clz();
      const uint32_t rem = br.read_bits(static_cast<int>(k));
      z[p] = static_cast<uint8_t>(((q << k) | rem) & 0xFFU);
    }
    const int word_id = base_word + w;
    if (word_id < num_words) {
      void* out8 = static_cast<char*>(out_base) +
                   static_cast<size_t>(word_id) * 8 * kElemBytes;
      inverse_e8int_word<OD>(z, alpha, out8);
    }
  }
}

// Cooperative-load variant of the fused kernel. A warp owns
// 32 consecutive decoder streams, whose Rice bits occupy a contiguous bitstream
// region [offset[warp_base], offset[warp_base+32]). The warp first stages that
// region into shared memory with fully coalesced loads (lane i grabs word i,
// i+32, ...), then each lane runs its serial BitReader against SHARED memory
// instead of L2. This attacks the dominant long-scoreboard stall (load latency
// of the serial per-stream bit chain) WITHOUT changing the stored format or
// spending any extra DRAM. If a group's region exceeds the smem cap (rare; only
// with abnormally long codes) the warp's lanes fall back to reading from global,
// so output is identical regardless of cap. Block size must be a multiple of 32.
template <int CAP_WORDS>
__global__ void decode_fused_coop_stage2_to_bf16_kernel(
    const uint32_t* __restrict__ encoded_words,
    int word_count,
    const uint32_t* __restrict__ stream_offsets_bits,
    const uint8_t* __restrict__ stream_k,
    int num_streams,
    int symbols_per_stream,
    int num_words,
    float alpha,
    uint16_t* __restrict__ bf16_out) {
  extern __shared__ uint32_t s_words[];  // (warps_per_block * CAP_WORDS) words

  const int lane = static_cast<int>(threadIdx.x & 31U);
  const int warp_in_block = static_cast<int>(threadIdx.x >> 5);
  const int global_warp =
      static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
  const int warp_base = global_warp * 32;  // first stream of this group
  if (warp_base >= num_streams) return;

  uint32_t* sm = s_words + warp_in_block * CAP_WORDS;

  // Contiguous word span covering all 32 streams of this group.
  const int start_word = static_cast<int>(stream_offsets_bits[warp_base] >> 5);
  const int next_first = warp_base + 32;
  const int end_word = (next_first < num_streams)
      ? static_cast<int>((stream_offsets_bits[next_first] + 31U) >> 5)
      : word_count;
  const int region = end_word - start_word;
  const bool use_smem = (region <= CAP_WORDS);

  if (use_smem) {
    for (int i = lane; i < region; i += 32) {
      sm[i] = load_word_safe(encoded_words, word_count, start_word + i);
    }
    __syncwarp();
  }

  const int stream_id = warp_base + lane;
  if (stream_id >= num_streams) return;

  const uint32_t k = static_cast<uint32_t>(stream_k[stream_id]);
  const int base_word = (stream_id * symbols_per_stream) / 8;
  const int words_per_stream = symbols_per_stream / 8;

  BitReader br = use_smem
      ? BitReader(sm, region,
                  stream_offsets_bits[stream_id] -
                      static_cast<uint32_t>(start_word) * 32U)
      : BitReader(encoded_words, word_count, stream_offsets_bits[stream_id]);

  for (int w = 0; w < words_per_stream; ++w) {
    uint8_t z[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t q = br.read_unary_quotient_clz();
      const uint32_t rem = br.read_bits(static_cast<int>(k));
      z[p] = static_cast<uint8_t>(((q << k) | rem) & 0xFFU);
    }
    const int word_id = base_word + w;
    if (word_id < num_words) {
      inverse_e8int_word_to_bf16(z, alpha, &bf16_out[word_id * 8]);
    }
  }
}

// Interleaved-layout fused decode. Streams are grouped 32 to
// a warp; within a group the 32 streams' 32-bit words are stored round-robin
// (interleaved[group_base + row*32 + lane]). Lane l reads its stream's word n at
// group_base + n*32 + l, so a warp's 32 reads at the same row are contiguous ->
// coalesced global reads with NO shared memory and full occupancy, at ANY S
// (this is what cooperative-smem coop could not achieve at large S). Each stream
// is word-aligned in its column, so the strided BitReader starts at skip_bits=0.
__global__ void decode_interleaved_stage2_to_bf16_kernel(
    const uint32_t* __restrict__ words,
    int word_count,
    const uint32_t* __restrict__ group_base_words,
    const uint8_t* __restrict__ stream_k,
    int num_streams,
    int symbols_per_stream,
    int num_words,
    float alpha,
    uint16_t* __restrict__ bf16_out) {
  const int stream_id = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (stream_id >= num_streams) return;

  const int group = stream_id >> 5;   // 32 streams per interleaved group
  const int lane = stream_id & 31;
  const int start_word = static_cast<int>(group_base_words[group]) + lane;

  const uint32_t k = static_cast<uint32_t>(stream_k[stream_id]);
  const int base_word = (stream_id * symbols_per_stream) / 8;
  const int words_per_stream = symbols_per_stream / 8;

  BitReader br(words, word_count, start_word, /*skip_bits=*/0, /*stride=*/32);

  for (int w = 0; w < words_per_stream; ++w) {
    uint8_t z[8];
#pragma unroll
    for (int p = 0; p < 8; ++p) {
      const uint32_t q = br.read_unary_quotient_clz();
      const uint32_t rem = br.read_bits(static_cast<int>(k));
      z[p] = static_cast<uint8_t>(((q << k) | rem) & 0xFFU);
    }
    const int word_id = base_word + w;
    if (word_id < num_words) {
      inverse_e8int_word_to_bf16(z, alpha, &bf16_out[word_id * 8]);
    }
  }
}

}  // namespace

Stage2CudaDecoder::Stage2CudaDecoder(
    const std::vector<uint32_t>& encoded_words,
    const std::vector<uint32_t>& stream_offsets_bits,
    const std::vector<uint8_t>& stream_k,
    int symbols_per_stream,
    int total_symbols,
    int threads_per_block)
    : num_streams_(static_cast<int>(stream_offsets_bits.size())),
      symbols_per_stream_(symbols_per_stream),
      total_symbols_(total_symbols),
      threads_per_block_(threads_per_block),
      word_count_(static_cast<int>(encoded_words.size())),
      num_words_(total_symbols / 8) {
  if (num_streams_ <= 0 || symbols_per_stream_ <= 0 || total_symbols_ <= 0) {
    error_message_ = "Invalid decoder dimensions.";
    return;
  }
  if (stream_k.size() != stream_offsets_bits.size()) {
    error_message_ = "stream_k and stream_offsets_bits size mismatch.";
    return;
  }

  if (!check_cuda(cudaMalloc(&d_words_, encoded_words.size() * sizeof(uint32_t)), error_message_, "cudaMalloc d_words")) return;
  if (!check_cuda(cudaMalloc(&d_offsets_, stream_offsets_bits.size() * sizeof(uint32_t)), error_message_, "cudaMalloc d_offsets")) return;
  if (!check_cuda(cudaMalloc(&d_k_, stream_k.size() * sizeof(uint8_t)), error_message_, "cudaMalloc d_k")) return;
  if (!check_cuda(cudaMalloc(&d_out_, static_cast<size_t>(total_symbols_) * sizeof(uint8_t)), error_message_, "cudaMalloc d_out")) return;
  if (num_words_ > 0) {
    if (!check_cuda(
            cudaMalloc(&d_bf16_out_, static_cast<size_t>(num_words_) * 8U * sizeof(uint16_t)),
            error_message_,
            "cudaMalloc d_bf16_out")) {
      return;
    }
  }

  if (!check_cuda(cudaMemcpy(d_words_, encoded_words.data(), encoded_words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy words H2D")) return;
  if (!check_cuda(cudaMemcpy(d_offsets_, stream_offsets_bits.data(), stream_offsets_bits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy offsets H2D")) return;
  if (!check_cuda(cudaMemcpy(d_k_, stream_k.data(), stream_k.size() * sizeof(uint8_t), cudaMemcpyHostToDevice),
                  error_message_, "cudaMemcpy k H2D")) return;

  ok_ = true;
}

Stage2CudaDecoder::~Stage2CudaDecoder() {
  if (d_words_) cudaFree(d_words_);
  if (d_offsets_) cudaFree(d_offsets_);
  if (d_k_) cudaFree(d_k_);
  if (d_out_) cudaFree(d_out_);
  if (d_bf16_out_) cudaFree(d_bf16_out_);
  if (d_byte_out_) cudaFree(d_byte_out_);
}

bool Stage2CudaDecoder::decode(std::vector<uint8_t>& out, bool copy_output_to_host) {
  if (!ok_) return false;
  last_timing_stats_ = TimingStats{};

  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_rice_done = nullptr;
  if (!check_cuda(cudaEventCreate(&ev_start), error_message_, "cudaEventCreate ev_start")) return false;
  if (!check_cuda(cudaEventCreate(&ev_rice_done), error_message_, "cudaEventCreate ev_rice_done")) return false;

  const int blocks = (num_streams_ + threads_per_block_ - 1) / threads_per_block_;
  if (!check_cuda(cudaEventRecord(ev_start), error_message_, "cudaEventRecord ev_start")) return false;
  decode_stage2_kernel<<<blocks, threads_per_block_>>>(
      d_words_,
      word_count_,
      d_offsets_,
      d_k_,
      num_streams_,
      symbols_per_stream_,
      total_symbols_,
      d_out_);
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch decode_stage2_kernel")) return false;
  if (!check_cuda(cudaEventRecord(ev_rice_done), error_message_, "cudaEventRecord ev_rice_done")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize")) return false;

  float ms_rice = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&ms_rice, ev_start, ev_rice_done), error_message_, "cudaEventElapsedTime rice")) return false;
  last_timing_stats_.rice_ms = ms_rice;
  last_timing_stats_.total_gpu_ms = ms_rice;

  if (copy_output_to_host) {
    out.resize(static_cast<size_t>(total_symbols_));
    if (!check_cuda(
            cudaMemcpy(
                out.data(),
                d_out_,
                static_cast<size_t>(total_symbols_) * sizeof(uint8_t),
                cudaMemcpyDeviceToHost),
            error_message_,
            "cudaMemcpy out D2H")) {
      return false;
    }
  } else {
    out.clear();
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_rice_done);
  return true;
}

bool Stage2CudaDecoder::decode_to_bf16(std::vector<uint16_t>& out_bf16, float alpha, bool copy_output_to_host) {
  if (!ok_) return false;
  if (!(alpha > 0.0f)) {
    error_message_ = "Invalid alpha (must be > 0).";
    return false;
  }
  if (num_words_ <= 0 || d_bf16_out_ == nullptr) {
    error_message_ = "Invalid bf16 decode dimensions.";
    return false;
  }
  last_timing_stats_ = TimingStats{};

  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_rice_done = nullptr;
  cudaEvent_t ev_inverse_done = nullptr;
  if (!check_cuda(cudaEventCreate(&ev_start), error_message_, "cudaEventCreate ev_start")) return false;
  if (!check_cuda(cudaEventCreate(&ev_rice_done), error_message_, "cudaEventCreate ev_rice_done")) return false;
  if (!check_cuda(cudaEventCreate(&ev_inverse_done), error_message_, "cudaEventCreate ev_inverse_done")) return false;

  const int blocks = (num_streams_ + threads_per_block_ - 1) / threads_per_block_;
  if (!check_cuda(cudaEventRecord(ev_start), error_message_, "cudaEventRecord ev_start")) return false;
  decode_stage2_kernel<<<blocks, threads_per_block_>>>(
      d_words_,
      word_count_,
      d_offsets_,
      d_k_,
      num_streams_,
      symbols_per_stream_,
      total_symbols_,
      d_out_);
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch decode_stage2_kernel")) return false;
  if (!check_cuda(cudaEventRecord(ev_rice_done), error_message_, "cudaEventRecord ev_rice_done")) return false;

  const int inverse_blocks = (num_words_ + threads_per_block_ - 1) / threads_per_block_;
  inverse_e8int_z_to_bf16_kernel<<<inverse_blocks, threads_per_block_>>>(
      d_out_,
      num_words_,
      alpha,
      d_bf16_out_);
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch inverse_e8int_z_to_bf16_kernel")) return false;
  if (!check_cuda(cudaEventRecord(ev_inverse_done), error_message_, "cudaEventRecord ev_inverse_done")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize")) return false;

  float ms_rice = 0.0f;
  float ms_inverse = 0.0f;
  float ms_total = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&ms_rice, ev_start, ev_rice_done), error_message_, "cudaEventElapsedTime rice")) return false;
  if (!check_cuda(cudaEventElapsedTime(&ms_inverse, ev_rice_done, ev_inverse_done), error_message_, "cudaEventElapsedTime inverse")) return false;
  if (!check_cuda(cudaEventElapsedTime(&ms_total, ev_start, ev_inverse_done), error_message_, "cudaEventElapsedTime total")) return false;
  last_timing_stats_.rice_ms = ms_rice;
  last_timing_stats_.inverse_ms = ms_inverse;
  last_timing_stats_.total_gpu_ms = ms_total;

  if (copy_output_to_host) {
    out_bf16.resize(static_cast<size_t>(num_words_) * 8U);
    if (!check_cuda(
            cudaMemcpy(
                out_bf16.data(),
                d_bf16_out_,
                static_cast<size_t>(num_words_) * 8U * sizeof(uint16_t),
                cudaMemcpyDeviceToHost),
            error_message_,
            "cudaMemcpy bf16_out D2H")) {
      return false;
    }
  } else {
    out_bf16.clear();
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_rice_done);
  cudaEventDestroy(ev_inverse_done);
  return true;
}

bool Stage2CudaDecoder::decode_fused_to_bf16(std::vector<uint16_t>& out_bf16, float alpha, bool copy_output_to_host) {
  if (!ok_) return false;
  if (!(alpha > 0.0f)) {
    error_message_ = "Invalid alpha (must be > 0).";
    return false;
  }
  if (num_words_ <= 0 || d_bf16_out_ == nullptr) {
    error_message_ = "Invalid bf16 decode dimensions.";
    return false;
  }
  last_timing_stats_ = TimingStats{};

  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_done = nullptr;
  if (!check_cuda(cudaEventCreate(&ev_start), error_message_, "cudaEventCreate ev_start")) return false;
  if (!check_cuda(cudaEventCreate(&ev_done), error_message_, "cudaEventCreate ev_done")) return false;

  const int blocks = (num_streams_ + threads_per_block_ - 1) / threads_per_block_;
  if (!check_cuda(cudaEventRecord(ev_start), error_message_, "cudaEventRecord ev_start")) return false;
  decode_fused_stage2_to_bf16_kernel<<<blocks, threads_per_block_>>>(
      d_words_,
      word_count_,
      d_offsets_,
      d_k_,
      num_streams_,
      symbols_per_stream_,
      num_words_,
      alpha,
      d_bf16_out_);
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch decode_fused_stage2_to_bf16_kernel")) return false;
  if (!check_cuda(cudaEventRecord(ev_done), error_message_, "cudaEventRecord ev_done")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize")) return false;

  float ms_fused = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&ms_fused, ev_start, ev_done), error_message_, "cudaEventElapsedTime fused")) return false;
  last_timing_stats_.fused_ms = ms_fused;
  last_timing_stats_.total_gpu_ms = ms_fused;

  if (copy_output_to_host) {
    out_bf16.resize(static_cast<size_t>(num_words_) * 8U);
    if (!check_cuda(
            cudaMemcpy(
                out_bf16.data(),
                d_bf16_out_,
                static_cast<size_t>(num_words_) * 8U * sizeof(uint16_t),
                cudaMemcpyDeviceToHost),
            error_message_,
            "cudaMemcpy bf16_out D2H")) {
      return false;
    }
  } else {
    out_bf16.clear();
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_done);
  return true;
}

bool Stage2CudaDecoder::decode_fused_to(OutputDtype dtype, std::vector<uint8_t>& out_bytes, float alpha, bool copy_output_to_host) {
  if (!ok_) return false;
  if (!(alpha > 0.0f)) {
    error_message_ = "Invalid alpha (must be > 0).";
    return false;
  }
  if (num_words_ <= 0) {
    error_message_ = "Invalid decode dimensions.";
    return false;
  }
  const int elem_bytes = (dtype == OutputDtype::kBf16) ? 2 : 1;
  const size_t total_elems = static_cast<size_t>(num_words_) * 8U;

  // Pick the output device buffer: reuse d_bf16_out_ for bf16, lazily alloc the
  // 1-byte buffer for fp8/int8.
  void* d_out = nullptr;
  if (dtype == OutputDtype::kBf16) {
    if (d_bf16_out_ == nullptr) { error_message_ = "bf16 output buffer not allocated."; return false; }
    d_out = d_bf16_out_;
  } else {
    if (d_byte_out_ == nullptr) {
      if (!check_cuda(cudaMalloc(&d_byte_out_, total_elems), error_message_, "cudaMalloc d_byte_out")) return false;
    }
    d_out = d_byte_out_;
  }

  last_timing_stats_ = TimingStats{};
  cudaEvent_t ev_start = nullptr, ev_done = nullptr;
  if (!check_cuda(cudaEventCreate(&ev_start), error_message_, "cudaEventCreate ev_start")) return false;
  if (!check_cuda(cudaEventCreate(&ev_done), error_message_, "cudaEventCreate ev_done")) return false;

  const int blocks = (num_streams_ + threads_per_block_ - 1) / threads_per_block_;
  if (!check_cuda(cudaEventRecord(ev_start), error_message_, "cudaEventRecord ev_start")) return false;
  switch (dtype) {
    case OutputDtype::kBf16:
      decode_fused_out_kernel<OutputDtype::kBf16><<<blocks, threads_per_block_>>>(
          d_words_, word_count_, d_offsets_, d_k_, num_streams_, symbols_per_stream_, num_words_, alpha, d_out);
      break;
    case OutputDtype::kFp8E4M3:
      decode_fused_out_kernel<OutputDtype::kFp8E4M3><<<blocks, threads_per_block_>>>(
          d_words_, word_count_, d_offsets_, d_k_, num_streams_, symbols_per_stream_, num_words_, alpha, d_out);
      break;
    case OutputDtype::kInt8:
      decode_fused_out_kernel<OutputDtype::kInt8><<<blocks, threads_per_block_>>>(
          d_words_, word_count_, d_offsets_, d_k_, num_streams_, symbols_per_stream_, num_words_, alpha, d_out);
      break;
  }
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch decode_fused_out_kernel")) return false;
  if (!check_cuda(cudaEventRecord(ev_done), error_message_, "cudaEventRecord ev_done")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize")) return false;

  float ms = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&ms, ev_start, ev_done), error_message_, "cudaEventElapsedTime fused_cast")) return false;
  last_timing_stats_.fused_cast_ms = ms;
  last_timing_stats_.total_gpu_ms = ms;

  if (copy_output_to_host) {
    out_bytes.resize(total_elems * elem_bytes);
    if (!check_cuda(cudaMemcpy(out_bytes.data(), d_out, total_elems * elem_bytes, cudaMemcpyDeviceToHost),
                    error_message_, "cudaMemcpy out_bytes D2H")) {
      return false;
    }
  } else {
    out_bytes.clear();
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_done);
  return true;
}

bool Stage2CudaDecoder::decode_fused_coop_to_bf16(std::vector<uint16_t>& out_bf16, float alpha, bool copy_output_to_host) {
  if (!ok_) return false;
  if (!(alpha > 0.0f)) {
    error_message_ = "Invalid alpha (must be > 0).";
    return false;
  }
  if (num_words_ <= 0 || d_bf16_out_ == nullptr) {
    error_message_ = "Invalid bf16 decode dimensions.";
    return false;
  }
  last_timing_stats_ = TimingStats{};

  cudaEvent_t ev_start = nullptr;
  cudaEvent_t ev_done = nullptr;
  if (!check_cuda(cudaEventCreate(&ev_start), error_message_, "cudaEventCreate ev_start")) return false;
  if (!check_cuda(cudaEventCreate(&ev_done), error_message_, "cudaEventCreate ev_done")) return false;

  // A warp's 32 streams form a contiguous bitstream region of (roughly)
  // 32 * (word_count / num_streams) words; it must fit the smem cap or the warp
  // falls back to global reads. Pick the smallest cap that covers the estimate
  // (with margin for length variance), and choose warps/block so each block's
  // smem stays ~constant (~24 KB) — i.e. fewer warps/block as the cap grows.
  // This lets coop keep working at large S (the compression-optimal regime),
  // trading occupancy for coalesced+low-latency reads.
  const long long avg_words_per_stream =
      (static_cast<long long>(word_count_) + num_streams_ - 1) / num_streams_;
  const long long region_est = (32LL * avg_words_per_stream) * 5 / 4 + 64;  // +25% +slack

  // Pick the smallest cap covering the estimate, with warps/block chosen to keep
  // block smem ~constant (~24 KB). Beyond cap=3072 the only fit is 1 warp/block,
  // whose occupancy collapse (~22%) loses to plain fused on this latency-bound
  // kernel — so for large regions (large S, the compression-optimal regime) we
  // launch the plain fused kernel instead. This makes coop a strict improvement
  // over fused: it adds coalesced+low-latency reads where they pay off and is
  // identical to fused where they don't. (True large-S read coalescing needs the
  // interleaved layout, which costs stored memory.)
  int cap_words = 768;
  int warps_per_block = 8;
  bool use_coop = true;
  if (region_est > 3072) { use_coop = false; }
  else if (region_est > 1536) { cap_words = 3072; warps_per_block = 2; }
  else if (region_est > 768) { cap_words = 1536; warps_per_block = 4; }

  if (!check_cuda(cudaEventRecord(ev_start), error_message_, "cudaEventRecord ev_start")) return false;
  if (use_coop) {
    const int threads = warps_per_block * 32;
    const int total_warps = (num_streams_ + 31) / 32;
    const int blocks = (total_warps + warps_per_block - 1) / warps_per_block;
    const size_t smem_bytes = static_cast<size_t>(warps_per_block) * cap_words * sizeof(uint32_t);
#define LAUNCH_COOP(CAP)                                                       \
    decode_fused_coop_stage2_to_bf16_kernel<CAP><<<blocks, threads, smem_bytes>>>( \
        d_words_, word_count_, d_offsets_, d_k_, num_streams_,                 \
        symbols_per_stream_, num_words_, alpha, d_bf16_out_)
    switch (cap_words) {
      case 3072: LAUNCH_COOP(3072); break;
      case 1536: LAUNCH_COOP(1536); break;
      default:   LAUNCH_COOP(768);  break;
    }
#undef LAUNCH_COOP
  } else {
    const int blocks = (num_streams_ + threads_per_block_ - 1) / threads_per_block_;
    decode_fused_stage2_to_bf16_kernel<<<blocks, threads_per_block_>>>(
        d_words_, word_count_, d_offsets_, d_k_, num_streams_,
        symbols_per_stream_, num_words_, alpha, d_bf16_out_);
  }
  if (!check_cuda(cudaGetLastError(), error_message_, "kernel launch decode_fused_coop_stage2_to_bf16_kernel")) return false;
  if (!check_cuda(cudaEventRecord(ev_done), error_message_, "cudaEventRecord ev_done")) return false;
  if (!check_cuda(cudaDeviceSynchronize(), error_message_, "cudaDeviceSynchronize")) return false;

  float ms_coop = 0.0f;
  if (!check_cuda(cudaEventElapsedTime(&ms_coop, ev_start, ev_done), error_message_, "cudaEventElapsedTime coop")) return false;
  last_timing_stats_.coop_ms = ms_coop;
  last_timing_stats_.total_gpu_ms = ms_coop;

  if (copy_output_to_host) {
    out_bf16.resize(static_cast<size_t>(num_words_) * 8U);
    if (!check_cuda(
            cudaMemcpy(
                out_bf16.data(),
                d_bf16_out_,
                static_cast<size_t>(num_words_) * 8U * sizeof(uint16_t),
                cudaMemcpyDeviceToHost),
            error_message_,
            "cudaMemcpy bf16_out D2H")) {
      return false;
    }
  } else {
    out_bf16.clear();
  }

  cudaEventDestroy(ev_start);
  cudaEventDestroy(ev_done);
  return true;
}

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
    bool copy_output_to_host) {
  auto fail = [&](const char* m, cudaError_t e) {
    if (err) *err = std::string(m) + ": " + cudaGetErrorString(e);
    return false;
  };
  if (interleaved_words.empty() || group_base_words.empty() || num_streams <= 0 ||
      symbols_per_stream <= 0 || total_symbols <= 0 || !(alpha > 0.0f)) {
    if (err) *err = "decode_interleaved_layout: invalid arguments.";
    return false;
  }
  const int num_words = total_symbols / 8;

  uint32_t* d_words = nullptr;
  uint32_t* d_gbase = nullptr;
  uint8_t* d_k = nullptr;
  uint16_t* d_out = nullptr;
  cudaError_t e;
  if ((e = cudaMalloc(&d_words, interleaved_words.size() * sizeof(uint32_t))) != cudaSuccess) return fail("cudaMalloc d_words", e);
  if ((e = cudaMalloc(&d_gbase, group_base_words.size() * sizeof(uint32_t))) != cudaSuccess) return fail("cudaMalloc d_gbase", e);
  if ((e = cudaMalloc(&d_k, stream_k.size() * sizeof(uint8_t))) != cudaSuccess) return fail("cudaMalloc d_k", e);
  if ((e = cudaMalloc(&d_out, static_cast<size_t>(num_words) * 8U * sizeof(uint16_t))) != cudaSuccess) return fail("cudaMalloc d_out", e);

  if ((e = cudaMemcpy(d_words, interleaved_words.data(), interleaved_words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)) != cudaSuccess) return fail("memcpy words", e);
  if ((e = cudaMemcpy(d_gbase, group_base_words.data(), group_base_words.size() * sizeof(uint32_t), cudaMemcpyHostToDevice)) != cudaSuccess) return fail("memcpy gbase", e);
  if ((e = cudaMemcpy(d_k, stream_k.data(), stream_k.size() * sizeof(uint8_t), cudaMemcpyHostToDevice)) != cudaSuccess) return fail("memcpy k", e);

  cudaEvent_t ev0, ev1;
  cudaEventCreate(&ev0);
  cudaEventCreate(&ev1);
  const int blocks = (num_streams + threads_per_block - 1) / threads_per_block;
  cudaEventRecord(ev0);
  decode_interleaved_stage2_to_bf16_kernel<<<blocks, threads_per_block>>>(
      d_words, static_cast<int>(interleaved_words.size()), d_gbase, d_k,
      num_streams, symbols_per_stream, num_words, alpha, d_out);
  if ((e = cudaGetLastError()) != cudaSuccess) return fail("kernel launch interleaved", e);
  cudaEventRecord(ev1);
  if ((e = cudaDeviceSynchronize()) != cudaSuccess) return fail("sync", e);
  if (out_ms) { float ms = 0.0f; cudaEventElapsedTime(&ms, ev0, ev1); *out_ms = ms; }

  if (copy_output_to_host) {
    out_bf16.resize(static_cast<size_t>(num_words) * 8U);
    if ((e = cudaMemcpy(out_bf16.data(), d_out, static_cast<size_t>(num_words) * 8U * sizeof(uint16_t), cudaMemcpyDeviceToHost)) != cudaSuccess) return fail("memcpy out", e);
  } else {
    out_bf16.clear();
  }

  cudaEventDestroy(ev0);
  cudaEventDestroy(ev1);
  cudaFree(d_words);
  cudaFree(d_gbase);
  cudaFree(d_k);
  cudaFree(d_out);
  return true;
}

// ---------------------------------------------------------------------------
// Integration launcher: decode from caller-owned device buffers into a
// caller-provided bf16 buffer. Defined outside the anonymous namespace.
// ---------------------------------------------------------------------------
void metalrice_launch_fused_decode_bf16(
    const uint32_t* d_words, int word_count,
    const uint32_t* d_offsets, const uint8_t* d_k,
    int num_streams, int symbols_per_stream, int num_words,
    float alpha, uint16_t* d_bf16_out, int threads_per_block, void* cuda_stream) {
  const int blocks = (num_streams + threads_per_block - 1) / threads_per_block;
  decode_fused_stage2_to_bf16_kernel<<<blocks, threads_per_block, 0,
      static_cast<cudaStream_t>(cuda_stream)>>>(
      d_words, word_count, d_offsets, d_k, num_streams, symbols_per_stream,
      num_words, alpha, d_bf16_out);
}
