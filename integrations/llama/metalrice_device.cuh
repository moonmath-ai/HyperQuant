// metalrice_device.cuh
// Shared device-side primitives for MetalRice kernels.
// Bit-exact with stage2_cuda_decoder.cu (same BitReader / E8 inverse logic).
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>

// ─────────────────────────────────────────────────────────────────────────────
// Bit-stream reader (MSB-first, contiguous uint32 words)
// Matches the BitReader in stage2_cuda_decoder.cu exactly.
// ─────────────────────────────────────────────────────────────────────────────

struct MRBitReader {
    const uint32_t* words;
    int             word_count;
    int             next_word_idx;
    uint32_t        c1, c2;
    int             c1_bits, c2_bits;
    uint64_t        reg;
    int             bits_in_reg;

    __device__ __forceinline__
    static uint32_t safe_load(const uint32_t* w, int wc, int idx) {
        return (idx >= 0 && idx < wc) ? w[idx] : 0U;
    }

    __device__ __forceinline__
    static uint32_t shl32z(uint32_t v, int s) {
        return (s >= 32) ? 0U : (v << s);
    }

    __device__
    MRBitReader(const uint32_t* w, int wc, uint32_t bit_offset)
        : words(w), word_count(wc)
    {
        const int sw = static_cast<int>(bit_offset >> 5);
        const int sb = static_cast<int>(bit_offset & 31U);
        c1 = safe_load(w, wc, sw);
        c2 = safe_load(w, wc, sw + 1);
        c1_bits = 32;  c2_bits = 32;
        next_word_idx = sw + 2;
        reg = 0;  bits_in_reg = 0;
        if (sb > 0) {
            c1 = shl32z(c1, sb);
            c1_bits -= sb;
            if (c1_bits == 0) {
                c1 = c2;  c1_bits = 32;
                c2 = safe_load(w, wc, next_word_idx++);
            }
        }
    }

    __device__ __forceinline__ void promote() {
        c1 = c2;  c1_bits = c2_bits;
        c2 = safe_load(words, word_count, next_word_idx++);
        c2_bits = 32;
    }

    __device__ __forceinline__ void refill(int need) {
        while (bits_in_reg < need) {
            if (c1_bits == 0) promote();
            const int room = 64 - bits_in_reg;
            const int take = (c1_bits < room) ? c1_bits : room;
            const uint32_t piece = c1 >> (32 - take);
            reg |= static_cast<uint64_t>(piece) << (64 - bits_in_reg - take);
            c1 = shl32z(c1, take);
            c1_bits -= take;  bits_in_reg += take;
        }
    }

    __device__ __forceinline__ uint32_t read_bits(int n) {
        if (n <= 0) return 0U;
        refill(n);
        const uint32_t v = static_cast<uint32_t>(reg >> (64 - n));
        reg <<= n;  bits_in_reg -= n;
        return v;
    }

    // Count leading zeros in valid bits, consume quotient + terminator.
    __device__ __forceinline__ uint32_t read_unary() {
        uint32_t q = 0;
        while (true) {
            refill(1);
            uint64_t probe = reg;
            if (bits_in_reg < 64)
                probe |= (1ULL << (64 - bits_in_reg)) - 1ULL;
            const uint32_t lz = (probe == 0ULL)
                ? 64U : static_cast<uint32_t>(__clzll(probe));
            q += lz;
            if (lz < static_cast<uint32_t>(bits_in_reg)) {
                const int consumed = static_cast<int>(lz) + 1;
                reg <<= consumed;  bits_in_reg -= consumed;
                break;
            }
            reg = 0;  bits_in_reg = 0;
        }
        return q;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
// E8int inverse transform: 8 zigzag symbols → 8 float lattice values Y/alpha
// Matches inverse_e8int_word in stage2_cuda_decoder.cu.
// ─────────────────────────────────────────────────────────────────────────────

__device__ __forceinline__ int mr_inv_zigzag(uint32_t zz) {
    return (zz & 1U) ? -static_cast<int>((zz + 1U) >> 1U)
                     :  static_cast<int>(zz >> 1U);
}

__device__ __forceinline__
void mr_decode_e8_word(const uint8_t z[8], float inv_alpha, float out[8]) {
    int s[8];
    for (int i = 0; i < 7; ++i)
        s[i] = mr_inv_zigzag(static_cast<uint32_t>(z[i]));
    const uint32_t z7 = z[7];
    const int c = static_cast<int>(z7 & 1U);
    const int t = mr_inv_zigzag(z7 >> 1U);
    int p8 = 0;
    for (int i = 0; i < 7; ++i) p8 += s[i];
    p8 &= 1;
    s[7] = (t << 1) + p8;
    for (int i = 0; i < 8; ++i)
        out[i] = static_cast<float>((s[i] << 1) + c) * inv_alpha;
}

// Decode one full Rice stream into float accumulators (dot product with x).
// One thread calls this for its assigned stream; `base_x` is the start index
// of the x slice that this stream's symbols correspond to.
__device__ __forceinline__
float mr_decode_stream_and_dot(
    const uint32_t* words, int word_count,
    uint32_t bit_offset, uint32_t k,
    int words_per_stream,          // = sps / 8
    float inv_alpha,
    const __nv_bfloat16* __restrict__ s_x,  // shared x, [in_features]
    int base_x)                    // starting index into s_x
{
    MRBitReader br(words, word_count, bit_offset);
    float acc = 0.0f;
    for (int w = 0; w < words_per_stream; ++w) {
        uint8_t z[8];
#pragma unroll
        for (int p = 0; p < 8; ++p) {
            const uint32_t q   = br.read_unary();
            const uint32_t rem = br.read_bits(static_cast<int>(k));
            z[p] = static_cast<uint8_t>(((q << k) | rem) & 0xFFU);
        }
        float dec[8];
        mr_decode_e8_word(z, inv_alpha, dec);
#pragma unroll
        for (int i = 0; i < 8; ++i)
            acc += dec[i] * __bfloat162float(s_x[base_x + w * 8 + i]);
    }
    return acc;
}
