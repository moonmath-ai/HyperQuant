// rice_fused_attention.cu — Fused Rice decode + scaled dot-product attention.
//
// Replaces the three-step decode path
//   (1) rice_decode_into  → bf16 scratch K  (one call per chunk)
//   (2) denorm + unrotate → bf16 K          (one GEMM per chunk)
//   (3) standard attention Q·K^T, softmax, ·V
// with a single kernel per attention layer that reads the Rice bitstream
// directly and computes the softmax-weighted output without any bf16
// scratch tensor in global memory.
//
// ─────────────────────────────────────────────────────────────────────────────
// Algorithm (Flash-Rice attention, two-pass, for single decode step T_q = 1):
//
//   Pre-processing (Python, before kernel):
//     Q_prerot[h_q] = Q[h_q] @ R_k      (Q in K's rotated frame; R_k = rotation_k)
//       [because K_rot = K @ R_k.T, so Q·K = (Q@R_k)·K_rot]
//
//   Pass 1 — decode K, compute scores:
//     For each Rice chunk c, thread 0 decodes stream (c·n_kv + kv_head)
//     from the consolidated bitstream into shared tile s_K[CT][D].  After
//     __syncthreads(), all 128 threads compute scores[base+t][q] for all
//     (t, q) pairs in the chunk via warp-reduce dot products.
//     Pending tokens (unfilled partial chunk, stored as normalised bf16)
//     are handled similarly without Rice decode.
//
//   Softmax:  in-place on scores[T_total][n_q_per_kv] in shared memory.
//
//   Pass 2 — decode V, accumulate output:
//     Same structure as Pass 1 but thread 0 decodes V bitstream and all
//     threads accumulate s_out[q][d] += score[t][q] * V_tile[t][d].
//
//   Post-processing (Python, after kernel):
//     output[h_q] = out_rot[h_q] @ R_v   (undo V rotation)
//
// ─────────────────────────────────────────────────────────────────────────────
// Grid  : (n_kv_heads,)  — one block per KV head
// Block : ATTN_BLOCK = 128 threads  (n_q_per_kv × 32 threads/Q-head)
//
// Shared memory (T_max = 512, n_q_per_kv = 4, D = 128):
//   s_K   [chunk_tokens × D]  float = 4×128×4 = 2 KB  (reused for V)
//   s_Q   [n_q_per_kv × D]    float = 4×128×4 = 2 KB
//   s_sc  [T_max × n_q_per_kv] float = 512×4×4 = 8 KB
//   s_out [n_q_per_kv × D]    float = 4×128×4 = 2 KB
//   Total ≈ 14 KB per block — well within H100's 228 KB per SM.
//
// Falls back (score tensor not populated) if T_total > T_max; caller then
// uses the standard two-kernel path for this step.

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdint.h>
#include "metalrice_device.cuh"

static constexpr int ATTN_BLOCK  = 128;   // threads per block = n_q_per_kv × 32
static constexpr int ATTN_T_MAX  = 512;   // max past tokens in shared scores
static constexpr int ATTN_CT_MAX = 8;     // max chunk_tokens (sps/head_dim)
static constexpr int ATTN_D_MAX  = 128;   // head_dim (Llama-style)
static constexpr int ATTN_Q_MAX  = 8;     // max n_q_per_kv

// ─────────────────────────────────────────────────────────────────────────────
// Warp-level reduce and score helpers
// ─────────────────────────────────────────────────────────────────────────────

// Reduce a float within a single warp using shuffles.  Returns the full sum
// in lane 0 (other lanes hold intermediate results).
__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1)
        v += __shfl_down_sync(0xFFFFFFFF, v, mask);
    return v;
}

// Compute all scores for one decoded K tile.
// s_K     : [chunk_tokens][D]  float (decoded, denormalised)
// s_Q     : [n_q_per_kv][D]   float
// s_sc    : [T_total][n_q_per_kv] float (write target)
// base_t  : token index of s_K[0] in the full T dimension
// scale   : = 1.0 / D  (combines attention scale and the norm normalisation)
__device__ void compute_scores_tile(
    const float* __restrict__ s_K,       // [CT][D]
    const float* __restrict__ s_Q,       // [n_q_per_kv][D]
    float*       __restrict__ s_sc,      // [T_max][n_q_per_kv]
    int base_t, int chunk_tokens, int head_dim, int n_q_per_kv,
    float score_inv_D                    // = 1.0f / head_dim
) {
    // Thread layout: 32 threads per Q-head → threadIdx.x / 32 = q_head index.
    // Each group of 32 threads reduces over head_dim elements.
    const int q   = threadIdx.x >> 5;          // Q-head in [0, n_q_per_kv)
    const int lane = threadIdx.x & 31;
    if (q >= n_q_per_kv) return;

    // Each thread contributes lane and lane+32*k for k=0..D/32-1.
    for (int t = 0; t < chunk_tokens; ++t) {
        float partial = 0.0f;
        for (int d = lane; d < head_dim; d += 32)
            partial += s_Q[q * head_dim + d] * s_K[t * head_dim + d];
        partial = warp_reduce_sum(partial);
        if (lane == 0)
            s_sc[(base_t + t) * n_q_per_kv + q] = partial * score_inv_D;
    }
}

// Accumulate output for one decoded V tile using precomputed softmax weights.
__device__ void accumulate_output_tile(
    const float* __restrict__ s_V,    // [CT][D]
    const float* __restrict__ s_sc,   // [T_max][n_q_per_kv] (softmax weights)
    float*       __restrict__ s_out,  // [n_q_per_kv][D]
    int base_t, int chunk_tokens, int head_dim, int n_q_per_kv
) {
    const int q   = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (q >= n_q_per_kv) return;

    for (int t = 0; t < chunk_tokens; ++t) {
        float w = s_sc[(base_t + t) * n_q_per_kv + q];
        for (int d = lane; d < head_dim; d += 32)
            s_out[q * head_dim + d] += w * s_V[t * head_dim + d];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Main fused kernel
// ─────────────────────────────────────────────────────────────────────────────

__global__ void rice_fused_attn_decode_kernel(
    // ── K consolidated bitstream
    const uint32_t* __restrict__ words_k, int wc_k,
    const uint32_t* __restrict__ offsets_k,   // [total_streams] global bit offsets
    const uint8_t*  __restrict__ ks_k,
    int n_streams_k,                           // = n_chunks × n_kv_heads
    // ── K norms: [n_kv_heads, n_complete] float16 raw L2 norms
    const __nv_bfloat16* __restrict__ norms_k,
    // ── V consolidated bitstream (same layout)
    const uint32_t* __restrict__ words_v, int wc_v,
    const uint32_t* __restrict__ offsets_v,
    const uint8_t*  __restrict__ ks_v,
    int n_streams_v,
    const __nv_bfloat16* __restrict__ norms_v,
    // ── Pending buffer (normalised, rotated, bf16): [n_kv_heads, n_pend, D]
    const __nv_bfloat16* __restrict__ pend_uk,  // may be nullptr if n_pend==0
    const __nv_bfloat16* __restrict__ pend_nk,  // [n_kv_heads, n_pend]
    const __nv_bfloat16* __restrict__ pend_uv,
    const __nv_bfloat16* __restrict__ pend_nv,
    int n_pend,
    // ── Query (pre-rotated to K's frame): [n_q_heads, D] float16
    const __nv_bfloat16* __restrict__ Q_prerot,
    // ── Output (in V's rotated normalised frame): [n_q_heads, D] float32
    float* __restrict__ out_rot,
    // ── Dimensions
    int n_kv_heads, int n_q_per_kv, int head_dim,
    int n_complete, int chunk_tokens, int sps, float alpha
) {
    const int kv_head = blockIdx.x;            // one block per KV head
    const int T_total = n_complete + n_pend;
    const int n_chunks= n_complete / chunk_tokens;

    if (T_total > ATTN_T_MAX) return;          // safety: caller falls back

    // ── Shared memory layout (static sizes bounded by ATTN_* constants).
    __shared__ float s_K [ATTN_CT_MAX * ATTN_D_MAX];  // decoded K tile (reused V)
    __shared__ float s_Q [ATTN_Q_MAX  * ATTN_D_MAX];  // query vectors
    __shared__ float s_sc[ATTN_T_MAX  * ATTN_Q_MAX];  // scores [T][n_q_per_kv]
    __shared__ float s_out[ATTN_Q_MAX * ATTN_D_MAX];  // output accumulator

    // ── Load Q into shared (cooperative, all 128 threads).
    for (int q = 0; q < n_q_per_kv; ++q) {
        int h_q = kv_head * n_q_per_kv + q;
        for (int d = threadIdx.x; d < head_dim; d += ATTN_BLOCK)
            s_Q[q * head_dim + d] = __bfloat162float(Q_prerot[h_q * head_dim + d]);
    }
    // ── Initialise scores to -inf and output to 0.
    for (int i = threadIdx.x; i < T_total * n_q_per_kv; i += ATTN_BLOCK)
        s_sc[i] = -1e30f;
    for (int i = threadIdx.x; i < n_q_per_kv * head_dim; i += ATTN_BLOCK)
        s_out[i] = 0.0f;
    __syncthreads();

    // score = Q_prerot · K_rot / sqrt(D).  K_rot stored as u_hat·norm/sqrt(D),
    // so score = dot(Q_prerot, s_K) / sqrt(D).  inv_D encodes 1/sqrt(D).
    const float inv_D = 1.0f / sqrtf(static_cast<float>(head_dim));
    const float inv_alpha = 1.0f / alpha;

    // ──────────────────────── Pass 1: decode K, compute scores ──────────────
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        // Stream index in the consolidated bitstream for this KV head + chunk.
        // Layout: chunk 0 → streams [0..n_kv_heads-1], chunk 1 → [n_kv_heads..2n-1]
        const int sid = chunk * n_kv_heads + kv_head;

        // ── Thread 0: serial Rice decode for this stream.
        if (threadIdx.x == 0) {
            MRBitReader br(words_k, wc_k, offsets_k[sid]);
            const uint32_t k = ks_k[sid];
            const int base_tok  = chunk * chunk_tokens;
            const int wps       = sps / 8;           // words (E8) per stream
            const int wppt      = head_dim / 8;      // words per token per head
            for (int w = 0; w < wps; ++w) {
                uint8_t z[8];
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const uint32_t q   = br.read_unary();
                    const uint32_t rem = br.read_bits(static_cast<int>(k));
                    z[p] = static_cast<uint8_t>(((q << k) | rem) & 0xFFU);
                }
                float dec[8];
                mr_decode_e8_word(z, inv_alpha, dec);
                // Map word w → (token, dim_start) within the tile.
                const int tok = w / wppt;
                const int d0  = (w % wppt) * 8;
                // Denormalise: multiply by raw_norm / sqrt(D).
                const float norm_factor = __bfloat162float(
                    norms_k[kv_head * n_complete + base_tok + tok])
                    / sqrtf(static_cast<float>(head_dim));
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    s_K[tok * head_dim + d0 + i] = dec[i] * norm_factor;
            }
        }
        __syncthreads();

        compute_scores_tile(s_K, s_Q, s_sc, chunk * chunk_tokens,
                            chunk_tokens, head_dim, n_q_per_kv, inv_D);
        __syncthreads();
    }

    // ── Pending tokens (normalised bf16 directly).
    if (n_pend > 0 && pend_uk != nullptr) {
        if (threadIdx.x == 0) {
            const int base_tok = n_complete;
            for (int t = 0; t < n_pend; ++t) {
                const float nf = __bfloat162float(
                    pend_nk[kv_head * n_pend + t])
                    / sqrtf(static_cast<float>(head_dim));
                for (int d = 0; d < head_dim; ++d)
                    s_K[t * head_dim + d] = __bfloat162float(
                        pend_uk[(kv_head * n_pend + t) * head_dim + d]) * nf;
            }
        }
        __syncthreads();
        compute_scores_tile(s_K, s_Q, s_sc, n_complete,
                            n_pend, head_dim, n_q_per_kv, inv_D);
        __syncthreads();
    }

    // ──────────────────────── Softmax ───────────────────────────────────────
    // Each Q-head's scores are T_total values; compute softmax in parallel
    // across Q-heads (one warp per Q-head, iterating over T_total serially).
    {
        const int q   = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        if (q < n_q_per_kv && lane == 0) {
            float m = -1e30f, l = 0.0f;
            for (int t = 0; t < T_total; ++t) {
                const float x = s_sc[t * n_q_per_kv + q];
                const float m2 = fmaxf(m, x);
                l = l * expf(m - m2) + expf(x - m2);
                m = m2;
            }
            for (int t = 0; t < T_total; ++t) {
                float& sc = s_sc[t * n_q_per_kv + q];
                sc = expf(sc - m) / l;
            }
        }
    }
    __syncthreads();

    // ──────────────────────── Pass 2: decode V, accumulate output ───────────
    for (int chunk = 0; chunk < n_chunks; ++chunk) {
        const int sid = chunk * n_kv_heads + kv_head;

        if (threadIdx.x == 0) {
            MRBitReader br(words_v, wc_v, offsets_v[sid]);
            const uint32_t k = ks_v[sid];
            const int base_tok = chunk * chunk_tokens;
            const int wps = sps / 8;
            const int wppt = head_dim / 8;
            for (int w = 0; w < wps; ++w) {
                uint8_t z[8];
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const uint32_t q   = br.read_unary();
                    const uint32_t rem = br.read_bits(static_cast<int>(k));
                    z[p] = static_cast<uint8_t>(((q << k) | rem) & 0xFFU);
                }
                float dec[8];
                mr_decode_e8_word(z, inv_alpha, dec);
                const int tok = w / wppt;
                const int d0  = (w % wppt) * 8;
                const float nf = __bfloat162float(
                    norms_v[kv_head * n_complete + base_tok + tok])
                    / sqrtf(static_cast<float>(head_dim));
#pragma unroll
                for (int i = 0; i < 8; ++i)
                    s_K[tok * head_dim + d0 + i] = dec[i] * nf;
            }
        }
        __syncthreads();

        accumulate_output_tile(s_K, s_sc, s_out, chunk * chunk_tokens,
                               chunk_tokens, head_dim, n_q_per_kv);
        __syncthreads();
    }

    // ── Pending V.
    if (n_pend > 0 && pend_uv != nullptr) {
        if (threadIdx.x == 0) {
            for (int t = 0; t < n_pend; ++t) {
                const float nf = __bfloat162float(
                    pend_nv[kv_head * n_pend + t])
                    / sqrtf(static_cast<float>(head_dim));
                for (int d = 0; d < head_dim; ++d)
                    s_K[t * head_dim + d] = __bfloat162float(
                        pend_uv[(kv_head * n_pend + t) * head_dim + d]) * nf;
            }
        }
        __syncthreads();
        accumulate_output_tile(s_K, s_sc, s_out, n_complete,
                               n_pend, head_dim, n_q_per_kv);
        __syncthreads();
    }

    // ── Write output: [n_q_per_kv, D] → out_rot[kv_head*n_q_per_kv .. ][D].
    for (int q = 0; q < n_q_per_kv; ++q) {
        int h_q = kv_head * n_q_per_kv + q;
        for (int d = threadIdx.x; d < head_dim; d += ATTN_BLOCK)
            out_rot[h_q * head_dim + d] = s_out[q * head_dim + d];
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// C-linkage launcher called from binding.cu
// ─────────────────────────────────────────────────────────────────────────────
void launch_rice_fused_attn_decode(
    // K bitstream
    const uint32_t* words_k, int wc_k,
    const uint32_t* offsets_k, const uint8_t* ks_k, int n_streams_k,
    const uint16_t* norms_k,
    // V bitstream
    const uint32_t* words_v, int wc_v,
    const uint32_t* offsets_v, const uint8_t* ks_v, int n_streams_v,
    const uint16_t* norms_v,
    // Pending
    const uint16_t* pend_uk, const uint16_t* pend_nk,
    const uint16_t* pend_uv, const uint16_t* pend_nv,
    int n_pend,
    // Query (pre-rotated)
    const uint16_t* Q_prerot,
    // Output
    float* out_rot,
    // Dims
    int n_kv_heads, int n_q_per_kv, int head_dim,
    int n_complete, int chunk_tokens, int sps, float alpha,
    cudaStream_t stream)
{
    if (n_complete + n_pend > ATTN_T_MAX) return;  // too long: caller falls back
    if (n_q_per_kv > ATTN_Q_MAX)          return;
    if (chunk_tokens > ATTN_CT_MAX)        return;
    if (head_dim != ATTN_D_MAX)            return;  // only 128-D supported

    rice_fused_attn_decode_kernel<<<n_kv_heads, ATTN_BLOCK, 0, stream>>>(
        words_k, wc_k,
        offsets_k, ks_k, n_streams_k,
        reinterpret_cast<const __nv_bfloat16*>(norms_k),
        words_v, wc_v,
        offsets_v, ks_v, n_streams_v,
        reinterpret_cast<const __nv_bfloat16*>(norms_v),
        reinterpret_cast<const __nv_bfloat16*>(pend_uk),
        reinterpret_cast<const __nv_bfloat16*>(pend_nk),
        reinterpret_cast<const __nv_bfloat16*>(pend_uv),
        reinterpret_cast<const __nv_bfloat16*>(pend_nv),
        n_pend,
        reinterpret_cast<const __nv_bfloat16*>(Q_prerot),
        out_rot,
        n_kv_heads, n_q_per_kv, head_dim,
        n_complete, chunk_tokens, sps, alpha);
}

// Returns ATTN_T_MAX so Python can query it.
int rice_fused_attn_t_max() { return ATTN_T_MAX; }
