// fused_decode_gemv.cu — Warp-specialized fused Rice decode + linear projection.
//
// Eliminates the global-memory scratch buffer of the two-kernel path:
//   Two-kernel: read bitstream (0.5 B/s) + WRITE scratch (2 B/s) + READ scratch (2 B/s)
//   Fused GEMV : read bitstream (0.5 B/s) + read x (smem, once per block)
//
// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1 – rice_fused_gemv_kernel (batch = 1)
//
//   Grid  : (ceil(out_features / ROWS_PER_BLOCK),)    block : (256,)
//   ROWS_PER_BLOCK = floor(256 / streams_per_row)
//
//   Phase 1 (all 256 threads)   : cooperatively load x_rot into shared memory.
//   Phase 2 (first ROWS_PER_BLOCK * streams_per_row threads):
//                                  Each thread independently decodes one Rice
//                                  stream from global memory and accumulates a
//                                  partial dot product against the matching
//                                  x_rot slice already in shared memory.
//   Phase 3 (stream-0 threads)  : reduce STREAMS_PER_ROW partial sums → y[row].
//
//   Critical: threads beyond the first (rows_per_block * streams_per_row) are
//   NEVER active (they write 0 to s_acc and skip all global-memory accesses),
//   preventing OOB reads on boundary blocks.
//
// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2 – rice_fused_gemm_ws_kernel (batch ≥ 1, warp-specialized)
//
//   WS_TILE_ROWS = 4  output rows  per tile
//   WS_TILE_BATCH = 8 batch rows   per tile
//   Block : 64 threads = 2 warps
//     Warp 0 (producer, threads  0–31): decodes WS_TILE_ROWS weight tiles
//       into smem_w[stage][r][j] via a double-buffered ping-pong.  Each lane
//       decodes one row's stream-k sequentially, writing 8 decoded floats per
//       E8 word.
//     Warp 1 (consumer, threads 32–63): while the producer fills the NEXT
//       tile, computes the partial GEMM: for each (b, r) pair assigned to
//       this thread, accumulates sum_j smem_w[stage][r][j] * x[b, stream*sps+j].
//       Exactly 32 (b, r) pairs exist (4 rows × 8 batch), so every consumer
//       thread handles exactly one pair with no idle threads or OOB accesses.
//
//   This is the H100-ready skeleton for warp-specialized decode+MMA.  Replacing
//   the consumer GEMM with wgmma.mma_async and the smem fill with cp.async
//   would give full async pipeline overlap (future work: requires PTX + CUTLASS).

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <stdint.h>
#include "metalrice_device.cuh"

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 1: fused GEMV (batch = 1)
// ─────────────────────────────────────────────────────────────────────────────
// Shared memory:
//   [0 .. in_features*2)          : __nv_bfloat16  s_x[in_features]
//   [in_features*2 .. +BLOCK_THREADS*4) : float s_acc[BLOCK_THREADS]
template <int BLOCK_THREADS>
__global__ void rice_fused_gemv_kernel(
    const uint32_t* __restrict__ words,
    int                           word_count,
    const uint32_t* __restrict__ offsets,
    const uint8_t*  __restrict__ ks,
    int   num_streams,
    int   sps,
    float alpha,
    const __nv_bfloat16* __restrict__ x_rot,
    const __nv_bfloat16* __restrict__ sigma,
    __nv_bfloat16*       __restrict__ y_out,
    int out_features,
    int in_features,
    int streams_per_row)
{
    extern __shared__ char smem[];
    __nv_bfloat16* s_x   = reinterpret_cast<__nv_bfloat16*>(smem);
    float*         s_acc = reinterpret_cast<float*>(smem + in_features * 2);

    // Phase 1: cooperative x load (all 256 threads).
    for (int i = threadIdx.x; i < in_features; i += BLOCK_THREADS)
        s_x[i] = x_rot[i];
    __syncthreads();

    // Thread assignment.
    const int rows_per_block = BLOCK_THREADS / streams_per_row;
    // Only the first (rows_per_block * streams_per_row) threads are active.
    // Threads beyond this limit skip all global-memory access to avoid OOB
    // on boundary blocks where global_row would still be in-bounds.
    const int active_limit   = rows_per_block * streams_per_row;
    const int local_row      = threadIdx.x / streams_per_row;
    const int stream_in_row  = threadIdx.x % streams_per_row;
    const int global_row     = blockIdx.x * rows_per_block + local_row;

    float acc = 0.0f;
    if (threadIdx.x < active_limit && global_row < out_features) {
        const int global_stream = global_row * streams_per_row + stream_in_row;
        if (global_stream < num_streams) {
            acc = mr_decode_stream_and_dot(
                words, word_count,
                offsets[global_stream],
                static_cast<uint32_t>(ks[global_stream]),
                sps / 8,
                1.0f / alpha,
                s_x,
                stream_in_row * sps);
        }
    }
    s_acc[threadIdx.x] = acc;
    __syncthreads();

    // Phase 3: reduce streams_per_row partial sums → output.
    if (stream_in_row == 0 && threadIdx.x < active_limit && global_row < out_features) {
        float total = 0.0f;
        const int base = local_row * streams_per_row;
        for (int s = 0; s < streams_per_row; ++s)
            total += s_acc[base + s];
        y_out[global_row] = __float2bfloat16_rn(total * __bfloat162float(sigma[global_row]));
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Kernel 2: warp-specialized fused GEMM (batch ≥ 1)
//
//   Block : 64 threads = 2 warps
//     Warp 0 (producer) : decodes WS_TILE_ROWS weight tiles into smem_w.
//     Warp 1 (consumer) : accumulates the GEMM over WS_TILE_ROWS × WS_TILE_BATCH.
//
//   Tile: WS_TILE_ROWS rows × WS_SPS columns (= one stream's sps symbols).
//   The producer iterates over streams_per_row streams (all K tiles for these
//   rows), filling smem_w double-buffered while the consumer processes the
//   previous fill.
// ─────────────────────────────────────────────────────────────────────────────
static constexpr int WS_BLOCK_THREADS = 64;   // 2 warps: 0=producer, 1=consumer
static constexpr int WS_TILE_ROWS     = 4;    // output rows per producer tile
static constexpr int WS_TILE_BATCH    = 8;    // batch elements per consumer tile
                                               // WS_TILE_ROWS * WS_TILE_BATCH = 32 = 1 warp ✓
static constexpr int WS_SPS           = 512;  // symbols per stream (matches sps arg)

__global__ void rice_fused_gemm_ws_kernel(
    const uint32_t* __restrict__ words,
    int                           word_count,
    const uint32_t* __restrict__ offsets,
    const uint8_t*  __restrict__ ks,
    int   num_streams,
    int   /*sps*/,          // must equal WS_SPS; checked in binding
    float alpha,
    const __nv_bfloat16* __restrict__ x_rot,  // [batch, in_features]
    const __nv_bfloat16* __restrict__ sigma,  // [out_features]
    __nv_bfloat16*       __restrict__ y_out,  // [batch, out_features]
    int out_features,
    int in_features,
    int batch,
    int streams_per_row)
{
    // Shared memory:
    //   smem_w[2][WS_TILE_ROWS][WS_SPS] – double-buffered decoded weight tile (float32)
    //   smem_x[WS_TILE_BATCH][WS_SPS]   – input activation tile (bfloat16)
    //   smem_acc[WS_TILE_BATCH][WS_TILE_ROWS] – accumulator (float32)
    __shared__ float         smem_w[2][WS_TILE_ROWS][WS_SPS];
    __shared__ __nv_bfloat16 smem_x[WS_TILE_BATCH][WS_SPS];
    __shared__ float         smem_acc[WS_TILE_BATCH][WS_TILE_ROWS];

    const int warp_id = threadIdx.x >> 5;   // 0 = producer, 1 = consumer
    const int lane    = threadIdx.x & 31;

    const int blk_row   = blockIdx.x * WS_TILE_ROWS;
    const int blk_batch = blockIdx.y * WS_TILE_BATCH;

    // Initialise accumulators (consumer warp = threads 32–63).
    if (warp_id == 1) {
        // lane maps to one (b, r) pair: r = lane % WS_TILE_ROWS, b = lane / WS_TILE_ROWS
        const int r = lane % WS_TILE_ROWS;
        const int b = lane / WS_TILE_ROWS;
        smem_acc[b][r] = 0.0f;
    }

    const float inv_alpha = 1.0f / alpha;
    int stage = 0;

    // ── Prime: producer fills stage 0 for stream_in_row = 0.
    if (warp_id == 0) {
        // Each of the first WS_TILE_ROWS lanes decodes one weight row sequentially.
        // Rice decode is serial per stream, so only one lane per row; the other
        // 28 lanes in the producer warp are idle during the prime step.
        const int global_row_p = blk_row + lane;  // one lane per output row
        if (lane < WS_TILE_ROWS && global_row_p < out_features) {
            const int gstream = global_row_p * streams_per_row; // stream_in_row=0
            if (gstream < num_streams) {
                MRBitReader br(words, word_count, offsets[gstream]);
                const uint32_t k = ks[gstream];
                for (int w = 0; w < WS_SPS / 8; ++w) {
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
                        smem_w[0][lane][w * 8 + i] = dec[i];
                }
            } else {
                for (int j = 0; j < WS_SPS; ++j) smem_w[0][lane][j] = 0.0f;
            }
        } else if (lane < WS_TILE_ROWS) {
            for (int j = 0; j < WS_SPS; ++j) smem_w[0][lane][j] = 0.0f;
        }
    }
    __syncthreads();

    // ── Main loop over K-tiles (streams).
    for (int sk = 0; sk < streams_per_row; ++sk) {
        const int base_x    = sk * WS_SPS;
        const int next_sk   = sk + 1;
        const int ns        = 1 - stage;

        // Producer: decode next weight tile into smem_w[ns].
        if (warp_id == 0 && next_sk < streams_per_row) {
            const int global_row_p = blk_row + lane;
            if (lane < WS_TILE_ROWS && global_row_p < out_features) {
                const int gstream = global_row_p * streams_per_row + next_sk;
                if (gstream < num_streams) {
                    MRBitReader br(words, word_count, offsets[gstream]);
                    const uint32_t k = ks[gstream];
                    for (int w = 0; w < WS_SPS / 8; ++w) {
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
                            smem_w[ns][lane][w * 8 + i] = dec[i];
                    }
                } else {
                    for (int j = 0; j < WS_SPS; ++j) smem_w[ns][lane][j] = 0.0f;
                }
            } else if (lane < WS_TILE_ROWS) {
                for (int j = 0; j < WS_SPS; ++j) smem_w[ns][lane][j] = 0.0f;
            }
        }

        // Consumer: load x tile, then accumulate partial GEMM on current stage.
        if (warp_id == 1) {
            // Load smem_x[b][j] = x_rot[blk_batch+b, base_x+j].
            // Consumer warp has 32 lanes; each loads WS_SPS/32 = 16 elements per batch row.
            for (int b = 0; b < WS_TILE_BATCH; ++b) {
                const int gb = blk_batch + b;
                for (int j = lane; j < WS_SPS; j += 32) {
                    const int gj = base_x + j;
                    smem_x[b][j] = (gb < batch && gj < in_features)
                        ? x_rot[gb * in_features + gj]
                        : __float2bfloat16(0.0f);
                }
            }
        }
        __syncthreads();  // smem_x and smem_w[stage] both ready

        // Consumer accumulation: thread lane → (b, r) pair.
        // lane 0..31: r = lane%WS_TILE_ROWS (0..3), b = lane/WS_TILE_ROWS (0..7).
        if (warp_id == 1) {
            const int r = lane % WS_TILE_ROWS;
            const int b = lane / WS_TILE_ROWS;
            float s = 0.0f;
            const float* __restrict__ wrow = smem_w[stage][r];
            const __nv_bfloat16* __restrict__ xrow = smem_x[b];
#pragma unroll 8
            for (int j = 0; j < WS_SPS; ++j)
                s += wrow[j] * __bfloat162float(xrow[j]);
            smem_acc[b][r] += s;
        }
        __syncthreads();  // producer may now overwrite this stage

        stage = 1 - stage;
    }

    // Write outputs (consumer warp).
    if (warp_id == 1) {
        const int r = lane % WS_TILE_ROWS;
        const int b = lane / WS_TILE_ROWS;
        const int global_row   = blk_row   + r;
        const int global_batch = blk_batch + b;
        if (global_row < out_features && global_batch < batch) {
            y_out[global_batch * out_features + global_row] =
                __float2bfloat16_rn(smem_acc[b][r] * __bfloat162float(sigma[global_row]));
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Launchers
// ─────────────────────────────────────────────────────────────────────────────

void launch_rice_fused_gemv(
    const uint32_t* words, int word_count,
    const uint32_t* offsets, const uint8_t* ks,
    int num_streams, int sps, float alpha,
    const uint16_t* x_rot,
    const uint16_t* sigma,
    uint16_t*       y_out,
    int out_features, int in_features,
    cudaStream_t stream)
{
    constexpr int BT = 256;
    const int streams_per_row = in_features / sps;
    const int rows_per_block  = BT / streams_per_row;
    const int grid = (out_features + rows_per_block - 1) / rows_per_block;
    const size_t smem = static_cast<size_t>(in_features) * 2
                      + static_cast<size_t>(BT) * 4;

    rice_fused_gemv_kernel<BT><<<grid, BT, smem, stream>>>(
        words, word_count, offsets, ks,
        num_streams, sps, alpha,
        reinterpret_cast<const __nv_bfloat16*>(x_rot),
        reinterpret_cast<const __nv_bfloat16*>(sigma),
        reinterpret_cast<__nv_bfloat16*>(y_out),
        out_features, in_features, streams_per_row);
}

void launch_rice_fused_gemm_ws(
    const uint32_t* words, int word_count,
    const uint32_t* offsets, const uint8_t* ks,
    int num_streams, int sps, float alpha,
    const uint16_t* x_rot,
    const uint16_t* sigma,
    uint16_t*       y_out,
    int out_features, int in_features, int batch,
    cudaStream_t stream)
{
    const int streams_per_row = in_features / sps;
    const dim3 grid(
        (out_features + WS_TILE_ROWS  - 1) / WS_TILE_ROWS,
        (batch         + WS_TILE_BATCH - 1) / WS_TILE_BATCH);

    rice_fused_gemm_ws_kernel<<<grid, WS_BLOCK_THREADS, 0, stream>>>(
        words, word_count, offsets, ks,
        num_streams, sps, alpha,
        reinterpret_cast<const __nv_bfloat16*>(x_rot),
        reinterpret_cast<const __nv_bfloat16*>(sigma),
        reinterpret_cast<__nv_bfloat16*>(y_out),
        out_features, in_features, batch, streams_per_row);
}
