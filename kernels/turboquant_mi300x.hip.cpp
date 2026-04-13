/**
 * turboquant_mi300x.hip.cpp — TurboQuant HIP Kernels for AMD Instinct MI300X
 *
 * Target: gfx942 (CDNA3), ROCm 7.2, Wave64 (wavefront = 64 threads)
 *
 * Key optimizations vs. domvox/turboquant-hip (gfx1100/RDNA3 baseline):
 *
 *  1. BALLOT-BASED BIT PACKING (Opt-A)
 *     Replaces domvox's atomicOr pattern (~32 serialized LDS atomics per word)
 *     with __ballot() wavefront instructions. On Wave64, two 64-bit ballot masks
 *     pack all 128 index bits per plane with zero LDS atomics. Estimated savings:
 *     ~960 cycles → 6 cycles per packed 3-bit word.
 *
 *  2. REGISTER-BASED INDEX STORAGE (Opt-B)
 *     Quantized indices are kept in registers (not LDS), eliminating the 128-byte
 *     s_indices LDS allocation. The ballot instruction reads bits directly from
 *     each thread's register. LDS usage drops from ~700 bytes to ~524 bytes per
 *     block, enabling 24% more blocks per CU (better occupancy).
 *
 *  3. FUSED WRITE (Opt-C)
 *     Output bitplanes are written directly to global memory by warp leaders
 *     (one uint64_t store per warp per bit plane). The domvox single-thread
 *     sequential write (48-iteration loop in thread 0) is eliminated.
 *
 *  4. QJL KEYS KERNEL (Opt-D, new — not in domvox)
 *     Implements TurboQuant Algorithm 2 (TurboQuant_prod): adds 1-bit QJL
 *     residual correction for key vectors, addressing the systematic MSE bias
 *     in key dot products. The QJL correction yields the full 3-bit quality
 *     of the paper (2-bit MSE + 1-bit QJL) vs. domvox's MSE-only approach.
 *
 *  5. WAVE64-NATIVE WARP REDUCTION (Opt-E)
 *     Explicit Wave64 warp reduction starting at offset=32 (6 iterations).
 *     Domvox uses runtime warpSize; ours is statically compiled for gfx942.
 *
 * Bit-plane format:
 *   For TQ3, bit plane p (p=0 LSB, p=2 MSB) is 128 bits:
 *     bytes [p*16..p*16+8)  = ballot mask from wave 0 (threads 0-63)
 *     bytes [p*16+8..p*16+16) = ballot mask from wave 1 (threads 64-127)
 *   bit i of wave 0 mask = bit p of index[i]
 *   bit i of wave 1 mask = bit p of index[64+i]
 *
 * Authors: Adaptation of domvox/turboquant-hip for CDNA3/MI300X
 * Date: April 2026
 */

#include "turboquant_mi300x.h"
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdint.h>

// ──────────────────────────────────────────────────────────────────────────────
// Constant memory codebooks (uploaded once via tqm_init)
// ──────────────────────────────────────────────────────────────────────────────

__constant__ float d_cb3[8];   // TQ3 Lloyd-Max centroids
__constant__ float d_cb4[16];  // TQ4 Lloyd-Max centroids
__constant__ float d_cb2[4];   // TQ2 Lloyd-Max centroids

// ──────────────────────────────────────────────────────────────────────────────
// Device utilities
// ──────────────────────────────────────────────────────────────────────────────

/**
 * Wave64-optimized warp reduction (gfx942: wavefront = 64 threads).
 * Starts at offset=32 and runs 6 iterations.
 */
__device__ __forceinline__
float warp_reduce_sum_w64(float val) {
    val += __shfl_down(val, 32);
    val += __shfl_down(val, 16);
    val += __shfl_down(val, 8);
    val += __shfl_down(val, 4);
    val += __shfl_down(val, 2);
    val += __shfl_down(val, 1);
    return val;
}

/**
 * Find nearest TQ3 centroid (8-level Lloyd-Max).
 * Linear scan is optimal for n_levels=8 (constant memory lookup table).
 */
__device__ __forceinline__
uint8_t tqm_nearest_3(float val) {
    float best = 1e30f;
    uint8_t idx = 0;
    for (int c = 0; c < 8; c++) {
        float d = val - d_cb3[c];
        d = d * d;
        if (d < best) { best = d; idx = (uint8_t)c; }
    }
    return idx;
}

/**
 * Find nearest TQ4 centroid (16-level).
 */
__device__ __forceinline__
uint8_t tqm_nearest_4(float val) {
    float best = 1e30f;
    uint8_t idx = 0;
    for (int c = 0; c < 16; c++) {
        float d = val - d_cb4[c];
        d = d * d;
        if (d < best) { best = d; idx = (uint8_t)c; }
    }
    return idx;
}

/**
 * Find nearest TQ2 centroid (4-level).
 */
__device__ __forceinline__
uint8_t tqm_nearest_2(float val) {
    float best = 1e30f;
    uint8_t idx = 0;
    for (int c = 0; c < 4; c++) {
        float d = val - d_cb2[c];
        d = d * d;
        if (d < best) { best = d; idx = (uint8_t)c; }
    }
    return idx;
}

// ──────────────────────────────────────────────────────────────────────────────
// TQ3 Quantize Kernel — MI300X Optimized
// ──────────────────────────────────────────────────────────────────────────────

/**
 * tqm_quantize_kernel_tq3
 *
 * Grid:  (n_vectors, 1, 1)
 * Block: (TQ_HEAD_DIM=128, 1, 1)  — 2 Wave64 wavefronts
 *
 * Per-block:
 *   - LDS: s_input[128] + s_norm_sq + s_warp_sums[2] = 524 bytes
 *   - Rotation: read from global memory (256 MB LLC keeps it warm after first block)
 *   - No LDS atomics (Opt-A), no s_indices LDS (Opt-B), fused write (Opt-C)
 */
__global__ void tqm_quantize_kernel_tq3(
    const float        * __restrict__ src,
    const float        * __restrict__ rotation,
    block_tq3_mi300x   * __restrict__ dst,
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid  = threadIdx.x;  // 0..127
    const int lane = tid & 63;     // lane within wavefront (0..63)
    const int warp = tid >> 6;     // wavefront index (0 or 1)
    const int d    = TQ_HEAD_DIM;  // 128

    // ── LDS: input vector + norm reduction ──────────────────────────────────
    __shared__ float s_input[TQ_HEAD_DIM];
    __shared__ float s_norm_sq;
    __shared__ float s_warp_sums[2];  // one per wavefront

    // Load input
    s_input[tid] = src[vec_idx * d + tid];
    __syncthreads();

    // ── Norm computation: 2-level reduction ─────────────────────────────────
    float val_sq = s_input[tid] * s_input[tid];
    val_sq = warp_reduce_sum_w64(val_sq);

    if (lane == 0) s_warp_sums[warp] = val_sq;
    __syncthreads();

    if (tid == 0) s_norm_sq = s_warp_sums[0] + s_warp_sums[1];
    __syncthreads();

    const float norm = sqrtf(s_norm_sq);

    // ── Output block pointer ─────────────────────────────────────────────────
    block_tq3_mi300x *blk = dst + vec_idx;

    // ── Zero-vector fast path ────────────────────────────────────────────────
    if (norm < 1e-15f) {
        if (tid == 0) blk->norm = 0.0f;
        // Write zeros to all 3 bit-planes (48 bytes = 6 × uint64_t)
        if (tid < 6) {
            ((uint64_t *)blk->planes)[tid] = 0ULL;
        }
        return;
    }

    const float inv_norm = 1.0f / norm;

    // ── Rotation: y[tid] = Π[tid, :] · x_unit ───────────────────────────────
    // Each thread computes its own rotated coordinate.
    // s_input is in LDS (fast); rotation row comes from global memory / LLC.
    float y_val = 0.0f;
    {
        const float * __restrict__ row = rotation + tid * d;
        for (int j = 0; j < d; j++) {
            y_val += row[j] * s_input[j];
        }
        y_val *= inv_norm;
    }

    // ── Nearest centroid (register, no LDS write — Opt-B) ───────────────────
    const uint8_t my_idx = tqm_nearest_3(y_val);

    // ── Ballot-based bitplane packing (Opt-A) ────────────────────────────────
    // For each bit plane b (0=LSB, 2=MSB):
    //   Collect the b-th bit of each thread's index via __ballot().
    //   __ballot() reads from registers, returns a 64-bit mask with no LDS atomics.
    //   Warp leader (lane==0) stores the 8-byte mask to global memory (Opt-C).
    if (tid == 0) blk->norm = norm;

    for (int b = 0; b < 3; b++) {
        const int bit = (my_idx >> b) & 1;
        const unsigned long long mask = __ballot(bit);
        // lane==0 of each wavefront is the warp leader
        if (lane == 0) {
            // Wave 0 → plane[b*16..b*16+8), Wave 1 → plane[b*16+8..b*16+16)
            ((uint64_t *)(blk->planes + b * 16))[warp] = mask;
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TQ3 Dequantize Kernel — MI300X Optimized
// ──────────────────────────────────────────────────────────────────────────────

/**
 * tqm_dequantize_kernel_tq3
 *
 * Reads bit-plane format, extracts per-thread index, looks up centroid,
 * applies inverse rotation, scales by stored norm.
 *
 * Grid:  (n_vectors, 1, 1)
 * Block: (TQ_HEAD_DIM=128, 1, 1)
 *
 * LDS: s_y_hat[128] (centroids after index lookup) + s_norm = 516 bytes
 */
__global__ void tqm_dequantize_kernel_tq3(
    const block_tq3_mi300x * __restrict__ src,
    const float            * __restrict__ rotation,
    float                  * __restrict__ dst,
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int warp = tid >> 6;
    const int d    = TQ_HEAD_DIM;

    __shared__ float s_y_hat[TQ_HEAD_DIM];
    __shared__ float s_norm;

    const block_tq3_mi300x *blk = src + vec_idx;

    if (tid == 0) s_norm = blk->norm;
    __syncthreads();

    if (fabsf(s_norm) < 1e-15f) {
        dst[vec_idx * d + tid] = 0.0f;
        return;
    }

    // ── Extract index from bit-plane format ──────────────────────────────────
    // Thread tid's index is at bit `lane` of the warp's plane word.
    uint8_t my_idx = 0;
    for (int b = 0; b < 3; b++) {
        // Load the 64-bit plane word for this wavefront
        const uint64_t plane_word = ((const uint64_t *)(blk->planes + b * 16))[warp];
        const int bit_b = (plane_word >> lane) & 1ULL;
        my_idx |= (uint8_t)(bit_b << b);
    }

    // ── Centroid lookup ──────────────────────────────────────────────────────
    s_y_hat[tid] = d_cb3[my_idx];
    __syncthreads();

    // ── Inverse rotation: x_hat[tid] = Π^T[tid, :] · y_hat = Π[:, tid] · y_hat ──
    // Π is orthogonal: Π^{-1} = Π^T.
    // Π^T[tid, j] = Π[j, tid] = rotation[j * d + tid]
    float x_val = 0.0f;
    for (int j = 0; j < d; j++) {
        x_val += rotation[j * d + tid] * s_y_hat[j];
    }

    dst[vec_idx * d + tid] = x_val * s_norm;
}

// ──────────────────────────────────────────────────────────────────────────────
// TQ3 Fused Dot Product Kernel
// ──────────────────────────────────────────────────────────────────────────────

/**
 * tqm_fused_dot_kernel_tq3
 *
 * Computes scores[q, k] = q_rotated[q] · dequant(kv[k]) without full inverse rotation.
 * Key insight: if q is already in rotated space (q_rotated = Π · q_unit), then:
 *   q · k = (Π^T q_rotated) · (norm × Π^T centroid)
 *          = norm × q_rotated · (Π Π^T centroid)
 *          = norm × q_rotated · centroid   (since Π is orthogonal)
 * So: score = norm × dot(q_rotated, codebook[indices])
 *
 * Grid:  (n_kv, n_queries)
 * Block: (TQ_HEAD_DIM=128, 1, 1)
 */
__global__ void tqm_fused_dot_kernel_tq3(
    const float             * __restrict__ q_rotated,
    const block_tq3_mi300x  * __restrict__ kv_blocks,
    float                   * __restrict__ scores,
    int n_queries,
    int n_kv
) {
    const int kv_idx = blockIdx.x;
    const int q_idx  = blockIdx.y;
    if (kv_idx >= n_kv || q_idx >= n_queries) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int warp = tid >> 6;
    const int d    = TQ_HEAD_DIM;

    const block_tq3_mi300x *blk = kv_blocks + kv_idx;

    __shared__ float s_norm;
    __shared__ float s_warp_dots[2];

    if (tid == 0) s_norm = blk->norm;
    __syncthreads();

    if (fabsf(s_norm) < 1e-15f) {
        if (tid == 0) scores[q_idx * n_kv + kv_idx] = 0.0f;
        return;
    }

    // ── Extract index from bit-plane ─────────────────────────────────────────
    uint8_t my_idx = 0;
    for (int b = 0; b < 3; b++) {
        const uint64_t plane_word = ((const uint64_t *)(blk->planes + b * 16))[warp];
        my_idx |= (uint8_t)(((plane_word >> lane) & 1ULL) << b);
    }

    // ── Fused dot: q_rotated[tid] × codebook[my_idx] ────────────────────────
    const float centroid = d_cb3[my_idx];
    const float q_val    = q_rotated[q_idx * d + tid];
    float partial = q_val * centroid;

    // Warp-level reduction
    partial = warp_reduce_sum_w64(partial);

    // Cross-warp reduction (2 warps)
    if (lane == 0) s_warp_dots[warp] = partial;
    __syncthreads();

    if (tid == 0) {
        scores[q_idx * n_kv + kv_idx] = (s_warp_dots[0] + s_warp_dots[1]) * s_norm;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// TQ4 Kernels (4-bit, 16 centroids — nibble-pack format, bitplane variant)
// ──────────────────────────────────────────────────────────────────────────────

__global__ void tqm_quantize_kernel_tq4(
    const float        * __restrict__ src,
    const float        * __restrict__ rotation,
    block_tq4_mi300x   * __restrict__ dst,
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int warp = tid >> 6;
    const int d    = TQ_HEAD_DIM;

    __shared__ float s_input[TQ_HEAD_DIM];
    __shared__ float s_norm_sq;
    __shared__ float s_warp_sums[2];

    s_input[tid] = src[vec_idx * d + tid];
    __syncthreads();

    float val_sq = s_input[tid] * s_input[tid];
    val_sq = warp_reduce_sum_w64(val_sq);
    if (lane == 0) s_warp_sums[warp] = val_sq;
    __syncthreads();

    if (tid == 0) s_norm_sq = s_warp_sums[0] + s_warp_sums[1];
    __syncthreads();

    const float norm = sqrtf(s_norm_sq);
    block_tq4_mi300x *blk = dst + vec_idx;

    if (norm < 1e-15f) {
        if (tid == 0) blk->norm = 0.0f;
        if (tid < 8) ((uint64_t *)blk->planes)[tid] = 0ULL;  // 64 bytes = 8 × uint64
        return;
    }

    const float inv_norm = 1.0f / norm;

    float y_val = 0.0f;
    {
        const float * __restrict__ row = rotation + tid * d;
        for (int j = 0; j < d; j++) y_val += row[j] * s_input[j];
        y_val *= inv_norm;
    }

    const uint8_t my_idx = tqm_nearest_4(y_val);

    if (tid == 0) blk->norm = norm;

    // 4 bit-planes, ballot per plane
    for (int b = 0; b < 4; b++) {
        const int bit = (my_idx >> b) & 1;
        const unsigned long long mask = __ballot(bit);
        if (lane == 0) {
            ((uint64_t *)(blk->planes + b * 16))[warp] = mask;
        }
    }
}

__global__ void tqm_dequantize_kernel_tq4(
    const block_tq4_mi300x * __restrict__ src,
    const float            * __restrict__ rotation,
    float                  * __restrict__ dst,
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int warp = tid >> 6;
    const int d    = TQ_HEAD_DIM;

    __shared__ float s_y_hat[TQ_HEAD_DIM];
    __shared__ float s_norm;

    const block_tq4_mi300x *blk = src + vec_idx;

    if (tid == 0) s_norm = blk->norm;
    __syncthreads();

    if (fabsf(s_norm) < 1e-15f) {
        dst[vec_idx * d + tid] = 0.0f;
        return;
    }

    uint8_t my_idx = 0;
    for (int b = 0; b < 4; b++) {
        const uint64_t plane_word = ((const uint64_t *)(blk->planes + b * 16))[warp];
        my_idx |= (uint8_t)(((plane_word >> lane) & 1ULL) << b);
    }

    s_y_hat[tid] = d_cb4[my_idx];
    __syncthreads();

    float x_val = 0.0f;
    for (int j = 0; j < d; j++) x_val += rotation[j * d + tid] * s_y_hat[j];

    dst[vec_idx * d + tid] = x_val * s_norm;
}

// ──────────────────────────────────────────────────────────────────────────────
// QJL Keys Kernel (Algorithm 2 — not in domvox baseline)
// ──────────────────────────────────────────────────────────────────────────────

/**
 * tqm_qjl_kernel
 *
 * For each key vector, computes QJL correction data:
 *   1. Dequantize MSE block → k_mse (on-chip, from constant centroids)
 *   2. Compute residual in rotated space: r_rot = y_unit - centroid
 *   3. Compute residual norm (in original space, approximated)
 *   4. Project residual with QJL matrix S: p = S · (Π^T r_rot)
 *   5. Store sign(p[j]) for j=0..127, packed into 16 bytes via ballot
 *
 * Note: k_unit is provided pre-normalized; it is used to compute residual
 * accurately without re-normalizing. The rotation is applied to go back to
 * original space before projecting with S.
 *
 * Grid:  (n_vectors, 1, 1)
 * Block: (TQ_HEAD_DIM=128, 1, 1)
 */
__global__ void tqm_qjl_kernel(
    const float            * __restrict__ k_unit,       // [n_vectors × d] pre-normalized
    const block_tq3_mi300x * __restrict__ mse_blk,      // [n_vectors] MSE quantized blocks
    const float            * __restrict__ rotation,     // [d × d] same rotation as compress
    const float            * __restrict__ S,            // [d × d] QJL Gaussian matrix
    block_qjl_mi300x       * __restrict__ qjl_out,      // [n_vectors] output
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid  = threadIdx.x;
    const int lane = tid & 63;
    const int warp = tid >> 6;
    const int d    = TQ_HEAD_DIM;

    // LDS for cross-thread communication
    __shared__ float s_residual[TQ_HEAD_DIM];   // r in original space
    __shared__ float s_norm_sq;
    __shared__ float s_warp_sums[2];

    // ── Step 1: Extract centroid in rotated space ────────────────────────────
    const block_tq3_mi300x *blk = mse_blk + vec_idx;
    uint8_t my_idx = 0;
    for (int b = 0; b < 3; b++) {
        const uint64_t plane_word = ((const uint64_t *)(blk->planes + b * 16))[warp];
        my_idx |= (uint8_t)(((plane_word >> lane) & 1ULL) << b);
    }
    const float centroid_val = d_cb3[my_idx]; // y_hat[tid] in rotated space

    // ── Step 2: Inverse rotate centroid → k_mse in original (unit) space ────
    // k_mse_unit[tid] = (Π^T y_hat)[tid] = sum_j Π[j, tid] * y_hat[j]
    // Requires all y_hat values → use LDS (s_residual temporarily holds y_hat)
    s_residual[tid] = centroid_val;
    __syncthreads();

    float k_mse_unit_val = 0.0f;
    for (int j = 0; j < d; j++) {
        k_mse_unit_val += rotation[j * d + tid] * s_residual[j];
    }

    // ── Step 3: Compute residual r = k_unit - k_mse_unit ────────────────────
    const float k_unit_val = k_unit[vec_idx * d + tid];
    s_residual[tid] = k_unit_val - k_mse_unit_val;
    __syncthreads();

    // ── Step 4: Compute residual norm ||r||_2 ────────────────────────────────
    float r_sq = s_residual[tid] * s_residual[tid];
    r_sq = warp_reduce_sum_w64(r_sq);
    if (lane == 0) s_warp_sums[warp] = r_sq;
    __syncthreads();
    if (tid == 0) s_norm_sq = s_warp_sums[0] + s_warp_sums[1];
    __syncthreads();
    const float residual_norm = sqrtf(s_norm_sq);

    // ── Step 5: Project residual: p[tid] = S[tid, :] · r ───────────────────
    float p_val = 0.0f;
    {
        const float * __restrict__ S_row = S + tid * d;
        for (int j = 0; j < d; j++) {
            p_val += S_row[j] * s_residual[j];
        }
    }

    // ── Step 6: Pack sign bits via ballot ────────────────────────────────────
    const int sign_bit = (p_val > 0.0f) ? 1 : 0;
    const unsigned long long sign_mask = __ballot(sign_bit);

    block_qjl_mi300x *out = qjl_out + vec_idx;
    if (lane == 0) {
        ((uint64_t *)out->signs)[warp] = sign_mask;
    }
    if (tid == 0) {
        out->residual_norm = residual_norm;
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// Host Functions
// ──────────────────────────────────────────────────────────────────────────────

extern "C"
void tqm_init(void) {
    hipMemcpyToSymbol(HIP_SYMBOL(d_cb3), TQ3_CODEBOOK_MI300X, 8  * sizeof(float), 0, hipMemcpyHostToDevice);
    hipMemcpyToSymbol(HIP_SYMBOL(d_cb4), TQ4_CODEBOOK_MI300X, 16 * sizeof(float), 0, hipMemcpyHostToDevice);
    hipMemcpyToSymbol(HIP_SYMBOL(d_cb2), TQ2_CODEBOOK_MI300X, 4  * sizeof(float), 0, hipMemcpyHostToDevice);
}

extern "C"
void tqm_quantize_tq3(
    const float *d_src, const float *d_rotation,
    block_tq3_mi300x *d_dst, int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    dim3 grid(n_vectors), block(TQ_BLOCK_THREADS);
    tqm_quantize_kernel_tq3<<<grid, block, 0, stream>>>(d_src, d_rotation, d_dst, n_vectors);
}

extern "C"
void tqm_dequantize_tq3(
    const block_tq3_mi300x *d_src, const float *d_rotation,
    float *d_dst, int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    dim3 grid(n_vectors), block(TQ_BLOCK_THREADS);
    tqm_dequantize_kernel_tq3<<<grid, block, 0, stream>>>(d_src, d_rotation, d_dst, n_vectors);
}

extern "C"
void tqm_fused_dot_tq3(
    const float *d_q_rotated, const block_tq3_mi300x *d_kv_blocks,
    float *d_scores, int n_queries, int n_kv, hipStream_t stream
) {
    if (n_queries <= 0 || n_kv <= 0) return;
    dim3 grid(n_kv, n_queries), block(TQ_BLOCK_THREADS);
    tqm_fused_dot_kernel_tq3<<<grid, block, 0, stream>>>(
        d_q_rotated, d_kv_blocks, d_scores, n_queries, n_kv);
}

extern "C"
void tqm_quantize_tq4(
    const float *d_src, const float *d_rotation,
    block_tq4_mi300x *d_dst, int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    dim3 grid(n_vectors), block(TQ_BLOCK_THREADS);
    tqm_quantize_kernel_tq4<<<grid, block, 0, stream>>>(d_src, d_rotation, d_dst, n_vectors);
}

extern "C"
void tqm_dequantize_tq4(
    const block_tq4_mi300x *d_src, const float *d_rotation,
    float *d_dst, int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    dim3 grid(n_vectors), block(TQ_BLOCK_THREADS);
    tqm_dequantize_kernel_tq4<<<grid, block, 0, stream>>>(d_src, d_rotation, d_dst, n_vectors);
}

extern "C"
void tqm_qjl_keys(
    const float *d_k_unit, const block_tq3_mi300x *d_mse_blk,
    const float *d_rotation, const float *d_S,
    block_qjl_mi300x *d_qjl_out, int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    dim3 grid(n_vectors), block(TQ_BLOCK_THREADS);
    tqm_qjl_kernel<<<grid, block, 0, stream>>>(
        d_k_unit, d_mse_blk, d_rotation, d_S, d_qjl_out, n_vectors);
}

// ──────────────────────────────────────────────────────────────────────────────
// Rotation/QJL matrix generation (CPU side, uploads to device)
// ──────────────────────────────────────────────────────────────────────────────

/*
 * Simple seeded Gaussian matrix generation using a linear congruential generator.
 * For production use, replace with a proper orthogonalization (e.g., LAPACK dgeqrf).
 * The rotation here is NOT orthogonal — callers expecting orthogonal rotation
 * should use the Python wrapper which calls scipy.linalg.qr.
 *
 * This is intentionally minimal to avoid LAPACK dependencies in the kernel library.
 * The Python wrapper tqm.py generates proper rotation matrices.
 */

static float lcg_randn(uint64_t *state) {
    // Box-Muller using two LCG samples
    *state = 6364136223846793005ULL * (*state) + 1442695040888963407ULL;
    uint64_t s1 = *state;
    *state = 6364136223846793005ULL * (*state) + 1442695040888963407ULL;
    uint64_t s2 = *state;
    // Convert to [0,1] uniform
    double u1 = (double)(s1 >> 11) / (double)(1ULL << 53);
    double u2 = (double)(s2 >> 11) / (double)(1ULL << 53);
    if (u1 < 1e-30) u1 = 1e-30;
    double r = sqrt(-2.0 * log(u1));
    double theta = 2.0 * 3.14159265358979323846 * u2;
    return (float)(r * cos(theta));
}

static void gram_schmidt_inplace(float *mat, int n) {
    // Simple Gram-Schmidt orthogonalization on rows of n×n matrix
    for (int i = 0; i < n; i++) {
        float *row_i = mat + i * n;
        // Subtract projections onto previous rows
        for (int j = 0; j < i; j++) {
            const float *row_j = mat + j * n;
            float dot = 0.0f;
            for (int k = 0; k < n; k++) dot += row_i[k] * row_j[k];
            for (int k = 0; k < n; k++) row_i[k] -= dot * row_j[k];
        }
        // Normalize
        float norm_sq = 0.0f;
        for (int k = 0; k < n; k++) norm_sq += row_i[k] * row_i[k];
        float inv = 1.0f / sqrtf(norm_sq);
        for (int k = 0; k < n; k++) row_i[k] *= inv;
    }
}

extern "C"
void tqm_alloc_rotation(uint64_t seed, float **d_rot_out) {
    const int n = TQ_HEAD_DIM;
    float *h_mat = (float *)malloc(n * n * sizeof(float));
    uint64_t state = seed ^ 0xDEADBEEFCAFEBABEULL;
    for (int i = 0; i < n * n; i++) h_mat[i] = lcg_randn(&state);
    gram_schmidt_inplace(h_mat, n);

    float *d_mat = nullptr;
    hipMalloc(&d_mat, n * n * sizeof(float));
    hipMemcpy(d_mat, h_mat, n * n * sizeof(float), hipMemcpyHostToDevice);
    free(h_mat);
    *d_rot_out = d_mat;
}

extern "C"
void tqm_alloc_qjl_matrix(uint64_t seed, float **d_S_out) {
    const int n = TQ_HEAD_DIM;
    float *h_mat = (float *)malloc(n * n * sizeof(float));
    // QJL projection does NOT need to be orthogonal — just random Gaussian
    uint64_t state = seed ^ 0xBEEFDEADCAFEF00DULL;
    float inv_sqrt_d = 1.0f / sqrtf((float)n);
    for (int i = 0; i < n * n; i++) h_mat[i] = lcg_randn(&state) * inv_sqrt_d;

    float *d_mat = nullptr;
    hipMalloc(&d_mat, n * n * sizeof(float));
    hipMemcpy(d_mat, h_mat, n * n * sizeof(float), hipMemcpyHostToDevice);
    free(h_mat);
    *d_S_out = d_mat;
}
