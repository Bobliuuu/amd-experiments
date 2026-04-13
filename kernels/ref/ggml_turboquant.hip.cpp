/*
 * TurboQuant: HIP Kernel Implementation for AMD RDNA3 (gfx1100)
 * Ported from CUDA reference by domvox, 2026-04-06
 *
 * Changes from CUDA:
 *   - hip/hip_runtime.h instead of cuda_runtime.h
 *   - hipMemcpyToSymbol / hipStream_t
 *   - __shfl_down without sync mask (HIP style)
 *   - Single-thread bit-packing (avoids atomicOr alignment issues on RDNA3)
 *   - warpSize-aware reduction (RDNA3 wavefront can be 32 or 64)
 */

#include "ggml_turboquant.h"
#include <hip/hip_runtime.h>

/* Codebooks in constant memory */
__constant__ float d_codebook_3[8];
__constant__ float d_codebook_4[16];

/* Find nearest centroid (linear scan, n_levels <= 16) */
__device__ __forceinline__
uint8_t tq_find_nearest(float val, const float *codebook, int n_levels) {
    float best_dist = 1e30f;
    uint8_t best_idx = 0;
    for (int c = 0; c < n_levels; c++) {
        float d = (val - codebook[c]);
        d = d * d;
        if (d < best_dist) {
            best_dist = d;
            best_idx = (uint8_t)c;
        }
    }
    return best_idx;
}

/* Warp reduction — works for both wavefront 32 and 64 */
__device__ __forceinline__
float warp_reduce_sum(float val) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        val += __shfl_down(val, offset);
    }
    return val;
}

/* =========================================================================
 * Quantize Kernel — one block (128 threads) per vector
 * ========================================================================= */
__global__ void tq_quantize_kernel_tq3(
    const float * __restrict__ src,
    void        * __restrict__ dst,
    const float * __restrict__ rotation,
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid = threadIdx.x;
    const int d = TQ_HEAD_DIM;

    __shared__ float s_input[TQ_HEAD_DIM];
    __shared__ float s_norm_sq;
    __shared__ uint8_t s_indices[TQ_HEAD_DIM];

    /* Load input */
    s_input[tid] = src[vec_idx * d + tid];
    __syncthreads();

    /* L2 norm via warp reduction */
    float val_sq = s_input[tid] * s_input[tid];
    val_sq = warp_reduce_sum(val_sq);

    /* Cross-warp reduction */
    const int n_warps = (TQ_HEAD_DIM + warpSize - 1) / warpSize;
    __shared__ float s_warp_sums[8]; /* max 8 warps (128/16 if wavefront 16, unlikely) */
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) {
        s_warp_sums[warp_id] = val_sq;
    }
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += s_warp_sums[w];
        s_norm_sq = total;
    }
    __syncthreads();

    float norm = sqrtf(s_norm_sq);

    /* Zero vector */
    if (norm < 1e-15f) {
        if (tid == 0) {
            block_tq3 *blk = (block_tq3 *)((uint8_t *)dst + vec_idx * sizeof(block_tq3));
            blk->norm = 0.0f;
            for (int i = 0; i < TQ3_INDEX_BYTES; i++) blk->indices[i] = 0;
        }
        return;
    }

    float inv_norm = 1.0f / norm;

    /* Rotate: y[tid] = row tid of Π · x_unit */
    float y_val = 0.0f;
    const float *my_row = rotation + tid * d;
    for (int j = 0; j < d; j++) {
        y_val += my_row[j] * s_input[j] * inv_norm;
    }

    /* Nearest centroid */
    s_indices[tid] = tq_find_nearest(y_val, d_codebook_3, 8);
    __syncthreads();

    /* Parallel bit-packing: each thread packs its own 3 bits */
    __shared__ uint8_t s_packed[TQ3_INDEX_BYTES];
    if (tid < TQ3_INDEX_BYTES) s_packed[tid] = 0;
    __syncthreads();

    {
        uint8_t val = s_indices[tid];
        int bit_start = tid * 3;
        for (int b = 0; b < 3; b++) {
            if (val & (1 << b)) {
                int bit_pos = bit_start + b;
                int byte_idx = bit_pos / 8;
                int bit_in_byte = bit_pos % 8;
                atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)), 1u << (bit_in_byte + 8 * (byte_idx & 3)));
            }
        }
    }
    __syncthreads();

    /* Write output */
    if (tid == 0) {
        block_tq3 *blk = (block_tq3 *)((uint8_t *)dst + vec_idx * sizeof(block_tq3));
        blk->norm = norm;
        for (int i = 0; i < TQ3_INDEX_BYTES; i++) blk->indices[i] = s_packed[i];
    }
}

/* =========================================================================
 * Dequantize Kernel
 * ========================================================================= */
__global__ void tq_dequantize_kernel_tq3(
    const void  * __restrict__ src,
    float       * __restrict__ dst,
    const float * __restrict__ rotation,
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;

    const int tid = threadIdx.x;
    const int d = TQ_HEAD_DIM;

    __shared__ float s_y_hat[TQ_HEAD_DIM];
    __shared__ uint8_t s_packed[TQ3_INDEX_BYTES];
    __shared__ float s_norm;

    const block_tq3 *blk = (const block_tq3 *)((const uint8_t *)src + vec_idx * sizeof(block_tq3));

    if (tid == 0) {
        s_norm = blk->norm;
        for (int i = 0; i < TQ3_INDEX_BYTES; i++) s_packed[i] = blk->indices[i];
    }
    __syncthreads();

    if (fabsf(s_norm) < 1e-15f) {
        dst[vec_idx * d + tid] = 0.0f;
        return;
    }

    /* Unpack 3-bit index */
    uint8_t my_idx = 0;
    {
        int bit_start = tid * 3;
        for (int b = 0; b < 3; b++) {
            int bit_pos = bit_start + b;
            if (s_packed[bit_pos / 8] & (1 << (bit_pos % 8))) {
                my_idx |= (1 << b);
            }
        }
    }

    /* Codebook lookup */
    s_y_hat[tid] = d_codebook_3[my_idx];
    __syncthreads();

    /* Inverse rotation: x_hat[tid] = sum_j Π[j][tid] * y_hat[j] */
    float x_val = 0.0f;
    for (int j = 0; j < d; j++) {
        x_val += rotation[j * d + tid] * s_y_hat[j];
    }

    dst[vec_idx * d + tid] = x_val * s_norm;
}

/* =========================================================================
 * Fused dot product: Q_rotated · dequant(KV) without materializing
 * ========================================================================= */
__global__ void tq_fused_dot_tq3(
    const float * __restrict__ q_rotated,
    const void  * __restrict__ kv_blocks,
    float       * __restrict__ scores,
    int n_queries,
    int n_kv
) {
    const int kv_idx = blockIdx.x;
    const int q_idx  = blockIdx.y;
    if (kv_idx >= n_kv || q_idx >= n_queries) return;

    const int tid = threadIdx.x;
    const int d = TQ_HEAD_DIM;

    const block_tq3 *blk = (const block_tq3 *)((const uint8_t *)kv_blocks + kv_idx * sizeof(block_tq3));

    __shared__ float s_norm;
    __shared__ uint8_t s_packed[TQ3_INDEX_BYTES];

    if (tid == 0) {
        s_norm = blk->norm;
        for (int i = 0; i < TQ3_INDEX_BYTES; i++) s_packed[i] = blk->indices[i];
    }
    __syncthreads();

    /* Unpack */
    uint8_t my_idx = 0;
    {
        int bit_start = tid * 3;
        for (int b = 0; b < 3; b++) {
            int bit_pos = bit_start + b;
            if (s_packed[bit_pos / 8] & (1 << (bit_pos % 8))) {
                my_idx |= (1 << b);
            }
        }
    }

    float y_hat_val = d_codebook_3[my_idx];
    float q_val = q_rotated[q_idx * d + tid];
    float partial = q_val * y_hat_val;

    /* Warp reduction */
    partial = warp_reduce_sum(partial);

    /* Cross-warp reduction */
    const int n_warps = (TQ_HEAD_DIM + warpSize - 1) / warpSize;
    __shared__ float s_warp_dots[8];
    int warp_id = tid / warpSize;
    int lane_id = tid % warpSize;
    if (lane_id == 0) {
        s_warp_dots[warp_id] = partial;
    }
    __syncthreads();

    if (tid == 0) {
        float dot = 0.0f;
        for (int w = 0; w < n_warps; w++) dot += s_warp_dots[w];
        scores[q_idx * n_kv + kv_idx] = dot * s_norm;
    }
}

/* =========================================================================
 * Host wrappers
 * ========================================================================= */

extern "C"
void tq_hip_init_codebooks(void) {
    hipMemcpyToSymbol(HIP_SYMBOL(d_codebook_3), TQ_CODEBOOK_3, 8 * sizeof(float), 0, hipMemcpyHostToDevice);
    hipMemcpyToSymbol(HIP_SYMBOL(d_codebook_4), TQ_CODEBOOK_4, 16 * sizeof(float), 0, hipMemcpyHostToDevice);
}

extern "C"
void tq_hip_quantize_tq3(
    const float *d_src, void *d_dst, const float *d_rotation,
    int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    tq_quantize_kernel_tq3<<<n_vectors, TQ_HEAD_DIM, 0, stream>>>(
        d_src, d_dst, d_rotation, n_vectors);
}

extern "C"
void tq_hip_dequantize_tq3(
    const void *d_src, float *d_dst, const float *d_rotation,
    int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    tq_dequantize_kernel_tq3<<<n_vectors, TQ_HEAD_DIM, 0, stream>>>(
        d_src, d_dst, d_rotation, n_vectors);
}

extern "C"
void tq_hip_fused_dot_tq3(
    const float *d_q_rotated, const void *d_kv_blocks, float *d_scores,
    int n_queries, int n_kv, hipStream_t stream
) {
    dim3 grid(n_kv, n_queries);
    dim3 block(TQ_HEAD_DIM);
    tq_fused_dot_tq3<<<grid, block, 0, stream>>>(
        d_q_rotated, d_kv_blocks, d_scores, n_queries, n_kv);
}

/* =========================================================================
 * TQ4 Kernels (4-bit, 16 centroids, 68 bytes per vector, 3.8x compression)
 * ========================================================================= */

__global__ void tq_quantize_kernel_tq4(
    const float * __restrict__ src,
    void        * __restrict__ dst,
    const float * __restrict__ rotation,
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;
    const int tid = threadIdx.x;
    const int d = TQ_HEAD_DIM;

    __shared__ float s_input[TQ_HEAD_DIM];
    __shared__ float s_norm_sq;
    __shared__ uint8_t s_indices[TQ_HEAD_DIM];

    s_input[tid] = src[vec_idx * d + tid];
    __syncthreads();

    float val_sq = s_input[tid] * s_input[tid];
    val_sq = warp_reduce_sum(val_sq);

    const int n_warps = (TQ_HEAD_DIM + warpSize - 1) / warpSize;
    __shared__ float s_warp_sums[8];
    if (tid % warpSize == 0) s_warp_sums[tid / warpSize] = val_sq;
    __syncthreads();

    if (tid == 0) {
        float total = 0.0f;
        for (int w = 0; w < n_warps; w++) total += s_warp_sums[w];
        s_norm_sq = total;
    }
    __syncthreads();

    float norm = sqrtf(s_norm_sq);
    if (norm < 1e-15f) {
        if (tid == 0) {
            block_tq4 *blk = (block_tq4 *)((uint8_t *)dst + vec_idx * sizeof(block_tq4));
            blk->norm = 0.0f;
            for (int i = 0; i < TQ4_INDEX_BYTES; i++) blk->indices[i] = 0;
        }
        return;
    }

    float inv_norm = 1.0f / norm;
    float y_val = 0.0f;
    const float *my_row = rotation + tid * d;
    for (int j = 0; j < d; j++) y_val += my_row[j] * s_input[j] * inv_norm;

    s_indices[tid] = tq_find_nearest(y_val, d_codebook_4, 16);
    __syncthreads();

    /* TQ4: 4 bits per value = simple nibble packing, 2 values per byte */
    if (tid < TQ_HEAD_DIM / 2) {
        block_tq4 *blk = (block_tq4 *)((uint8_t *)dst + vec_idx * sizeof(block_tq4));
        if (tid == 0) blk->norm = norm;
        blk->indices[tid] = (s_indices[tid * 2] & 0x0F) | (s_indices[tid * 2 + 1] << 4);
    }
}

__global__ void tq_dequantize_kernel_tq4(
    const void  * __restrict__ src,
    float       * __restrict__ dst,
    const float * __restrict__ rotation,
    int n_vectors
) {
    const int vec_idx = blockIdx.x;
    if (vec_idx >= n_vectors) return;
    const int tid = threadIdx.x;
    const int d = TQ_HEAD_DIM;

    __shared__ float s_y_hat[TQ_HEAD_DIM];
    __shared__ float s_norm;

    const block_tq4 *blk = (const block_tq4 *)((const uint8_t *)src + vec_idx * sizeof(block_tq4));

    if (tid == 0) s_norm = blk->norm;
    __syncthreads();

    if (fabsf(s_norm) < 1e-15f) {
        dst[vec_idx * d + tid] = 0.0f;
        return;
    }

    /* Unpack 4-bit nibble */
    uint8_t packed_byte = blk->indices[tid / 2];
    uint8_t my_idx = (tid % 2 == 0) ? (packed_byte & 0x0F) : (packed_byte >> 4);

    s_y_hat[tid] = d_codebook_4[my_idx];
    __syncthreads();

    float x_val = 0.0f;
    for (int j = 0; j < d; j++) x_val += rotation[j * d + tid] * s_y_hat[j];

    dst[vec_idx * d + tid] = x_val * s_norm;
}

extern "C"
void tq_hip_quantize_tq4(
    const float *d_src, void *d_dst, const float *d_rotation,
    int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    tq_quantize_kernel_tq4<<<n_vectors, TQ_HEAD_DIM, 0, stream>>>(
        d_src, d_dst, d_rotation, n_vectors);
}

extern "C"
void tq_hip_dequantize_tq4(
    const void *d_src, float *d_dst, const float *d_rotation,
    int n_vectors, hipStream_t stream
) {
    if (n_vectors <= 0) return;
    tq_dequantize_kernel_tq4<<<n_vectors, TQ_HEAD_DIM, 0, stream>>>(
        d_src, d_dst, d_rotation, n_vectors);
}
