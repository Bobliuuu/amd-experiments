/**
 * tq_mfma_rotate.hip.cpp
 * MFMA-accelerated rotation matrix kernel for TurboQuant on AMD MI300X (gfx942)
 *
 * Replaces the per-thread scalar dot-product loop in tqm_quantize/dequantize
 * with a proper tiled GEMM using the native v_mfma_f32_16x16x16f16 instruction.
 *
 * Ops:
 *   tq_rotate_forward : Y = X @ R^T   (used in compress:   y = x_unit @ R.T)
 *   tq_rotate_inverse : Y = X @ R     (used in decompress: x = y_hat  @ R  )
 *
 * Matrix dimensions:
 *   X : (n, 128)  float32  — batch of head vectors
 *   R : (128, 128) float32 — fixed orthogonal rotation matrix
 *   Y : (n, 128)  float32  — rotated output
 *
 * Algorithm — tiled 16×16×16 MFMA GEMM:
 *   Grid : (ceil(n/TILE), 1)    where TILE=16
 *   Block: (64, 1, 1)           one Wave64 wavefront per block
 *
 *   Each wavefront computes one 16-row output tile across all 128 columns:
 *     For n_tile in 0..7   (128/16 N-tiles)
 *       For k_tile in 0..7 (128/16 K-tiles, contraction dimension)
 *         MFMA call: C[16×16] += A_tile[16×16] × B_tile[16×16]
 *     Total: 64 MFMA calls per wavefront, 64 threads per wavefront.
 *
 * MFMA register layout (v_mfma_f32_16x16x16f16, wave64):
 *   Thread t (0..63) holds:
 *     A_reg[4 fp16]: A[t%16][ (t/16)*4 : (t/16)*4+4 ]
 *     B_reg[4 fp16]: B[ (t/16)*4 : (t/16)*4+4 ][ t%16 ]
 *     D_reg[4 fp32]: D[ (t/16)*4 : (t/16)*4+4 ][ t%16 ]
 *
 * For tq_rotate_forward (Y = X @ R.T):
 *   A_tile = X[block_m:block_m+16, k_tile*16:(k_tile+1)*16]
 *   B_tile = R.T[k_tile*16:(k_tile+1)*16, n_tile*16:(n_tile+1)*16]
 *          = R[n_tile*16:(n_tile+1)*16, k_tile*16:(k_tile+1)*16]  (accessed transposed)
 *
 * For tq_rotate_inverse (Y = X @ R):
 *   A_tile = same
 *   B_tile = R[k_tile*16:(k_tile+1)*16, n_tile*16:(n_tile+1)*16]  (accessed directly)
 *
 * Precision: inputs are fp32, converted to fp16 for MFMA, accumulated in fp32.
 *   FP16 precision is sufficient — rotation rounding error is negligible vs
 *   the subsequent 3-bit quantization noise.
 *
 * Target: gfx942 (CDNA3), -mcode-object-version=5 for COV5 / hipModuleLoad compatibility.
 */

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

#define TQ_HEAD_DIM 128
#define TILE        16    /* 16×16×16 MFMA tile size */
#define N_TILES     8     /* 128 / 16 = 8 tiles along each dimension */

/* ──────────────────────────────────────────────────────────────────────────
 * Helper: load 4 fp16 values from a float32 array into an ext_vector_type(4)
 * ────────────────────────────────────────────────────────────────────────── */
typedef __fp16   v4hf __attribute__((ext_vector_type(4)));
typedef float    v4sf __attribute__((ext_vector_type(4)));

__device__ __forceinline__
v4hf load4_f32_to_f16(const float * __restrict__ ptr, int base) {
    return (v4hf){(__fp16)ptr[base+0], (__fp16)ptr[base+1],
                  (__fp16)ptr[base+2], (__fp16)ptr[base+3]};
}

__device__ __forceinline__
v4hf load4_f32_to_f16_checked(const float * __restrict__ ptr, int base,
                               bool valid) {
    if (!valid) return (v4hf){0.0f, 0.0f, 0.0f, 0.0f};
    return load4_f32_to_f16(ptr, base);
}

/* ──────────────────────────────────────────────────────────────────────────
 * Core GEMM tile function (shared between forward and inverse)
 *
 * Computes Y[block_m:block_m+16, :] = X[block_m:block_m+16, :] @ B_src
 * where B_src is either R or R.T depending on `transpose_b`.
 *
 * transpose_b = true  → B_tile = R[n_col_start:n_col_start+16, k_start:k_start+16].T
 *                                (i.e. B[k][n] = R[n_col_start+n][k_start+k])
 * transpose_b = false → B_tile = R[k_start:k_start+16, n_col_start:n_col_start+16]
 *                                (i.e. B[k][n] = R[k_start+k][n_col_start+n])
 * ────────────────────────────────────────────────────────────────────────── */
template <bool TRANSPOSE_B>
__device__ __forceinline__
void tq_rotate_core(
    const float * __restrict__ X,   /* (n, 128) row-major */
    const float * __restrict__ R,   /* (128, 128) row-major */
    float       * __restrict__ Y,   /* (n, 128) row-major */
    int n
) {
    const int block_m = blockIdx.x * TILE;
    if (block_m >= n) return;

    const int t = threadIdx.x;   /* 0..63: one wavefront */

    /* Thread t's position in the A (input) and B (rotation) tiles:
     *   A[row][col_grp*4 : col_grp*4+4]  where row=t%16, col_grp=t/16
     *   B[row_grp*4 : ...][col]           where col=t%16, row_grp=t/16
     *   D[row_grp*4 : ...][col]           same layout as B
     */
    const int a_row   = t % TILE;      /* row in A = which input vector */
    const int a_kgrp  = t / TILE;      /* K-group in A (0..3, each covers 4 K) */
    const int m_abs   = block_m + a_row;

    const int b_col   = t % TILE;      /* col in B = output feature within tile */
    const int b_kgrp  = t / TILE;      /* K-group in B = same as a_kgrp */

    const int d_col   = b_col;
    const int d_rgrp  = b_kgrp;        /* output row group (0..3) */

    /* 8 N-tiles: cover all 128 output columns */
    for (int nt = 0; nt < N_TILES; nt++) {
        v4sf C = {0.0f, 0.0f, 0.0f, 0.0f};   /* 4-element fp32 accumulator */

        const int n_abs = nt * TILE + b_col;   /* global output column for this thread */

        /* 8 K-tiles: accumulate the inner product across the 128 K dimension */
        for (int kt = 0; kt < N_TILES; kt++) {
            /* Load A: X[m_abs, kt*16 + a_kgrp*4 : kt*16 + a_kgrp*4+4] */
            v4hf A_reg = load4_f32_to_f16_checked(
                X, m_abs * TQ_HEAD_DIM + kt * TILE + a_kgrp * 4,
                m_abs < n
            );

            /* Load B depending on transpose flag:
             *   TRANSPOSE_B=true  → B = R.T, so B[k][n] = R[n][k]
             *                        thread holds B[(b_kgrp*4:b_kgrp*4+4)][b_col]
             *                        = R[nt*16 + b_col][(kt*16 + b_kgrp*4):...]
             *   TRANSPOSE_B=false → B = R,   so B[k][n] = R[k][n]
             *                        = R[(kt*16 + b_kgrp*4):...][nt*16 + b_col]
             */
            v4hf B_reg;
            if constexpr (TRANSPOSE_B) {
                /* B[k][n] = R[n_abs][k_start + j] */
                B_reg = load4_f32_to_f16(
                    R, n_abs * TQ_HEAD_DIM + kt * TILE + b_kgrp * 4
                );
            } else {
                /* B[k][n] = R[k_start + j][n_abs]
                 * Need 4 consecutive elements in column n_abs, at rows
                 * kt*16 + b_kgrp*4 .. kt*16 + b_kgrp*4+3
                 * R is row-major, so these are NOT consecutive in memory.
                 * Load individually.
                 */
                const int k_base = kt * TILE + b_kgrp * 4;
                B_reg = (v4hf){
                    (__fp16)R[(k_base + 0) * TQ_HEAD_DIM + n_abs],
                    (__fp16)R[(k_base + 1) * TQ_HEAD_DIM + n_abs],
                    (__fp16)R[(k_base + 2) * TQ_HEAD_DIM + n_abs],
                    (__fp16)R[(k_base + 3) * TQ_HEAD_DIM + n_abs],
                };
            }

            /* MFMA: C += A_tile × B_tile (16×16×16, fp16 in, fp32 out) */
            C = __builtin_amdgcn_mfma_f32_16x16x16f16(A_reg, B_reg, C, 0, 0, 0);
        }

        /* Write output: thread t stores 4 consecutive rows at fixed column.
         * D[d_rgrp*4 + j][d_col_within_tile] → Y[block_m + d_rgrp*4 + j][nt*16 + d_col]
         */
        const int out_n = nt * TILE + d_col;
        for (int j = 0; j < 4; j++) {
            const int out_m = block_m + d_rgrp * 4 + j;
            if (out_m < n) {
                Y[out_m * TQ_HEAD_DIM + out_n] = C[j];
            }
        }
    }
}

/* ──────────────────────────────────────────────────────────────────────────
 * Exported kernels (C linkage for hipModuleGetFunction)
 * ────────────────────────────────────────────────────────────────────────── */

/**
 * tq_rotate_forward — Y = X @ R^T
 *
 * Grid:  (ceil(n / TILE), 1, 1)
 * Block: (64, 1, 1)  — one Wave64 wavefront
 *
 * Args:
 *   X   : (n, 128) float32
 *   R   : (128, 128) float32 (row-major)
 *   Y   : (n, 128) float32 output
 *   n   : int — number of vectors
 */
extern "C"
__global__ void tq_rotate_forward(
    const float * __restrict__ X,
    const float * __restrict__ R,
    float       * __restrict__ Y,
    int n
) {
    tq_rotate_core<true>(X, R, Y, n);
}

/**
 * tq_rotate_inverse — Y = X @ R
 *
 * Same grid/block as tq_rotate_forward.
 * Used during decompress: x_hat_unit = y_hat @ R
 */
extern "C"
__global__ void tq_rotate_inverse(
    const float * __restrict__ X,
    const float * __restrict__ R,
    float       * __restrict__ Y,
    int n
) {
    tq_rotate_core<false>(X, R, Y, n);
}
