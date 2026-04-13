/**
 * turboquant_mi300x.h — TurboQuant KV Cache Compression for AMD Instinct MI300X
 *
 * Target: gfx942 (CDNA3), ROCm 7.2, Wave64
 *
 * Implements TQ2/TQ3/TQ4 PolarQuant compression + QJL keys (Algorithm 2).
 *
 * Bit layout: BITPLANE format (differs from domvox sequential layout)
 *   For b-bit quantization, the b bit-planes are stored contiguously.
 *   Plane p (p=0 is LSB, p=b-1 is MSB) contains bit p of index[tid] for
 *   all 128 threads packed into 128/8=16 bytes. This enables zero-atomic
 *   packing via wavefront ballot.
 *
 * TQ3 block layout (52 bytes per 128-dim vector):
 *   [0..3]   float norm (L2 norm of original vector)
 *   [4..19]  bit-plane 0 (LSB of all 128 indices, 16 bytes)
 *   [20..35] bit-plane 1
 *   [36..51] bit-plane 2 (MSB)
 *
 * Compression ratio: 256 bytes (FP16) → 52 bytes = 4.92×
 *
 * Codebooks: Lloyd-Max optimal for Beta(d=128) distribution.
 * Source: turboquant_codebooks.json (verified against turboquant.py)
 */

#pragma once

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stddef.h>

// ──────────────────────────────────────────────────────────────────────────────
// Configuration
// ──────────────────────────────────────────────────────────────────────────────

#define TQ_HEAD_DIM          128
#define TQ_BLOCK_THREADS     128   // 1 thread per dimension, one wavefront pair on Wave64

// Bytes per compressed block (HEAD_DIM=128)
#define TQ3_INDEX_BYTES      48    // 3 bit-planes × 16 bytes = 48 bytes
#define TQ4_INDEX_BYTES      64    // 4 bit-planes × 16 bytes = 64 bytes
#define TQ2_INDEX_BYTES      32    // 2 bit-planes × 16 bytes = 32 bytes

// QJL: 1 sign bit per dimension = 128/8 = 16 bytes per key vector
#define TQK_QJL_SIGN_BYTES   16

// ──────────────────────────────────────────────────────────────────────────────
// Compressed block types
// ──────────────────────────────────────────────────────────────────────────────

// TQ3: 3-bit per element, 128 elements → 4 + 48 = 52 bytes
typedef struct __attribute__((packed)) {
    float   norm;                     // L2 norm of original unquantized vector
    uint8_t planes[TQ3_INDEX_BYTES];  // 3 bitplanes, 16 bytes each
} block_tq3_mi300x;

// TQ4: 4-bit per element → 4 + 64 = 68 bytes
typedef struct __attribute__((packed)) {
    float   norm;
    uint8_t planes[TQ4_INDEX_BYTES];
} block_tq4_mi300x;

// TQ2: 2-bit per element → 4 + 32 = 36 bytes
typedef struct __attribute__((packed)) {
    float   norm;
    uint8_t planes[TQ2_INDEX_BYTES];
} block_tq2_mi300x;

// QJL key supplement: packed sign bits + residual norm
typedef struct __attribute__((packed)) {
    uint8_t signs[TQK_QJL_SIGN_BYTES]; // sign bit for each of 128 dimensions
    float   residual_norm;              // ||r||_2 where r = k_unit - k_mse_reconstructed
} block_qjl_mi300x;

// ──────────────────────────────────────────────────────────────────────────────
// Lloyd-Max codebooks (precomputed for Beta distribution, d=128)
// These centroids are for coordinates of uniformly random unit vectors in R^128.
// The coordinate distribution is approximately N(0, 1/128) = N(0, 0.0078).
// Source: turboquant_codebooks.json (computed by turboquant.py LloydMax solver)
// ──────────────────────────────────────────────────────────────────────────────

// TQ2: 4 centroids, expected MSE = 0.116 (for unit vectors in R^128)
static const float TQ2_CODEBOOK_MI300X[4] = {
    -0.13311451677280386f,
    -0.04002746648341520f,
     0.04002746648341517f,
     0.13311451677280380f,
};

// TQ3: 8 centroids, expected MSE = 0.0340 (matches paper: 0.0337 ± 5%)
static const float TQ3_CODEBOOK_MI300X[8] = {
    -0.18904037194348838f,
    -0.11879501670185091f,
    -0.06702922184405663f,
    -0.02174971334976657f,
     0.02174971334976654f,
     0.06702922184405660f,
     0.11879501670185087f,
     0.18904037194348833f,
};

// TQ4: 16 centroids, expected MSE = 0.00933
static const float TQ4_CODEBOOK_MI300X[16] = {
    -0.23961253307138700f,
    -0.18317108415643454f,
    -0.14430970076906538f,
    -0.11276586366299288f,
    -0.08507481024405737f,
    -0.05962130616889217f,
    -0.03539017687270855f,
    -0.01173284981923122f,
     0.01173284981923120f,
     0.03539017687270851f,
     0.05962130616889214f,
     0.08507481024405730f,
     0.11276586366299284f,
     0.14430970076906535f,
     0.18317108415643450f,
     0.23961253307138697f,
};

// ──────────────────────────────────────────────────────────────────────────────
// Public C API
// ──────────────────────────────────────────────────────────────────────────────

#ifdef __cplusplus
extern "C" {
#endif

/**
 * tqm_init — Upload codebooks to GPU constant memory. Call once before kernels.
 */
void tqm_init(void);

/**
 * tqm_quantize_tq3 — Compress n_vectors FP32 head vectors to TQ3 bitplane format.
 *
 * @param d_src      [n_vectors × 128] float32, device memory
 * @param d_rotation [128 × 128] float32 orthogonal rotation matrix, device memory
 * @param d_dst      [n_vectors] block_tq3_mi300x, device memory (pre-allocated)
 * @param n_vectors  number of vectors to compress
 * @param stream     HIP stream (or 0 for default stream)
 */
void tqm_quantize_tq3(
    const float         *d_src,
    const float         *d_rotation,
    block_tq3_mi300x    *d_dst,
    int                  n_vectors,
    hipStream_t          stream
);

/**
 * tqm_dequantize_tq3 — Decompress n_vectors TQ3 blocks back to FP32.
 *
 * @param d_src      [n_vectors] block_tq3_mi300x, device memory
 * @param d_rotation [128 × 128] float32 rotation matrix (same as compress), device memory
 * @param d_dst      [n_vectors × 128] float32 output, device memory
 * @param n_vectors  number of vectors to decompress
 * @param stream     HIP stream
 */
void tqm_dequantize_tq3(
    const block_tq3_mi300x *d_src,
    const float             *d_rotation,
    float                   *d_dst,
    int                      n_vectors,
    hipStream_t              stream
);

/**
 * tqm_fused_dot_tq3 — Compute dot products between queries and compressed KV blocks.
 * Avoids materializing full FP16 KV; decodes on-chip.
 *
 * scores[q, k] = dot(query[q], dequantize(kv[k]))  for all (q, k)
 *
 * @param d_q_rotated [n_queries × 128] float32 queries already rotated by Π
 * @param d_kv_blocks [n_kv] block_tq3_mi300x compressed KV blocks
 * @param d_scores    [n_queries × n_kv] float32 output scores
 * @param n_queries   number of query vectors
 * @param n_kv        number of KV blocks
 * @param stream      HIP stream
 */
void tqm_fused_dot_tq3(
    const float             *d_q_rotated,
    const block_tq3_mi300x  *d_kv_blocks,
    float                   *d_scores,
    int                      n_queries,
    int                      n_kv,
    hipStream_t              stream
);

/**
 * tqm_quantize_tq4 — 4-bit variant.
 */
void tqm_quantize_tq4(
    const float         *d_src,
    const float         *d_rotation,
    block_tq4_mi300x    *d_dst,
    int                  n_vectors,
    hipStream_t          stream
);

/**
 * tqm_dequantize_tq4 — 4-bit dequantize.
 */
void tqm_dequantize_tq4(
    const block_tq4_mi300x *d_src,
    const float             *d_rotation,
    float                   *d_dst,
    int                      n_vectors,
    hipStream_t              stream
);

/**
 * tqm_qjl_keys — Compute QJL correction data for key vectors (Algorithm 2).
 * Run AFTER tqm_quantize_tq3 on keys.
 *
 * @param d_k_unit   [n_vectors × 128] L2-normalized key vectors (FP32)
 * @param d_mse_blk  [n_vectors] block_tq3_mi300x from tqm_quantize_tq3
 * @param d_rotation [128 × 128] same rotation matrix used in quantize
 * @param d_S        [128 × 128] QJL Gaussian projection matrix (different seed)
 * @param d_qjl_out  [n_vectors] block_qjl_mi300x output
 * @param n_vectors  number of key vectors
 * @param stream     HIP stream
 */
void tqm_qjl_keys(
    const float             *d_k_unit,
    const block_tq3_mi300x  *d_mse_blk,
    const float             *d_rotation,
    const float             *d_S,
    block_qjl_mi300x        *d_qjl_out,
    int                      n_vectors,
    hipStream_t              stream
);

/**
 * tqm_alloc_rotation — Generate and upload an orthogonal rotation matrix.
 * Uses Gram-Schmidt on seeded Gaussian matrix. Caller must hipFree(*d_rot_out).
 *
 * @param seed       RNG seed (use same seed for both compress and decompress)
 * @param d_rot_out  output device pointer to [128 × 128] float32
 */
void tqm_alloc_rotation(uint64_t seed, float **d_rot_out);

/**
 * tqm_alloc_qjl_matrix — Generate and upload a Gaussian QJL projection matrix.
 * Caller must hipFree(*d_S_out).
 *
 * @param seed       RNG seed (must differ from rotation seed)
 * @param d_S_out    output device pointer to [128 × 128] float32
 */
void tqm_alloc_qjl_matrix(uint64_t seed, float **d_S_out);

#ifdef __cplusplus
}
#endif
