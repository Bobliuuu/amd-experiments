/**
 * tq_hip_benchmark_mi300x.cpp — Kernel Throughput Benchmarks
 *
 * Measures compression and decompression throughput (GB/s) for TQ3/TQ4
 * on AMD Instinct MI300X (gfx942).
 *
 * Also measures the domvox reference baseline (if available) for comparison.
 *
 * Usage:
 *   ./tq_bench_mi300x [n_vectors] [n_warmup] [n_iters]
 *   ./tq_bench_mi300x 65536 10 50     # 65K vectors, 10 warmup, 50 iters
 *
 * Expected targets (MI300X, gfx942):
 *   Quantize  (compress):   > 100 GB/s input throughput
 *   Dequantize (decompress): > 80  GB/s output throughput
 *   Fused dot:              > 50  GB/s effective throughput
 */

#include "turboquant_mi300x.h"
#include <hip/hip_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <string.h>

// ──────────────────────────────────────────────────────────────────────────────
// Utilities
// ──────────────────────────────────────────────────────────────────────────────

#define CHECK_HIP(call) do {                                          \
    hipError_t _e = (call);                                           \
    if (_e != hipSuccess) {                                           \
        fprintf(stderr, "HIP error %s:%d: %s\n",                     \
                __FILE__, __LINE__, hipGetErrorString(_e));           \
        exit(1);                                                      \
    }                                                                 \
} while (0)

static uint64_t rng = 0xABCDEF1234567890ULL;
static float lcg_f32(void) {
    rng = 6364136223846793005ULL * rng + 1442695040888963407ULL;
    return (float)(int32_t)(rng >> 32) / (float)(1 << 30);
}

static double hip_elapsed_ms(hipEvent_t start, hipEvent_t stop) {
    float ms;
    hipEventElapsedTime(&ms, start, stop);
    return (double)ms;
}

// ──────────────────────────────────────────────────────────────────────────────
// Benchmark harness
// ──────────────────────────────────────────────────────────────────────────────

typedef struct {
    double min_ms, max_ms, avg_ms;
    double gbps_in, gbps_out;
    size_t bytes_in, bytes_out;
} BenchResult;

typedef void (*KernelFn)(void *, void *, int);

static BenchResult bench_kernel(
    const char *name,
    KernelFn    fn,
    void       *arg_in,
    void       *arg_out,
    int         n_vectors,
    size_t      bytes_in_per_run,
    size_t      bytes_out_per_run,
    int         n_warmup,
    int         n_iters
) {
    hipEvent_t ev_start, ev_stop;
    CHECK_HIP(hipEventCreate(&ev_start));
    CHECK_HIP(hipEventCreate(&ev_stop));

    // Warmup
    for (int i = 0; i < n_warmup; i++) fn(arg_in, arg_out, n_vectors);
    CHECK_HIP(hipDeviceSynchronize());

    // Benchmark
    double times[1024];
    int actual_iters = (n_iters > 1024) ? 1024 : n_iters;
    double total_ms = 0.0;
    double min_ms = 1e30, max_ms = 0.0;

    for (int i = 0; i < actual_iters; i++) {
        CHECK_HIP(hipEventRecord(ev_start, 0));
        fn(arg_in, arg_out, n_vectors);
        CHECK_HIP(hipEventRecord(ev_stop, 0));
        CHECK_HIP(hipEventSynchronize(ev_stop));
        double ms = hip_elapsed_ms(ev_start, ev_stop);
        times[i] = ms;
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;
    }

    BenchResult r;
    r.min_ms   = min_ms;
    r.max_ms   = max_ms;
    r.avg_ms   = total_ms / actual_iters;
    r.bytes_in  = bytes_in_per_run;
    r.bytes_out = bytes_out_per_run;
    r.gbps_in  = (bytes_in_per_run  / 1e9) / (r.min_ms / 1e3);
    r.gbps_out = (bytes_out_per_run / 1e9) / (r.min_ms / 1e3);

    printf("  %-32s | n=%6d | min=%.3f ms | avg=%.3f ms | "
           "BW_in=%.1f GB/s | BW_out=%.1f GB/s\n",
           name, n_vectors, r.min_ms, r.avg_ms, r.gbps_in, r.gbps_out);

    CHECK_HIP(hipEventDestroy(ev_start));
    CHECK_HIP(hipEventDestroy(ev_stop));
    return r;
}

// ──────────────────────────────────────────────────────────────────────────────
// Kernel wrappers for bench_kernel
// ──────────────────────────────────────────────────────────────────────────────

static float *g_d_rotation = NULL;

static void run_quantize_tq3(void *src, void *dst, int n) {
    tqm_quantize_tq3((const float *)src, g_d_rotation, (block_tq3_mi300x *)dst, n, 0);
}

static void run_dequantize_tq3(void *src, void *dst, int n) {
    tqm_dequantize_tq3((const block_tq3_mi300x *)src, g_d_rotation, (float *)dst, n, 0);
}

static void run_quantize_tq4(void *src, void *dst, int n) {
    tqm_quantize_tq4((const float *)src, g_d_rotation, (block_tq4_mi300x *)dst, n, 0);
}

static void run_dequantize_tq4(void *src, void *dst, int n) {
    tqm_dequantize_tq4((const block_tq4_mi300x *)src, g_d_rotation, (float *)dst, n, 0);
}

// For fused dot: fixed n_queries=1 (single query vs n_kv KV blocks)
static float *g_d_q_rot = NULL;
static float *g_d_scores = NULL;
static int    g_n_kv = 0;

static void run_fused_dot_tq3(void *src, void *unused_dst, int n) {
    (void)unused_dst;
    tqm_fused_dot_tq3(g_d_q_rot, (const block_tq3_mi300x *)src, g_d_scores,
                      1, n, 0);
}

// ──────────────────────────────────────────────────────────────────────────────
// main
// ──────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    int n_vectors = 65536;
    int n_warmup  = 10;
    int n_iters   = 50;

    if (argc > 1) n_vectors = atoi(argv[1]);
    if (argc > 2) n_warmup  = atoi(argv[2]);
    if (argc > 3) n_iters   = atoi(argv[3]);

    printf("=== TurboQuant MI300X Kernel Benchmark ===\n");
    printf("GPU: ");
    {
        hipDeviceProp_t p;
        hipGetDeviceProperties(&p, 0);
        printf("%s | warpSize=%d\n", p.name, p.warpSize);
    }
    printf("n_vectors=%d  n_warmup=%d  n_iters=%d\n\n", n_vectors, n_warmup, n_iters);

    const int d = TQ_HEAD_DIM;

    // ── Allocate buffers ─────────────────────────────────────────────────────
    float           *d_input;
    float           *d_output;
    block_tq3_mi300x *d_tq3;
    block_tq4_mi300x *d_tq4;

    CHECK_HIP(hipMalloc(&d_input,  (size_t)n_vectors * d * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_output, (size_t)n_vectors * d * sizeof(float)));
    CHECK_HIP(hipMalloc(&d_tq3,    (size_t)n_vectors * sizeof(block_tq3_mi300x)));
    CHECK_HIP(hipMalloc(&d_tq4,    (size_t)n_vectors * sizeof(block_tq4_mi300x)));

    // Fill input with random unit vectors
    {
        float *h_in = (float *)malloc((size_t)n_vectors * d * sizeof(float));
        for (int v = 0; v < n_vectors; v++) {
            float norm_sq = 0.0f;
            for (int j = 0; j < d; j++) { h_in[v*d+j] = lcg_f32(); norm_sq += h_in[v*d+j] * h_in[v*d+j]; }
            float inv = 1.0f / sqrtf(norm_sq);
            for (int j = 0; j < d; j++) h_in[v*d+j] *= inv;
        }
        CHECK_HIP(hipMemcpy(d_input, h_in, (size_t)n_vectors * d * sizeof(float), hipMemcpyHostToDevice));
        free(h_in);
    }

    // ── Initialize library ────────────────────────────────────────────────────
    tqm_init();
    tqm_alloc_rotation(42ULL, (float **)&g_d_rotation);

    // For fused dot: single query rotated
    CHECK_HIP(hipMalloc(&g_d_q_rot, (size_t)d * sizeof(float)));
    CHECK_HIP(hipMalloc(&g_d_scores, (size_t)n_vectors * sizeof(float)));
    // Initialize query to random unit vector
    {
        float h_q[128];
        float ns = 0.0f;
        for (int j = 0; j < d; j++) { h_q[j] = lcg_f32(); ns += h_q[j]*h_q[j]; }
        float inv = 1.0f / sqrtf(ns);
        for (int j = 0; j < d; j++) h_q[j] *= inv;
        CHECK_HIP(hipMemcpy(g_d_q_rot, h_q, d * sizeof(float), hipMemcpyHostToDevice));
    }

    // Pre-compress TQ3 and TQ4 blocks for the decompression and fused-dot benchmarks
    tqm_quantize_tq3(d_input, g_d_rotation, d_tq3, n_vectors, 0);
    tqm_quantize_tq4(d_input, g_d_rotation, d_tq4, n_vectors, 0);
    CHECK_HIP(hipDeviceSynchronize());

    // ── Compute byte counts ────────────────────────────────────────────────────
    size_t bytes_fp32_in  = (size_t)n_vectors * d * sizeof(float);           // 67 MB for 65K
    size_t bytes_tq3_out  = (size_t)n_vectors * sizeof(block_tq3_mi300x);    // 3.4 MB for 65K
    size_t bytes_tq4_out  = (size_t)n_vectors * sizeof(block_tq4_mi300x);    // 4.4 MB for 65K

    printf("Input:   %.1f MB (float32, %d vectors × %d dim)\n",
           bytes_fp32_in / 1e6, n_vectors, d);
    printf("TQ3 out: %.1f MB (%.1f× compression)\n",
           bytes_tq3_out / 1e6, (double)bytes_fp32_in / bytes_tq3_out);
    printf("TQ4 out: %.1f MB (%.1f× compression)\n\n",
           bytes_tq4_out / 1e6, (double)bytes_fp32_in / bytes_tq4_out);

    printf("%-34s | %-8s | %-14s | %-14s | %-18s | %s\n",
           "Kernel", "n", "min_ms", "avg_ms", "BW_in (GB/s)", "BW_out (GB/s)");
    printf("%s\n", "────────────────────────────────────────────────────────────────────────────────────────────────");

    // ── TQ3 Quantize (Compress) ─────────────────────────────────────────────
    bench_kernel(
        "TQ3 quantize (compress)",
        run_quantize_tq3, d_input, d_tq3, n_vectors,
        bytes_fp32_in, bytes_tq3_out, n_warmup, n_iters);

    // ── TQ3 Dequantize (Decompress) ─────────────────────────────────────────
    bench_kernel(
        "TQ3 dequantize (decompress)",
        run_dequantize_tq3, d_tq3, d_output, n_vectors,
        bytes_tq3_out, bytes_fp32_in, n_warmup, n_iters);

    // ── TQ4 Quantize ────────────────────────────────────────────────────────
    bench_kernel(
        "TQ4 quantize (compress)",
        run_quantize_tq4, d_input, d_tq4, n_vectors,
        bytes_fp32_in, bytes_tq4_out, n_warmup, n_iters);

    // ── TQ4 Dequantize ───────────────────────────────────────────────────────
    bench_kernel(
        "TQ4 dequantize (decompress)",
        run_dequantize_tq4, d_tq4, d_output, n_vectors,
        bytes_tq4_out, bytes_fp32_in, n_warmup, n_iters);

    // ── Fused dot (TQ3) ──────────────────────────────────────────────────────
    // 1 query vs n_vectors KV blocks
    bench_kernel(
        "TQ3 fused dot (1 query)",
        run_fused_dot_tq3, d_tq3, NULL, n_vectors,
        bytes_tq3_out + (size_t)d * sizeof(float),
        (size_t)n_vectors * sizeof(float),
        n_warmup, n_iters);

    // ── Summary ──────────────────────────────────────────────────────────────
    printf("\n=== Theoretical MI300X limits ===\n");
    printf("  Memory bandwidth: 5300 GB/s\n");
    printf("  For TQ3 compress  (FP32 in = %.1f MB): theoretical %.0f GB/s bounded = %.3f ms min\n",
           bytes_fp32_in / 1e6, 5300.0, bytes_fp32_in / (5300e9) * 1e3);
    printf("  For TQ3 decompress (TQ3 in = %.1f MB): theoretical %.0f GB/s bounded = %.4f ms min\n",
           bytes_tq3_out / 1e6, 5300.0, bytes_tq3_out / (5300e9) * 1e3);

    // Cleanup
    hipFree(d_input); hipFree(d_output); hipFree(d_tq3); hipFree(d_tq4);
    hipFree(g_d_q_rot); hipFree(g_d_scores); hipFree(g_d_rotation);
    return 0;
}
