/*
 * TurboQuant HIP Validation Suite
 * Tests correctness, not just speed.
 *
 * Build: hipcc -O3 --offload-arch=gfx1100 -o tq_validate ggml_turboquant.c ggml_turboquant.hip.cpp tq_validate.cpp -lm
 */

#include "ggml_turboquant.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <chrono>

#define CHECK_HIP(call) do { \
    hipError_t err = (call); \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error at %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

extern "C" {
    int tq_context_init(tq_context *ctx, int bits, uint64_t seed);
    void tq_quantize(const tq_context *ctx, const float *src, void *dst);
    void tq_dequantize(const tq_context *ctx, const void *src, float *dst);
    void tq_hip_init_codebooks(void);
    void tq_hip_quantize_tq3(const float *, void *, const float *, int, hipStream_t);
    void tq_hip_dequantize_tq3(const void *, float *, const float *, int, hipStream_t);
    void tq_hip_quantize_tq4(const float *, void *, const float *, int, hipStream_t);
    void tq_hip_dequantize_tq4(const void *, float *, const float *, int, hipStream_t);
}

static int tests_passed = 0;
static int tests_failed = 0;

static void check(bool cond, const char *name) {
    if (cond) {
        printf("  \033[32m✓ PASS\033[0m %s\n", name);
        tests_passed++;
    } else {
        printf("  \033[31m✗ FAIL\033[0m %s\n", name);
        tests_failed++;
    }
}

/* Box-Muller for normal distribution */
static float randn() {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    if (u1 < 1e-10f) u1 = 1e-10f;
    return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
}

int main() {
    printf("=========================================================\n");
    printf("TurboQuant HIP Validation Suite\n");
    printf("=========================================================\n\n");

    tq_context ctx;
    tq_context_init(&ctx, 3, TQ_ROTATION_SEED);

    /* ---- Test 1: GPU vs CPU bit-exact comparison ---- */
    printf("[Test 1] GPU vs CPU Output Comparison\n");
    {
        const int N = 256;
        float *h_src = (float *)malloc(N * TQ_HEAD_DIM * sizeof(float));
        srand(42);
        for (int i = 0; i < N * TQ_HEAD_DIM; i++) h_src[i] = randn() * 0.5f;

        /* CPU path */
        void *cpu_quant = malloc(N * sizeof(block_tq3));
        float *cpu_out = (float *)malloc(N * TQ_HEAD_DIM * sizeof(float));
        for (int v = 0; v < N; v++) {
            tq_quantize(&ctx, h_src + v * TQ_HEAD_DIM, (uint8_t *)cpu_quant + v * sizeof(block_tq3));
            tq_dequantize(&ctx, (uint8_t *)cpu_quant + v * sizeof(block_tq3), cpu_out + v * TQ_HEAD_DIM);
        }

        /* GPU path */
        float *d_src, *d_out, *d_rotation;
        void *d_quant;
        CHECK_HIP(hipMalloc(&d_src, N * TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_quant, N * sizeof(block_tq3)));
        CHECK_HIP(hipMalloc(&d_out, N * TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float)));

        CHECK_HIP(hipMemcpy(d_src, h_src, N * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_rotation, ctx.rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));
        tq_hip_init_codebooks();

        tq_hip_quantize_tq3(d_src, d_quant, d_rotation, N, 0);
        tq_hip_dequantize_tq3(d_quant, d_out, d_rotation, N, 0);
        CHECK_HIP(hipDeviceSynchronize());

        float *gpu_out = (float *)malloc(N * TQ_HEAD_DIM * sizeof(float));
        CHECK_HIP(hipMemcpy(gpu_out, d_out, N * TQ_HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

        /* Compare */
        double max_diff = 0.0;
        double total_diff = 0.0;
        int mismatches = 0;
        for (int i = 0; i < N * TQ_HEAD_DIM; i++) {
            double diff = fabs(cpu_out[i] - gpu_out[i]);
            if (diff > max_diff) max_diff = diff;
            total_diff += diff;
            if (diff > 0.01) mismatches++;
        }
        double avg_diff = total_diff / (N * TQ_HEAD_DIM);

        printf("  Max diff:  %.6f\n", max_diff);
        printf("  Avg diff:  %.6f\n", avg_diff);
        printf("  Mismatches (>0.01): %d / %d\n", mismatches, N * TQ_HEAD_DIM);
        check(max_diff < 0.05, "GPU matches CPU within tolerance");

        CHECK_HIP(hipFree(d_src)); CHECK_HIP(hipFree(d_quant));
        CHECK_HIP(hipFree(d_out)); CHECK_HIP(hipFree(d_rotation));
        free(h_src); free(cpu_quant); free(cpu_out); free(gpu_out);
    }

    /* ---- Test 2: MSE with normal distribution (like real KV cache) ---- */
    printf("\n[Test 2] MSE with Normal Distribution\n");
    {
        const int N = 10000;
        float *h_src = (float *)malloc(N * TQ_HEAD_DIM * sizeof(float));
        srand(123);
        for (int i = 0; i < N * TQ_HEAD_DIM; i++) h_src[i] = randn();

        float *d_src, *d_out, *d_rotation;
        void *d_quant;
        CHECK_HIP(hipMalloc(&d_src, N * TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_quant, N * sizeof(block_tq3)));
        CHECK_HIP(hipMalloc(&d_out, N * TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float)));

        CHECK_HIP(hipMemcpy(d_src, h_src, N * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_rotation, ctx.rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));

        tq_hip_quantize_tq3(d_src, d_quant, d_rotation, N, 0);
        tq_hip_dequantize_tq3(d_quant, d_out, d_rotation, N, 0);
        CHECK_HIP(hipDeviceSynchronize());

        float *gpu_out = (float *)malloc(N * TQ_HEAD_DIM * sizeof(float));
        CHECK_HIP(hipMemcpy(gpu_out, d_out, N * TQ_HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

        double total_mse = 0.0;
        for (int v = 0; v < N; v++) {
            double mse = 0.0;
            for (int j = 0; j < TQ_HEAD_DIM; j++) {
                double diff = h_src[v * TQ_HEAD_DIM + j] - gpu_out[v * TQ_HEAD_DIM + j];
                mse += diff * diff;
            }
            mse /= TQ_HEAD_DIM;
            total_mse += mse;
        }
        total_mse /= N;

        printf("  GPU MSE (normal dist): %.6f  (paper: ~0.034)\n", total_mse);
        printf("  Ratio to paper: %.2f\n", total_mse / 0.034);
        check(total_mse < 0.05, "MSE within 1.5x of paper");
        check(total_mse > 0.02, "MSE not suspiciously low");

        CHECK_HIP(hipFree(d_src)); CHECK_HIP(hipFree(d_quant));
        CHECK_HIP(hipFree(d_out)); CHECK_HIP(hipFree(d_rotation));
        free(h_src); free(gpu_out);
    }

    /* ---- Test 3: Zero vector handling ---- */
    printf("\n[Test 3] Zero Vector\n");
    {
        float zeros[TQ_HEAD_DIM] = {0};
        float *d_src, *d_out, *d_rotation;
        void *d_quant;
        CHECK_HIP(hipMalloc(&d_src, TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_quant, sizeof(block_tq3)));
        CHECK_HIP(hipMalloc(&d_out, TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float)));

        CHECK_HIP(hipMemcpy(d_src, zeros, TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_rotation, ctx.rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));

        tq_hip_quantize_tq3(d_src, d_quant, d_rotation, 1, 0);
        tq_hip_dequantize_tq3(d_quant, d_out, d_rotation, 1, 0);
        CHECK_HIP(hipDeviceSynchronize());

        float out[TQ_HEAD_DIM];
        CHECK_HIP(hipMemcpy(out, d_out, TQ_HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

        double norm = 0.0;
        for (int i = 0; i < TQ_HEAD_DIM; i++) norm += out[i] * out[i];
        check(norm < 1e-10, "Zero vector round-trips to zero");

        CHECK_HIP(hipFree(d_src)); CHECK_HIP(hipFree(d_quant));
        CHECK_HIP(hipFree(d_out)); CHECK_HIP(hipFree(d_rotation));
    }

    /* ---- Test 4: Norm preservation ---- */
    printf("\n[Test 4] Norm Preservation\n");
    {
        float vec[TQ_HEAD_DIM];
        srand(999);
        double orig_norm = 0.0;
        for (int i = 0; i < TQ_HEAD_DIM; i++) {
            vec[i] = randn() * 3.7f;
            orig_norm += vec[i] * vec[i];
        }
        orig_norm = sqrt(orig_norm);

        float *d_src, *d_out, *d_rotation;
        void *d_quant;
        CHECK_HIP(hipMalloc(&d_src, TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_quant, sizeof(block_tq3)));
        CHECK_HIP(hipMalloc(&d_out, TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float)));

        CHECK_HIP(hipMemcpy(d_src, vec, TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_rotation, ctx.rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));

        tq_hip_quantize_tq3(d_src, d_quant, d_rotation, 1, 0);
        tq_hip_dequantize_tq3(d_quant, d_out, d_rotation, 1, 0);
        CHECK_HIP(hipDeviceSynchronize());

        float out[TQ_HEAD_DIM];
        CHECK_HIP(hipMemcpy(out, d_out, TQ_HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

        double recon_norm = 0.0;
        for (int i = 0; i < TQ_HEAD_DIM; i++) recon_norm += out[i] * out[i];
        recon_norm = sqrt(recon_norm);

        double ratio = recon_norm / orig_norm;
        printf("  Original norm: %.4f  Reconstructed: %.4f  Ratio: %.4f\n", orig_norm, recon_norm, ratio);
        check(fabs(ratio - 1.0) < 0.1, "Norm preserved within 10%");

        CHECK_HIP(hipFree(d_src)); CHECK_HIP(hipFree(d_quant));
        CHECK_HIP(hipFree(d_out)); CHECK_HIP(hipFree(d_rotation));
    }

    /* ---- Test 5: Quantize throughput with parallel bit-packing ---- */
    printf("\n[Test 5] Quantize Throughput (bottleneck analysis)\n");
    {
        const int N = 65536;
        const int iters = 20;
        float *d_src, *d_out, *d_rotation;
        void *d_quant;
        size_t src_bytes = (size_t)N * TQ_HEAD_DIM * sizeof(float);

        float *h_src = (float *)malloc(src_bytes);
        srand(77);
        for (size_t i = 0; i < (size_t)N * TQ_HEAD_DIM; i++) h_src[i] = randn();

        CHECK_HIP(hipMalloc(&d_src, src_bytes));
        CHECK_HIP(hipMalloc(&d_quant, N * sizeof(block_tq3)));
        CHECK_HIP(hipMalloc(&d_out, src_bytes));
        CHECK_HIP(hipMalloc(&d_rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float)));

        CHECK_HIP(hipMemcpy(d_src, h_src, src_bytes, hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_rotation, ctx.rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));

        /* Warmup */
        tq_hip_quantize_tq3(d_src, d_quant, d_rotation, N, 0);
        CHECK_HIP(hipDeviceSynchronize());

        auto t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; it++) {
            tq_hip_quantize_tq3(d_src, d_quant, d_rotation, N, 0);
        }
        CHECK_HIP(hipDeviceSynchronize());
        auto t1 = std::chrono::high_resolution_clock::now();
        double quant_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        /* Dequant for comparison */
        tq_hip_dequantize_tq3(d_quant, d_out, d_rotation, N, 0);
        CHECK_HIP(hipDeviceSynchronize());

        t0 = std::chrono::high_resolution_clock::now();
        for (int it = 0; it < iters; it++) {
            tq_hip_dequantize_tq3(d_quant, d_out, d_rotation, N, 0);
        }
        CHECK_HIP(hipDeviceSynchronize());
        t1 = std::chrono::high_resolution_clock::now();
        double dequant_ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;

        printf("  Quantize:   %.3f ms (%.0f vec/s)\n", quant_ms, N / (quant_ms / 1000.0));
        printf("  Dequantize: %.3f ms (%.0f vec/s)\n", dequant_ms, N / (dequant_ms / 1000.0));
        printf("  Ratio Q/DQ: %.1fx\n", quant_ms / dequant_ms);
        check(quant_ms < 50.0, "Quantize < 50ms for 65K vectors");
        check(dequant_ms < 5.0, "Dequantize < 5ms for 65K vectors");

        CHECK_HIP(hipFree(d_src)); CHECK_HIP(hipFree(d_quant));
        CHECK_HIP(hipFree(d_out)); CHECK_HIP(hipFree(d_rotation));
        free(h_src);
    }

    /* ---- Test 6: TQ4 MSE and GPU vs CPU ---- */
    printf("\n[Test 6] TQ4 (4-bit) Validation\n");
    {
        tq_context ctx4;
        tq_context_init(&ctx4, 4, TQ_ROTATION_SEED);

        const int N = 1000;
        float *h_src = (float *)malloc(N * TQ_HEAD_DIM * sizeof(float));
        srand(555);
        for (int i = 0; i < N * TQ_HEAD_DIM; i++) h_src[i] = randn();

        /* CPU reference */
        float *cpu_out = (float *)malloc(N * TQ_HEAD_DIM * sizeof(float));
        void *cpu_quant = malloc(N * sizeof(block_tq4));
        for (int v = 0; v < N; v++) {
            tq_quantize(&ctx4, h_src + v * TQ_HEAD_DIM, (uint8_t *)cpu_quant + v * sizeof(block_tq4));
            tq_dequantize(&ctx4, (uint8_t *)cpu_quant + v * sizeof(block_tq4), cpu_out + v * TQ_HEAD_DIM);
        }

        double cpu_mse = 0.0;
        for (int v = 0; v < N; v++) {
            double mse = 0.0;
            for (int j = 0; j < TQ_HEAD_DIM; j++) {
                double diff = h_src[v * TQ_HEAD_DIM + j] - cpu_out[v * TQ_HEAD_DIM + j];
                mse += diff * diff;
            }
            cpu_mse += mse / TQ_HEAD_DIM;
        }
        cpu_mse /= N;

        /* GPU */
        float *d_src, *d_out, *d_rotation;
        void *d_quant;
        CHECK_HIP(hipMalloc(&d_src, N * TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_quant, N * sizeof(block_tq4)));
        CHECK_HIP(hipMalloc(&d_out, N * TQ_HEAD_DIM * sizeof(float)));
        CHECK_HIP(hipMalloc(&d_rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float)));

        CHECK_HIP(hipMemcpy(d_src, h_src, N * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));
        CHECK_HIP(hipMemcpy(d_rotation, ctx4.rotation, TQ_HEAD_DIM * TQ_HEAD_DIM * sizeof(float), hipMemcpyHostToDevice));

        tq_hip_quantize_tq4(d_src, d_quant, d_rotation, N, 0);
        tq_hip_dequantize_tq4(d_quant, d_out, d_rotation, N, 0);
        CHECK_HIP(hipDeviceSynchronize());

        float *gpu_out = (float *)malloc(N * TQ_HEAD_DIM * sizeof(float));
        CHECK_HIP(hipMemcpy(gpu_out, d_out, N * TQ_HEAD_DIM * sizeof(float), hipMemcpyDeviceToHost));

        double gpu_mse = 0.0;
        double max_diff = 0.0;
        for (int v = 0; v < N; v++) {
            double mse = 0.0;
            for (int j = 0; j < TQ_HEAD_DIM; j++) {
                double diff_src = h_src[v * TQ_HEAD_DIM + j] - gpu_out[v * TQ_HEAD_DIM + j];
                mse += diff_src * diff_src;
                double diff_cpu = fabs(cpu_out[v * TQ_HEAD_DIM + j] - gpu_out[v * TQ_HEAD_DIM + j]);
                if (diff_cpu > max_diff) max_diff = diff_cpu;
            }
            gpu_mse += mse / TQ_HEAD_DIM;
        }
        gpu_mse /= N;

        printf("  CPU MSE (TQ4): %.6f  (paper: ~0.009)\n", cpu_mse);
        printf("  GPU MSE (TQ4): %.6f\n", gpu_mse);
        printf("  GPU vs CPU max diff: %.6f\n", max_diff);
        check(gpu_mse < 0.015, "TQ4 GPU MSE within range");
        check(max_diff < 0.05, "TQ4 GPU matches CPU");

        CHECK_HIP(hipFree(d_src)); CHECK_HIP(hipFree(d_quant));
        CHECK_HIP(hipFree(d_out)); CHECK_HIP(hipFree(d_rotation));
        free(h_src); free(cpu_quant); free(cpu_out); free(gpu_out);
    }

    printf("\n=========================================================\n");
    printf("Results: %d passed, %d failed\n", tests_passed, tests_failed);
    printf("=========================================================\n");

    return tests_failed > 0 ? 1 : 0;
}
