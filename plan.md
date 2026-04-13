# AMD ROCm TurboQuant — Implementation & Benchmarking Plan

**Goal**: Adapt TurboQuant 3-bit KV cache compression for AMD Instinct MI300X (192 GB HBM3, CDNA3, gfx942, ROCm 7.2) and rigorously benchmark it against FP16/FP8 baselines across long context lengths (512–256K tokens), producing a complete performance, memory capacity, and model quality evaluation. Code is designed to be portable to any ROCm-compatible GPU (gfx942, gfx1100, gfx1201).

---

## Project Overview

```
Phase 0: Environment Setup (Days 1–2)
Phase 1: Baseline Benchmarks — FP16, FP8, INT4 (Days 3–5)
Phase 2: TurboQuant HIP Port to gfx942 (Days 6–10)
Phase 3: Fused Triton Attention Kernel (Days 11–15)
Phase 4: Integration & End-to-End Benchmarks (Days 16–20)
Phase 5: Analysis, Visualizations, Report (Days 21–25)
```

---

## Phase 0: Environment Setup

### 0.1 System Requirements

- OS: Ubuntu 24.04 LTS (pre-installed on devcloud)
- GPU: AMD Instinct MI300X — gfx942, 192 GB HBM3, 5.3 TB/s
- ROCm: 7.2 (pre-installed on devcloud — do not downgrade)
- Python: 3.11+ (pre-installed in devcloud image)

### 0.2 Environment Verification & Setup

ROCm 7.2 is pre-installed. Verify the environment and install Python dependencies:

```bash
# Verify MI300X and ROCm 7.2
rocminfo | grep -E "Name|gfx942"
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.hip)"

# Python dependencies (ROCm 7.2 compatible)
pip install transformers accelerate scipy numpy matplotlib pandas tqdm
pip install triton   # ROCm-compatible Triton (likely pre-installed)

# Performance flags for MI300X
export PYTORCH_TUNABLEOP_ENABLED=1        # GEMM autotuning
export HIP_FORCE_DEV_KERNARG=1            # faster kernel launch (default in devcloud images)
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library

# Verify GPU is visible and usable
python -c "import torch; x = torch.randn(1024,1024,device='cuda'); print('GPU OK, free VRAM:', torch.cuda.mem_get_info()[0]//1e9, 'GB')"
```

### 0.3 Model Selection

With 192 GB VRAM, we can run much larger models than a consumer GPU allows:

Primary model: **Mistral-7B-v0.1** (FP16, ~14 GB)
- Used in the original TurboQuant paper — direct quality comparison possible
- Short enough that we can test many context lengths quickly

Secondary model: **Llama-3.1-70B** (FP16, ~140 GB) — **key MI300X showcase**
- Fits in a single MI300X at FP16
- With TQ3, frees ~30 GB for KV at very long contexts
- Demonstrates the capacity unlock story that consumer GPUs cannot tell

Tertiary: **Llama-3.1-8B-Instruct** (used in original TurboQuant LongBench evaluation — enables direct paper comparison)

### 0.4 Repository Structure

```
turboquant-mi300x/
│
├── kernels/                            ← Core MI300X kernel library
│   ├── turboquant_mi300x.hip.cpp      ← OUR CONTRIBUTION: MI300X-optimized kernels
│   ├── turboquant_mi300x.h            ← API header (extends domvox's header)
│   ├── turboquant_mi300x_test.cpp     ← Validation test suite (9+ tests)
│   ├── turboquant_mi300x.py           ← Python ctypes wrapper
│   ├── tq_triton.py                   ← Triton fused dequant-attention
│   ├── build_mi300x.sh                ← Build script (gfx942)
│   └── ref/                           ← domvox reference (unmodified)
│       ├── ggml_turboquant.hip.cpp
│       ├── ggml_turboquant.h
│       ├── ggml_turboquant.c
│       └── turboquant.py
│
├── benchmarks/
│   ├── bench_kernels.py               ← Kernel throughput (GB/s, vectors/sec)
│   ├── bench_attention.py             ← Tokens/sec vs context length
│   ├── bench_memory.py                ← VRAM measurement
│   ├── bench_batch.py                 ← Batch size scaling (concurrent users)
│   └── bench_quality.py              ← Perplexity + cosine similarity
│
├── baselines/
│   ├── fp16_baseline.py
│   ├── fp8_baseline.py
│   └── int4_baseline.py
│
├── analysis/
│   ├── plot_throughput.py
│   ├── plot_memory.py
│   ├── plot_quality.py
│   ├── plot_kernel_breakdown.py
│   └── plot_batch_scaling.py
│
├── report/
│   └── final_report.md
│
└── requirements.txt
```

---

## Phase 1: Baseline Benchmarks

**Objective**: Establish reference numbers for FP16, FP8, and INT4 KV cache on gfx1201.

### 1.1 Benchmarking Harness

```python
# benchmarks/bench_attention.py (pseudocode)
import torch, time

def benchmark_decode_loop(model, tokenizer, prompt, seq_len, kv_dtype, n_tokens=100):
    """Measure tokens/sec for decode phase at given context length."""
    # Prefill to target seq_len
    inputs = pad_prompt_to_len(prompt, seq_len)
    with torch.no_grad():
        out = model(**inputs, use_cache=True)
        kv = convert_kv_cache(out.past_key_values, dtype=kv_dtype)
    
    # Decode loop
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_tokens):
        token_out = model(next_token, past_key_values=kv, use_cache=True)
        kv = token_out.past_key_values
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    return n_tokens / elapsed  # tokens/sec
```

### 1.2 Baseline Configurations

| Name | KV dtype | Method | Notes |
|------|----------|--------|-------|
| `fp16` | FP16 (2 bytes) | Standard PyTorch | Reference baseline |
| `fp8_e4m3` | FP8 E4M3 (1 byte) | `torch.float8_e4m3fn` cast | Supported natively on MI300X |
| `int8` | INT8 (1 byte) | `torch.quantize_per_tensor` | Software dequant |
| `int4_naive` | INT4 (0.5 byte) | PyTorch manual packing | No fused kernel, quality check only |

### 1.3 Metrics to Collect

For each baseline at `seq_len ∈ {512, 2048, 8192, 32768, 65536, 131072}` (extended range for MI300X):

- **Tokens/sec** (decode phase, 100 tokens, median of 5 runs)
- **Latency per token** (ms)
- **VRAM peak** (`torch.cuda.max_memory_allocated()`)
- **Memory bandwidth utilization** (estimated from VRAM bytes read / elapsed time)
- **Prefill time** (time to process prompt, one measurement)

For quality (run once at `seq_len=4096`):
- **Perplexity** on WikiText-103 (2048 token chunks, 50 samples)
- **Cosine similarity** of hidden states vs FP16 reference

---

## Phase 2: TurboQuant HIP — Validate & Build Our Own MI300X Library

This phase has two sub-parts:
- **2a**: Validate the domvox baseline on gfx942 (days 6–7, quick)
- **2b**: Write our own MI300X-optimized library `turboquant_mi300x.hip.cpp` (days 8–10, the core contribution)

### 2.1 Phase 2a — Validate domvox Baseline on gfx942

Clone domvox/turboquant-hip and recompile for MI300X:

```bash
git clone https://github.com/domvox/turboquant-hip.git
cd turboquant-hip

# Patch build script for gfx942
sed -i 's/gfx1100/gfx942/g' build.sh

# Build
./build.sh

# Validate — 9 GPU correctness tests
hipcc -O3 --offload-arch=gfx942 -o tq_validate \
    ggml_turboquant.c ggml_turboquant.hip.cpp tq_validate.cpp -lm
./tq_validate  # Must show: 9/9 PASS

# Benchmark baseline throughput
./tq_bench 65536 128  # 65K vectors, dim=128
```

**Expected**: Most code is Wave64-safe (uses `warpSize` at runtime). The `atomicOr` bit-packing and rotation matrix access patterns are suboptimal for MI300X but functionally correct.

**Baseline measurements to record**:
- Quantize throughput (GB/s) — domvox got 36 GB/s on gfx1100
- Dequantize throughput (GB/s) — domvox got 101 GB/s on gfx1100
- Fused dot throughput (dot/s)
- MSE vs paper: should be 0.0337 ± 5% for TQ3

### 2.2 Phase 2b — Our Own MI300X-Optimized Library

We write `turboquant_mi300x.hip.cpp` as a **new, purpose-built library** for CDNA3. This is the primary research contribution: **the first TurboQuant HIP implementation optimized for MI300X**.

Repository layout:
```
turboquant-rdna4/kernels/
├── turboquant_mi300x.hip.cpp   ← NEW: our MI300X-optimized kernels
├── turboquant_mi300x.h         ← NEW: API, data structures (extends domvox header)
├── turboquant_mi300x_test.cpp  ← NEW: validation suite
├── turboquant_mi300x.py        ← NEW: Python ctypes wrapper
├── build_mi300x.sh             ← NEW: build script
│
│   (Reference files from domvox — not modified, used as baseline)
├── ref/ggml_turboquant.hip.cpp
├── ref/ggml_turboquant.h
├── ref/ggml_turboquant.c
└── ref/turboquant.py
```

### 2.3 MI300X Kernel Optimizations to Implement

**Optimization 1: Ballot-based 3-bit packing (eliminates LDS atomics)**

Replace the `atomicOr` pattern with wavefront ballot operations. On Wave64, `__builtin_amdgcn_ballot_w64` returns a 64-bit mask of which threads have a true predicate:

```cpp
// Current (domvox): atomicOr into LDS — ~32 serialized atomics per word
atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)), ...);

// New (MI300X): ballot-based, zero LDS atomics
__shared__ uint64_t s_wave_ballot[3];  // one 64-bit mask per bit plane
if (tid < warpSize) {  // wave 0
    for (int b = 0; b < 3; b++) {
        int bit_val = (s_indices[tid] >> b) & 1;
        s_wave_ballot[b] = __builtin_amdgcn_ballot_w64(bit_val);
    }
} else {  // wave 1 (threads 64-127)
    for (int b = 0; b < 3; b++) {
        int bit_val = (s_indices[tid] >> b) & 1;
        // Atomically OR into second half of packed array
        s_wave_ballot[b] |= ((uint64_t)__builtin_amdgcn_ballot_w64(bit_val) << ...);
    }
}
__syncthreads();
// Thread 0 assembles the 3 ballot masks into 48 packed bytes
```

**Optimization 2: LDS rotation matrix caching (eliminates HBM3 traffic)**

Load the 128×128 rotation matrix into LDS once per CTA, then run multiple vectors:

```cpp
// Block of VECTORS_PER_BLOCK (e.g., 2) × HEAD_DIM threads
// LDS layout: [rotation: 128×128 floats = 64 KB] + [input vecs: 2×128 floats]
__shared__ float s_rot[TQ_HEAD_DIM * TQ_HEAD_DIM];  // 64 KB
__shared__ float s_vecs[VECTORS_PER_BLOCK][TQ_HEAD_DIM];

// Load rotation once (distribute across all threads)
const int total_rot = TQ_HEAD_DIM * TQ_HEAD_DIM;
for (int i = tid; i < total_rot; i += blockDim.x)
    s_rot[i] = rotation[i];  // Only done ONCE per block, amortized over N vectors
__syncthreads();

// Process VECTORS_PER_BLOCK vectors using s_rot
for (int v = 0; v < VECTORS_PER_BLOCK; v++) {
    // ... quantize vector v using s_rot ...
}
```

**Optimization 3: MFMA-accelerated rotation (replaces naive loop)**

Use `__builtin_amdgcn_mfma_f32_16x16x16f16` for the 128×128 matrix × vector multiply.
The rotation step (y = Π · x_unit) decomposes into 8 blocks of 16 rows:

```cpp
// For HEAD_DIM=128: 8 tiles of 16×128×1 (each tile = 8 mfma calls of 16×16×16)
// Use AGPR accumulators for intermediate sums
__builtin_amdgcn_mfma_f32_16x16x16f16(
    a_frag,      // 16×16 tile of Π (FP16)
    b_frag,      // 16×16 tile of x_unit column-broadcast (FP16)
    c_frag,      // FP32 accumulator (stored in AGPRs)
    0, 0, 0      // MFMA control bits
);
// Extract results from AGPR to VGPR for quantization
```

**Optimization 4: Vectorized output write**

Replace single-thread sequential write with parallel uint32 stores:

```cpp
// 48 bytes = 12 × uint32. Distribute across 12 threads.
if (tid < 12) {
    uint32_t *dst_int = (uint32_t *)(blk->indices);
    uint32_t *src_int = (uint32_t *)(s_packed);
    dst_int[tid] = src_int[tid];
}
if (tid == 0) blk->norm = norm;  // 4 bytes
```

**Optimization 5: QJL keys kernel (not in domvox)**

The domvox library only implements MSE quantization (Algorithm 1). The full TurboQuant uses Algorithm 2 (TurboQuant_prod) for keys — adding a 1-bit QJL residual correction. We add:

```cpp
// After MSE quantize of keys:
// 1. Dequantize MSE result
// 2. Compute residual: r = k_unit - k_mse_reconstructed
// 3. Project: p = S · r  (S is d×d Gaussian matrix, stored in __constant__ memory)
// 4. Store signs: qjl_signs[j] = (p[j] > 0) ? 1 : -1

__global__ void tq_qjl_keys_kernel(
    const float * __restrict__ k_unit,     // Already normalized keys
    const void  * __restrict__ mse_blocks, // Output of tq_quantize_kernel
    const float * __restrict__ Pi,         // Rotation matrix
    const float * __restrict__ S,          // QJL projection matrix (d×d)
    uint8_t     * __restrict__ qjl_signs,  // 1-bit packed signs, d/8 bytes per vector
    float       * __restrict__ residual_norms, // ||r||_2, one float per vector
    int n_vectors
);
```

**Optimization 6: Fused compress+dot (for attention score computation)**

Inspired by `tq_fused_dot_tq3` but using MFMA:

```cpp
// For each KV block:
// 1. Unpack 3-bit indices from LDS (already loaded)
// 2. Map indices to centroids (LDS lookup table)
// 3. Apply QJL correction: dot += scale * (q_proj · signs)
// 4. Accumulate via warp reduction using MFMA-friendly layout
__global__ void tq_fused_dot_qjl_tq3(
    const float * __restrict__ q_rotated,
    const float * __restrict__ q_proj,     // q projected by S (for QJL)
    const void  * __restrict__ kv_blocks,
    const uint8_t * __restrict__ qjl_signs,
    const float * __restrict__ residual_norms,
    float       * __restrict__ scores,
    float correction_scale,
    int n_queries, int n_kv
);
```

### 2.4 Validation Test Suite

Our validation tests (in `turboquant_mi300x_test.cpp` and Python):

| Test | Method | Pass Criterion |
|------|--------|----------------|
| Codebook constants | Compare to `turboquant.py` output | Exact float match (hardcoded, not computed) |
| TQ3 round-trip (GPU) | compress + decompress vs original | Cosine similarity > 0.99 |
| TQ3 MSE | 1000 random unit vectors | MSE = 0.0337 ± 5% |
| Bit-exact match | GPU result vs CPU reference (`ggml_turboquant.c`) | Max diff < 1e-6 |
| TQ4 round-trip | compress + decompress | Cosine similarity > 0.995 |
| QJL signs | Sign pattern vs Python reference | Exact match (same seed) |
| MFMA rotation vs naive | Both paths on same input | Max diff < 1e-4 |
| Throughput regression | vs domvox baseline on gfx1100 (if available) | ≥ same GB/s |
| Large batch | 8M vectors (256K ctx × 32 heads) | No OOM, correct MSE |

### 2.5 Build System

```bash
# kernels/build_mi300x.sh
#!/bin/bash
set -e

ARCH=gfx942
ROCM=/opt/rocm

# Compile MI300X-optimized library
hipcc -O3 \
    --offload-arch=${ARCH} \
    -DUSE_MFMA=1 \
    -DTARGET_MI300X=1 \
    -mwavefrontsize64 \                   # Enforce Wave64 (default on CDNA)
    -DCDNA3 -DAMD_MFMA_AVAILABLE \
    -I${ROCM}/include \
    -ffast-math \
    -o libturboquant_mi300x.so \
    --shared -fPIC \
    turboquant_mi300x.hip.cpp

# Build validation suite
hipcc -O3 --offload-arch=${ARCH} \
    -DCDNA3 \
    -o tq_validate_mi300x \
    ref/ggml_turboquant.c turboquant_mi300x.hip.cpp turboquant_mi300x_test.cpp \
    -lm

# Build benchmark
hipcc -O3 --offload-arch=${ARCH} \
    -DCDNA3 \
    -o tq_bench_mi300x \
    turboquant_mi300x.hip.cpp tq_hip_benchmark_mi300x.cpp -lm

echo "Build complete."
./tq_validate_mi300x
```

### 2.3 Standalone Benchmark Kernel

Write `tq_hip_gfx1201.hip.cpp` as a standalone benchmark:

```cpp
// Compress a random tensor (B, H, S, D) in TQ3 format
// Measure: compression throughput (GB/s), decompression throughput
// Validate: MSE vs FP16 reference
```

Targets:
- Compression throughput: >100 GB/s (memory-bound)
- Decompression throughput: >80 GB/s
- MSE vs paper: within 5% of 0.0337 (TQ3)

### 2.4 Python Wrapper

```python
# kernels/tq_hip.py
import ctypes, torch

lib = ctypes.CDLL("./libturbo_hip.so")

def compress_kv_tq3(keys: torch.Tensor, values: torch.Tensor):
    """Compress (B, H, S, D) KV tensors to TQ3 format using HIP kernels."""
    assert keys.is_cuda
    assert keys.dtype == torch.float16
    # Call HIP kernel via ctypes or hipFFI
    ...

def decompress_kv_tq3(compressed_keys, compressed_values) -> tuple:
    """Decompress back to FP16."""
    ...
```

Alternative: expose via a PyTorch C++ extension using `hip_extension` build system.

### 2.5 Validation Tests

| Test | Pass Criterion |
|------|---------------|
| Round-trip: compress + decompress | Cosine similarity > 0.98 |
| MSE vs FP16 | < 0.040 (TQ3 target) |
| Bit-pack correctness | Exact match with CPU reference |
| QJL signs dot product | Attention score error < 0.01 |
| Large batch (B=16, H=32, S=4096) | No OOM, no crash |

---

## Phase 3: Fused Triton Dequant-Attention Kernel

**Objective**: Write a Triton kernel that fuses TQ3 decompression + attention score computation, avoiding a full FP16 KV materialization.

This is the key performance lever: without kernel fusion, decompression to FP16 and then running attention means two global memory passes. Fusion eliminates the intermediate FP16 KV store.

### 3.1 Algorithm (Fused Triton Kernel)

```
For each query token q and each KV block:
  1. Load compressed K (3-bit indices + norms) from global memory  [48 + 4 bytes per K vector]
  2. Dequantize K on-chip:
     - indices → centroids (lookup table in shared memory)
     - rotate back: y = Π · centroids
     - scale by norm: k_fp16 = norm × y
  3. Compute attention score: s = q · k_fp16
  4. Apply QJL correction: s += correction_factor × (q_proj · signs)
  5. Accumulate softmax numerator online (Flash Attention 2 style)
  
  For values:
  6. Load compressed V (3-bit indices + norms)
  7. Dequantize V on-chip: v_fp16 = norm × Π · centroids
  8. Accumulate: output += softmax_weight × v_fp16
```

### 3.2 Triton Implementation Strategy

Use the existing Flash Attention Triton AMD kernel as a base
(`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`). Patch the K/V loading section:

```python
# kernels/tq_triton.py
import triton
import triton.language as tl

@triton.jit
def turboquant_attention_fwd(
    Q_ptr, K_idx_ptr, K_norm_ptr, K_qjl_signs_ptr,
    V_idx_ptr, V_norm_ptr,
    O_ptr,
    CENTROIDS: tl.constexpr,  # lookup table, const memory
    PI: tl.constexpr,         # rotation matrix, const memory
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    # Load Q block (FP16, full precision)
    q = tl.load(Q_ptr + ...)
    
    # Iterate over K/V blocks
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    lse = tl.zeros([BLOCK_M], dtype=tl.float32)  # log-sum-exp
    
    for block_n in range(0, seq_len, BLOCK_N):
        # Load compressed K (3-bit indices)
        k_idx = tl.load(K_idx_ptr + ...)   # uint8, packed
        k_norm = tl.load(K_norm_ptr + ...)  # float16
        # Unpack 3-bit indices → 8-level centroids
        k_centroids = tl.gather(CENTROIDS, k_idx)
        # Apply rotation Π^T (matrix multiply with const)
        k_rot = tl.dot(PI, k_centroids)  # HEAD_DIM × HEAD_DIM matmul
        k_fp16 = k_norm[:, None] * k_rot
        
        # Compute attention scores
        scores = tl.dot(q, k_fp16.T)  # [BLOCK_M, BLOCK_N]
        
        # QJL correction
        signs = tl.load(K_qjl_signs_ptr + ...)  # packed bits
        q_proj = tl.dot(q, S.T)  # S is QJL projection matrix
        correction = correction_scale * (q_proj * signs.to(tl.float32))
        scores += correction.sum(axis=-1)
        
        # Online softmax (Flash Attention 2)
        # ... standard FA2 accumulation ...
    
    tl.store(O_ptr + ..., acc)
```

### 3.3 Triton Autotuning Configuration

```python
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'waves_per_eu': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 64, 'waves_per_eu': 1}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 128, 'waves_per_eu': 2}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'waves_per_eu': 1}, num_warps=8),
    ],
    key=['seq_len', 'head_dim'],
)
```

Based on the gfx1100 Flash Attention autotuning finding (`BLOCK_M=128, BLOCK_N=64, waves_per_eu=1, num_warps=8`), start with this as the default for gfx1201 and tune from there.

### 3.4 Fallback Path

If Triton fusion proves difficult (e.g., the rotation matrix multiply is problematic within a single kernel), fall back to a two-pass approach:

1. HIP kernel: batch decompress K/V to FP16 scratch buffer
2. Triton kernel: standard Flash Attention on FP16 K/V

This is slower (two global memory passes) but still benchmarks correctly.

---

## Phase 4: Integration & End-to-End Benchmarks

### 4.1 Integration with llama.cpp HIP Build (MI300X)

Extend `domvox/llama.cpp-turboquant-hip` for gfx942. Note that animehacker/llama-turboquant already supports gfx942 — use that as the build base:

```bash
git clone https://github.com/animehacker/llama-turboquant.git
cd llama-turboquant

mkdir build-hip && cd build-hip
cmake .. \
  -DGGML_HIP=ON \
  -DCMAKE_BUILD_TYPE=Release \
  -DAMDGPU_TARGETS="gfx942"   # MI300X
make -j$(nproc)
```

Run with TQ3 on MI300X:
```bash
HIP_VISIBLE_DEVICES=0 ./llama-cli \
  -m mistral-7b-v0.1.gguf \
  -c 32768 -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3
```

For the 70B model at long context:
```bash
HIP_VISIBLE_DEVICES=0 ./llama-cli \
  -m llama-3.1-70b.gguf \
  -c 131072 -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3
```

### 4.2 Full Benchmark Matrix

For each `(model, seq_len, kv_config)`:

```
models:       [mistral-7b, llama-3.1-8b-instruct, llama-3.1-70b (capacity demo)]
seq_lens:     [512, 2048, 8192, 32768, 65536, 131072]
kv_configs:   [fp16, fp8_e4m3, int4_naive, tq3_pytorch, tq3_hip, tq3_hip_fused]
```

**MI300X-specific benchmark**: Batch size scaling at fixed seq_len=32K:
```
batch_sizes:  [1, 2, 4, 8, 16, 32]  (tokens/sec/user, VRAM usage)
```
This quantifies the "4.9× more concurrent users" hypothesis.

Metrics per cell:
- Tokens/sec (100 decode steps, median 5 runs)
- VRAM peak (MB)
- Perplexity on 50 WikiText-103 samples (quality)
- Time breakdown: attention %, KV load %, softmax %

### 4.3 Profiling with rocprof (MI300X specific)

```bash
# ROCm 7.2 uses rocprofv2
rocprofv2 --hip-api --kernel-trace \
  python bench_attention.py --model mistral-7b --seq-len 32768 --kv tq3_hip_fused

# Key MI300X metrics to extract:
# - FETCH_SIZE (bytes fetched from global memory per kernel)
# - VALUUtilization (ALU utilization)
# - MemUnitBusy (HBM3 busy %)
# - L2CacheHit (256 MB LLC effectiveness)
# - MFMA_F16_16X16X16_F32 (matrix unit utilization — key for rotation kernel)

# Alternative: use PyTorch profiler with ROCm backend
import torch
with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as p:
    run_decode_step(...)
print(p.key_averages().table(sort_by="cuda_time_total"))
```

---

## Phase 5: Analysis & Report

### 5.1 Visualizations to Produce

**Plot 1: Tokens/sec vs context length**
```
Y-axis: tokens/sec (decode phase)
X-axis: seq_len (log scale: 512 → 131072)
Lines: fp16, fp8, tq3_pytorch, tq3_hip, tq3_hip_fused
Model: Mistral-7B
Highlight: crossover point where TQ3 > FP16 (expected ~32K on MI300X)
```

**Plot 2: VRAM usage vs context length (capacity unlock)**
```
Y-axis: VRAM usage (GB)
X-axis: seq_len (log scale)
Lines: fp16, fp8, tq3, tq4
Shade: 192 GB MI300X limit
Annotate: seq_len at which 70B + KV exceeds 192 GB for each scheme
```

**Plot 3: Quality vs compression level**
```
Y-axis: perplexity (WikiText-103)
X-axis: bits per element (16, 8, 4, 3, 2)
FP16 perplexity as dashed baseline
```

**Plot 4: Kernel time breakdown**
```
Stacked bar chart for each (seq_len, kv_config):
- Attention computation
- KV memory load
- Compression/decompression overhead
- Softmax
```

**Plot 5: AMD vs NVIDIA comparison (stretch)**
```
If H100/A100 baseline available, compare:
- Bandwidth efficiency (tokens/sec per GB/s bandwidth)
- Compression crossover point
```

### 5.2 Analysis Checklist

- [ ] Identify crossover point: minimum seq_len where TQ3 outperforms FP16 at 5.3 TB/s (expected ~32K)
- [ ] Quantify bandwidth savings: (FP16 bytes - TQ3 bytes) / FP16 bytes = 79.7% KV reduction
- [ ] Compute achieved bandwidth for KV-load kernel: `bytes_loaded / kernel_time`
- [ ] Compare achieved vs theoretical 5.3 TB/s; check LLC (256 MB) hit rate
- [ ] Assess whether bottleneck is compute (MFMA rotation) or memory (HBM3 index/norm load)
- [ ] Batch scaling: max concurrent users per 192 GB at seq_len=32K (fp16 vs tq3)
- [ ] 70B capacity test: longest context fitting in 192 GB for each scheme
- [ ] Document any MI300X / gfx942 specific issues (wave64, MFMA availability, ROCm 7.2 quirks)

### 5.3 Expected Outcomes

| Scenario | Expectation | Basis |
|----------|-------------|-------|
| TQ3 vs FP16 at seq_len ≥ 32K | TQ3 faster | HBM3 KV traffic reduction dominates at long contexts |
| TQ3 vs FP16 at seq_len < 8K | FP16 faster or equal | 5.3 TB/s bandwidth; dequant compute overhead dominates |
| 70B max context (fp16) | ~80K tokens in 192 GB | Weights ~140 GB; 52 GB for KV |
| 70B max context (tq3) | ~400K tokens in 192 GB | 4.9× KV compression releases ~42 GB |
| Quality (perplexity) | <1% degradation vs FP16 | Paper: zero accuracy loss on similar models |
| Fused vs unfused TQ3 | Fused 2–3× faster | Eliminates second global memory pass |
| Batch size multiplier (32K ctx) | ~4–5× more users with TQ3 | 4.9× KV compression → proportional VRAM saving |

---

## Engineering Decisions & Trade-offs

### Why llama.cpp over vLLM as the primary integration target?

The domvox/llama.cpp-turboquant-hip fork is the most complete end-to-end TurboQuant implementation. It has been validated for gfx942 via the animehacker fork. For benchmarking purposes, llama.cpp gives us direct control over KV cache behavior. vLLM integration is a stretch goal (adding TQ3 as a new `kv_cache_dtype` in vLLM's paged attention).

### Why Triton over CK Tile for the fused attention kernel?

On MI300X, CK Flash Attention (via FlashInfer) is also available and production-quality. Both are viable:
- **Triton**: Python-level, rapid iteration, good for prototyping the fused kernel. Use first.
- **CK (Composable Kernel)**: C++ templates, production performance, CDNA-native MFMA tiling. Use for final optimized implementation if Triton performance is insufficient.

Start with Triton, switch to CK if there is > 20% gap vs theoretical peak.

### Why ROCm 7.2 (and not 6.4.1)?

ROCm 7.2 is the pre-installed production version on the MI300X devcloud. The VRAM regression observed by the domvox author was on consumer RDNA3 GPUs (gfx1100) and may be an upstream llama.cpp quantized-KV issue, not a MI300X / ROCm 7.2 issue. We will test and document this explicitly.

### Rotation Matrix Storage

The 128×128 rotation matrix `Π` is 128×128×2 bytes = 32 KB. On MI300X, each CU has 64 KB shared memory (LDS), so the rotation matrix fits in shared memory if pre-loaded once per CTA. Strategy: pass as a const pointer to HIP kernels; load into LDS at the start of the compression kernel to avoid repeated HBM3 access. In the Triton kernel, load `Π` into a shared tensor before the K/V loop.

### Wave64 Impact on Warp Reductions

The domvox kernels use `__shfl_down` for per-warp reductions (used in norm computation and dot products). On gfx942 with Wave64, verify that the reduction loop runs for `log2(64)=6` iterations (vs `log2(32)=5` on gfx1100). The `warpSize` runtime read should handle this automatically if the code is written correctly.

---

## Stretch Goals

1. **vLLM integration**: Add TurboQuant as a new `kv_cache_dtype` option in vLLM's paged attention for MI300X (gfx942).
2. **GQA + TurboQuant**: Test with Llama-3.1-8B (GQA model) — does sharing compressed KV across query groups affect quality?
3. **MFMA-accelerated rotation kernel**: Replace the naive rotation matmul in the HIP kernel with `__builtin_amdgcn_mfma_f32_16x16x16f16` for peak throughput on MI300X matrix cores.
4. **2-bit KV (TQ2)**: Evaluate quality at 2-bit (6.4× compression) — would enable 70B at ~700K context in 192 GB.
5. **Continuous batching / multi-user**: Simulate 4–32 concurrent users at seq_len=32K; measure aggregate tokens/sec and per-user latency vs FP16.
6. **NVIDIA paper comparison**: Reproduce the H100 attention speedup chart from the TurboQuant paper and compare MI300X results — does the 5.3 TB/s bandwidth change the speedup profile?

---

## Deliverables Summary

| Deliverable | Location | Status |
|-------------|----------|--------|
| **MI300X HIP library** | `kernels/turboquant_mi300x.hip.cpp` | Core contribution |
| MI300X API header | `kernels/turboquant_mi300x.h` | TQ2/TQ3/TQ4 + QJL |
| Validation test suite | `kernels/turboquant_mi300x_test.cpp` | 9+ tests, bit-exact |
| Python ctypes wrapper | `kernels/turboquant_mi300x.py` | Callable from PyTorch |
| Triton fused dequant-attention | `kernels/tq_triton.py` | Stretch: fused kernel |
| Benchmark scripts | `benchmarks/` | All phases |
| Raw results (CSV) | `results/` | All benchmark cells |
| Plots (5+) | `analysis/figures/` | PNG + PDF |
| Technical report | `report/final_report.md` | Final writeup |
| Research notes | `research.md` | Algorithm + porting deep-dive |

---

## Timeline

| Days | Phase | Key Milestones |
|------|-------|----------------|
| 1–2 | Setup | ROCm 7.2 verified on MI300X, models downloaded |
| 3–5 | Baselines | FP16/FP8/INT4 benchmark numbers at 6 context lengths |
| 6–7 | Phase 2a | domvox port compiles and passes 9 tests on gfx942 |
| 8–10 | Phase 2b | `turboquant_mi300x.hip.cpp` written, validated, benchmarked |
| 11–12 | Phase 2b polish | MFMA optimization profiled; Python wrapper callable |
| 13–15 | Triton kernel | Initial fused dequant-attention running on MI300X |
| 16–18 | Integration | animehacker llama.cpp fork working with TQ3 on MI300X |
| 19–20 | Full benchmark | All (model × seq_len × kv_config) combinations |
| 21–22 | Profiling | `rocprofv2` kernel breakdown, HBM3 bandwidth analysis |
| 23–24 | Visualizations | 5+ plots generated |
| 25 | Report | Final technical report written |
