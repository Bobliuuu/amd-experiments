# TurboQuant for AMD GPUs — Research Notes

**AMD ROCm TurboQuant Benchmarking & KV Cache Optimization Study**
Date: April 2026

---

## 1. TurboQuant — The Algorithm (Google Research)

Source: [Google Research Blog](https://research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/) | Paper: [arXiv 2504.19874](https://arxiv.org/abs/2504.19874) (ICLR 2026)

TurboQuant is a two-stage KV cache compression algorithm combining **PolarQuant** and **QJL** (Quantized Johnson-Lindenstrauss).

### 1.1 Core Problem

The KV cache grows linearly with sequence length. At FP16 (2 bytes/element), a single attention head with `head_dim=128` at sequence length 32k costs `32768 × 128 × 2 × 2 (K+V) = 16 MB` per layer. Modern 8B models have 32 layers × 8 heads, making this ~4 GB per sequence — a dominant bottleneck.

Traditional vector quantization requires storing quantization constants per block, adding 1–2 bits of overhead that partially defeats the compression goal. TurboQuant eliminates this overhead entirely.

### 1.2 Algorithm Stages

**Stage 1: PolarQuant (key + value compression)**

1. **Normalize**: Compute the L2 norm of each head vector, store as a 4-byte scalar.
2. **Random Orthogonal Rotation**: Apply a fixed random rotation matrix `Π` to the unit vector. After rotation, each coordinate is approximately `N(0, 1/d)` (Gaussian) by the Central Limit Theorem. This "Gaussianizes" the distribution.
3. **Lloyd-Max Scalar Quantization**: With a Gaussian distribution, the optimal quantizer is pre-computable offline. No per-block calibration is needed. For 3-bit (8 centroids): `{-2.157, -1.334, -0.743, -0.243, +0.243, +0.743, +1.334, +2.157} × σ`.
4. **Bit-pack**: Store 128 indices at 3 bits each = 48 bytes total (+ 4 bytes norm = 52 bytes/vector, vs 256 bytes FP16 → **4.9× compression**).

**Stage 2: QJL (key-only residual bias correction)**

After MSE quantization of keys, a systematic bias remains. QJL addresses this using the Johnson-Lindenstrauss Transform:

1. Compute residual: `r = k_unit - k_mse_reconstructed`
2. Project residual with a Gaussian random matrix `S`: `s = S · r`
3. Store only the **sign bit** of `s` (1 bit per dimension)
4. At attention time, use the stored sign bits to apply an unbiased correction to the attention logit, eliminating the quantization bias without storing high-precision data.

**Bit allocation (3-bit total)**:
- Keys: 2-bit MSE (PolarQuant) + 1-bit QJL signs
- Values: 3-bit MSE (PolarQuant only — no QJL needed for values)

**Resulting compression**: ~4.9× at 3-bit (52 bytes vs 256 bytes per 128-dim FP16 vector)

### 1.3 Benchmarked Results (Google, H100)

- 4-bit TurboQuant: **up to 8× speedup** on attention logit computation vs FP32
- **Zero accuracy loss** on LongBench, Needle-in-Haystack, RULER, ZeroSCROLLS
- Models: Mistral-7B, Gemma, Llama-3.1-8B-Instruct
- **5.02× KV cache compression** (vs 3.76× MXFP4, 3.56× NVFP4)
- Works without retraining or fine-tuning

---

## 2. Reference CUDA Implementation: turboquant-gpu

Repository: [DevTechJr/turboquant-gpu](https://github.com/DevTechJr/turboquant-gpu)

### 2.1 Architecture

```
turboquant_gpu/
├── __init__.py          # Version 0.1.4, lazy export of TurboQuantEngine
├── host.py              # TurboQuantEngine: orchestration, HF cache integration
├── constants.py         # HEAD_DIM=128, block sizes, defaults
├── codebook.py          # LloydMaxCodebook (scipy-based offline solver)
├── compress.py          # cuTile @ct.kernel compress kernels
├── decompress.py        # cuTile @ct.kernel decompress kernels
└── attention.py         # cuTile @ct.kernel attention: scores + softmax + V
```

### 2.2 Kernel Design (cuTile)

All GPU kernels are written as Python `@ct.kernel` functions using NVIDIA's **cuTile** framework (CUDA 13.0+, not standard CUDA). Block sizes: `BLOCK_S=64` (compress), `BLOCK_Q=16`, `BLOCK_KV=64` (attention), `HEAD_DIM=128`.

Key kernels:
- `turboquant_compress_kv_3bit` — fused K+V in a single kernel launch
- `turboquant_compress_keys` — normalize → rotate (`Pi_T`) → Lloyd-Max → QJL signs
- `turboquant_compress_values` — normalize → rotate → Lloyd-Max
- `turboquant_decompress_values` — codebook lookup → matrix multiply (`Pi`) → scale
- `turboquant_attention_scores` — K_mse dot products + QJL correction term
- `turboquant_fused_attention_vfused_3bit` — on-chip V dequant + softmax + accumulate

PyTorch fallback paths exist for all operations when cuTile is unavailable.

### 2.3 Public API

```python
engine = TurboQuantEngine(head_dim=128, total_bits=3, device="cuda")
compressed = engine.compress_kv_cache(past_key_values)
cache = engine.build_cache(compressed)
result = engine.generate(model, tokenizer, "prompt")
stats = engine.compression_stats(past_key_values)  # ratio, VRAM saved
engine.auto_tune(seq_len=512)  # benchmark cuTile vs PyTorch, pick best
```

### 2.4 Critical Limitation

cuTile is **NVIDIA-specific** (CUDA Tile IR, CUDA 13.0+). It cannot run on AMD GPUs. The PyTorch fallback path is the only portable code, but it is unoptimized.

---

## 3. Existing AMD / HIP Community Work

### 3.1 domvox/turboquant-hip (RDNA3, gfx1100)

Repository: [domvox/turboquant-hip](https://github.com/domvox/turboquant-hip)

A standalone HIP port of TurboQuant compression kernels targeting AMD RDNA3 (gfx1100, RX 7900 XTX series). **This is the most directly useful prior work.**

Files:
- `ggml_turboquant.hip.cpp` — HIP kernels: quantize, dequantize, fused dot (TQ3 + TQ4)
- `ggml_turboquant.h` — types, codebooks, API
- `ggml_turboquant.c` — CPU reference implementation
- `ggml_turboquant.cu` — Original CUDA kernels (reference)

Performance (gfx1100, ROCm 6.4):
| Metric | TQ3 (3-bit) | TQ4 (4-bit) |
|--------|------------|------------|
| MSE | 0.0337 | 0.0093 |
| Compression | 4.9× | 3.8× |
| Block size | 52 bytes | 68 bytes |

HIP porting notes from the code:
- `__shfl_down` without sync mask (HIP-style warp reduction)
- `warpSize`-aware reduction — works with RDNA wave32 **and** wave64
- TQ4 uses nibble packing (2 values per byte, no atomics)
- `hipMemcpyToSymbol` for constant memory codebooks

### 3.2 domvox/llama.cpp-turboquant-hip (gfx1100, full inference pipeline)

Repository: [domvox/llama.cpp-turboquant-hip](https://github.com/domvox/llama.cpp-turboquant-hip), branch `feature/turboquant-hip-port-clean`

Full llama.cpp integration with new GGML types: `TURBO2_0` (2-bit), `TURBO3_0` (3-bit), `TURBO4_0` (4-bit).

Usage:
```bash
cmake -B build -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j16
HIP_VISIBLE_DEVICES=0 ./build/bin/llama-cli \
  -m model.gguf -c 4096 -ngl 99 \
  --cache-type-k turbo3 --cache-type-v turbo3
```

Known issues: Symmetric K+V turbo gives correct results; asymmetric falls back to slower non-fused path. VRAM growth regression observed on ROCm 7.2.1 (not present on 6.4) — upstream ROCm issue, not TurboQuant-specific.

### 3.3 animehacker/llama-turboquant (gfx1151, Strix Halo APU)

Benchmarked on AMD Radeon 8060S (Strix Halo APU, 128 GB UMA), ROCm 7.2. Demonstrates TurboQuant TQ3_0 on a different AMD architecture. Supports gfx1151 (Strix Halo), gfx1100 (RDNA3), gfx1030 (RDNA2), gfx942 (MI300X).

**Notable**: RX 9060 XT (gfx1201) is NOT yet listed as a supported target in any of these ports.

### 3.4 Flash Attention Triton Backend (RDNA4, gfx12)

vLLM PR #32944 (merged Jan 2026): Flash Attention Triton backend enabled for RDNA3/RDNA4 in vLLM.

Key finding: CK (Composable Kernel) Flash Attention only supports CDNA (gfx90a/gfx942/gfx950). For RDNA4 (gfx1201), the **Triton backend** is the correct approach.

Activation:
```bash
export FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE
```

Observed speedup: 1.8–6.5× over PyTorch SDPA for ViT attention. Autotuning (finding optimal `BLOCK_M`, `BLOCK_N`, `waves_per_eu`) further improves performance.

### 3.5 FP8 KV Cache on RDNA4 (gfx12)

vLLM PR #34741 (Feb 2026): FP8 KV cache for gfx12 (RDNA4) custom paged attention via software dequantization. Uses `convert_b8x8_to_b16x8()` HIP runtime `fp8::vec_conversion`. Results in 2× token capacity increase (e.g., 132K → 264K tokens on R9700 with Qwen3-30B).

This is the **baseline 8-bit reference** for our benchmarks. No sub-8-bit KV cache exists yet for ROCm.

---

## 4. Target Hardware: AMD Instinct MI300X

> **Note on hardware**: The benchmarking environment is an AMD Instinct MI300X (192 GB, ROCm 7.2) datacenter GPU, not a consumer RDNA4 card. The implementation is architected to be portable across any ROCm-compatible GPU (gfx942, gfx1100, gfx1201), but all empirical results are from MI300X.

### 4.1 Specifications

| Attribute | Value |
|-----------|-------|
| Architecture | CDNA3 |
| GPU Target (ROCm) | **gfx942** |
| Compute Units | 304 CUs |
| Matrix Cores | 1,216 |
| Stream Processors | 19,456 |
| Peak Engine Clock | 2100 MHz |
| FP8 Performance | **2.61 PFLOPs** (5.22 PFLOPs sparse) |
| FP16/BF16 Performance | **1.3 PFLOPs** |
| FP32 Performance | 163.4 TFLOPs |
| Memory | **192 GB HBM3** |
| Memory Interface | 8192-bit |
| Memory Bandwidth | **5.3 TB/s** |
| Last Level Cache (LLC) | 256 MB |
| TDP | 750 W |
| Process Node | 5nm (XCDs) + 6nm (IOD), TSMC |

### 4.2 Memory Bandwidth Analysis

The MI300X has **5.3 TB/s** HBM3 bandwidth — the highest in its generation, exceeding the H100 SXM5 (3.35 TB/s) by ~58%. This fundamentally changes the compression value proposition compared to consumer GPUs.

| GPU | Bandwidth | Class |
|-----|-----------|-------|
| MI300X (this study) | 5.3 TB/s | Datacenter |
| H100 SXM5 | 3.35 TB/s | Datacenter |
| A100 80GB | 2.0 TB/s | Datacenter |
| RX 7900 XTX (RDNA3) | 960 GB/s | Consumer |
| RX 9060 XT (RDNA4) | 320 GB/s | Consumer |

**Revised hypothesis for MI300X**: At short-to-medium context lengths (< 32K tokens), the MI300X is NOT the bandwidth bottleneck — it is instead a **compute bottleneck** during matrix operations. The value of KV compression on MI300X shifts from raw bandwidth savings to:

1. **Fitting larger models at longer context**: A 70B model at FP16 uses ~140 GB of VRAM, leaving only ~52 GB for KV cache. At seq_len=128K with 32 layers × 8 KV heads (GQA) × 128 head_dim: `32 × 8 × 128 × 131072 × 2 (K+V) × 2 bytes ≈ 17 GB` — still fits, but for multi-head architectures or larger models it becomes critical.
2. **Higher batch sizes**: TQ3 frees up ~80% of KV memory, enabling 4.9× more concurrent users at the same context length.
3. **Very long context (> 128K tokens)**: At 1M token contexts, even 5.3 TB/s becomes the bottleneck. The crossover moves to > 32K tokens on MI300X vs ~1–2K on a consumer GPU.

For a Mistral-7B at FP16 with seq_len=4096:
- KV cache: 32 layers × 32 heads × 128 dim × 4096 tokens × 2 (K+V) × 2 bytes ≈ **2 GB**
- Load time at 5.3 TB/s: 2 GB / 5300 GB/s ≈ **0.38 ms** per decode step
- With TQ3 (4.9×): 0.41 GB → **0.077 ms** — compute overhead of dequant likely exceeds this at short seq_len

Estimated crossover for MI300X: **~32K–64K tokens** (vs ~1K–2K tokens on RX 9060 XT). At long context, the bandwidth savings become the dominant factor.

### 4.3 ROCm 7.2 Support Status for gfx942 (MI300X)

ROCm 7.2 is the production-ready, well-tested release for MI300X. Status:

| Component | Status on MI300X |
|-----------|-----------------|
| ROCm 7.2 base support | Full production support |
| PyTorch ROCm | Full support |
| Triton on ROCm | Works — use for custom kernels |
| Flash Attention (CK backend) | **Fully supported** (CDNA-native) |
| Flash Attention (Triton backend) | Also works (`FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`) |
| FlashInfer 0.5.3 | **Supported** (gfx942 / CDNA3) |
| AITER (AMD Inference Tensor Engine) | **Supported** (CDNA-class only) |
| vLLM FP8 KV cache | **Supported** and well-tested |
| hipBLASLt autotuning | Full support with GEMM tuning |
| LLM inference (vLLM, llama.cpp) | Production quality |
| TurboQuant HIP (3-bit) | animehacker/llama-turboquant lists **gfx942** as a target |

**Key advantage over RDNA4**: MI300X is a first-class ROCm target. All major inference libraries (vLLM, FlashInfer, AITER) support gfx942 natively. No workarounds required.

### 4.4 RDNA3 → CDNA3 Porting Differences

The community TurboQuant HIP ports primarily target gfx1100 (RDNA3 Wave32). CDNA3 (gfx942) is a different architecture:

| Feature | RDNA3 (gfx1100) | CDNA3 (gfx942) |
|---------|----------------|----------------|
| Wave size | Wave32 | Wave64 |
| FP8 support | Software (gfx1100) | Native hardware |
| Matrix units | WMMA | MFMA (Matrix Fused Multiply-Add) |
| Shared memory | 64 KB/CU | 64 KB/CU |
| Memory | GDDR6 (320–960 GB/s) | HBM3 (5.3 TB/s) |
| Infinity Cache | 32–96 MB | 256 MB LLC |
| HIP warp shuffle | Maskless, wave32 | Wave64, `__shfl_down` syntax differs |

**Critical**: The domvox kernels use `warpSize`-aware code (runtime `warpSize` check) to handle wave32/wave64. Verify this is active. MFMA instructions on gfx942 can accelerate the rotation matrix multiply — a key optimization opportunity beyond a basic recompile.

animehacker/llama-turboquant already lists gfx942 in its AMDGPU_TARGETS table, so a recompile with `-DAMDGPU_TARGETS="gfx942"` is the correct starting point.

---

## 5. Deep Dive: Porting turboquant-hip to MI300X (CDNA3/gfx942)

This section documents the exact code-level analysis of the domvox `ggml_turboquant.hip.cpp` and identifies every change required — and every optimization opportunity — when targeting MI300X.

### 5.1 Source Code Structure (domvox baseline)

The kernel file contains three main GPU functions plus host wrappers:

| Kernel | Block size | Purpose |
|--------|-----------|---------|
| `tq_quantize_kernel_tq3` | 128 threads × 1 block/vector | Normalize → rotate → quantize → bit-pack |
| `tq_dequantize_kernel_tq3` | 128 threads × 1 block/vector | Unpack → centroid → inverse rotate → scale |
| `tq_fused_dot_tq3` | 128 threads × (n_kv × n_q) blocks | Unpack → dot product with query (no full decompression) |
| `tq_quantize_kernel_tq4` | 128 threads × 1 block/vector | Same as TQ3 but 4-bit nibble packing (simpler) |
| `tq_dequantize_kernel_tq4` | 128 threads × 1 block/vector | TQ4 dequant |

Codebooks are stored as `__constant__ float d_codebook_3[8]` and `d_codebook_4[16]` in constant memory, loaded via `hipMemcpyToSymbol`. The 128×128 rotation matrix is passed as a `const float *` global memory pointer.

### 5.2 Wave64 Correctness Analysis

The kernel uses `warpSize` at runtime in two places:

**`warp_reduce_sum()`**:
```cpp
for (int offset = warpSize / 2; offset > 0; offset >>= 1)
    val += __shfl_down(val, offset);
```
On Wave32 (gfx1100): starts at offset=16, runs 5 iterations.
On Wave64 (gfx942): starts at offset=32, runs 6 iterations. **Correct — runtime `warpSize` handles this automatically.**

**`s_warp_sums[8]` sizing**:
- Wave32: 128 threads / 32 = **4 warps** → indices 0–3 used
- Wave64: 128 threads / 64 = **2 warps** → indices 0–1 used
- Array size 8 is sufficient for both. **Correct.**

**Conclusion**: The warp reduction code is Wave64-safe as written. No changes needed for correctness.

### 5.3 Bit-Packing atomicOr Analysis

The 3-bit packing kernel uses:
```cpp
atomicOr((unsigned int *)(s_packed + (byte_idx & ~3)),
         1u << (bit_in_byte + 8 * (byte_idx & 3)));
```

This aligns all atomic operations to 4-byte boundaries, using `unsigned int` atomics on LDS. `TQ3_INDEX_BYTES = 48` bytes → 12 × 4-byte words.

**Contention analysis on Wave64**:
- 128 threads, each writing 3 bits = 384 total bit-writes into 12 words
- Average: 32 threads per 4-byte word
- On Wave64, a single wave does 64 threads: ~16 threads hit the same word on average
- LDS `atomicOr` on MI300X: ~20–40 cycle latency per atomic, serialized per word
- Worst case: 32 serialized atomics × 30 cycles = ~960 cycles of LDS atomic overhead
- This is the primary bottleneck in the quantize kernel — avoidable with ballot

**MI300X opportunity**: Replace with wavefront ballot — zero LDS atomics:
```cpp
// For bit b of each thread's index:
uint32_t bit_val = (s_indices[tid] >> b) & 1;
uint64_t mask = __builtin_amdgcn_ballot_w64(bit_val);  // Wave64 ballot
// Thread 0 unpacks the 64-bit mask into bytes
if (tid < TQ3_INDEX_BYTES) { /* extract the right 8 bits */ }
```

### 5.4 Rotation Matrix Memory Traffic

The rotation matrix is `128 × 128 × 4 bytes = 64 KB`. Each kernel invocation (one vector):
- `tq_quantize_kernel_tq3`: Each of 128 threads reads its row: 128 floats = 512 bytes
- Total per block: 128 × 512 bytes = 64 KB from global memory (HBM3)

For a large batch (e.g., 32K context × 32 layers × 8 KV heads = 8,192,000 vectors per forward pass):
- Total rotation reads: **524 GB** of HBM3 reads for the rotation alone
- At 5.3 TB/s: ~99 ms of pure HBM3 time just for rotation reads

**Fix**: Load rotation into LDS once per CTA, reuse across multiple vectors per block:
```cpp
__shared__ float s_rot[TQ_HEAD_DIM][TQ_HEAD_DIM];  // 64 KB — fits in gfx942 LDS

// Thread 0 loads rotation (or distribute across threads)
for (int row = tid; row < d; row += blockDim.x)
    for (int col = 0; col < d; col++)
        s_rot[row][col] = rotation[row * d + col];
__syncthreads();
// Then use s_rot instead of rotation[...]
```

But wait: MI300X LDS per CU is 64 KB. The rotation matrix IS 64 KB. This means loading the rotation into LDS occupies the entire LDS bank. For a single-vector-per-block design, this works. For a batched design (multiple vectors per block), we'd need to either use a smaller LDS chunk or process multiple vectors per pass with the rotation loaded once.

**Better**: Process **multiple vectors per CTA** (e.g., 4 vectors × 32 threads = 128 threads). Load rotation once into LDS, then process 4 vectors using it. This amortizes the LDS load cost by 4×.

### 5.5 Rotation Matrix Multiply — MFMA Opportunity

Current implementation:
```cpp
float y_val = 0.0f;
const float *my_row = rotation + tid * d;
for (int j = 0; j < d; j++)
    y_val += my_row[j] * s_input[j] * inv_norm;
```

This is a row-times-vector dot product: 128 FMAs per thread, 16,384 FMAs per block, no vectorization.

**MFMA replacement** for MI300X (gfx942):
The `__builtin_amdgcn_mfma_f32_16x16x16f16` instruction computes a 16×16×16 half-precision matrix multiply into FP32 accumulator in a single wave-level instruction.

To compute `y = Π × x_unit` (128×1 = 128×128 × 128×1):
- Decompose into 8×1 tiles of 16-element rows: 8 tiles × 16 rows/tile = 128 rows
- Use MFMA to compute 16×16 blocks (need to pack x into 16-wide column)
- One `mfma_f32_16x16x16f16` instruction replaces 16×16 = 256 scalar FMAs

Expected speedup: ~4–8× over naive loop for the rotation step.

### 5.6 Output Write Bottleneck

Current TQ3 quantize:
```cpp
if (tid == 0) {
    block_tq3 *blk = ...;
    blk->norm = norm;
    for (int i = 0; i < TQ3_INDEX_BYTES; i++)  // 48 iterations!
        blk->indices[i] = s_packed[i];
}
```

Thread 0 writes 52 bytes (4-byte norm + 48-byte indices) sequentially — a single-threaded memory write.

**Fix**: Distribute write across 13 threads using `uint4` (16-byte) stores:
```cpp
// 48 bytes = 3 × 16-byte uint4 stores
if (tid < 12)  // 12 × 4 bytes = 48 bytes
    ((uint32_t*)(blk->indices))[tid] = ((uint32_t*)s_packed)[tid];
```

### 5.7 Summary: domvox → MI300X Changes Required

| Issue | Severity | Action |
|-------|----------|--------|
| Wave64 warp reduction | None — already correct | No change needed |
| `atomicOr` bit-packing contention | High | Replace with `__builtin_amdgcn_ballot_w64` |
| Rotation from global memory every call | High | Cache in LDS, amortize over multiple vectors per CTA |
| Naive rotation matmul loop | Medium | Replace with MFMA tiles |
| Single-thread output write | Low | Distribute across 12 threads with uint32 stores |
| Only gfx1100 in build flags | Critical | Add `--offload-arch=gfx942` |
| No head_dim=256 support | Medium | Templatize on `HEAD_DIM` |
| No QJL kernel (keys only) | Medium | Add QJL sign kernel per Algorithm 2 |

### 5.8 CDNA3 vs RDNA3 Architecture Reference

| Feature | RDNA3 (gfx1100) | CDNA3 (gfx942) | Impact on TurboQuant |
|---------|----------------|----------------|----------------------|
| Wavefront size | Wave32 (default) | Wave64 (fixed) | Reduction loops run 6 not 5 iterations |
| Matrix units | WMMA (16×16 only) | MFMA v3 (4×4 to 32×32) | MFMA can accelerate rotation matmul |
| LDS per CU | 128 KB | 64 KB | Rotation (64 KB) fills entire LDS on CDNA3 |
| L2/LLC cache | 4–8 MB | 256 MB | Rotation stays hot across blocks |
| Memory | GDDR6, 128-bit bus | HBM3, 8192-bit | Far higher bandwidth, different coalescing |
| AGPR registers | Not available | 256 AGPRs | MFMA results stored in AGPRs |
| VGPR file | 1024 VGPRs per CU | Shared VGPR/AGPR | Must track AGPR usage for MFMA |
| `__ballot` width | 32-bit or 64-bit | 64-bit only | Ballot-based packing uses 64-bit masks |
| `atomicOr` LDS | Supported | Supported | Works but contended on Wave64 |
| `hipMemcpyToSymbol` | Supported | Supported | No change |
| Compile flag | `--offload-arch=gfx1100` | `--offload-arch=gfx942` | Required change |

### 5.9 Our Custom Library Design: turboquant_mi300x

We will write `turboquant_mi300x.hip.cpp` as a new file (not modifying domvox's code) that is:
1. **Correct**: bit-exact on unpacking, MSE matches paper (TQ3: 0.0337 ± 5%)
2. **MI300X-first**: exploits MFMA, Wave64 ballot, LDS rotation caching
3. **General**: templatized on `HEAD_DIM` (128 and 256)
4. **Complete**: TQ2, TQ3, TQ4 quantize + dequantize + fused dot
5. **Validated**: against `turboquant.py` Python reference (CPU bit-exact match)

The library is a standalone shared object: `libturboquant_mi300x.so`, callable from Python via ctypes or as a PyTorch custom op.

## 6. Software Stack

### 5.1 Kernel Programming Options

| Option | Suitability for MI300X | Notes |
|--------|------------------------|-------|
| **Raw HIP (hipcc)** | Excellent for bit-pack compression | Direct port from gfx1100 community kernels; gfx942 already a listed target |
| **Triton on ROCm** | Good for attention kernels | Python-level, JIT-compiled, flexible for prototyping |
| **CK Flash Attention** | Best for fused attention on CDNA | CDNA-native, production quality, used in FlashInfer |
| **AITER** | Excellent for MI300X primitives | Fully supported on gfx942; provides optimized attention ops |
| **MFMA intrinsics** | Best for rotation matmul in HIP | Native MI300X matrix unit; replaces WMMA |
| **cuTile (NVIDIA)** | Not applicable | NVIDIA-specific |

**Recommended stack for MI300X**:
- Compression/decompression (bit-pack, codebook): Raw HIP with gfx942 MFMA for rotation matrix
- Fused dequant-attention: CK Flash Attention extension, or Triton for rapid prototyping
- Orchestration: PyTorch ROCm backend (transformers / direct)
- Integration: vLLM (production, full MI300X support) or llama.cpp HIP (simpler, TurboQuant fork exists)

### 5.2 Inference Framework Options

| Framework | MI300X / gfx942 Status | TurboQuant Integration Path |
|-----------|------------------------|------------------------------|
| **vLLM** | Production quality | Add TQ3 as new `kv_cache_dtype`; custom paged attention kernel |
| **llama.cpp** | Works well | domvox/llama.cpp-turboquant-hip; recompile for gfx942 |
| **FlashInfer** | FlashInfer 0.5.3 supports gfx942 | Could wrap TQ3 decompression before attention |
| **PyTorch direct** | Always works | TurboQuant PyTorch fallback from turboquant-gpu |

### 5.3 Environment Setup (MI300X, ROCm 7.2)

ROCm 7.2 is pre-installed on the devcloud environment. Verify and configure:

```bash
# Verify MI300X and ROCm 7.2
rocminfo | grep -E "gfx942|Name"
python -c "import torch; print(torch.cuda.get_device_name(0)); print(torch.version.hip)"

# Dependencies
pip install transformers accelerate scipy numpy matplotlib pandas
pip install triton  # ROCm-compatible Triton (pre-installed in devcloud image)

# Performance flags for MI300X
export PYTORCH_TUNABLEOP_ENABLED=1       # GEMM autotuning
export HIP_FORCE_DEV_KERNARG=1           # faster kernel launch
export ROCBLAS_TENSILE_LIBPATH=/opt/rocm/lib/rocblas/library  # rocBLAS tuning
```

---

## 6. Key Findings and Gaps

### What Exists

1. The TurboQuant algorithm is fully described and theoretically sound (ICLR 2026).
2. A CUDA/cuTile GPU implementation exists (DevTechJr/turboquant-gpu) with PyTorch fallback.
3. Community HIP port exists (domvox/turboquant-hip) targeting gfx1100 (RDNA3), with **gfx942 listed** as a supported target in animehacker/llama-turboquant.
4. Full llama.cpp integration exists (domvox/llama.cpp-turboquant-hip), compilable for gfx942.
5. CK Flash Attention, AITER, and FlashInfer all fully support MI300X (gfx942).
6. FP8 KV cache is available in vLLM on MI300X as a baseline (well-tested, production quality).
7. ROCm 7.2 on MI300X is production-grade — no stability workarounds needed.

### What Does Not Yet Exist (Our Contribution)

1. **Rigorous end-to-end TurboQuant benchmarks on MI300X** at long context lengths (32K–256K).
2. **Fused dequant-attention HIP/Triton kernel for gfx942** using MFMA matrix instructions for the rotation step.
3. **Quality evaluation** (perplexity, LongBench) at sub-4-bit KV precision on MI300X hardware.
4. **Systematic comparison**: FP16 vs FP8 vs TQ3 vs TQ4 across 6+ context lengths on the same hardware.
5. **Batch size scaling analysis**: How does TQ3 affect concurrent user capacity on 192 GB VRAM?
6. **Crossover point identification** for MI300X at the specific bandwidth regime (5.3 TB/s).

### Hardware Hypothesis (updated for MI300X)

**Short context (seq_len ≤ 8K)**: MI300X is compute-bound during GEMM phases; memory bandwidth not saturated by KV ops. TQ3 dequant overhead may make it slightly slower than FP16 here.

**Long context (seq_len ≥ 32K)**: KV memory traffic dominates. TQ3's 4.9× compression reduces KV load from global memory by ~80%, providing measurable throughput benefit even at 5.3 TB/s.

Quantified estimate for Mistral-7B at seq_len=32K:
- FP16 KV: 32 × 32 × 128 × 32768 × 4 bytes ≈ **16 GB** per decode step load
- At 5.3 TB/s: **3.0 ms** memory-bound
- With TQ3: 3.27 GB → **0.62 ms** → **net speedup ≈ 4× on the KV portion**
- At seq_len=128K: FP16 KV ≈ 64 GB → 12 ms; TQ3 ≈ 13 GB → 2.5 ms

**Primary value proposition for MI300X**:
- Enables 70B models at 128K+ context on a single GPU (capacity unlock)
- Enables 4.9× more concurrent users at same context length (batch multiplier)
- Provides throughput gains starting at ~32K tokens (measured crossover)
- Maintains exact same quality as FP16 (zero accuracy loss per paper)

---

## 7. Related Work Summary

| Work | Method | Bits | Hardware | Status |
|------|--------|------|----------|--------|
| TurboQuant (Google, 2026) | PolarQuant + QJL | 3–4 | H100 (CUDA) | Published, ICLR 2026 |
| turboquant-gpu (DevTechJr) | PolarQuant + QJL (cuTile) | 2–4 | NVIDIA CUDA | Public, v0.1.4 |
| domvox/turboquant-hip | PolarQuant + QJL (HIP) | 3–4 | RDNA3 gfx1100 | Community, working |
| animehacker/llama-turboquant | TQ3_0 (HIP) | 3 | gfx942, gfx1100, gfx1151 | Community, gfx942 listed |
| vLLM FP8 KV (AMD MI300X) | FP8 quantization | 8 | gfx942 (MI300X) | Production, ROCm 7.2+ |
| FlashInfer on ROCm | Optimized attention | FP16/FP8 | gfx942 (CDNA3/4) | v0.5.3, production |
| KIVI (baseline) | INT4 KV | 4 | CUDA | Published |
| **This work** | TurboQuant HIP/Triton for CDNA3 | 2–4 | **gfx942 (MI300X)** | Planned |

---

## 8. Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Wave64 vs Wave32 mismatch in domvox kernels | Medium | High | Verify `warpSize` read at runtime; test with `__AMDGCN_WAVEFRONT_SIZE__` |
| gfx942 MFMA intrinsics not used (leaving perf on table) | Medium | Medium | Profile with `rocprof --stats`; add `__builtin_amdgcn_mfma_f32_16x16x16f16` for rotation |
| Compression crossover at short seq_len barely visible | High | Low | Expected — document and focus benchmark on ≥ 32K tokens; capacity story remains strong |
| Triton kernel autotune takes long first run | Low | Low | Pre-cache with `TRITON_CACHE_DIR`; disable autotune after finding best config |
| ROCm 7.2 regression on quantized KV (observed on RDNA consumer) | Low | Medium | ROCm 7.2 on MI300X is production-tested; likely consumer-GPU-specific issue |
| vLLM TurboQuant integration is complex | Medium | Medium | Use llama.cpp fork as fallback; vLLM is stretch goal |
| Quality degradation at 2-bit (TQ2) | Medium | Low | Expected per paper; document and recommend 3-bit minimum |
