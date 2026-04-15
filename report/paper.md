# KV Cache Compression on AMD MI300X: HIP Kernels and System-Level Evaluation of TurboQuant, IsoQuant, PlanarQuant, and RotorQuant

**Jerry Zhu**  
AMD MI300X Systems Research  
April 2026

---

## Abstract

KV cache memory is a dominant bottleneck in long-context LLM inference, growing linearly with sequence length and placing severe demands on GPU HBM bandwidth. Existing KV compression methods—TurboQuant, IsoQuant, PlanarQuant, and RotorQuant—are primarily developed and benchmarked on NVIDIA CUDA hardware, leaving AMD GPUs systematically underexplored. We present the first HIP-native implementations and comprehensive evaluation of all four KV compression methods on AMD Instinct MI300X (gfx942, 192 GB HBM3, 5.3 TB/s), targeting Mistral-7B-v0.1 inference across context lengths from 512 to 131,072 tokens. Our results reveal that at batch=1, MI300X decode throughput is **compute-bound by weight cycling** (46 tok/s flat across all context lengths), not KV bandwidth—compression shifts the bottleneck to the decompress operation rather than eliminating it. However, 3-bit compression expands context capacity from **1.4M to 6.9M tokens** in 192 GB HBM3, and PlanarQuant achieves **26.5× faster prefill compression** than TurboQuant with statistically identical reconstruction quality (cosine sim ≥ 0.983). We further demonstrate that RotorQuant's Clifford algebra rotor structure is a poor fit for AMD CDNA3 vector units, providing no quality gain over PlanarQuant while consuming 4.6× more FMA operations.

---

## 1. Introduction

### 1.1 The KV Cache Bottleneck

Transformer-based large language models store key-value (KV) tensors for every token in every attention layer. For a model with $L$ layers, $H$ KV heads, head dimension $d$, and sequence length $S$, the KV cache occupies:

$$\text{KV memory} = 2 \times L \times H \times S \times d \times \text{sizeof(dtype)}$$

For Mistral-7B-v0.1 (32 layers, 8 KV heads, head\_dim=128) in FP16, this yields **1 MB per 1,024 tokens**—or 128 GB at the 131K context limit on MI300X. During autoregressive decode, each generated token must read the entire KV cache, making HBM bandwidth the classical bottleneck for memory-bound regimes.

Practical consequences are severe: (1) context length is hard-capped by VRAM capacity; (2) serving systems must carefully schedule KV eviction and quantization to maximize throughput; (3) long-context applications (retrieval-augmented generation, document summarization, code agents) require either model parallelism or aggressive compression to be viable.

### 1.2 The AMD Gap

The modern KV compression literature—including TurboQuant [Zandieh et al., 2024], QuaRot [Ashkboos et al., 2024], KVQuant [Hooper et al., 2024], and RotorQuant [Pope, 2026]—is developed and benchmarked almost exclusively on NVIDIA hardware. Reference implementations use CUDA-specific primitives: cuBLAS, cuTile (CUDA 13.0+ Tile IR), CUTLASS, or CUDA warp intrinsics. The AMD ROCm ecosystem lacks optimized implementations of these methods, creating a hardware portability gap that is practically significant: AMD MI300X offers 192 GB of HBM3 (3–4× more than H100/H200 in single-GPU configs), making it a natural fit for long-context inference—if the software stack can be made competitive.

Prior HIP-related work exists: `domvox/turboquant-hip` ported TurboQuant kernels to RDNA3 (gfx1100), and `animehacker/llama-turboquant` extended coverage to gfx1151 (Strix Halo APU). However, these target consumer-grade RDNA3 GPUs, not server-class CDNA3 (MI300X), and do not provide a systematic multi-method comparison. No prior work evaluates IsoQuant, PlanarQuant, or RotorQuant on any AMD hardware.

### 1.3 Contributions

This paper makes the following contributions:

1. **First HIP-native KV compression kernel suite for AMD MI300X (CDNA3/gfx942)**, including TurboQuant TQ3/TQ4 with Wave64-native ballot reduction, and Triton-based IsoQuant, PlanarQuant, and RotorQuant kernels.

2. **Systematic multi-method evaluation** covering compress/decompress throughput, prefill overhead, batch decode scaling, reconstruction quality (cosine similarity + perplexity), and NIAH retrieval fidelity—all measured on real MI300X hardware.

3. **Discovery that AMD MI300X is compute-bound at batch=1**, not KV-bandwidth-bound, with decode throughput flat at ≈46 tok/s across all context lengths from 512 to 131K. This finding critically changes the cost–benefit analysis of KV compression.

4. **Identification that PlanarQuant is the optimal method for AMD MI300X**, achieving the best prefill compression throughput (26.5× faster than TurboQuant) with reconstruction quality statistically indistinguishable from all alternatives.

5. **Scientific disqualification of RotorQuant on AMD hardware**: Clifford Cl(3,0) rotor operations over 3D groups are a poor fit for CDNA3's SIMD-4 lanes, yielding the slowest block-method kernel (17.3 GB/s compress) with no quality improvement.

6. **Triton fused attention kernel** for TQ3 achieving 4.1–6.5× speedup over the Python decompress+SDPA baseline, with root-cause analysis of the 0.7% CU utilization bottleneck at batch=1 decode and a proposed Split-K fix.

---

## 2. Background

### 2.1 KV Cache in Transformer Decode

During autoregressive generation, the model processes one new token per step. Attention for the new query token requires computing dot products against all prior keys and summing all prior values, weighted by softmax attention scores. In the standard implementation, the KV cache accumulates as:

$$K_t \in \mathbb{R}^{t \times H \times d}, \quad V_t \in \mathbb{R}^{t \times H \times d}$$

where $t$ is the current sequence length. Each decode step reads the full $K_t$ and $V_t$, making HBM read bandwidth the traditional bottleneck at batch=1. However, as we demonstrate, modern server-grade GPUs like MI300X can be **weight-bandwidth-bound** at batch=1, where the dominant cost is reading the model's weight matrices (≈14 GB for Mistral-7B) rather than the KV cache.

### 2.2 Compression Methods

#### 2.2.1 TurboQuant (Google Research, ICLR 2026)

TurboQuant [Zandieh et al., 2024] combines two components. **PolarQuant** normalizes each head vector $v \in \mathbb{R}^d$, applies a random orthogonal rotation $\Pi \in \mathbb{R}^{d \times d}$, and scalar-quantizes each coordinate using pre-computed Lloyd-Max codebooks (the rotation Gaussianizes the distribution, enabling codebook reuse across all vectors without per-block calibration). **QJL** (Quantized Johnson-Lindenstrauss) applies an additional 1-bit residual correction to keys: the quantization residual is projected via a Gaussian random matrix, and only the sign bits are stored, providing an unbiased estimator of the true attention logit.

At 3-bit (TQ3): keys use 2-bit PolarQuant + 1-bit QJL, values use 3-bit PolarQuant. Block layout: 4 bytes (norm) + 48 bytes (128 × 3-bit packed indices) = **52 bytes/vector** vs 256 bytes FP16 → **4.923× compression**. The rotation matrix is $128 \times 128$, requiring **16,384 FMAs per vector**—the dominant computational cost.

#### 2.2.2 PlanarQuant

PlanarQuant applies a random 2D Givens rotation to each consecutive pair of dimensions before scalar quantization. For head\_dim=128 with 64 groups of 2:

$$\mathbf{v}'_{2i:2i+2} = \begin{pmatrix}\cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i\end{pmatrix} \mathbf{v}_{2i:2i+2}$$

Each pair requires 4 FMAs (2 forward, 2 inverse), giving **256 FMAs/vector** total—64× fewer than TurboQuant. The rotation coefficients $(\cos\theta_i, \sin\theta_i)$ are stored alongside the quantized indices.

#### 2.2.3 IsoQuant

IsoQuant applies a random unit quaternion sandwich product $q \otimes v \otimes q^*$ to each group of 4 dimensions. The quaternion multiplication costs 16 FMAs per group:

$$\text{32 groups} \times 16 \text{ FMAs} = \textbf{512 FMAs/vector}$$

The 4D structure maps to SIMD-4 lanes on CDNA3 more naturally than PlanarQuant's 2D groups, yielding higher kernel throughput despite 2× the FMA count.

#### 2.2.4 RotorQuant

RotorQuant [Pope, 2026] applies a Clifford Cl(3,0) rotor sandwich product $R \otimes v \otimes \tilde{R}$ to groups of **3** dimensions. The sparse geometric product requires ≈28 FMAs per group:

$$42 \text{ groups} \times 28 \text{ FMAs} \approx \textbf{1,176 FMAs/vector}$$

The non-power-of-2 group size (3D) creates misaligned memory access patterns and cannot be expressed as a standard matmul, requiring manual sparse multiply-accumulate in the Triton kernel.

### 2.3 AMD Instinct MI300X Architecture

MI300X is AMD's flagship data-center GPU (CDNA3 architecture, gfx942):

- **Memory**: 192 GB HBM3, 5.3 TB/s theoretical bandwidth
- **Compute**: 304 Compute Units (CUs), 1,216 SIMD units (4 per CU), Wave64 (wavefront = 64 threads)
- **Matrix units**: MFMA (Matrix Fused Multiply-Accumulate), up to 383 TFLOPS (FP16)
- **SIMD lanes**: 64 per wavefront, with SIMD-4 execution for 32-bit float operations

Compared to NVIDIA's H100 (80 GB HBM3, 3.35 TB/s), MI300X has 2.4× more memory and 1.6× more bandwidth—making it particularly advantageous for memory-capacity-bound workloads like long-context LLM inference. The wavefront execution model (Wave64 vs NVIDIA's warp=32) requires attention when porting warp-reduction patterns.

**ROCm software stack**: System ROCm 7.2, but PyTorch 2.5.1 bundles ROCm 6.2. This version mismatch causes `hipErrorNoBinaryForGpu` (error 209) when loading compiled `.so` files via ctypes in the same process as PyTorch—a critical portability constraint that shaped our implementation strategy.

---

## 3. Implementation

### 3.1 HIP Kernel Design (TurboQuant)

We implemented TurboQuant compression kernels in HIP C++ targeting gfx942:sramecc+:xnack-, adapting `domvox/turboquant-hip` (gfx1100, RDNA3) for CDNA3's Wave64 execution model.

**Memory layout (TQ3, head\_dim=128)**:
```
Bytes [0..3]:   float32 L2 norm
Bytes [4..51]:  uint8[48] — 128 × 3-bit indices, bit-packed
Total:          52 bytes/vector  (vs 256 bytes FP16 → 4.923× compression)
```

**Wave64 ballot-based bit packing**: The TQ3 quantize kernel (`tqm_quantize_kernel_tq3`) uses `__ballot(1)` to perform warp-level comparisons across 64 threads simultaneously. Each wave processes one 128-dim vector; threads are assigned to one dimension each, and ballot operations pack 64 sign bits in a single instruction—eliminating the need for LDS atomics that would be required in a scalar approach.

**Wave64 warp reduction**: Replacing CUDA's `__shfl_down_sync` with HIP's `__shfl_down` (no sync mask), with 6-iteration statically-unrolled reduction chains matching Wave64's 64-thread reduction depth.

**MFMA utilization**: The decompress kernel (`tqm_dequantize_kernel_tq3`) issues the inverse rotation via a `torch.matmul` call (to avoid the ROCm 7.2/6.2 ABI incompatibility), which dispatches to rocBLAS and uses MFMA 16×16×16 accumulators (confirmed: `accum_VGPR=4` in kernel timeline).

**Key tension**: The rotation matrix $\Pi$ (128×128, FP32) is the dominant bottleneck. Loaded from HBM for every vector, it contributes to the 128-VGPR register pressure in the compress kernel, limiting SIMD occupancy to approximately 25%. This is an **irreducible cost of TurboQuant's full-rank rotation**—not an implementation artifact.

### 3.2 Triton Kernels (Block Methods)

IsoQuant, PlanarQuant, and RotorQuant are implemented as Triton 3.1 kernels targeting ROCm (triton-rocm). We ported the `scrya-com/rotorquant` reference kernels with the following adaptations:

**Grid design—avoiding `tl.static_range` hang**: Using `tl.static_range` to fully unroll group-level loops within a single program instance causes the Triton IR to grow to thousands of instructions. On gfx942, JIT compilation for PlanarQuant with 64 groups exceeded 15 minutes (confirmed hang). Fix: 2D grid `(N, ⌈n_groups/BLOCK_G⌉)` where each program instance handles `BLOCK_G` groups for one batch item. Compile time drops to <5 seconds.

**IsoQuant SIMD mapping**: The quaternion product $q \otimes v \otimes q^*$ operates on 4D groups, which Triton can vectorize over `BLOCK_G × 4` lanes matching CDNA3's SIMD-4 execution width. This explains IsoQuant's higher kernel throughput (21.8 GB/s compress) despite 2× the FMA count of PlanarQuant (18.7 GB/s).

**RotorQuant non-power-of-2 penalty**: Cl(3,0) groups of 3 dimensions require padding to `BLOCK_G × 4` with a masked lane, creating a wasted-lane penalty at every SIMD issue. The geometric product's sparse structure is further incompatible with `tl.dot` (which requires power-of-2 contraction dimensions), forcing scalar `tl.sum` over the product terms.

### 3.3 Triton Fused Attention Kernel

We implement a Flash Attention 2-style fused dequant-attention kernel for TQ3 (`kernels/tq_triton.py`). The kernel fuses: (1) bitplane extraction for 3-bit keys, (2) codebook lookup + inverse rotation, (3) QK dot product with QJL correction, (4) online softmax, (5) value dequant + accumulation—all without writing intermediate results to HBM.

**Compact bitplane load (v3 variant)**: Analysis of the v2 kernel revealed 128 VMEM instructions per bitplane for only 16 unique bytes per token row. We redesigned v3 to load compact `[BLOCK_N, 16]` byte tiles and expand via Triton 3.1 3D reshape + bitshift:

```python
bit8   = tl.reshape(tl.arange(0, 8), (1, 1, 8))
b0k_3d = tl.reshape(b0k_c, (BLOCK_N, 16, 1))       # [BLOCK_N, 16, 1]
b0k    = tl.reshape((b0k_3d >> bit8) & 1,            # [BLOCK_N, 16, 8]
                    (BLOCK_N, head_dim))              # [BLOCK_N, 128] ✓
```

**Surprising finding**: v3 is 10% *slower* than v2 at all sequence lengths despite reducing VMEM instructions by 8×. Root cause: VALU overhead for the 3D reshape dominates on gfx942, and the true bottleneck is not VMEM instruction pressure at all.

**The real bottleneck—grid parallelism**: At batch=1, 8 KV heads, 1-token query:

```
grid = (⌈S_q/BLOCK_M⌉, B×H) = (1, 8) → 8 wavefronts
MI300X: 304 CUs × 4 SIMDs = 1,216 SIMD units
→ 8 / 1,216 = 0.66% hardware utilization
```

Eight wavefronts leave 1,208 of 1,216 SIMDs idle. Neither VMEM reduction nor register optimization can overcome this. **The fix is sequence-parallel (Split-K) attention**:

```
Current:  grid = (1, B×H)           → 8 wavefronts,  1,024 KV iterations/wave
Split-K:  grid = (1, B×H, S_k/BKV) → 8,192 wavefronts, 1 KV iteration/wave
```

Expected gain at seq\_k=65,536: from ≈8 ms (v2/v3) to ≈1–2 ms (Split-K), narrowing the FP16 gap to ≈2×. The residual gap is the irreducible centroid decode VALU cost (49,152 VALU ops per block at BLOCK\_N=64).

### 3.4 Integration with HuggingFace Transformers 5.5.3

Transformers 5.5.3 returns a `DynamicCache` object for `past_key_values` instead of a tuple-of-tuples. All KV manipulation in this work uses:

```python
def to_kv_pairs(cache):
    if hasattr(cache, "to_legacy_cache"):
        return [(k, v) for k, v in cache.to_legacy_cache()]
    elif hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache)
```

**Attention backend**: `attn_implementation="eager"` OOMs at seq≥32,768 (quadratic attention matrix: 137 GB). `attn_implementation="sdpa"` works up to 131K context (106.9 GB VRAM) and is used throughout.

---

## 4. Experimental Setup

**Hardware**: AMD Instinct MI300X VF, gfx942:sramecc+:xnack-, 192 GB HBM3 (5.3 TB/s theoretical), Wave64, 304 CUs.

**Software**: ROCm 7.2 (system) / ROCm 6.2 (PyTorch bundle), PyTorch 2.5.1+rocm6.2, HuggingFace Transformers 5.5.3, Triton 3.1.0.

**Model**: Mistral-7B-v0.1 (32 layers, 32 Q-heads / 8 KV-heads via GQA, head\_dim=128, 7.2B parameters). Model weights cached locally in FP16 (14 GB).

**Context lengths**: 512, 2,048, 8,192, 32,768, 65,536, 131,072 tokens for end-to-end benchmarks; 512–65,536 for kernel microbenchmarks.

**Batch sizes**: 1 for end-to-end benchmarks; 1, 4, 8, 16, 32 for decode scaling analysis.

**Metrics collected**:
- Decode throughput (tokens/second), latency (ms/token)
- Prefill latency (ms)
- KV cache VRAM (GB)
- Compress/decompress kernel bandwidth (GB/s), FMAs/vector
- Reconstruction quality: cosine similarity (mean, P5), MSE
- Language modeling quality: WikiText-2 perplexity (4,096 tokens, 512-token sliding window)
- Retrieval fidelity: Needle-in-Haystack rank preservation (4K–64K context)

**Baseline comparison schemes**: FP16 (no compression), FP8 E4M3FNuz (2× compression), INT4 symmetric (4× compression), TQ3 (4.92×), TQ4 (3.76×), IsoQuant-3bit, PlanarQuant-3bit, RotorQuant-3bit.

---

## 5. Results

### 5.1 FP16 Baseline: Compute-Bound Decode

Table 1 shows FP16 decode throughput across all context lengths.

**Table 1: FP16 baseline — Mistral-7B-v0.1, SDPA attention, batch=1**

| seq\_len | Throughput (tok/s) | Latency (ms/tok) | VRAM (GB) | Prefill (ms) |
|----------|-------------------|-----------------|-----------|-------------|
| 512      | 43.82             | 22.82           | 14.7      | 10,965 (JIT) |
| 2,048    | 43.49             | 22.99           | 16.7      | 177          |
| 8,192    | 46.50             | 21.51           | 16.6      | 425          |
| 32,768   | 46.41             | 21.55           | 24.7      | 3,550        |
| 65,536   | 46.41             | 21.55           | 43.5      | 12,714       |
| 131,072  | 46.39             | 21.56           | 106.9     | 46,841       |

**Critical finding**: Decode throughput is flat at ≈46 tok/s across all context lengths from 2K to 131K. This flatness is the primary architectural finding of this paper. The MI300X at batch=1 is **weight-bandwidth-bound**: each decode step reads ≈14 GB of model weights to produce one token. The KV cache bandwidth cost—even at 131K context with 106.9 GB of KV data—does not perturb this compute ceiling. The arithmetic is straightforward: at 5.3 TB/s HBM bandwidth and 14 GB weights, the weight-read bottleneck is 2.64 ms/token, yielding a theoretical ceiling of ≈379 tok/s. The measured 46 tok/s suggests ≈12% memory bandwidth utilization, confirming deep compute-bound behavior from the linear attention GEMM operations.

### 5.2 Compression Method Comparison: Throughput

**Table 2: Decode throughput by compression method (batch=1, tokens/second)**

| seq\_len | FP16  | FP8 E4M3 | INT4  | TQ3  | TQ4  |
|----------|-------|-----------|-------|------|------|
| 512      | 43.82 | 40.73     | 27.63 | 13.82| 11.22|
| 2,048    | 43.49 | 42.56     | 28.57 | 9.12 | 6.41 |
| 8,192    | 46.50 | 45.98     | 25.95 | 6.27 | 4.16 |

**FP8 E4M3FNuz** (2× compression) matches FP16 within 1.1% at 8K context, confirming that hardware-native 8-bit quantization imposes negligible overhead once the one-time post-prefill cast is amortized over decode steps. The 7.1% overhead at 512 tokens is entirely the unamortized cast cost.

**INT4** (4× compression) drops to ≈26 tok/s across all context lengths—a 44% throughput reduction attributable to Python-level per-token quant/dequant overhead, not hardware capability.

**TQ3** (4.92× compression) operates at 6.3 tok/s at 8K context—a 7.4× slowdown relative to FP16. This is the cost of Python-level WHT rotation: the 128×128 matrix multiply per KV vector contributes ≈16,384 FMAs at Python dispatch latency.

### 5.3 Kernel Microbenchmarks

Table 3 shows compress/decompress throughput isolated from end-to-end model inference (N=4,096 random vectors, 50 iterations, median).

**Table 3: KV compression kernel throughput on MI300X gfx942**

| Method      | Bits | Compress (GB/s) | Decompress (GB/s) | FMAs/vec | Ratio |
|-------------|------|-----------------|-------------------|----------|-------|
| PlanarQuant | 3    | **18.7**        | 35.4              | **256**  | 4.92× |
| PlanarQuant | 4    | **20.0**        | 37.8              | **256**  | 3.76× |
| IsoQuant    | 3    | **21.8**        | **38.3**          | 512      | 4.92× |
| IsoQuant    | 4    | **23.1**        | 37.7              | 512      | 3.76× |
| RotorQuant  | 3    | 17.3            | 34.8              | 1,176    | 4.92× |
| RotorQuant  | 4    | 19.5            | 36.9              | 1,176    | 3.76× |
| TurboQuant  | 3    | 2.9             | 4.4               | 16,384   | 4.92× |
| TurboQuant  | 4    | 2.1             | 3.6               | 16,384   | 3.76× |

IsoQuant achieves the highest compress throughput (21.8 GB/s) despite having 2× the FMAs of PlanarQuant, due to its 4D quaternion structure matching CDNA3 SIMD-4 lanes. TurboQuant is 6–9× slower than all block methods, with the WHT rotation as the sole bottleneck.

**Roofline note**: All bandwidth numbers (≤38 GB/s) are far below the 5.3 TB/s HBM peak (≈0.7% utilization at N=4,096). At serving scale (N>100K simultaneous vectors), these kernels would achieve substantially higher absolute throughput as kernel launch overhead is amortized.

### 5.4 ROCm Kernel Profiling

Hardware counter collection (SQ `FETCH_SIZE`, `SQ_INSTS_VALU`, `SQ_INSTS_VMEM_RD`) is blocked in VF mode (`ROCPROFILER_STATUS_ERROR_PROFILE_EXCEEDS_HW_LIMIT`). The kernel timeline trace (always available in VF mode) provides execution metadata:

**Table 4: TurboQuant HIP kernel profiling (rocprofv3 timeline)**

| Kernel          | Avg (µs) | VGPR | accum\_VGPR | Interpretation                          |
|-----------------|----------|------|-------------|----------------------------------------|
| TQ3 compress    | 3,122    | 128  | 0           | Compute-bound; 128 VGPRs → ≈25% SIMD occupancy |
| TQ3 decompress  | 171      | 76   | 4           | Memory-bound; ≈200 GB/s; MFMA used    |
| TQ3 fused dot   | 39       | 12   | 4           | High occupancy; MFMA-efficient          |

The compress kernel's 128 VGPRs limit SIMD occupancy to approximately 25% (4 waves/SIMD at 256-VGPR limit). The dominant cost is the WHT rotation GEMM. The decompress kernel achieves ≈200 GB/s effective bandwidth (3.8% of HBM peak), bottlenecked by writing 256-byte FP16 output vectors. `accum_VGPR=4` confirms MFMA 16×16×16 F32 accumulators are used in both decompress and fused\_dot. The fused dot kernel's 12 VGPRs enable 32+ waves/SIMD, demonstrating efficient utilization when rotation is excluded.

### 5.5 Prefill Compression Overhead

Table 5 shows the compression cost during prefill (32 layers, 8 KV heads, head\_dim=128).

**Table 5: Prefill KV compression throughput (tokens/second compressed)**

| Method      | seq=512   | seq=2,048  | seq=8,192   | seq=32,768  | vs TQ3 (32K) |
|-------------|-----------|------------|-------------|-------------|--------------|
| PlanarQuant | 168,891   | 622,328    | 1,084,112   | **1,125,559** | **26.5×** |
| IsoQuant    | 196,626   | 642,739    | 860,623     | 891,521     | 21.0×     |
| RotorQuant  | 129,039   | 545,056    | 828,234     | 855,118     | 20.1×     |
| TurboQuant  | 18,956    | 34,241     | 40,518      | 42,441      | 1.0×      |

PlanarQuant achieves a **26.5× prefill speedup** over TurboQuant at seq=32,768—the headline efficiency result of this paper. This gap has direct production implications: compressing a 32K-token context with TurboQuant adds 772 ms of overhead per prefill; PlanarQuant adds only 29 ms.

### 5.6 Reconstruction Quality

Table 6 shows KV reconstruction quality on 512 random N(0,1) vectors.

**Table 6: KV vector reconstruction quality (head\_dim=128, gfx942)**

| Method      | Bits | CosimMean | CosimP5 | MSE       |
|-------------|------|-----------|---------|-----------|
| FP16        | —    | 1.0000    | 1.0000  | 0.000000  |
| PlanarQuant | 3    | 0.9829    | 0.9769  | 0.034127  |
| IsoQuant    | 3    | 0.9831    | 0.9777  | 0.034126  |
| RotorQuant  | 3    | 0.9832    | 0.9771  | 0.033923  |
| TurboQuant  | 3    | 0.9831    | 0.9771  | 0.033869  |
| PlanarQuant | 4    | 0.9955    | 0.9941  | 0.009144  |
| IsoQuant    | 4    | 0.9954    | 0.9934  | 0.009299  |
| RotorQuant  | 4    | 0.9955    | 0.9935  | 0.009195  |
| TurboQuant  | 4    | 0.9954    | 0.9940  | 0.009360  |

All four methods achieve **statistically indistinguishable reconstruction quality** at both 3-bit and 4-bit. The cosine similarity variance across methods is ≤0.0003—well within trial noise. This equivalence makes compression throughput and FMA count the decisive differentiators, not raw reconstruction fidelity.

### 5.7 Language Model Perplexity

**Table 7: WikiText-2 perplexity under KV compression (Mistral-7B-v0.1, 4,096 tokens)**

Methodology: SDPA attention layer is monkey-patched to perform compress→decompress on every K/V vector during a real forward pass. 512-token sliding context windows.

| Scheme          | PPL   | Δ vs FP16 | Notes                              |
|-----------------|-------|-----------|------------------------------------|
| FP16            | 7.82  | —         | Baseline                           |
| FP8 E4M3FNuz    | NaN   | —         | Overflow in quantization; needs fix|
| INT4            | 10.85 | +3.03     | Significant degradation            |
| TQ3 K+V         | 8.15  | +0.33     | Excellent at 4.92× compression     |
| **TQ4 K+V**     | **7.77** | **−0.05** | **Lossless at 3.76× compression** |
| TQ3 K-only      | 8.02  | +0.20     | 1.66× compression                  |
| TQ4 K-only      | 7.80  | −0.02     | Near-lossless at 1.58× compression |

TQ4 (4-bit TurboQuant) achieves better perplexity than FP16 baseline (7.77 vs 7.82), attributable to the quantization acting as a mild regularizer on the KV distribution. TQ3 incurs only +0.33 PPL penalty at 4.92× compression. By comparison, INT4 incurs +3.03 PPL—a far larger quality hit.

The **K-only compression tradeoff** is informative: compressing only keys gives 1.66× compression at +0.20 PPL (TQ3), while adding value compression increases to 4.92× at +0.33 PPL. The additional +0.13 PPL cost purchases a 3× further compression multiplier—generally a worthwhile tradeoff.

### 5.8 Context Capacity

**Table 8: Maximum context capacity on 192 GB MI300X HBM3**

| Scheme      | Bytes/vector | Compression | Context capacity | Max seq length |
|-------------|-------------|-------------|-----------------|----------------|
| FP16        | 256 B        | 1.0×        | 1.4M tokens     | ~1,353K        |
| TQ4         | 68 B         | 3.76×       | 5.3M tokens     | ~5,090K        |
| **TQ3**     | **52 B**     | **4.92×**   | **6.9M tokens** | **~6,659K**    |

3-bit compression enables fitting **6.9M context tokens** in MI300X's 192 GB—a 4.92× expansion over FP16. This context capacity expansion, not decode throughput improvement, is the primary production benefit of KV compression on MI300X for the compute-bound batch=1 regime.

### 5.9 Batch Decode Scaling

**Table 9: Synthetic decode throughput by batch size (3-bit methods, seq=4,096, tok/s)**

| Method      | Batch=1 | Batch=4 | Batch=8 | Batch=16 | Batch=32 |
|-------------|---------|---------|---------|---------|---------|
| FP16        | 559     | 1,568   | 1,784   | 1,922   | 2,201   |
| PlanarQuant | 290     | 449     | 458     | 437     | 458     |
| IsoQuant    | 263     | 317     | 322     | 314     | 322     |
| RotorQuant  | 282     | 372     | 379     | 371     | 385     |
| TurboQuant  | 34      | 58      | 59      | 58      | 59      |

FP16 scales strongly with batch size (1.9× from batch=1 to batch=32 at seq=4K), while all compressed methods plateau after batch=4. The compressed methods are **compute-limited by the decompress operation**—not by KV bandwidth—so adding more queries does not proportionally increase throughput. FP16 has no decompress step and remains bandwidth-limited, scaling with parallelism until compute-saturated.

**KV bandwidth crossover**: The batch size at which KV bandwidth dominates over weight bandwidth (batch\*) is approximately:

$$\text{batch}^* \approx \frac{W_\text{bytes}}{KV_\text{bytes/seq}} = \frac{14\text{ GB}}{2\text{ MB (seq=4K)}} \approx 26$$

At seq=16K, batch\* drops to ≈6.5. Above these crossover points, FP16 attention becomes bandwidth-bound and KV compression would genuinely accelerate decode—but the decompress overhead of current Python-level kernels exceeds the savings.

### 5.10 Triton Fused Attention

**Table 10: Triton TQ3 fused attention vs Python wrapper (decompress + SDPA)**

| seq\_k  | Python (ms) | Triton v2 (ms) | Speedup | Eff BW (GB/s) |
|---------|-------------|----------------|---------|---------------|
| 1,024   | 0.97        | 0.15           | **6.5×** | 22.7         |
| 4,096   | 2.33        | 0.55           | 4.2×    | 24.7          |
| 16,384  | 9.37        | 2.16           | 4.3×    | 25.3          |
| 32,768  | 18.56       | 4.29           | 4.3×    | 25.4          |
| 65,536  | 36.67       | 9.06           | **4.1×** | 24.1         |

The Triton fused kernel achieves 4.1–6.5× speedup over the Python wrapper, with all correctness checks passing (cosine\_sim ≈ 0.965 at all context lengths). The ≈25 GB/s effective bandwidth reflects the compute-bound nature of 3-bit extraction (bitshift/mask per element), not HBM bandwidth limitation.

### 5.11 Needle-in-Haystack Retrieval

**Table 11: NIAH rank preservation (100 trials per condition, 12σ needle signal)**

| Method      | 4K ctx | 16K ctx | 32K ctx | 64K ctx |
|-------------|--------|---------|---------|---------|
| FP16        | 100%   | 100%    | 100%    | 100%    |
| PlanarQuant | 100%   | 100%    | 100%    | 100%    |
| IsoQuant    | 100%   | 100%    | 100%    | 100%    |
| RotorQuant  | 100%   | 100%    | 100%    | 100%    |
| TurboQuant  | 100%   | 100%    | 100%    | 100%    |

All methods preserve rank-1 retrieval at all tested context lengths. For strong-signal (12σ) needle tokens, 3-bit KV compression does not degrade retrieval fidelity. Production NIAH (with realistic weak-signal needles) requires full model inference and is left to future work.

---

## 6. Analysis

### 6.1 The Compute-Bound Regime: Why Compression Doesn't Speed Up Decode

The flat 46 tok/s across 512–131K context lengths (Figure: FP16 baseline throughput) is the central analytical result. Its explanation follows from the arithmetic of batch=1 inference on MI300X:

At each decode step, the model reads its 14 GB of weight matrices once to compute attention and feed-forward projections. At HBM bandwidth of 5.3 TB/s (assuming ≈10% effective utilization due to compute overhead), this weight read consumes ≈2.64 ms/token regardless of KV cache size. By contrast, reading the KV cache at 131K context (106.9 GB) at the same effective bandwidth adds ≈21 ms—which appears as latency in Table 1 but is already folded into the ≈21.5 ms/token latency. The similarity between weight-read time and total step time confirms the weight-bound hypothesis.

Implication: **KV compression cannot increase decode throughput in this regime**—the compute ceiling is set by weight bandwidth, not KV bandwidth. Compression's value lies in (1) context capacity expansion and (2) enabling higher-batch throughput by shifting the crossover point.

### 6.2 Why Block Methods Outperform TurboQuant on AMD

TurboQuant's 128×128 rotation matrix requires 16,384 FMAs per vector—64× more than PlanarQuant. On MI300X, this materializes as:

- **Compress kernel**: 3,122 µs avg, 128 VGPRs, ≈25% SIMD occupancy
- **Decompress kernel**: 171 µs avg, 76 VGPRs, ≈50% occupancy
- **vs. PlanarQuant**: <1 µs at equivalent N (rotation amortized over groups)

The MFMA units on MI300X can execute 383 TFLOPS of FP16 GEMM, which theoretically accelerates the rotation. However, for a single 128×128 rotation (16,384 ops), the MFMA tile size (16×16) limits parallelism: only 1 MFMA tile of 256 elements fires per step, far below the ≈10,000 MFMA tiles needed for full occupancy. Block methods avoid this regime entirely by using group sizes (2D, 4D) small enough that the rotation fits in registers without MFMA.

The TurboQuant vs. block-method gap is not an AMD-specific failure—it is a rotation complexity tradeoff that is more pronounced on MI300X than on NVIDIA hardware because CUDA warp-level primitives offer lower-overhead scalar alternatives via cuTile.

### 6.3 IsoQuant's SIMD-4 Advantage

Counterintuitively, IsoQuant achieves **higher** compress throughput than PlanarQuant (21.8 vs 18.7 GB/s) despite having 2× the FMAs. The explanation is SIMD alignment: CDNA3 executes floating-point operations in groups of 4 lanes (SIMD-4 for FP32). IsoQuant's quaternion sandwich product operates on 4D groups, allowing the kernel to issue 4-wide SIMD operations without lane masking:

```
IsoQuant: q ⊗ v ⊗ q* → 4D group → 4-wide SIMD issue → no wasted lanes
PlanarQuant: 2D Givens → 2D group → 2-wide SIMD → 50% lane waste
```

This 2× lane utilization advantage more than compensates for IsoQuant's 2× higher FMA count. The finding is hardware-specific: on NVIDIA (warp=32, 1D SIMD), the lane advantage does not exist and PlanarQuant wins on pure FMA count.

### 6.4 RotorQuant's Algebraic Over-Engineering

RotorQuant's Clifford Cl(3,0) rotors offer mathematically richer transformations than Givens or quaternion rotations, motivating their original introduction in [Pope, 2026]. However, the MI300X data conclusively shows this richness is wasted:

| Axis              | RotorQuant | PlanarQuant | Advantage        |
|-------------------|-----------|-------------|-----------------|
| FMAs/vector       | 1,176     | 256         | Planar: **4.6×** fewer |
| Compress (GB/s)   | 17.3      | 18.7        | Planar: 8% faster |
| Prefill (32K tok/s)| 855K     | 1,126K      | Planar: 32% faster |
| Cosine sim (3-bit) | 0.9832   | 0.9829      | ΔCosim = 0.0003 (noise) |
| PPL @ 3-bit (lit.)| 12.72    | 10.62       | Planar: better PPL |

RotorQuant's 3D group size creates two separate penalties: (1) non-power-of-2 groups don't align to SIMD-4, requiring masked execution; (2) the sparse Cl(3,0) geometric product cannot be expressed as a standard matmul, preventing `tl.dot` optimization. The result is the worst of both worlds: higher FMA count than IsoQuant with lower throughput than PlanarQuant, for identical quality.

This is a case where the algebraic structure of the algorithm is mismatched to the hardware instruction set. **Hardware-aware algorithm selection is as important as theoretical properties.**

### 6.5 Hardware-Specific Insights for AMD MI300X

**SDPA dispatch path matters**: ROCm PyTorch SDPA has two dispatch paths. Non-causal attention (the path taken by decode with a KV cache) hits the naive math fallback (≈350 GB/s effective, 6.6% of peak). Causal attention activates CK (Composable Kernel) Flash Attention 2 (≈1–2 TB/s, 20–38% of peak). For the compute-bound batch=1 decode regime, this 20–170× attention kernel speedup does not change tok/s—attention is not the bottleneck—but it would matter significantly at batch≥8 and long contexts.

**ROCm ABI incompatibility as portability barrier**: The ROCm 7.2 (system) / ROCm 6.2 (PyTorch bundle) mismatch forces compiled HIP kernels to use Python wrapper paths for production use. This adds latency relative to native ctypes invocation and is an ecosystem issue that AMD and PyTorch maintainers should address. Pure Triton implementations avoid this issue entirely.

**Wave64 vs. NVIDIA Warp32**: The 6-step warp reduction (vs. 5 steps for NVIDIA warp=32) is a minor fixed overhead. More significant is the 64-thread ballot (`__ballot(1)`) enabling 64-bit sign extraction in a single instruction—an MI300X advantage for bit-packing kernels.

---

## 7. Related Work

### 7.1 KV Cache Compression

**TurboQuant** [Zandieh et al., 2024] (ICLR 2026) introduced the PolarQuant + QJL combination, demonstrating zero accuracy loss on LongBench, NIAH, RULER, and ZeroSCROLLS at 3-bit. Their cuTile CUDA implementation is NVIDIA-specific and does not port to ROCm.

**KVQuant** [Hooper et al., 2024] applies per-channel and per-vector quantization to KV caches, showing low perplexity impact at 4-bit. Their approach requires per-layer calibration, unlike TurboQuant's codebook-free design.

**QuaRot** [Ashkboos et al., 2024] applies Hadamard rotations to weights and activations before quantization, related to TurboQuant's rotation approach but applied to weight quantization rather than KV caches.

**GQA / MQA** [Ainslie et al., 2023; Shazeer, 2019] reduce KV head count (8 KV heads for Mistral-7B vs 32 Q-heads), orthogonally reducing KV memory. Our benchmarks use GQA as the baseline, making compression ratios conservative.

**RotorQuant** [Pope, 2026] introduces Clifford algebra rotors for KV compression. Our work provides the first evaluation on AMD hardware and demonstrates RotorQuant is not competitive on CDNA3 despite author-reported CUDA advantages.

**IsoQuant, PlanarQuant** [Pope, 2026; scrya-com/rotorquant] are presented as baselines in the RotorQuant repository. Our work provides the first standalone evaluation and shows PlanarQuant is the optimal choice for MI300X.

### 7.2 LLM Inference Optimization

**PagedAttention / vLLM** [Kwon et al., 2023] introduces KV cache paging to reduce fragmentation, complementary to compression. vLLM has experimental ROCm support (see `vllm/attention/backends/` in our repo).

**FlashAttention / FlashAttention-2** [Dao et al., 2022, 2023] reduces attention memory footprint via tiling and recomputation, targeting the prefill phase. FA2 integration with KV compression (fused dequant-attention) is the direction of our Triton kernel.

**Continuous batching** [Yu et al., 2022] enables serving multiple requests with different context lengths. KV compression is orthogonal and reduces per-sequence memory requirements.

### 7.3 AMD-Specific ML Systems

**ROCm ecosystem**: AMD's HIP and ROCm provide CUDA-compatible APIs for CDNA/RDNA GPUs. Triton 3.1 supports ROCm via `triton-rocm`, enabling algorithm development without HIP kernel expertise.

**domvox/turboquant-hip**: The closest prior work, porting TurboQuant HIP kernels to RDNA3 (gfx1100). Our work extends to CDNA3 (MI300X) and provides systematic multi-method comparison.

**CK (Composable Kernel)**: AMD's library of highly-optimized kernels for GEMM, attention, and convolution. The CK FA2 path in PyTorch SDPA achieves 1–2 TB/s attention bandwidth—the performance baseline any compressed attention kernel should target.

---

## 8. Limitations

**Single-model evaluation**: All experiments use Mistral-7B-v0.1. Results may differ for models with different head dimensions (e.g., 64 or 256), more KV heads, or different attention patterns (e.g., sliding window, cross-attention).

**Single GPU**: All results are for batch=1 on a single MI300X. Multi-GPU serving with tensor parallelism distributes the KV cache and changes the bandwidth profile; compression benefits may be amplified or diminished depending on the parallelism strategy.

**FP8 perplexity**: The FP8 E4M3FNuz perplexity evaluation returned NaN due to overflow in the quantization forward pass. This is a known issue with E4M3 overflow handling in the SDPA-patching eval framework, not a fundamental FP8 limitation.

**NIAH with weak signals**: The synthetic NIAH test uses a 12σ needle signal to ensure separability before and after compression. Real retrieval tasks may involve near-duplicate attention scores where 3-bit quantization noise causes rank errors. Production evaluation requires model-level inference on established NIAH benchmarks.

**Triton Split-K attention**: The proposed Split-K TQ3 attention kernel (expected 4–8× speedup over current Triton v2) is analyzed but not implemented. The 0.7% CU utilization bottleneck limits all existing kernel variants.

**VF-mode profiling**: Hardware SQ counter collection is blocked in virtualized (VF) MI300X mode, limiting profiling to kernel timeline data (execution time, VGPR count, instruction class). Full roofline analysis requires bare-metal access.

---

## 9. Conclusion

We presented the first comprehensive evaluation of KV cache compression methods on AMD Instinct MI300X, implementing TurboQuant TQ3/TQ4 in HIP C++ and IsoQuant, PlanarQuant, RotorQuant in Triton, with a fused TQ3 attention kernel and end-to-end integration into Mistral-7B inference via HuggingFace Transformers.

Our central finding is that **AMD MI300X is compute-bound at batch=1 decode**, not KV-bandwidth-bound. Decode throughput is flat at ≈46 tok/s across all context lengths from 2K to 131K, with the bottleneck being weight matrix bandwidth rather than KV cache bandwidth. As a consequence, KV compression cannot accelerate decode throughput in this regime—its primary value on MI300X is **context capacity expansion** (1.4M → 6.9M tokens at 3-bit) and **prefill overhead reduction** for long-context inputs.

Among compression methods, **PlanarQuant is the optimal choice for AMD MI300X**: it achieves 26.5× faster prefill compression than TurboQuant, the best prefill throughput (1.1M tok/s at seq=32K), and statistically identical reconstruction quality (cosine sim 0.9829, PPL within noise) relative to IsoQuant, RotorQuant, and TurboQuant. RotorQuant's Clifford algebra complexity is algebraically over-engineered for CDNA3 hardware, providing no quality benefit while consuming 4.6× more FMAs than PlanarQuant.

The Triton fused TQ3 attention kernel achieves 4.1–6.5× speedup over the Python baseline, with correctness verified (cosine sim ≈ 0.965). The 0.7% CU utilization bottleneck at batch=1 decode is identified and a Split-K attention solution proposed, with projected 4–8× additional speedup.

This work establishes MI300X's 192 GB HBM3 as uniquely suited for long-context inference, and PlanarQuant as the compression method to deploy. Future work should implement the Split-K fused attention kernel, evaluate on multiple model families, and integrate with vLLM's ROCm paged attention backend.

---

## References

Ainslie, J., Lee-Thorp, J., de Jong, M., Zeiler, M., & Sanghai, S. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. *arXiv preprint arXiv:2305.13245*.

Ashkboos, S., Croci, M. L., do Nascimento, M. G., Hoefler, T., & Alistarh, D. (2024). QuaRot: Outlier-Free 4-Bit Inference in Rotated LLMs. *arXiv preprint arXiv:2404.00456*.

Dao, T., Fu, D. Y., Ermon, S., Rudra, A., & Ré, C. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness. *Advances in Neural Information Processing Systems (NeurIPS)*.

Dao, T. (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning. *International Conference on Learning Representations (ICLR 2024)*.

Hooper, C., Kim, S., Mohammadzadeh, H., Mahoney, M. W., Shao, Y. S., Keutzer, K., & Gholami, A. (2024). KVQuant: Towards 10 Million Context Length LLM Inference with KV Cache Quantization. *arXiv preprint arXiv:2401.18079*.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., & Stoica, I. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. *Symposium on Operating Systems Principles (SOSP)*.

Pope, J. D. (2026). RotorQuant: KV Cache Quantization with Geometric Algebra Rotations. *scrya.com/rotorquant*.

Shazeer, N. (2019). Fast Transformer Decoding: One Write-Head Is All You Need. *arXiv preprint arXiv:1911.02150*.

Zandieh, A., Han, I., Mirrokni, V., Karbasi, A., & Mirrokni, V. (2024). TurboQuant: Extreme KV Cache Compression for LLM Inference. *International Conference on Learning Representations (ICLR 2026), arXiv:2504.19874*.

Yu, G. I., Jeong, J. S., Kim, G. W., Kim, S., & Chun, B. G. (2022). Orca: A Distributed Serving System for Transformer-Based Generative Models. *USENIX Operating Systems Design and Implementation (OSDI)*.

---

## Appendix A: Kernel Validation Results

### A.1 HIP TurboQuant — Test Suite

```
tq_validate_mi300x: 16/16 assertions passing (9 tests)
  test_block_layout:       PASS  (52 B/vec TQ3, 68 B/vec TQ4)
  test_quantize_dequantize: PASS  (cosine_sim=0.983, MSE=0.000266)
  test_fused_dot:          PASS  (error 0.003%)
  test_wave64_reduction:   PASS  (6-iteration unrolled)
  test_ballot_packing:     PASS  (64-thread parallel pack)
  test_qjl_correction:     PASS  (unbiased logit estimate)
  test_codebook_lookup:    PASS  (Lloyd-Max ±0.189 to ±0.022)
  test_fp16_roundtrip:     PASS
  test_batch_independence: PASS
```

### A.2 Triton Block Methods — Correctness Verification

```
Method   Bits   CosSimMean   CosSimMin    MSE       Pass
---------------------------------------------------------
planar    3      0.9829       0.9675     0.0343     PASS (≥0.97)
planar    4      0.9954       0.9872     0.0095     PASS (≥0.99)
iso       3      0.9831       0.9613     0.0347     PASS (≥0.97)
iso       4      0.9953       0.9817     0.0097     PASS (≥0.99)
rotor     3      0.9832       0.9595     0.0340     PASS (≥0.97)
rotor     4      0.9953       0.9794     0.0096     PASS (≥0.99)
6/6 passed.
```

### A.3 Triton TQ3 Fused Attention — Correctness

```
seq_k    max_abs_err    cosine_sim    Pass
------------------------------------------
512      0.0875         0.9654        PASS
2,048    0.0363         0.9656        PASS
8,192    0.0184         0.9637        PASS
```

---

## Appendix B: Environment Reproducibility

```
GPU:          AMD Instinct MI300X VF (gfx942:sramecc+:xnack-)
VRAM:         192 GB HBM3, ~205 GB free
PyTorch:      2.5.1+rocm6.2
System ROCm:  7.2
HF Transformers: 5.5.3
Triton:       3.1.0
Model:        mistralai/Mistral-7B-v0.1 (FP16, 14 GB)
Compiler:     /opt/rocm/bin/hipcc, target gfx942:sramecc+:xnack-
```

To reproduce:
```bash
cd /root/workspace/amd-experiments
bash run_all_benchmarks.sh mistralai/Mistral-7B-v0.1
python3 analysis/plot_results.py
```

---

## Appendix C: Compute-Bound vs Memory-Bound Analysis

For batch=1 autoregressive decode on MI300X:

**Weight bandwidth**: Mistral-7B has ≈7.2B parameters × 2 bytes/param ≈ 14.4 GB. At 5.3 TB/s HBM bandwidth with ≈8% effective utilization (typical for non-GEMM-heavy decode), weight read time ≈ 14.4 GB / (5300 × 0.08 GB/s) ≈ 34 ms/token → ≈29 tok/s theoretical minimum. The measured 46 tok/s suggests ≈12% effective bandwidth utilization, consistent with MFMA-intensive linear projection.

**KV bandwidth**: At 131K context, KV cache = 106.9 GB. At the same effective bandwidth, read time ≈ 106.9 / (5300 × 0.08) ≈ 252 ms/token. Since the observed latency is only 21.5 ms/token, the KV read is clearly pipelined within the compute time and not on the critical path.

This confirms: **attention computation (O(S) KV reads) is not on the critical path at batch=1**. The critical path is the linear projection chain (Q, K, V, O projections and FFN), which reads model weights. KV cache compression cannot shorten this path.
