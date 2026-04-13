---
name: Next-Gen KV Benchmark Plan
overview: A detailed benchmarking and engineering improvement plan for the next AMD ROCm KV-cache compression report, extending TurboQuant-only coverage to a four-way comparison of TurboQuant, IsoQuant, PlanarQuant, and RotorQuant on MI300X-class hardware, with RotorQuant included specifically to prove empirically that the Clifford rotor overhead is not justified by quality gains.
todos: []
isProject: false
---

# Next-Generation KV-Cache Compression Benchmark Plan
## AMD ROCm / MI300X Focus — Technical Planning Memo, April 2026

---

## 1. Executive Recommendation

### Method Inclusion Decision

**Include all four methods. RotorQuant is included specifically to be falsified.**

| Method | Include? | Role | Rationale |
|---|---|---|---|
| TurboQuant (TQ3/TQ4) | Yes — primary baseline | Incumbent to beat | Existing MI300X HIP + Triton kernels. Strong coverage already exists. |
| IsoQuant (iso3/iso4) | Yes — primary contender | Best decode speed + quality | Quaternion 4D group, 512 FMAs vs 16,384 for WHT. Triton kernel exists in rotorquant repo. Production llama.cpp. |
| PlanarQuant (planar3/planar4) | Yes — primary contender | Best 3-bit quality overall | Givens 2D group, 256 FMAs. Best PPL at 3-bit (10.12 vs RotorQuant 12.22). Production llama.cpp. |
| RotorQuant (rotor3/rotor4) | Yes — included to be disproved | "Algebraic overkill" control | Clifford Cl(3,0), ~2,400 FMAs, 372 params. Triton-only, Research stage. Existing data already shows it loses to Planar/Iso on PPL AND speed. Prove it rigorously on AMD. |

### The Narrative RotorQuant Enables

The Clifford rotor hypothesis was that algebraic richness (geometric product, trivector components) would better decorrelate KV cache vectors. The RotorQuant repo's own PPL table already suggests otherwise:

- 3-bit: PlanarQuant 10.12 < **RotorQuant 12.22** < IsoQuant 12.35 (PlanarQuant wins)
- 4-bit: IsoQuant 9.03 < PlanarQuant 9.56 < **RotorQuant 10.03** (RotorQuant is strictly worst)

The AMD report should provide the hardware-grounded explanation: RotorQuant's Cl(3,0) sandwich product (RxR̃) incurs 2,400 FMAs vs 256 for PlanarQuant, with no quality benefit on ROCm hardware where you can measure the exact kernel cycles. This makes the finding *citable* and *platform-authoritative*.

### Final Benchmark Matrix (headline)

`{FP16, FP8, TQ3, TQ4, iso3, iso4, planar3, planar4, rotor3, rotor4}` × `{Mistral-7B, Llama-3-8B}` × `{batch=1,4,8,16,32,64}` × `{seq=1K,4K,8K,32K,64K,128K}`

INT4 symmetric remains as a simple-quantization reference point only — it has no rotation overhead and serves as the "dumb floor."

---

## 2. Gaps in the Current TurboQuant Report

### 2a. Missing System-Level Benchmarks

- **No multi-method comparison** — the report is TurboQuant-only. The FP16/FP8/INT4 baselines are present but IsoQuant/PlanarQuant/RotorQuant are entirely absent.
- **No measured batch-size scaling** — Section 13 provides a theoretical bandwidth model but no *measured* multi-batch decode numbers. The model predicts 4.58× speedup at batch=64/seq=32K; this is never validated experimentally.
- **No prefill throughput comparison** — the report doesn't benchmark prefill speed at all (only prefill latency as a timing artifact). The RotorQuant README shows a 5.3× prefill speed difference between PlanarQuant and TurboQuant — this is a major uncovered gap.
- **No continuous batching / serving throughput** — there are no requests/sec or time-to-first-token (TTFT) measurements under a simulated serving load.
- **No K-only vs KV-symmetric ablation** — the RotorQuant README shows `planar3/f16` achieves near-zero PPL loss at 5.1× compression. The current report only tests symmetric TQ3/TQ4.
- **No multi-GPU / tensor-parallel** measurement — MI300X is often deployed in 2×, 4×, 8× configurations for 70B models.

### 2b. Missing Kernel-Level Profiling

- **No achieved memory bandwidth measurement for Triton v2** — the report gives effective bandwidth for v1 (12 GB/s) and v2 estimate (24–42 GB/s), but these are derived from timing, not hardware counters.
- **No per-operation kernel timeline** — no rocprof trace showing where time is spent within a decode step (weight load vs KV load vs rotation vs attention).
- **No IsoQuant/PlanarQuant kernel implementation on ROCm** — all existing Triton kernels are TQ3-only. The Triton kernels for PlanarQuant/IsoQuant exist in the rotorquant repo (CUDA-targeting) but have never been ported or benchmarked on gfx942.
- **No RotorQuant kernel on ROCm** — `triton_rotor_full_fused` exists in the rotorquant repo but was developed for CUDA. Port and benchmark it on MI300X to get the honest AMD number.
- **No compress/decompress microbenchmark for the new methods** — only TQ3/TQ4 HIP kernels are benchmarked. Need standalone compress+decompress throughput for all four methods.

### 2c. Missing Quality / Accuracy Evaluations

- **No perplexity** — the existing report's perplexity entry (`bench_quality.py`) was written but not run due to the shell blocker. Actual PPL numbers on WikiText-2 are absent.
- **No IsoQuant/PlanarQuant/RotorQuant PPL on AMD hardware** — the RotorQuant README PPL numbers are from CUDA (likely RTX or similar). The PPL values may differ slightly under ROCm due to numerical differences in Triton codegen.
- **No end-task evaluation** — no downstream task accuracy (e.g., MMLU, HellaSwag, or any standard LLM eval harness run). Cosine similarity and PPL are proxies; a single MMLU 5-shot comparison would be publishable.
- **No Needle-in-Haystack** — the RotorQuant README mentions passing NiaH at 8K/32K/65K on CUDA. This should be reproduced on AMD for all four methods.
- **No attention pattern divergence analysis** — no visualization of how much the compressed KV changes attention weights (beyond cosine sim of the vectors themselves).

### 2d. Missing Serving / Deployment Benchmarks

- **No vLLM serving integration for IsoQuant/PlanarQuant** — the existing `TurboQuantROCmAttentionBackend` is TQ3-only. The new methods need equivalent vLLM backends or at least a clear description of what would be needed.
- **No time-to-first-token (TTFT) breakdown** — prefill phase is measured only as a latency total, not decomposed into attention vs FFN vs KV-write time.
- **No SLA-bound analysis** — no "what is the maximum batch size that keeps p99 latency under X ms?" analysis, which is how production serving is actually capacity-planned.
- **No paged cache fragmentation analysis** — the vLLM backend is implemented but the overhead of paged-cache with 52-byte-per-slot TQ3 (vs 128-byte FP8 or 64-byte INT4) has not been measured.
- **No multi-session / cache eviction behavior** — no measurement of how compression ratio interacts with paged eviction policies.

### 2e. Missing AMD-Specific Profiling

- **No rocprof kernel timeline** for any of the benchmark runs.
- **No wave occupancy measurement** — the HIP kernel achieves 198 GB/s decompress, but what is the achieved occupancy? Is it wavefront-limited?
- **No LDS usage reported beyond the one note** (524 B/block for TQ3 HIP kernel).
- **No MFMA utilization measurement** — the MFMA rotation kernel (`tq_mfma_rotate_cov5.hsaco`) achieves 798 GB/s at best, but MFMA theoretical peak for gfx942 is ~383 TFLOPS FP16. There's no report of MFMA utilization percentage.
- **No AMD-vs-CUDA comparison** — the RotorQuant README benchmarks are on RTX 5090. The AMD report should explicitly compare the MI300X vs RTX 5090 numbers where they exist, to make the AMD positioning concrete.

---

## 3. Additional Metrics to Track

### Throughput / Latency

- `tokens_per_sec` at (batch, seq_len) grid — both decode-only and prefill+decode
- `prefill_tokens_per_sec` separately — this is where PlanarQuant/IsoQuant show the largest gain over TurboQuant (5.3×)
- `time_to_first_token_ms` (= prefill latency for a single request)
- `p50 / p95 / p99 decode step latency` — not just median; tail latency matters for SLA analysis
- `step latency variance` (std dev across warm runs) — compression methods with variable-length codebook lookups may have higher variance

### Kernel-Level

- `compress_throughput_GBs` — input bytes per second for the compress kernel (for each method)
- `decompress_throughput_GBs` — output bytes per second for the decompress kernel
- `fused_attention_effective_bandwidth_GBs` — KV bytes read / attention step time
- `rotation_FMAs_per_vector` — theoretical (256 Planar, 512 Iso, 2400 Rotor, 16384 TQ) vs measured cycle counts
- `kernel_launch_overhead_us` — for Triton JIT kernels, first-call vs warm-call latency

### Hardware Counters (via rocprof)

- `FETCH_SIZE` / `WRITE_SIZE` — actual bytes read/written per kernel (memory traffic proxy)
- `WAVES_EXECUTED` and `WAVE_CYCLES` — wave occupancy proxy
- `SQ_INSTS_VMEM_RD` / `SQ_INSTS_VMEM_WR` — VMEM instruction counts (critical for identifying gather bottlenecks as found in TQ3 v1)
- `SQ_INSTS_VALU` — VALU instruction count (the Triton v2 fix moved work here)
- `TA_FLAT_READ_WAVEFRONTS` — texture/flat memory wavefronts (gather efficiency indicator)
- `LDS_BANK_CONFLICT` (if available on gfx942) — LDS contention for centroid table lookups

### Quality

- `perplexity_wikitext2` — standard, for all four methods at all bitwidths
- `cosine_similarity_mean` and `cosine_similarity_p5` (worst-case layers) — already tracked for TQ, extend to all methods
- `MSE_per_head` — track per-layer and per-head variance to find which heads are hardest to compress
- `MMLU_5shot_accuracy` — single downstream eval; enough to say "no statistically significant accuracy loss at 3-bit"
- `NiaH_pass_rate` at {8K, 32K, 64K, 128K} — Needle-in-Haystack, binary pass/fail at each depth

### Memory / Capacity

- `bytes_per_token` — measured (not just theoretical), including norm storage overhead
- `paged_cache_fragmentation_overhead_pct` — extra VRAM from block-granularity allocation
- `max_context_at_budget_GB` — for 50 GB, 100 GB, 150 GB KV budgets (useful for capacity planning tables)
- `compression_ratio_achieved` vs `compression_ratio_theoretical` — verify no hidden overhead

### Stability

- Run all hot benchmarks N=5, report mean ± std
- Flag any method with std/mean > 5% as "unstable" — important for Triton JIT methods on ROCm

---

## 4. Benchmark Categories to Add

### 4.1 Compress / Decompress / Rotation Microbenchmarks

**Why it matters**: This is the cleanest way to compare the four methods on equal footing. Rotation cost is the primary differentiator: PlanarQuant does 256 FMAs, RotorQuant does 2,400. On AMD the wave occupancy and VMEM/VALU ratio matter as much as the FMA count.

**What to measure**: Standalone compress latency, decompress latency, and (separately) rotation-only latency for vectors of size n = {4K, 16K, 65K, 262K} (simulating KV cache sizes at different context × batch configurations). Report in GB/s input and GB/s output.

**How to run fairly**: All four methods should use the same input tensor (random FP16, same seed), same head_dim=128, same device (gfx942). Use Triton for all — do not mix HIP-compiled (TurboQuant) with Triton-compiled (others) in the same table without a clear note. Run 20 warm-up iterations, 50 timed iterations, report median and p95.

**Pitfalls**: 
- TurboQuant has an existing HIP binary that achieves 198 GB/s decompress. Its Triton equivalent is slower. Report both, labeled clearly. Do not compare HIP TurboQuant vs Triton PlanarQuant — that is not a fair comparison.
- RotorQuant's `triton_rotor_full_fused` uses the Clifford sandwich (RxR̃) which requires two matrix multiplications and a geometric product. On ROCm, verify the Triton kernel compiles and produces correct results before reporting timing.
- The `triton_fused_planar_attention` and `triton_iso_full_fused` kernels in the rotorquant repo were written for CUDA. Verify ROCm correctness with a cosine similarity check before benchmarking.

### 4.2 End-to-End Single-Request Decode

**Why it matters**: The headline number for any KV compression paper. Mirrors the existing TQ3 measurement but extended to all methods.

**What to measure**: tok/s and latency (ms/token) for a single request (batch=1) at seq_len = {512, 2K, 8K, 32K, 64K, 128K}. Both prefill and decode phases separately.

**How to run fairly**: Same model (Mistral-7B-v0.1 as primary, Llama-3-8B as secondary), same prompt (repeating fixed token sequence to reach each context length), same warmup (1 full prefill + decode run discarded), 3 timed runs with median reported.

**Pitfalls**:
- IsoQuant and PlanarQuant's deferred-quantization mode (K stored as FP16 during prefill, quantized on insertion during decode) produces different PPL than roundtrip quantization. Test both modes and label them. The deferred mode is the production-relevant one.
- RotorQuant does not have a production llama.cpp integration — run it via the Python/Triton API. This means comparing Python-overhead-laden RotorQuant numbers against llama.cpp-native Planar/Iso numbers would be unfair. Either: (a) run all methods via the Python/Triton path, or (b) run Planar/Iso via llama.cpp and note clearly that RotorQuant numbers include Python overhead. Option (a) is fairer for a research report.

### 4.3 Long-Context Scaling

**Why it matters**: The core use case for MI300X with 192 GB HBM3. The existing report shows this for TQ3 only; the new report should show all four methods.

**What to measure**: At batch=1, sweep seq_len from 8K to 512K (or to VRAM limit). Record tok/s, latency, and VRAM. Also record `max_context_at_192GB` for each method.

**How to run fairly**: Same VRAM budget baseline. Note that Planar/Iso can achieve higher max context (10.3× compression vs TQ3's 4.92×) because they pack 3-bit as 10.3× not 4.92× — verify the byte layouts. The RotorQuant README gives 10.3× for iso3/planar3 and 4.923× for TurboQuant. This discrepancy matters: TurboQuant stores a separate FP32 norm (4 bytes overhead per 128-dim vector), while Planar/Iso appear to pack differently. Document the exact byte layout for each method before computing compression ratios.

**Pitfalls**: The byte-per-token discrepancy between methods (TQ3: 52 bytes including FP32 norm, vs iso3/planar3 at reportedly higher compression) needs to be verified precisely. If iso3 achieves true 10.3× (≈24.8 bytes per 128-dim vector at 3 bits), then max context on MI300X would be ~13.6M tokens for Mistral-7B — a dramatically stronger headline than TQ3's 6.7M. Verify this before putting it in a report.

### 4.4 Batch-Size Scaling

**Why it matters**: The existing theoretical model predicts TQ3 speedup at batch=64. This needs to be measured for all four methods, and the crossover batch size (`batch*`) should be empirically determined.

**What to measure**: At seq_len = {8K, 32K}, sweep batch_size = {1, 2, 4, 8, 16, 32, 64}. Record total tok/s (batch × 1/step_time), per-request latency, and estimated KV bandwidth. Plot speedup vs FP16 for each method.

**How to run fairly**: All methods via the same Python/Triton path (or all via llama.cpp if that integration exists). Ensure KV cache is pre-populated (prefill done, measure decode-only). Use `torch.cuda.synchronize()` / `torch.hip.synchronize()` correctly at each step boundary.

**Pitfalls**: At high batch sizes, the rotation kernel becomes a bottleneck for RotorQuant (2,400 FMAs per vector × batch × context). For PlanarQuant (256 FMAs), this cost is negligible. The relative ranking of methods will shift with batch size — show the full curve, not just a single batch point.

### 4.5 vLLM / Paged KV Cache Serving Benchmarks

**Why it matters**: The existing report implements `TurboQuantROCmAttentionBackend` for TQ3. The next report should extend this (at minimum as a design spec, ideally measured) to Planar/Iso, and benchmark the serving overhead under continuous batching.

**What to measure**: Using vLLM with the TQ3 backend (already implemented), measure: requests/sec at fixed VRAM budget, TTFT distribution (p50/p95), and `max_concurrent_requests` before OOM. If Planar/Iso backends are implemented, run the same metrics.

**How to run fairly**: Use `vllm benchmark_serving.py` with a fixed request distribution (ShareGPT or synthetic with realistic prompt/output lengths). Run FP16 and TQ3 at minimum; add Planar/Iso if implemented. Report at the same total VRAM budget (not the same VRAM usage) to normalize capacity.

**Pitfalls**: The paged-cache block size interacts with the per-token byte sizes. TQ3 at 52 bytes/token with block_size=16 uses 832 bytes/block. FP16 uses 4096 bytes/block. The number of blocks that fit in a given VRAM budget varies, which affects fragmentation. Measure and report actual block utilization.

### 4.6 Quality Benchmarks

**What to measure**:
- WikiText-2 perplexity for all methods at {2-bit, 3-bit, 4-bit} — use the same evaluation harness (`llama-perplexity` via llama.cpp or the Python path via the rotorquant repo).
- Needle-in-Haystack at {8K, 32K, 64K, 128K} context — binary pass/fail, run 10 trials each, report pass rate.
- MMLU 5-shot for the 3-bit configurations of each method — one number per method showing downstream task parity.

**Pitfalls**: PPL from llama.cpp (post-prefill, deferred quantization) will be different from PPL from the Python/Triton roundtrip path. The RotorQuant README PPL numbers are post-prefill. Make sure all PPL comparisons use the same evaluation mode. The roundtrip (compress-then-decompress each token during evaluation) is a harder test.

### 4.7 K-Only vs KV-Symmetric Ablation

**Why it matters**: The RotorQuant README shows `planar3/f16` achieves ~5.1× compression with near-zero PPL loss (PPL ≈ 6.63 vs FP16 baseline 6.63). This is a huge practical result — compressing only the K cache at 3-bit is essentially lossless in PPL. The V cache matters much less. This ablation is missing entirely from the current TQ3 report.

**What to measure**: For each method: {K=compressed + V=FP16}, {K=FP16 + V=compressed}, {K=V=compressed} at 3-bit and 4-bit. Report PPL and throughput for each asymmetric config.

**Pitfalls**: TurboQuant's WHT rotation is self-inverse (Hadamard property), so K-only compression has different mathematical behavior than for PlanarQuant/IsoQuant (which require explicit inverse rotation for V). The existing vLLM backend stores K and V in the same TQ3 block — it needs extension to support asymmetric K/V configs.

### 4.8 Bitwidth / Compression Level Ablation

**What to measure**: All four methods at {2-bit, 3-bit, 4-bit} where available. Plot the "PPL vs compression ratio" Pareto curve. This is the most important single plot for the report — it shows which method is on the efficient frontier.

**Pitfalls**: RotorQuant at 2-bit may not have a stable Lloyd-Max codebook implementation. Check the rotorquant repo for `bits=2` support before including it.

### 4.9 MHA vs GQA vs MQA

**Why it matters**: Mistral-7B uses GQA (32Q/8KV), Llama-3-70B uses GQA (64Q/8KV). The compression ratio per parameter is the same, but the rotation kernel launch pattern differs: with 8 KV heads, there are far fewer vectors to rotate per step than with 32 KV heads (MHA). The relative overhead of rotation vs attention is therefore larger in GQA models, which disadvantages RotorQuant more.

**What to measure**: Compare on both a GQA model (Mistral-7B) and an MHA model if one is available (e.g. GPT-2 or a synthetic MHA Llama variant). Show how rotation overhead scales with KV head count.

---

## 5. AMD/ROCm-Specific Profiling Plan

### rocprof / rocprofv2 Collection Strategy

Use `rocprofv2` (available in ROCm 7.2) for hardware counter collection. The key counters to collect on gfx942:

```bash
# Collect memory traffic + VALU/VMEM instruction counts
rocprofv2 --pmc FETCH_SIZE,WRITE_SIZE,SQ_INSTS_VMEM_RD,SQ_INSTS_VALU \
          --output-format csv python3 run_benchmark.py
```

**What is realistically collectible on MI300X gfx942:**
- `FETCH_SIZE`, `WRITE_SIZE` — bytes read/written per kernel dispatch. Directly gives achieved memory bandwidth when divided by kernel duration. This is the most important counter.
- `SQ_INSTS_VMEM_RD` — count of vector memory read instructions. The TQ3 v1 gather bottleneck showed up as excessive VMEM instructions; v2's `tl.where` fix should show dramatically lower counts. Collect this to confirm.
- `SQ_INSTS_VALU` — count of VALU instructions. Should be higher in v2 (moved work from VMEM to VALU) — confirm the tradeoff.
- `WAVE_CYCLES`, `WAVES_EXECUTED` — compute wave occupancy proxy. Compare TQ3's 524 B/block LDS usage (enabling higher occupancy) vs potential RotorQuant LDS usage.

**What is approximate or unavailable:**
- `MFMA_UTIL_PCT` — CDNA3 MFMA utilization percentage is not directly exposed as a single counter. Approximate via `SQ_INSTS_VALU_MFMA` divided by total cycles.
- `LDS_BANK_CONFLICT` — available on some gfx9 targets but not guaranteed on gfx942 VF (virtual function / partition mode). Try collecting; if unavailable, note this.
- Per-wavefront occupancy — requires `rocprof --hsa-trace` or ATT (Advanced Thread Trace), which may not be available in a virtualized VF environment. Approximate using achieved_bandwidth / theoretical_bandwidth.

### Kernel Timeline Collection

```bash
rocprofv2 --sys-trace --output-format json python3 run_single_decode_step.py
```

This produces a JSON trace viewable in Perfetto (https://ui.perfetto.dev). The goal is to see the wall-clock timeline of one decode step: weight load vs KV compress vs attention vs KV decompress. This is the only way to see if rotation has been correctly fused or is launching as a separate kernel.

**Realistic expectation**: On a VF (virtual function) partition, some hardware counters may be restricted. The timeline trace (kernel names + durations) should always be available. Hardware counters (FETCH_SIZE, VMEM counts) may require bare-metal or SRIOV-enabled access — verify availability first.

### HIP vs Triton Implementation Comparison

The existing report has a TurboQuant HIP kernel (198 GB/s decompress) and a TurboQuant Triton kernel (~42 GB/s fused attention). For the new methods:
- IsoQuant/PlanarQuant/RotorQuant only have Triton kernels (no HIP implementations).
- The report should explicitly note this asymmetry: TurboQuant has a hand-optimized HIP path that provides an upper bound, while the other methods are compared at the Triton JIT level.
- The recommendation for future work should be: implement HIP kernels for PlanarQuant/IsoQuant to see if the gap narrows further (PlanarQuant's 2D Givens rotation is simpler to implement in HIP than TurboQuant's butterfly network).

### HIP Graph Capture / Compilation Effects

- First-call Triton JIT compilation takes 10–30 seconds on gfx942. All benchmarks must include a warmup phase that triggers compilation *before* timing starts.
- HIP graph capture (`torch.cuda.CUDAGraph` / `torch.hip.graph`) can reduce kernel launch overhead for the decode loop. Measure the overhead of graph capture for the fused attention kernels. PlanarQuant/IsoQuant with simpler kernels may benefit more from graph capture than TurboQuant (fewer kernel arguments to rebind).
- rocBLAS plan caching: the 128×128 rotation GEMM in TurboQuant will be cached by rocBLAS after the first call. Include warmup to ensure fair comparison.

---

## 6. Fair Comparison Methodology

### Fixed Parameters Across All Methods

| Parameter | Value | Rationale |
|---|---|---|
| Model | Mistral-7B-v0.1 (primary), Llama-3-8B (secondary) | Both on MI300X without sharding |
| head_dim | 128 | Universal across all target models |
| Rotation seed | 42 (fixed random orthogonal matrix) | TurboQuant only; Planar/Iso/Rotor use learned rotation matrices from the rotorquant repo |
| Prompt | Fixed WikiText-2 prefix padded to target seq_len | Same text = same KV cache content = comparable quality |
| decode_steps | 50 (for throughput), 200 (for latency stability) | Enough to amortize JIT warmup |
| warmup_steps | 3 full decode steps discarded | Allow rocBLAS plan cache and Triton JIT to settle |
| n_runs | 5 (report median ± std) | Catch run-to-run variance |
| Attention implementation | SDPA (PyTorch), Flash Attention disabled | Consistent baseline; FA2 not installed |
| Bit budget comparison | iso3 vs planar3 vs turbo3 vs rotor3 (all 3-bit) | Equal bit budget comparison |
| Compression mode | symmetric K+V (primary), K-only (secondary ablation) | Cover the production and ideal cases |

### Implementation Maturity Handling

This is the thorniest issue. TurboQuant has a production HIP kernel; the others are Triton-only and CUDA-developed. The approach:

1. **Primary comparison table**: All methods via their best available *Triton* kernel on gfx942. This is a level playing field — no method has a specially tuned HIP path.
2. **Secondary column**: TurboQuant HIP kernel result, labeled "HIP (TQ3 only)." This shows the ceiling TurboQuant can reach with a hand-tuned kernel, and implies what Planar/Iso could achieve with equivalent engineering.
3. **Appendix note**: Explicitly state that PlanarQuant/IsoQuant's Triton kernels were ported from CUDA-targeting code. Any ROCm-specific optimizations (Wave64 ballot, LDS layout) could close the gap further.

### Normalization Axes

The report should include these specific normalized comparisons:

- **Speed at equal quality**: Fix PPL ≤ 7.2 (near-lossless for Mistral-7B), report tok/s for each method that achieves this. Expected result: PlanarQuant at K-only-3bit wins.
- **Quality at equal compression**: Fix compression ratio ≈ 10.3× (the iso3/planar3 target), report PPL for each method. Expected: IsoQuant and PlanarQuant beat TurboQuant, RotorQuant loses.
- **Speed at equal memory footprint**: Fix bytes_per_token to the iso3 level, report tok/s. TurboQuant at equivalent compression would be slower due to WHT overhead.
- **Throughput under fixed VRAM budget**: At 50 GB KV budget (Llama-3-70B on MI300X), report max total tokens/sec at batch=16. Each method will support a different max context, affecting the answer.
- **Max context under fixed VRAM budget**: At 192 GB total, subtract 14 GB (Mistral weights), report max context per method. This is where Planar/Iso's higher compression ratio (10.3× vs 4.92×) is most impactful.

---

## 7. Recommended Benchmark Matrix

### Models

| Model | Params | KV heads | head_dim | Weights VRAM | Architecture |
|---|---|---|---|---|---|
| Mistral-7B-v0.1 | 7B | 8 KV / 32 Q (GQA) | 128 | ~14 GB | Primary |
| Llama-3-8B | 8B | 8 KV / 32 Q (GQA) | 128 | ~16 GB | Secondary |
| Llama-3-70B | 70B | 8 KV / 64 Q (GQA) | 128 | ~140 GB | Capacity analysis only |

### Sequence Lengths

`{1K, 4K, 8K, 32K, 64K, 128K, 256K*}` — 256K only if time/VRAM permits

### Batch Sizes

`{1, 4, 8, 16, 32, 64}` — focus analysis on batch ≥ 8 where KV bandwidth dominates

### Methods × Precision

| Method | 3-bit | 4-bit | K-only-3bit | Notes |
|---|---|---|---|---|
| FP16 | — | — | — | Reference baseline |
| FP8 E4M3 | — | — | — | Simple quantization reference |
| INT4 symmetric | — | yes | — | Simple quantization reference |
| TurboQuant (TQ) | TQ3 | TQ4 | TQ3/f16 | HIP + Triton |
| IsoQuant (iso) | iso3 | iso4 | iso3/f16 | Triton (port to ROCm) |
| PlanarQuant (planar) | planar3 | planar4 | planar3/f16 | Triton (port to ROCm) |
| RotorQuant (rotor) | rotor3 | rotor4 | — | Triton (port to ROCm); included to prove inferiority |

### Serving Modes

- Single-request decode (batch=1, decode-only) — latency / throughput
- Multi-request batch decode (batch=4–64) — bandwidth-bound throughput
- vLLM paged serving (TQ3 backend) — requests/sec, TTFT

### Output Tables and Plots to Generate

**Table 1**: Compression ratio + bytes/token for all methods (replace report Table in §3.2)

**Table 2**: PPL (wikitext-2) × method × bitwidth — the core quality comparison

**Table 3**: Batch decode throughput matrix — tok/s at (batch, seq_len) for all methods

**Table 4**: Max context at 192 GB — per method, with weight VRAM subtracted

**Plot 1**: PPL vs compression ratio scatter — Pareto frontier plot, each method as a labeled point. This is the single most important new figure.

**Plot 2**: Decode tok/s vs seq_len for all methods at batch=1 — extends existing Fig 1

**Plot 3**: Decode tok/s vs batch_size at seq_len=32K — the bandwidth crossover plot

**Plot 4**: Prefill tok/s comparison — all methods at seq_len=8K; expected to show 5× gap between TurboQuant and Planar/Iso

**Plot 5**: Compress + decompress throughput bar chart — standalone microbenchmarks per method

**Plot 6**: Max context bar chart — one bar per method showing 192 GB KV capacity

**Plot 7**: Speed vs quality scatter at fixed bit budget (3-bit) — tok/s (x-axis) vs PPL (y-axis), each method labeled. The ideal method is bottom-right (fast + low PPL).

**Plot 8 (roofline-style)**: Effective memory bandwidth vs arithmetic intensity for each method's fused attention kernel, overlaid on the MI300X roofline (5.3 TB/s bandwidth, ~383 TFLOPS FP16 compute).

**Plot 9**: rocprof kernel timeline for one decode step — shows how much time each method spends in rotation vs quantization vs attention (qualitative, one figure per method).

---

## 8. Concrete Report Improvements

### New Narrative Arc

The current report is "TurboQuant on AMD — it works." The next report should be "Block-Diagonal Rotation vs Full-Rank WHT: AMD Hardware Settles the Debate." The AMD/ROCm framing makes the following claims possible that no CUDA paper can make:

1. On Wave64 hardware, the butterfly network's bitpacking advantage (2× vs Wave32) is largest for TurboQuant — yet Planar/Iso still wins on throughput because rotation cost dominates packing cost.
2. The 5.3 TB/s MI300X bandwidth means the compression-to-computation tradeoff shifts differently than on NVIDIA A100 (2 TB/s) — show this explicitly.
3. The 192 GB capacity enables context windows that are physically impossible on A100 (80 GB) or H100 (80 GB SXM) even with compression.

### AMD Positioning Claims (Safe to Make)

- "IsoQuant and PlanarQuant achieve [X]× higher decode throughput than TurboQuant at equal PPL on AMD MI300X" — provable from benchmarks.
- "RotorQuant's Clifford algebra rotation adds [Y]× more compute than PlanarQuant with no quality improvement on gfx942" — provable from FMAs + PPL.
- "MI300X enables [Z]M-token context for Llama-3-8B at 3-bit KV compression, vs [W]M on NVIDIA H100 at equivalent compression" — provable from VRAM arithmetic.

### Claims to Avoid

- Do not claim "RotorQuant is worse" based only on the CUDA PPL numbers from the RotorQuant README — run the AMD numbers. The claim needs to come from your own measurements.
- Do not claim PlanarQuant is "production-ready on ROCm" unless the Triton kernels are validated for correctness (cosine_sim check) and stability on gfx942.
- Do not compare HIP TurboQuant kernel speed to Triton PlanarQuant speed as an apples-to-apples kernel comparison.

### Plot Quality Improvements

- All plots should use a consistent color scheme: FP16=gray, FP8=light blue, INT4=yellow, TQ3=red, TQ4=orange, iso3=green, planar3=blue, rotor3=purple.
- The "PPL vs compression" Pareto plot should annotate each point with method name and show the Pareto frontier as a line.
- The roofline plot should show both the bandwidth-bound ridge (slope = 1/BW) and the compute-bound ceiling, with each method's fused attention kernel plotted as a point.
- Include a "deployment summary table" at the start of the report (before all plots) that summarizes: best method for latency, best for quality, best for max context, best for batch throughput — one recommendation per deployment scenario.

---

## 9. Prioritized Implementation Roadmap

### Highest Priority (do first — unblocks everything)

1. **Port IsoQuant and PlanarQuant Triton kernels to ROCm** — run correctness check (cosine_sim ≥ 0.99 at 4-bit, ≥ 0.97 at 3-bit) on gfx942 before any benchmarking. The kernels exist in `rotorquant/turboquant/triton_planarquant.py` and `triton_isoquant.py`. Likely require minor changes for ROCm Triton (e.g., `tl.constexpr` layout annotations, avoiding CUDA-specific intrinsics).
2. **Port RotorQuant Triton kernel to ROCm** — `triton_kernels.py::triton_rotor_full_fused`. Run correctness check. This is lower-priority than Planar/Iso but needed to include RotorQuant in the benchmark.
3. **Run PPL benchmarks for all four methods** — using the Python/Triton path. This gives the quality comparison table (Table 2) and the PPL vs compression Pareto plot. Can be done before fused attention is ready.
4. **Measure prefill throughput for all methods** — this is the 5.3× gap that the RotorQuant README claims. Reproducing it on AMD is the highest-impact single measurement. Use `llama-bench` from the llama.cpp fork if available; otherwise use the Python/Triton path with a prefill-only timing loop.

### Medium Priority (do after headline results are in)

5. **Implement measured batch-decode benchmark** — extend `bench_batch_decode.py` to support IsoQuant/PlanarQuant/RotorQuant. Replace the theoretical model with measured tok/s at each (batch, seq_len) point. This validates (or invalidates) the Section 13 theoretical predictions.
6. **Add compress/decompress microbenchmarks for all methods** — extend `bench_kernels.py` to benchmark all four methods' standalone compress and decompress kernels. This gives the "rotation cost" comparison table.
7. **Run NiaH (Needle-in-Haystack) at 32K and 64K** for all methods — binary pass/fail. Needs a long-context generation loop. Can use the rotorquant repo's `poc_high_context.py` as a starting point.
8. **Collect rocprof traces** for one representative run of each method — even a single kernel timeline showing compress→attention→decompress time split is publishable.
9. **Extend vLLM backend** to support at least IsoQuant — the symmetric Givens 2D rotation is simpler than TurboQuant's WHT and the paged-cache layout is similar (same norm + index structure).

### Nice-to-Have (if time permits)

10. **MMLU 5-shot accuracy** for all 3-bit methods — run via `lm-evaluation-harness` with the appropriate backend. One number per method, proving "no catastrophic accuracy loss."
11. **AMD vs NVIDIA comparison column** — using the RTX 5090 numbers from the RotorQuant README as the NVIDIA reference, add an AMD/NVIDIA column for the methods that appear in both. This makes the AMD positioning explicit.
12. **HIP kernel for PlanarQuant** — implement a native HIP decompress kernel for PlanarQuant (2D Givens is extremely simple: 2 multiplications per pair, no gather). Compare against Triton to show the engineering ceiling and motivate the "simpler rotation = easier hardware optimization" argument.
13. **K-only asymmetric mode benchmark** — `planar3/f16` achieving near-zero PPL loss at 5.1× compression is potentially the most practically useful result in the entire study. Measure it on MI300X with full decode throughput.
14. **MI325X / MI350X extrapolation section** — if MI325X (HBM3e, 288 GB) specs are available, add a one-page capacity analysis section projecting context lengths and batch throughputs. No measurements needed — just the bandwidth-model arithmetic, clearly labeled as projections.

### What Can Be Done Quickly (days, not weeks)

- PPL benchmarks for all four methods via the Python/Triton path — the rotorquant repo `benchmark_google_parity.py` does this already; run it on the MI300X.
- Compress/decompress microbenchmarks — extend existing `bench_kernels.py` structure.
- Max-context capacity table for all methods — pure arithmetic from bytes/token (verify byte layouts first).
- Prefill timing — add a `--prefill-only` flag to the existing decode benchmark scripts.

### What Requires Substantial Engineering (weeks)

- Full Triton kernel port + validation for all three new methods on gfx942.
- vLLM backend extension for IsoQuant/PlanarQuant.
- HIP kernel for PlanarQuant.
- NiaH at 128K context (requires generating 128K tokens, which takes ~45 minutes at 46 tok/s).
