# ROCm Bottleneck Report: Why Fused TQ Is Not Scaling End-to-End

## Scope
This report analyzes why the fused TurboQuant path does not yet deliver large serving throughput gains on MI300X, despite strong microbenchmark improvements.

Data sources:
- `results/bench_vllm_turboquant_ab_sweep_small.json`
- `results/bench_vllm_turboquant_ab_mistral_stress.json`
- `results/bench_vllm_turboquant_ab_sweep_kv_heavy.json`
- `results/bench_flash_attn_check.json`
- `results/bench_vllm_turboquant_ab_debug.json` (before fallback fix)
- `results/bench_vllm_turboquant_ab_debug_after_fix.json` (after fallback fix)
- `results/rocm_custom_paged_debug.jsonl` (runtime backend-decision trace)
- `results/bench_vllm_rocprof_timeline_summary.json` (rocprofv2 kernel-duration rollup, Apr 2026)
- vLLM runtime logs emitted during the above runs

## Update: ROCm custom paged-attention fallback fixed

We traced the fallback to the ROCm backend gate in `vllm/platforms/rocm.py`.

- Root cause: `use_rocm_custom_paged_attention()` only allowed `sliding_window` of `0` or `(-1, -1)`.
- Observed runtime value on MI300X VF: `sliding_window=4095` with `max_seq_len=4096`, which is effectively full causal context.
- Result: custom ROCm paged-attention was rejected even for otherwise valid MI300X shapes (`fp16`, `head_size=128`, `block_size=16`, `gqa_ratio=4`), causing Triton fallback.

Patch applied:
- Treat `sliding_window >= max_seq_len - 1` as "no effective sliding window" for the custom-kernel eligibility check.
- Added optional decision tracing (`VLLM_ROCM_CUSTOM_PAGED_DEBUG_PATH`) and force toggle (`VLLM_ROCM_FORCE_CUSTOM_PAGED_ATTN`) for controlled A/B diagnostics.

Validation signal:
- Before fix (`bench_vllm_turboquant_ab_debug.json`): ~632 tok/s.
- After fix (`bench_vllm_turboquant_ab_debug_after_fix.json`): ~697 tok/s.
- Relative gain in this short run: ~10%.
- Decision trace now records `decision=true` for the same call shapes where it previously rejected.

## Sweep Results (serving throughput)

| Workload | FP16 tok/s | TQ decompress tok/s | TQ fused tok/s | Fused vs Decompress |
|---|---:|---:|---:|---:|
| input=256, output=64, prompts=8 | 1096.1 | 1107.3 | 1105.8 | -0.14% |
| input=512, output=128, prompts=20 | 2017.3 | 2015.7 | 2018.2 | +0.12% |
| input=1024, output=256, prompts=32 | 2425.3 | 2434.7 | 2428.8 | -0.24% |

Observation: all three modes are effectively tied in end-to-end serving (sub-0.3% deltas, i.e., noise range).

## What this means

The fused kernel path is not the dominant bottleneck in these serving runs. If it were, fused should consistently beat decompress by a visible margin under larger context and concurrency. Instead, throughput parity indicates that runtime is dominated by non-KV-dequant components.

## Why this is happening (ranked bottlenecks)

1. **Decode path is not KV-dequant dominated at serving level**
   - Micro-level fused speedups exist, but end-to-end tokens/s does not move.
   - The bottleneck has shifted to model/engine orchestration costs.

2. **ROCm attention path fallback was active (now patched in this workspace)**
   - Prior repeated log line: `Cannot use ROCm custom paged attention kernel, falling back to Triton implementation.`
   - The specific gating issue has been fixed locally; this should be upstreamed/kept in the serving env.

3. **ROCm platform/runtime integration overhead is higher and less deterministic**
   - The workflow required explicit ROCm platform fixes (`amdsmi` for platform detection).
   - Build/runtime path selection is more fragile than CUDA defaults, increasing fallback probability and overhead.

4. **Profile counters are restricted in MI300X VF environment**
   - `rocprofv2 --list-metrics` is not fully available.
   - This limits low-level attribution and makes optimization loops slower on VF compared with bare-metal.

## Why it tends to show up more on ROCm

This behavior is disproportionately common on ROCm in this setup because:
- vLLM ROCm backend selection depends on ROCm-specific discovery (`amdsmi`) and can silently degrade to generic paths when environment or deps are incomplete.
- ROCm custom paged-attention path is less mature in this branch/config than CUDA equivalents; fallback to Triton is currently frequent.
- Tooling/observability constraints in VF mode reduce counter-level tuning feedback, so kernel-level wins are harder to convert into stable serving gains.

This is not a claim that ROCm cannot match performance. It is a statement that this specific software stack and platform mode currently route through non-ideal paths.

## Corroborating signal from decode-step matrix

From `results/bench_flash_attn_check.json`:
- Fused vs non-fused compressed decode-step speedup is large (about 4.8x to 12.9x over tested lengths).
- Yet serving A/B remains flat.

Conclusion: fused improvements are real in isolated decode-step compute, but masked at full serving stack level by higher-level bottlenecks.

## Practical next steps (after fallback fix)

1. **Collect timeline-level kernel breakdown per mode (fp16/decompress/fused)**
   - Even without full counters, kernel-duration timeline can show where decode time is spent.
2. **Run the same sweep on bare-metal MI300X (non-VF)**
   - Needed for full rocprof counter attribution and occupancy analysis.
3. **Only then retarget roofline claims**
   - Use measured serving bottlenecks, not aspirational fused arithmetic-intensity assumptions.

## Update: rocprofv2 timeline collection (blockers fixed)

### What was broken

- **`rocprofv2 --hip-trace --kernel-trace` + vLLM default (CUDA graphs / compile)** repeatedly ended in `hipErrorLaunchFailure` / `EngineDeadError` during graph capture or early decode.
- **Kernel-only tracing without `enforce_eager`** produced enormous `results_*.csv` traces (tens of thousands of kernel rows per short run) and impractical runtime.

### What we changed (harness)

- Script: `benchmarks/bench_vllm_rocprof_timeline.py`
  - Default **`--kernel-trace` only** (omit `--hip-trace` unless you pass `--with-hip-trace`).
  - Passes **`--enforce-eager`** through to `bench_vllm_turboquant_ab.py` so vLLM skips CUDA graph capture and torch.compile under the profiler (matches vLLM’s own “eager disables cudagraph” path).
  - Parses rocprofv2’s **`Kernel_Name` + `Start_Timestamp` / `End_Timestamp`** CSV format and merges all `results_<pid>.csv` files in the output directory (parent + `EngineCore` child).
- Script: `benchmarks/bench_vllm_turboquant_ab.py` adds **`--enforce-eager`** for profiling-friendly runs.

Reproduce:

```bash
cd /root/workspace
./amd-experiments/.venv-vllm-rocm/bin/python ./amd-experiments/benchmarks/bench_vllm_rocprof_timeline.py \
  --model mistralai/Mistral-7B-v0.1 --input-len 256 --output-len 64 --num-prompts 8
```

Output: `results/bench_vllm_rocprof_timeline_summary.json`.

### Measured kernel-time mix (Mistral-7B-v0.1, eager under rocprof)

Workload in this capture: `input_len=256`, `output_len=64`, `num_prompts=8`, `enforce_eager=True`, kernel trace only. Totals are **sum of per-kernel GPU durations** from the CSV (overlap across streams is possible, so treat totals as a **weighting signal**, not wall-clock).

| Mode | Σ kernel time (ms) | Dominant bucket (top kernel, share) | `paged_attention_ll4mi_*` | `fused_add_rms_norm` |
|------|-------------------:|--------------------------------------|--------------------------:|---------------------:|
| fp16 | ~1097 | rocBLAS MFMA GEMM (`Cijk_Alik_Bljk_*`), ~23% | ~2.1% | ~4.0% |
| turboquant_decompress | ~1102 | same GEMM family, ~23% | ~2.1% | ~4.1% |
| turboquant_fused | ~1099 | same GEMM family, ~23% | ~2.1% | ~4.1% |

**Interpretation for roofline / “KV-first” narratives:** on this short eager-profiled slice, **GEMM/MLP-style work dominates the kernel-duration budget**, while **paged attention is only a few percent** of summed kernel time. That aligns with end-to-end A/B parity: speeding up KV dequant/fusion cannot move tokens/s much until attention (or something else in the non-kernel path) becomes a larger fraction of wall time.

**Caveats**

- **`enforce_eager` changes the engine** (no CUDA graphs, no inductor compile). Use this mode for **timeline attribution**, not for absolute tokens/s ceilings.
- vLLM **0.19.0** logs `Unknown vLLM environment variable` for `VLLM_ATTENTION_BACKEND` / `VLLM_KV_CACHE_DTYPE` / `VLLM_TQ_USE_FUSED_KERNEL` in these runs: stock `AttentionBackendEnum` does not include TurboQuant. Until the custom backend is selected via **supported Engine args** (or a fork that registers those env vars), the **“decompress” vs “fused” labels may not change the actual attention implementation** in this interpreter build—kernel breakdowns matching across modes is therefore expected here.

## Bottom line

The proposed fused target is not being missed because fused kernels are ineffective. After fixing one major ROCm fallback gate, gains improved but fused-vs-decompress remains near-tied in serving A/B, indicating the next bottlenecks are now above the fused kernel itself in this MI300X VF stack.
