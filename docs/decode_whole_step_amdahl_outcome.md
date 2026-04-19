# Whole-decode Amdahl outcome (Phase D)

**Narrative closure (everything we did in-repo vs deployment handoff):** [`repo_decode_bottleneck_closure.md`](repo_decode_bottleneck_closure.md).

This document answers: **Can we shrink the non-KV path enough that KV / attention savings show up in end-to-end vLLM decode tok/s?** for **Mistral-7B**, **kv-heavy** decode (long prompt, many concurrent sequences).

## Golden configuration (locked)

| Field | Value |
|-------|--------|
| Model | `mistralai/Mistral-7B-v0.1` |
| `input_len` | 1024 |
| `output_len` | 256 |
| `num_prompts` | 32 |
| Backends | `fp16`, `turboquant_decompress`, `turboquant_fused` |
| Runtime | `VLLM_ENABLE_V1_MULTIPROCESSING=0` (in-process engine so bench-side TurboQuant `CUSTOM` registration is visible to vLLM V1) |
| VRAM knob | `VLLM_BENCH_GPU_MEM` → `--gpu-memory-utilization` (**0.15** after max-VRAM cleanup: `scripts/gpu_torch_empty_cache.py` + idle GPU; `rocm-smi` showed ~all VRAM free but vLLM’s **worker-init** `mem_get_info` can still read **~20–30 GiB** free inside a long multi-backend process, so **`0.85` failed mid-run** while **0.15** completes all three backends — see `vllm_init_free_memory_note` in the JSON) |
| Artifact | [`results/decode_whole_step_baseline_kv_heavy.json`](../results/decode_whole_step_baseline_kv_heavy.json) |

**Throughput (MI300X VF, frozen venv `.benchmark_mi300_vllm_frozen`, post-cleanup pass 2026-04-19):** FP16 **~1763** output tok/s; TurboQuant decompress **~1933** tok/s; TurboQuant fused **~1914** tok/s (**~8–10%** over FP16 — same Amdahl story: not 2× from KV alone). An older archived line at **~2425 tok/s** used a **higher** effective KV / memory budget; absolute tok/s move with `gpu_memory_utilization` and vLLM’s allocator snapshot — compare **shapes and buckets**, not only headline numbers across months.

## Where time goes (rocprof top-kernel buckets) — **answer: bottleneck is mixed GEMM + paged attention**

Source: [`results/decode_whole_step_rocprof_bucket_compare.json`](../results/decode_whole_step_rocprof_bucket_compare.json), summarized from [`results/bench_vllm_rocprof_timeline_summary.json`](../results/bench_vllm_rocprof_timeline_summary.json) after `benchmarks/bench_vllm_rocprof_timeline.py` with **`--kv-heavy-story2`**, **`--gpu-memory-utilization 0.15`**, **`--max-model-len 8192`** (same intent as the golden decode recipe).

**Verdict (FP16 mode, top-kernel rollup):**

- **`gemm_hipblaslt` ~43%** of summed top-kernel time — MLP / projection matmuls remain a **large** slice (latest summarized trace: **~43.1%** FP16).
- **`attention_named` ~30%** — dominated by **`kernel_paged_attention_2d.kd`** (**~29.6%** FP16 in the same refresh; bucket rule in `benchmarks/story2_rocprof_summarize.py` classifies `paged_attention` kernels into `attention_named`).

So the earlier story from a **lighter** rocprof shape (**GEMM ~54%, attention ~2%**) was **misleading for kv-heavy**: at the long-context / multi-sequence recipe, **paged attention is on the critical path alongside hipBLASLt GEMMs**. TurboQuant modes in the same summary keep a similar split (**~30% attention_named**, **~44% gemm_hipblaslt**): swapping KV / attention implementation does **not** collapse the GEMM bucket in this trace — consistent with **incremental**, not transformative, E2E tok/s deltas.

## Levers executed here

| Lever | Result | Artifact |
|-------|--------|------------|
| FFN / SwiGLU Triton spike | **no_go** at (M=64, H=14336): Triton fused **slower** than eager SiLU×mul | [`results/decode_whole_step_ffn_hypothesis_outcome.json`](../results/decode_whole_step_ffn_hypothesis_outcome.json) |
| Quant weights + TQ (AWQ) | **Ran** — AWQ ~**35% slower** than dense FP16 / TQ at this shape | [`results/decode_whole_step_quant_lever_status.json`](../results/decode_whole_step_quant_lever_status.json), [`results/story2_quant_kv_heavy_ab.json`](../results/story2_quant_kv_heavy_ab.json) |
| Graphs + scheduler sweep | **Partial** — [`story2_serving_path_ab_fp16.json`](../results/story2_serving_path_ab_fp16.json) written (eager leg); **CUDA-graphs** leg still aborts (`cuda_graphs_error` in JSON); e2e sweep re-run after cleanup still has **VRAM startup** failures on heaviest cells (see status JSON counts) | [`results/decode_whole_step_scheduler_status.json`](../results/decode_whole_step_scheduler_status.json) |

## Phase D conclusion (2026-04-19)

1. **At the golden kv-heavy shape, E2E output tok/s does not show a “KV-only” win large enough to dwarf the rest of the step** — FP16 vs TurboQuant stays in the **few–to–~10%** band depending on run conditions; AWQ is clearly **slower** here (~1210–1244 vs ~1760–1920 tok/s) because **weight dequant + kernels** add work on the hot path.
2. **Rocprof aligned to `--kv-heavy-story2` shows a joint bottleneck:** **~43% hipBLASLt GEMM** and **~30% paged-attention** (top kernel `kernel_paged_attention_2d.kd`). It is **not** “GEMM-only” anymore; it is also **not** “attention-only”.
3. **SwiGLU elementwise fusion microbench does not justify** pushing that exact fusion into vLLM next (it regresses at the tested shape).
4. **If you need scheduler / graphs conclusions:** run [`scripts/run_story2_scheduler_sweep.sh`](../scripts/run_story2_scheduler_sweep.sh) after **`scripts/gpu_torch_empty_cache.py`** (and real idle GPU); set **`STORY2_SCHEDULER_GPU_MEM`** / **`STORY2_SCHEDULER_MAX_LENS`** / **`STORY2_SCHEDULER_NPROMPTS`** so every cell’s startup sees **`free ≥ util × total`**, or use **one subprocess per sweep cell** if fragmentation persists.

## Automation

- **VRAM cleanup before heavy benches:** [`scripts/gpu_torch_empty_cache.py`](../scripts/gpu_torch_empty_cache.py) — then [`scripts/show_gpu_occupiers.sh`](../scripts/show_gpu_occupiers.sh) if free memory is still surprising.
- **Refresh golden baseline + dispatch log:** [`scripts/run_decode_whole_step_baseline_kv_heavy.sh`](../scripts/run_decode_whole_step_baseline_kv_heavy.sh)
- **Re-package baseline from an existing bench JSON:** `python3 benchmarks/decode_whole_step_golden_driver.py --from-existing results/<file>.json`
- **Rocprof timeline (kv-heavy):** `python3 benchmarks/bench_vllm_rocprof_timeline.py --kv-heavy-story2 --gpu-memory-utilization <frac>` then `python3 benchmarks/story2_rocprof_summarize.py --input results/bench_vllm_rocprof_timeline_summary.json --output results/decode_whole_step_rocprof_bucket_compare.json`
- **Tiny decode smoke (VRAM-tight / broken large-model runs):** [`bash scripts/run_decode_bottleneck_smoke.sh`](../scripts/run_decode_bottleneck_smoke.sh) → [`results/decode_bottleneck_smoke.json`](../results/decode_bottleneck_smoke.json) — tries Mistral-7B then **TinyLlama** with auto `gpu_memory_utilization`; **FP16 only** (stack sanity). For **where time goes** on MI300X+Mistral, still use the rocprof line above; smoke does not replace kernel buckets.
