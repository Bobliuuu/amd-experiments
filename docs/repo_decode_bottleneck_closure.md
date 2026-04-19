# Repository closure: vLLM decode bottleneck (MI300X)

This document is the **canonical narrative** for what **this repository** did to **shrink avoidable decode overhead** while preserving **accuracy**, **KV compression (TQ3 layout)**, and **lower HBM footprint**—and why **remaining batch=1 throughput** is no longer an **implementation** problem inside this tree.

Companion pieces:

- Operational levers and deployment checklist: [`bottleneck_improvement_mi300.md`](bottleneck_improvement_mi300.md)
- Golden numbers, rocprof recipe, and negative results (FFN spike, AWQ, scheduler): [`decode_whole_step_amdahl_outcome.md`](decode_whole_step_amdahl_outcome.md)
- Earlier ROCm paged-attention analysis: [`../report/rocm_kv_bottleneck_report.md`](../report/rocm_kv_bottleneck_report.md)

Figures (regenerate with `python3 report/generate_figures_v2.py`):

- **Fig 30** — `report/figures_v2/fig30_decode_whole_step_rocprof_buckets.png` — top-kernel **time share** on kv-heavy Mistral decode (`results/decode_whole_step_rocprof_bucket_compare.json`).
- **Fig 31** — `report/figures_v2/fig31_repo_engineering_closure_vs_deployment.png` — **in-repo** vs **outside-repo** handoff table.

### Tradeoff, “implementation,” and TurboQuant itself

**Tradeoff:** Fixing **accuracy**, **strong KV compression** (e.g. TQ3), and **lower KV memory** defines a **different objective** than maximizing **batch=1 decode tok/s** on **FP16 KV + stock** attention/GEMM. You are buying **HBM headroom and context**; you are **not** promised a multiplier on **whole-step** throughput, because each token still runs **MLPs and matmuls** that KV compression does not erase (see §1 and Fig **30**).

**Why the gap is not “unfinished implementation”:** After the closure work, typical wiring bugs, GQA issues, bogus paged-attention gates, and false fusion/quant wins are **closed or measured as regressions**. The remainder is **(i) Amdahl** — large **hipBLASLt** and **attention** buckets even with TQ; **(ii) TurboQuant’s design** — **bytes per vector** are paid for with **rotation + pack/unpack + how attention consumes KV**; code improves **constants**, not that **algebraic/storage choice**, unless you **change the product**; **(iii) deployment** — hipBLASLt, graphs, ROCm/vLLM on the **shipped image** (Fig **31**).

**Future work:** **(A) Product mix** — relax compression, faster block method, K-only, etc., to **move the Pareto** deliberately. **(B) Stack** — upgrade and re-profile on MI300X; validate graphs vs `enforce_eager`. Full prose also lives in **`report/paper.md` §5.14** and **`report/final_report_v2.md` §§14.5–14.6** / **`report/final_report.md` §14.5**.

---

## 1. What we proved (evidence, not opinion)

On the **kv-heavy** Mistral-7B vLLM configuration documented in `docs/decode_whole_step_amdahl_outcome.md`, rocprof-aligned bucket summaries show **where GPU time goes** in FP16 and TurboQuant paths alike:

- Roughly **40–44%** of summed top-kernel time in **`gemm_hipblaslt`** (MLP / projection matmuls).
- Roughly **30%** in **`attention_named`**, dominated by **`kernel_paged_attention_2d.kd`**.

So **end-to-end decode tok/s** cannot move like a “KV-only” multiplier: **Amdahl’s law** applies. TurboQuant correctly targets **KV traffic and attention implementation**, but **does not remove the GEMM bucket**—that is **library and stack** territory.

---

## 2. Everything we **implemented or measured** in this repository

| Area | What we did | Artifacts |
|------|-------------|-----------|
| **TurboQuant + vLLM V1** | Production-shaped backend in `tq_backends/`, copy-on-install, `tq3` `CacheDType` patch, registry helpers, wiring doc | `scripts/install_turboquant_vllm_backend.sh`, `tq_backends/attention/backends/rocm_flash_attn.py`, `docs/vllm_turboquant_wiring.md`, `benchmarks/vllm_turboquant_registry.py` |
| **GQA fused decode** | Correct expansion path for compressed KV with GQA | `expand_tq_compressed_for_gqa`, `results/bench_tq_gqa_decode_sweep.json` |
| **Attention microbenchmarks** | Split-K fused TQ3 vs FP16 SDPA vs Python; block-KV sweep | `results/bench_triton_attention.json`, `results/bench_block_kv_sweep.json` |
| **Whole-step vLLM baselines** | Golden kv-heavy JSON; throughput under controlled VRAM knob | `results/decode_whole_step_baseline_kv_heavy.json`, `scripts/run_decode_whole_step_baseline_kv_heavy.sh` |
| **Rocprof story** | Kernel timeline → bucket compare JSON | `benchmarks/bench_vllm_rocprof_timeline.py`, `benchmarks/story2_rocprof_summarize.py`, `results/decode_whole_step_rocprof_bucket_compare.json` |
| **Negative / guardrail experiments** | SwiGLU Triton fusion **no_go**; AWQ slower at this shape; partial scheduler/graphs status | `results/decode_whole_step_ffn_hypothesis_outcome.json`, `results/decode_whole_step_quant_lever_status.json`, `results/decode_whole_step_scheduler_status.json` |
| **ROCm custom paged-attention gate** | **Patch** so “full causal” `sliding_window == max_seq_len−1` does not spuriously disable the ROCm custom path | `scripts/patch_vllm_rocm_sliding_window_custom_paged.py` (invoked from install script) |
| **TurboQuant V1 bridge hot path** | Avoid syncing **all** query lengths to CPU when the batch is **uniform decode** or **uniform prefill** | `tq_backends/vllm_v1_turboquant_bridge.py` |
| **Stack fingerprinting** | Small script to print ROCm/torch/hipBLASLt-relevant environment for upgrade logs | `scripts/print_rocm_gemm_stack_info.py` |
| **Figures for reports and UI** | Fig 27–31 from checked-in JSON / static closure table | `report/generate_figures_v2.py` |

We also **removed** misleading layouts (e.g. in-repo `vllm/` shadowing PyPI) and **documented** Primus / ROCm 7.2 alignment in `report/paper.md`, `docker_run_amd_mi300x.sh`, and env docs.

---

## 3. How that **reduces** the bottleneck (what actually moved)

1. **Paged attention path:** When vLLM’s ROCm helper rejected custom paged attention because `sliding_window` was numerically “on” but **not shorter than context**, the stack could fall back to a **slower** attention implementation. The **idempotent patch** removes that **configuration-induced** penalty **without** changing model math—so we recover **wall-clock that was lost to a gate bug**, not “free” speed from compression.
2. **Bridge overhead:** Uniform batches no longer pay a full **CPU list materialization** just to classify decode vs prefill—reducing **host-side** drag on a path that already fights **GEMM-dominated** steps.
3. **Clarity of bottleneck:** Fig **30** makes the **joint GEMM + attention** story **auditable** from the same JSON the benches emit—so we stop debating anecdotes and start from **shared evidence**.

What **did not** materially shrink from repo-only work: **hipBLASLt matmul selection**, **CUDA graph / piecewise compile stability on ROCm+vLLM**, and **driver/ROCm release cadence**. Those dominate the next **dozen** percent only when addressed on the **shipped** interpreter and wheel.

---

## 4. What is **deliberately outside** this repository (not “unfinished,” **different ownership**)

The following are **real** levers; they are **not** expressed as durable patches inside `amd-experiments` because they bind to **vendor bits**, **cluster policy**, or **vLLM wheel internals** beyond what we can merge-test here:

| Lever | Why it is outside repo-only control |
|-------|-------------------------------------|
| **hipBLASLt / rocBLAS** performance and epilogue fusion | Delivered in **ROCm + PyTorch** builds; wins require **upgrading** and **re-profiling** on MI300X. |
| **vLLM graphs / `torch.compile` / `enforce_eager`** | **Stability matrix** per vLLM minor + ROCm; production enablement needs **logs and smoke** on the customer image. |
| **PagedAttention kernel choice** beyond our gate fix | **aiter**, flash-attn ROCm plugins, and **vLLM version** pin decide which `.so` loads; we document and patch **eligibility**, not replace AMD’s kernel packages. |
| **Scheduler / `max_num_batched_tokens`** | Changes **latency vs throughput** tradeoffs under **SLO**; not a single constant for the repo. |

That boundary is summarized graphically as **Fig 31**.

---

## 5. Closing statement

Within the stated product constraints—**keep accuracy**, **keep aggressive KV compression**, **keep lower memory**—this repository **exhausted** the **implementation-side** levers that are **safe to land in-tree**: wiring, microbenches, whole-step baselines, rocprof truth, guardrail experiments, **one ROCm eligibility patch**, and **bridge micro-optimizations**. **Further batch=1 decode gains** are **deployment and vendor-stack** work, each gated by a **short profile** on the **same** MI300X image you ship—not by additional speculative Python in this repo.
