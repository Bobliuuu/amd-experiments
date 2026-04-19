# MI300X decode bottleneck improvement (engineering)

**Canonical closure narrative (what we did, tried, and handed off):** [`repo_decode_bottleneck_closure.md`](repo_decode_bottleneck_closure.md).

This document implements the **bottleneck improvement roadmap**: move wall-clock in the two dominant buckets—**paged attention** (~30% of top-kernel time in kv-heavy Mistral decode) and **hipBLASLt GEMMs** (~43%)—rather than tuning benchmark harnesses.

Evidence: [`docs/decode_whole_step_amdahl_outcome.md`](decode_whole_step_amdahl_outcome.md), [`results/decode_whole_step_rocprof_bucket_compare.json`](../results/decode_whole_step_rocprof_bucket_compare.json).

---

## 1. ROCm custom paged attention vs Triton fallback

**Symptom:** vLLM may log that ROCm custom paged attention is unavailable and fall back to Triton when `sliding_window` is set to a value like `4095` while `max_seq_len` is `4096`—a configuration with **no effective sliding window** (full causal context).

**Fix (deployment):** run the idempotent patch against your vLLM install:

```bash
python3 scripts/patch_vllm_rocm_sliding_window_custom_paged.py
```

`bash scripts/install_turboquant_vllm_backend.sh` also invokes this patch (non-fatal if it fails).

**Background:** [`report/rocm_kv_bottleneck_report.md`](../report/rocm_kv_bottleneck_report.md).

**Validation:** short rocprof before/after on the same model and batch; success = higher tok/s and/or lower share in `kernel_paged_attention_2d` / attention bucket. Reconcile vLLM wheel version with ROCm / `aiter` per AMD guidance for gfx942.

---

## 2. GEMM / MLP stack (hipBLASLt)

Wins here are mostly **library selection, epilogue fusion, and ROCm cadence**—not Python churn.

- Track driver + ROCm + PyTorch builds together; record outputs of `python3 scripts/print_rocm_gemm_stack_info.py` when you change the stack.
- **Fusion policy:** only land vLLM-side fused MLP / custom matmul paths when a microbench or short profile proves a win on MI300X. Repo datapoint rejecting a SwiGLU Triton direction: [`results/decode_whole_step_ffn_hypothesis_outcome.json`](../results/decode_whole_step_ffn_hypothesis_outcome.json).

---

## 3. TurboQuant decode bridge (`tq_backends/`)

TurboQuant targets **KV / attention**; it does not remove the GEMM bucket by itself.

- Install path: `bash scripts/install_turboquant_vllm_backend.sh`, wiring doc: [`docs/vllm_turboquant_wiring.md`](vllm_turboquant_wiring.md).
- The V1 bridge [`tq_backends/vllm_v1_turboquant_bridge.py`](../tq_backends/vllm_v1_turboquant_bridge.py) avoids unnecessary host syncs on uniform decode / prefill batches when classifying `query_start_loc` segments.

---

## 4. Production serving: compile / graphs

**Goal:** sustained decode tok/s under load where ROCm + vLLM are stable.

- **Graphs / piecewise compile** are a **stability** task: enable only with logs and smoke tests on your exact vLLM + ROCm combo. A subprocess graphs leg failing with `exit=-6` in a bench harness is a **diagnostic signal**, not the production knob.
- Typical tension: **`--enforce-eager`** (or `enforce_eager=True`) disables graphs for easier debugging; turning eager off may improve throughput when graphs compile cleanly.
- **Scheduler / batching** (`max_num_batched_tokens`, concurrency) shifts the attention vs GEMM balance; tune against latency SLOs using the same two-bucket mental model.

---

## Out of scope

Benchmark-only refactors, `empty_cache` rituals, and chasing `gpu_memory_utilization` unless they unblock a **higher** real `max_model_len` or batch in production.
