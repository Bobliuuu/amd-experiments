# vLLM + TurboQuant (fused GQA) wiring

## Goal

Run **real** decode through vLLM with `TURBOQUANT_ROCM`, **tq3** KV cache, and optional **fused Triton** decode (`VLLM_TQ_USE_FUSED_KERNEL=1`), then benchmark and profile.

**After wiring:** for the full account of **in-repo mitigations** (patches, bridge tweaks, rocprof evidence) versus **deployment-only** throughput levers, read [`repo_decode_bottleneck_closure.md`](repo_decode_bottleneck_closure.md).

## Why the repo uses `tq_backends/`

The package **`tq_backends/`** holds drop-in attention modules. It must **not** be named `vllm`, or `import vllm` would resolve to the repo stub instead of PyPI vLLM.

When running vLLM benchmarks:

- Put **`kernels/`** on `PYTHONPATH` for `turboquant_mi300x` and `tq_triton`.
- Put the **repo root** on `PYTHONPATH` **only** for `register_turboquant_rocm_backend()` (imports `tq_backends`), or import the registry from a cwd that has the repo on path.

**Which Python?** In this repo, **PyTorch + vLLM for benchmarks** live in the **locked installer venv**: **`<repo>/.benchmark_mi300_vllm_frozen/.venv/bin/python`** (see **`docs/benchmark_mi300_locked_env.md`**). System `/usr/bin/python3` or a random **`.venv`** usually does **not** match that stack.

```bash
"$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/python" -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__))"
```

## Steps

1. **Install vLLM** built for your ROCm + torch (see `requirements-vllm-rocm.txt` for notes). Verify with the **venv** interpreter:

   ```bash
   "$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/python" -c "import vllm, pathlib; print(pathlib.Path(vllm.__file__))"
   ```

   The path should be under `site-packages`, not `amd-experiments`.

2. **Copy the backend** into site-packages:

   ```bash
   bash scripts/install_turboquant_vllm_backend.sh
   ```

3. **Register the backend** before constructing `LLM` (done automatically in `bench_vllm_turboquant_ab.py` and `bench_vllm_turboquant_e2e_sweep.py`):

   ```python
   from benchmarks.vllm_turboquant_registry import register_turboquant_rocm_backend
   register_turboquant_rocm_backend()
   ```

4. **Environment**

   - `VLLM_ATTENTION_BACKEND=TURBOQUANT_ROCM`
   - `VLLM_KV_CACHE_DTYPE=tq3` (some vLLM builds ignore this env; the benchmark also passes `kv_cache_dtype` to `LLM()` when supported.)
   - `VLLM_TQ_USE_FUSED_KERNEL=1` (fused) or `0` (decompress + SDPA)
   - `VLLM_TQ_LOG_DISPATCH=1` — prints `[TQ_DISPATCH] path=FUSED_TRITON|DECOMPRESS_SDPA ...` per decode forward

5. **Optional:** `export VLLM_USE_V1=0` if the V1 engine does not load custom backends.

## Profiling

- Use **`--enforce-eager`** on benchmarks when capturing `rocprofv2` / `torch.profiler` to avoid CUDA graph noise.
- With `VLLM_TQ_LOG_DISPATCH=1`, confirm **FUSED_TRITON** on Mistral-style GQA after the `expand_tq_compressed_for_gqa` fix.

## Weight-only quantization + TQ3 KV (serving bottleneck)

End-to-end tok/s is often **GEMM / MLP limited**; TQ3 KV helps most when the **non-KV**
path is small enough that attention matters. Combine:

1. **FP16 baseline** — default `mistralai/Mistral-7B-v0.1`, no `VLLM_ATTENTION_BACKEND`.
2. **TQ3 KV** — `VLLM_ATTENTION_BACKEND=TURBOQUANT_ROCM`, `kv_cache_dtype=tq3`, fused toggle as above.
3. **Quantized weights** — pass an AWQ/GPTQ HF checkpoint and vLLM `quantization=`:

```bash
# Six-way matrix (FP16 + TQ decompress/fused + same with AWQ weights) when --quant-model is set
python3 benchmarks/bench_vllm_turboquant_ab.py \
  --model mistralai/Mistral-7B-v0.1 \
  --quant-model TheBloke/Mistral-7B-v0.1-AWQ \
  --quantization awq \
  --input-len 512 --output-len 128 --num-prompts 20
```

`bench_vllm_turboquant_ab.py` backends:

| `--only-backend` | Weights | KV / attention |
|------------------|---------|----------------|
| `fp16` | FP16 checkpoint | FP16 + default ROCm attention |
| `turboquant_decompress` | FP16 | TQ3 + decompress + SDPA |
| `turboquant_fused` | FP16 | TQ3 + fused Triton decode |
| `quant_fp16_kv` | AWQ/GPTQ (`--quant-model`) | FP16 KV |
| `quant_turboquant_decompress` | AWQ/GPTQ | TQ3 + decompress |
| `quant_turboquant_fused` | AWQ/GPTQ | TQ3 + fused |

Unsupported `LLM()` kwargs for your vLLM version are **dropped** automatically; see `llm_kwargs_dropped` in each result JSON.

**Probe this ROCm build** (which quant formats load):

```bash
python3 benchmarks/spike_vllm_rocm_quant.py --quant-model TheBloke/Mistral-7B-v0.1-AWQ
# → results/spike_vllm_rocm_quant.json
```

**KV-sensitive sweeps** (higher concurrency, longer prompt, optional `max_num_batched_tokens`):

```bash
python3 benchmarks/bench_vllm_turboquant_e2e_sweep.py \
  --input-lens 512,4096 \
  --num-prompts-list 1,8,16 \
  --max-num-batched-tokens 8192 \
  --enforce-eager
```

**Eager vs CUDA graphs** (serving path A/B):

```bash
python3 benchmarks/bench_vllm_serving_path_ab.py --only-backend fp16
# → results/bench_vllm_serving_path_ab.json
```

**Quant vs FP16 greedy smoke** (short decode, first token mismatch index):

```bash
python3 benchmarks/bench_vllm_quant_quality_smoke.py \
  --quant-model TheBloke/Mistral-7B-v0.1-AWQ --quantization awq
```

**rocprof** including quantized modes (requires `--quant-model` for `quant_*`):

```bash
python3 benchmarks/bench_vllm_rocprof_timeline.py \
  --model mistralai/Mistral-7B-v0.1 \
  --quant-model TheBloke/Mistral-7B-v0.1-AWQ
# → results/bench_vllm_rocprof_timeline_summary.json
```

**Optional hardware counters** (same harness): append `--pmc FETCH_SIZE,WRITE_SIZE,SQ_INSTS_VALU` (comma-separated; exact names depend on `rocprofv2 --list-metrics` on your gfx target). On many **VF** partitions, large PMC sets fail with `ROCPROFILER_STATUS_ERROR_PROFILE_EXCEEDS_HW_LIMIT` — prefer default **kernel timeline only**, or a minimal counter set; see [`benchmarks/profile_rocprof.py`](../benchmarks/profile_rocprof.py) and `report/paper.md` (profiling / VF note).

**Custom FFN fusion spike** (not vLLM-integrated): [`kernels/ffn_fused_swiglu_spike.py`](../kernels/ffn_fused_swiglu_spike.py).

Scheduler-related kwargs to reproduce sweeps: `max_num_batched_tokens`, `max_num_seqs` (passed through A/B and sweep when supported).

## Whole-decode (Amdahl) — outcome doc

End-to-end question: *if KV/attention improves but tok/s is flat, what dominates the decode step?* See **[`docs/decode_whole_step_amdahl_outcome.md`](decode_whole_step_amdahl_outcome.md)** (golden kv-heavy baseline, rocprof buckets, lever status). Refresh baseline on MI300X: **`bash scripts/run_decode_whole_step_baseline_kv_heavy.sh`**.

## Story 2 profiling playbook (E2E vs isolated attention)

**Problem (from `report/final_report_v2.md` §13.2):** fused TQ3 attention wins in isolation at long `seq_k`, but full vLLM decode tok/s is often flat because **MLP/GEMM and framework** dominate wall time (Amdahl). This section is the **measure → reduce** loop: prove the stack, profile the **whole step**, then change the non-KV path.

### Phase 0 — Environment gate (artifact)

Run once after vLLM / torch upgrades or container rebuild:

```bash
bash scripts/story2_vllm_env_gate.sh
# import-only (no GPU load):
bash scripts/story2_vllm_env_gate.sh -- --skip-vllm-smoke
```

Writes **`results/story2_env_gate.json`**: interpreter path (use **`<repo>/.benchmark_mi300_vllm_frozen/.venv/bin/python`**, not a resolved `/usr/bin` symlink alone), `vllm_probe` (`pypi_or_system` vs stub), optional **short** `turboquant_fused` smoke with **`VLLM_TQ_LOG_DISPATCH=1`** and counts of **`[TQ_DISPATCH] path=FUSED_TRITON`** lines. Full smoke needs a working **torch ↔ vLLM** pair on GPU.

### Phase 1 — Dispatch logs + rocprof (same build as benchmarks)

1. **Dispatch logs** — every decode forward can log the attention path:

```bash
export VLLM_TQ_LOG_DISPATCH=1
export PYTHONPATH=amd-experiments/kernels:amd-experiments
# tee to results/logs/ for a permanent record
mkdir -p results/logs
```

2. **Kernel timeline (recommended: eager)** — [`benchmarks/bench_vllm_rocprof_timeline.py`](../benchmarks/bench_vllm_rocprof_timeline.py) wraps `bench_vllm_turboquant_ab.py` under **rocprofv2** (`--enforce-eager` default). Modes include **`fp16`**, **`turboquant_decompress`**, **`turboquant_fused`**, **`quant_fp16_kv`**, **`quant_turboquant_decompress`**, **`quant_turboquant_fused`** (quant modes need `--quant-model`).

   **Match fig29 / kv-heavy A/B shape** (`results/bench_vllm_turboquant_ab_sweep_kv_heavy.json`: `input_len=1024`, `output_len=256`, `num_prompts=32`):

```bash
bash scripts/run_story2_rocprof_matrix.sh -- \
  --kv-heavy-story2 \
  --quant-model TheBloke/Mistral-7B-v0.1-AWQ
```

   Lighter defaults remain for quick iteration on smaller shapes.

3. **Bucket summary** — merge top-kernel shares into coarse buckets (GEMM vs attention-ish vs other):

```bash
python3 benchmarks/story2_rocprof_summarize.py
# → results/story2_rocprof_bucket_compare.json
```

4. **`torch.profiler` on vLLM** — optional and often awkward with async engine + graphs; prefer **rocprof timeline** + **`VLLM_TQ_LOG_DISPATCH`** first. HF-only full-step buckets: [`benchmarks/profile_full_model_decode.py`](../benchmarks/profile_full_model_decode.py).

### Phase 2 — Quant weights, FFN hypothesis, scheduler

- **Quant + kv-heavy A/B** (after `spike_vllm_rocm_quant` proves AWQ loads on your ROCm build):

```bash
bash scripts/run_story2_quant_kv_heavy.sh
python3 benchmarks/story2_kv_heavy_quant_compare.py
# → results/story2_quant_kv_heavy_comparison.json
```

- **FFN / SwiGLU go/no-go** — criteria and integration notes: **`results/story2_ffn_integration_hypothesis.json`**, microbench **`kernels/ffn_fused_swiglu_spike.py`**.

- **Scheduler / graphs vs eager** — higher concurrency can surface KV/attention in E2E:

```bash
bash scripts/run_story2_scheduler_sweep.sh
```

## Scripts

| Script | Purpose |
|--------|---------|
| [`scripts/install_turboquant_vllm_backend.sh`](../scripts/install_turboquant_vllm_backend.sh) | Copy `rocm_flash_attn.py` into installed vLLM |
| [`scripts/install_isoquant_vllm_backend.sh`](../scripts/install_isoquant_vllm_backend.sh) | Patch `iq3` cache dtype + optional copy of IsoQuant backend into vLLM |
| [`benchmarks/vllm_isoquant_registry.py`](../benchmarks/vllm_isoquant_registry.py) | Register IsoQuant on V1 `AttentionBackendEnum.CUSTOM` (mutually exclusive with TurboQuant registry in the same process) |
| [`benchmarks/bench_vllm_isoquant_smoke.py`](../benchmarks/bench_vllm_isoquant_smoke.py) | Short decode smoke test for `kv_cache_dtype=iq3` |
| [`benchmarks/vllm_turboquant_registry.py`](../benchmarks/vllm_turboquant_registry.py) | Registry / monkey-patch |
| [`benchmarks/bench_vllm_turboquant_ab.py`](../benchmarks/bench_vllm_turboquant_ab.py) | FP16 / TQ / **quant weights** matrix + optional `max_num_batched_tokens` |
| [`benchmarks/bench_vllm_turboquant_e2e_sweep.py`](../benchmarks/bench_vllm_turboquant_e2e_sweep.py) | Context × concurrency × optional quant + batching |
| [`benchmarks/spike_vllm_rocm_quant.py`](../benchmarks/spike_vllm_rocm_quant.py) | Record which `LLM()` quant trials load on this stack |
| [`benchmarks/bench_vllm_serving_path_ab.py`](../benchmarks/bench_vllm_serving_path_ab.py) | Eager vs default graphs throughput comparison |
| [`benchmarks/bench_vllm_quant_quality_smoke.py`](../benchmarks/bench_vllm_quant_quality_smoke.py) | Greedy token prefix match vs FP16 |
| [`benchmarks/bench_vllm_rocprof_timeline.py`](../benchmarks/bench_vllm_rocprof_timeline.py) | rocprofv2 kernel rollup per backend |
| [`kernels/ffn_fused_swiglu_spike.py`](../kernels/ffn_fused_swiglu_spike.py) | Triton SwiGLU elementwise fusion microbench |
| [`scripts/story2_vllm_env_gate.sh`](../scripts/story2_vllm_env_gate.sh) | Story 2 Phase 0 → `results/story2_env_gate.json` |
| [`benchmarks/story2_env_gate.py`](../benchmarks/story2_env_gate.py) | Env gate implementation (import + optional TQ smoke) |
| [`scripts/run_story2_rocprof_matrix.sh`](../scripts/run_story2_rocprof_matrix.sh) | Wrapper for rocprof timeline matrix |
| [`benchmarks/story2_rocprof_summarize.py`](../benchmarks/story2_rocprof_summarize.py) | Kernel top-k → bucket JSON |
| [`scripts/run_story2_quant_kv_heavy.sh`](../scripts/run_story2_quant_kv_heavy.sh) | Quant + TQ kv-heavy A/B → `story2_quant_kv_heavy_ab.json` |
| [`benchmarks/story2_kv_heavy_quant_compare.py`](../benchmarks/story2_kv_heavy_quant_compare.py) | Merge reference + quant throughput JSON |
| [`scripts/run_story2_scheduler_sweep.sh`](../scripts/run_story2_scheduler_sweep.sh) | Serving path A/B + turboquant e2e sweep (scheduler story) |
| [`scripts/run_decode_whole_step_baseline_kv_heavy.sh`](../scripts/run_decode_whole_step_baseline_kv_heavy.sh) | Golden Mistral kv-heavy vLLM baseline + dispatch log |
| [`benchmarks/decode_whole_step_golden_driver.py`](../benchmarks/decode_whole_step_golden_driver.py) | Package baseline JSON from an existing bench JSON |

## See also (framing + figures)

- **`report/final_report.md` §14** / **`report/final_report_v2.md` §13** — two deployment stories (memory vs speed), **next step** question for e2e tok/s, **resolved issues** table, and **`fig29_story_e2e_vs_isolated_attention_comparison.png`** (two charts; interpretation in prose). Run **`python3 report/generate_figures_v2.py`** for **`fig27`–`fig29`** in `report/figures_v2/`.
