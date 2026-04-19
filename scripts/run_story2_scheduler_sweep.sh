#!/usr/bin/env bash
# Story 2 Phase 2: scheduler / batching knobs + eager vs graphs (Story 2 may surface under load).
#
# Writes JSON artifacts under results/ — edit OUT names if re-running to avoid overwrite.
#
# VRAM fraction (vLLM startup checks reported *free* VRAM vs utilization × total):
#   - If STORY2_SCHEDULER_GPU_MEM is set, that value wins.
#   - Else if STORY2_SCHEDULER_AUTO_GPU_MEM=1 (default): use scripts/estimate_vllm_safe_gpu_mem_frac.py
#     so a nearly-full GPU from other processes still runs (lower fraction).
#   - Else: VLLM_BENCH_GPU_MEM or 0.88.
#   Override auto: STORY2_SCHEDULER_GPU_MEM=0.90 STORY2_SCHEDULER_AUTO_GPU_MEM=0 bash ...
#
# Optional cleanup before heavy loads (torch empty_cache in this venv):
#   STORY2_SCHEDULER_CLEAN_GPU=0   — skip
#   STORY2_SCHEDULER_CLEAN_GPU=1   — default: run scripts/gpu_torch_empty_cache.py + short sleep
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ -n "${AMDEXP_PYTHON:-}" ]]; then
  PY="$AMDEXP_PYTHON"
elif [[ -x "$ROOT/.benchmark_mi300_vllm_frozen/.venv/bin/python3" ]]; then
  PY="$ROOT/.benchmark_mi300_vllm_frozen/.venv/bin/python3"
else
  PY="$ROOT/.venv/bin/python3"
fi
cd "$ROOT"
export PYTHONPATH="${ROOT}/kernels:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"

if [[ -n "${STORY2_SCHEDULER_GPU_MEM:-}" ]]; then
  GPU_MEM="$STORY2_SCHEDULER_GPU_MEM"
elif [[ "${STORY2_SCHEDULER_AUTO_GPU_MEM:-1}" != "0" ]]; then
  GPU_MEM="$("$PY" "$ROOT/scripts/estimate_vllm_safe_gpu_mem_frac.py")"
  echo "[run_story2_scheduler_sweep] auto GPU_MEM=$GPU_MEM (torch mem_get_info headroom)" >&2
else
  GPU_MEM="${VLLM_BENCH_GPU_MEM:-0.88}"
fi

if [[ "${STORY2_SCHEDULER_CLEAN_GPU:-1}" != "0" ]]; then
  echo "[run_story2_scheduler_sweep] idle GPU cleanup (torch cuda empty_cache)…" >&2
  "$PY" "$ROOT/scripts/gpu_torch_empty_cache.py" || true
  sleep "${STORY2_SCHEDULER_CLEAN_SLEEP_S:-2}"
fi

# Sweep grid: override with e.g.
#   STORY2_SCHEDULER_MAX_LENS=8192,16384 STORY2_SCHEDULER_NPROMPTS=1,8,16 STORY2_SCHEDULER_INLENS=512,1024,4096
# Default max_model_len grid: 8192 only avoids 16384×wide-batch startup failures on tight VRAM.
MAX_LENS="${STORY2_SCHEDULER_MAX_LENS:-8192}"
INLENS="${STORY2_SCHEDULER_INLENS:-512,1024,4096}"
NPROMPTS="${STORY2_SCHEDULER_NPROMPTS:-1,8,16}"

# 1) Eager vs default graphs (single backend smoke) — always writes JSON if eager leg succeeds
"$PY" benchmarks/bench_vllm_serving_path_ab.py --only-backend fp16 \
  --gpu-memory-utilization "$GPU_MEM" \
  --output "$ROOT/results/story2_serving_path_ab_fp16.json"

if [[ "${STORY2_SCHEDULER_CLEAN_GPU:-1}" != "0" ]]; then
  echo "[run_story2_scheduler_sweep] post–serving-path cleanup before e2e sweep…" >&2
  "$PY" "$ROOT/scripts/gpu_torch_empty_cache.py" || true
  sleep "${STORY2_SCHEDULER_CLEAN_SLEEP_S:-2}"
fi

# 2) TurboQuant e2e sweep: concurrency × context (scheduler-sensitive)
"$PY" benchmarks/bench_vllm_turboquant_e2e_sweep.py \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-lens "$MAX_LENS" \
  --input-lens "$INLENS" \
  --num-prompts-list "$NPROMPTS" \
  --max-num-batched-tokens 8192 \
  --enforce-eager \
  --output "$ROOT/results/story2_turboquant_e2e_scheduler_sweep.json"

echo "Artifacts:"
echo "  $ROOT/results/story2_serving_path_ab_fp16.json"
echo "  $ROOT/results/story2_turboquant_e2e_scheduler_sweep.json"
echo "Inspect for throughput deltas when num_prompts or input_len increases (attention/KV fraction)."
