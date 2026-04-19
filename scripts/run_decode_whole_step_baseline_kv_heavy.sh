#!/usr/bin/env bash
# Phase A (whole-decode plan): Mistral-7B kv-heavy golden baseline — matches
# results/bench_vllm_turboquant_ab_sweep_kv_heavy.json (input_len=1024, output_len=256, num_prompts=32).
#
# Run inside ROCm 7.2 Primus (docker_run_amd_mi300x.sh) with AMDEXP_USE_SYSTEM_PYTHON=1 recommended.
#
#   bash scripts/run_decode_whole_step_baseline_kv_heavy.sh
#
# Outputs:
#   results/decode_whole_step_baseline_kv_heavy.json
#   results/logs/decode_whole_step_dispatch_kv_heavy.log
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
# In-process EngineCore so TurboQuant CUSTOM backend registration (bench) is visible to vLLM V1.
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
export VLLM_TQ_LOG_DISPATCH="${VLLM_TQ_LOG_DISPATCH:-1}"
mkdir -p results/logs
LOG="$ROOT/results/logs/decode_whole_step_dispatch_kv_heavy.log"
OUT="$ROOT/results/decode_whole_step_baseline_kv_heavy.json"

echo "[decode_whole_step] logging dispatch + benchmark to $LOG" | tee "$LOG"
# If vLLM reports low free VRAM (e.g. after a crashed engine), set VLLM_BENCH_GPU_MEM lower.
GPU_MEM="${VLLM_BENCH_GPU_MEM:-0.85}"
set +o pipefail
"$PY" "$ROOT/benchmarks/bench_vllm_turboquant_ab.py" \
  --model mistralai/Mistral-7B-v0.1 \
  --input-len 1024 \
  --output-len 256 \
  --num-prompts 32 \
  --gpu-memory-utilization "$GPU_MEM" \
  --max-model-len 8192 \
  --enforce-eager \
  --output "$OUT" 2>&1 | tee -a "$LOG"
rc="${PIPESTATUS[0]}"
set -o pipefail
[[ "$rc" -eq 0 ]] || exit "$rc"

echo "[decode_whole_step] wrote $OUT"
