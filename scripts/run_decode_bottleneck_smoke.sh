#!/usr/bin/env bash
# One-command minimal decode benchmark (Mistral → TinyLlama fallback) for stack sanity.
#
#   bash scripts/run_decode_bottleneck_smoke.sh
#   BOTTLENECK_MODELS="TinyLlama/TinyLlama-1.1B-Chat-v1.0" bash scripts/run_decode_bottleneck_smoke.sh
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

EXTRA=( )
if [[ -n "${BOTTLENECK_MODELS:-}" ]]; then
  EXTRA+=( --models "$BOTTLENECK_MODELS" )
fi
if [[ -n "${BOTTLENECK_GPU_MEM:-}" ]]; then
  EXTRA+=( --gpu-memory-utilization "$BOTTLENECK_GPU_MEM" )
fi

"$PY" "$ROOT/benchmarks/decode_bottleneck_smoke.py" "${EXTRA[@]}"
echo "→ results/decode_bottleneck_smoke.json"
