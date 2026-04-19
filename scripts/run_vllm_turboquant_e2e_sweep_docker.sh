#!/usr/bin/env bash
# One-shot vLLM TurboQuant e2e decode sweep inside docker_run_amd_mi300x.sh.
#
# Why this exists:
#   - `docker run --rm` does not persist `pip install` across separate invocations.
#   - Primus images may ship a broken /workspace/FBGEMM on sys.path (torchrec); remove it.
#   - PYTHONPATH should be only kernels/ (see docs/vllm_turboquant_wiring.md).
#
# Usage (from repo root):
#   bash scripts/run_vllm_turboquant_e2e_sweep_docker.sh
#   bash scripts/run_vllm_turboquant_e2e_sweep_docker.sh --input-len 512 --output-len 64
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

EXTRA=( "$@" )
if [[ "${#EXTRA[@]}" -eq 0 ]]; then
  EXTRA=( --model mistralai/Mistral-7B-v0.1 --max-model-lens 8192 --input-len 4096 --num-prompts-list 1 --output-len 128 --enforce-eager --output results/bench_vllm_turboquant_e2e_sweep.json )
fi

SWEEP_CMD=( python3 benchmarks/bench_vllm_turboquant_e2e_sweep.py "${EXTRA[@]}" )
bash docker_run_amd_mi300x.sh -- bash -lc "rm -rf /workspace/FBGEMM 2>/dev/null || true; cd /workspace/amd-experiments && unset PYTHONPATH && export PYTHONPATH=/workspace/amd-experiments/kernels && (python3 -c 'import vllm._rocm_C' 2>/dev/null || python3 -m pip install -q vllm) && bash scripts/install_turboquant_vllm_backend.sh && $(printf '%q ' "${SWEEP_CMD[@]}")"
