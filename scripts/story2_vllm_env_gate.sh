#!/usr/bin/env bash
# Story 2 Phase 0: write results/story2_env_gate.json (vLLM probe + optional TQ smoke).
#
#   bash scripts/story2_vllm_env_gate.sh
#   bash scripts/story2_vllm_env_gate.sh -- --skip-vllm-smoke
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${AMDEXP_PYTHON:-$ROOT/.venv/bin/python3}"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3)"
fi
exec "$PY" "$ROOT/benchmarks/story2_env_gate.py" "$@"
