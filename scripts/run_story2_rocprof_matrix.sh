#!/usr/bin/env bash
# Story 2 Phase 1: rocprof kernel timeline for all MODES (see bench_vllm_rocprof_timeline.py).
# Requires: MI300X-class GPU, rocprofv2, vLLM in .venv, TurboQuant backend installed.
#
# Default: lighter shape for faster iteration. Use --kv-heavy for fig29-style params.
#
#   bash scripts/run_story2_rocprof_matrix.sh
#   bash scripts/run_story2_rocprof_matrix.sh -- --kv-heavy-story2 --quant-model TheBloke/Mistral-7B-v0.1-AWQ
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${AMDEXP_PYTHON:-$ROOT/.venv/bin/python3}"
cd "$ROOT"
exec "$PY" benchmarks/bench_vllm_rocprof_timeline.py "$@"
# Then: python3 benchmarks/story2_rocprof_summarize.py
