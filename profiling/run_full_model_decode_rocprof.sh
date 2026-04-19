#!/usr/bin/env bash
# Run profile_full_model_decode.py under rocprofv2 kernel trace (full model, MI300X).
#
# Usage (from amd-experiments/, host or inside ROCm container):
#   bash profiling/run_full_model_decode_rocprof.sh
#   ROCM_IMAGE=rocm/primus:v26.2 bash docker_run_amd_mi300x.sh -- bash -lc \
#     'cd /workspace/amd-experiments && AMDEXP_USE_SYSTEM_PYTHON=1 bash profiling/run_full_model_decode_rocprof.sh'
#
# Output directory: results/rocprof_full_model_decode_<timestamp>/

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
STAMP="$(date +%Y%m%d_%H%M%S)"
OUT="results/rocprof_full_model_decode_${STAMP}"
mkdir -p "$OUT"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"
SEQ_LEN="${SEQ_LEN:-8192}"
N_DECODE="${N_DECODE:-8}"
PY="${AMDEXP_PYTHON:-python3}"

echo "[rocprof] out_dir=$OUT model=$MODEL seq_len=$SEQ_LEN n_decode=$N_DECODE"

exec rocprofv2 -d "$OUT" --kernel-trace \
  "$PY" benchmarks/profile_full_model_decode.py \
  --model "$MODEL" \
  --seq-len "$SEQ_LEN" \
  --n-decode "$N_DECODE" \
  --output "$OUT/profile_full_model_decode.json"
