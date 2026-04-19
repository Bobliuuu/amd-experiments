#!/usr/bin/env bash
# Story 2 Phase 2: FP16 vs TQ vs quant+TQ at kv-heavy shape (matches sweep JSON defaults).
#
#   bash scripts/run_story2_quant_kv_heavy.sh
#   QUANT_MODEL=TheBloke/Mistral-7B-v0.1-AWQ bash scripts/run_story2_quant_kv_heavy.sh
#
# One Python process per backend: on some ROCm+vLLM builds, loading FP16 Mistral then
# AWQ in the same interpreter leaves torch.ops._C without awq_dequantize until restart.
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
QUANT_MODEL="${QUANT_MODEL:-TheBloke/Mistral-7B-v0.1-AWQ}"
OUT="${ROOT}/results/story2_quant_kv_heavy_ab.json"
cd "$ROOT"
export PYTHONPATH="${ROOT}/kernels:${ROOT}${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING:-0}"
# Match decode_whole_step baseline when vLLM sees low free VRAM (set lower, e.g. 0.15).
GPU_MEM="${VLLM_BENCH_GPU_MEM:-0.90}"

TMPD="$(mktemp -d "${TMPDIR:-/tmp}/story2_quant_kv_heavy.XXXXXX")"
trap 'rm -rf "$TMPD"' EXIT

COMMON=(
  benchmarks/bench_vllm_turboquant_ab.py
  --model mistralai/Mistral-7B-v0.1
  --quant-model "$QUANT_MODEL"
  --quantization awq
  --input-len 1024
  --output-len 256
  --num-prompts 32
  --gpu-memory-utilization "$GPU_MEM"
  --max-model-len 8192
  --enforce-eager
)

MODES=(fp16 turboquant_decompress turboquant_fused quant_fp16_kv quant_turboquant_decompress quant_turboquant_fused)
idx=0
for m in "${MODES[@]}"; do
  idx=$((idx + 1))
  echo "[run_story2_quant_kv_heavy] backend=$m ($idx/${#MODES[@]})" >&2
  "$PY" "${COMMON[@]}" --only-backend "$m" --output "$TMPD/part_${idx}.json"
done

if [[ "${STORY2_QUANT_SPIKE:-1}" != "0" ]]; then
  set +e
  "$PY" benchmarks/spike_vllm_rocm_quant.py \
    --quant-model "$QUANT_MODEL" \
    --gpu-memory-utilization "$GPU_MEM" \
    --max-model-len 8192 \
    --enforce-eager \
    2>&1 | tail -20
  set -e
fi

"$PY" - "$TMPD" "$OUT" <<'PY'
import json
import sys
from pathlib import Path

tmpd, out_path = Path(sys.argv[1]), Path(sys.argv[2])
for i in range(1, 7):
    p = tmpd / f"part_{i}.json"
    if not p.is_file():
        sys.stderr.write(f"missing expected partial: {p.name}\n")
        sys.exit(1)
p1 = tmpd / "part_1.json"
base = json.loads(p1.read_text())
base["results"] = []
merged = 0
for i in range(1, 7):
    p = tmpd / f"part_{i}.json"
    doc = json.loads(p.read_text())
    rows = doc.get("results") or []
    if len(rows) != 1:
        sys.stderr.write(f"{p.name}: expected exactly 1 result row, got {len(rows)}\n")
        sys.exit(1)
    base["results"].append(rows[0])
    merged += 1
if len(base["results"]) != 6:
    sys.stderr.write("expected 6 backend rows\n")
    sys.exit(1)
out_path.parent.mkdir(parents=True, exist_ok=True)
out_path.write_text(json.dumps(base, indent=2))
print(f"Merged {merged} part file(s) -> {out_path}")
PY

echo "Wrote $OUT"
echo "Compare throughput_output_tps across backends; expect GEMM-limited flatness without quant wins."
