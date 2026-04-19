#!/usr/bin/env bash
# install_and_run.sh — one-command environment setup + benchmark execution.
#
# Usage examples:
#   bash install_and_run.sh --setup-only
#   bash install_and_run.sh --focused
#   bash install_and_run.sh --full
#
# This script (ROCm 7.2 container-first):
#   1) creates a local virtualenv (default: $ROOT_DIR/.venv) or uses VENV_DIR
#   2) installs project requirements
#   3) validates torch+GPU availability
#   4) runs either focused bottleneck checks or full benchmark suite
#
# For vLLM + locked ROCm torch (project standard), use the installer venv instead:
#   VENV_DIR=$ROOT_DIR/.benchmark_mi300_vllm_frozen/.venv  (see docs/benchmark_mi300_locked_env.md)

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
MODE="setup-only"
MODEL="${MODEL:-mistralai/Mistral-7B-v0.1}"

usage() {
  cat <<'EOF'
install_and_run.sh

Options:
  --setup-only       Only create venv + install dependencies (default)
  --focused          Run focused KV bottleneck validation matrix
  --full             Run existing full benchmark suite (run_all_benchmarks.sh)
  --model <name>     Override model id (default: mistralai/Mistral-7B-v0.1)
  -h, --help         Show this help

Environment (container workflow):
  VENV_DIR=.../.venv           Override venv location (use .../.benchmark_mi300_vllm_frozen/.venv for locked stack)
  AMDEXP_USE_SYSTEM_PYTHON=1   Use preinstalled ROCm torch in container (no .venv)
  AMDEXP_PYTHON=python3       Override interpreter when using system mode
                              (Primus: defaults to /opt/venv/bin/python3 when present;
                              for locked vLLM stack use .../.benchmark_mi300_vllm_frozen/.venv/bin/python)
  AMDEXP_PIP_REQUIREMENTS     With system Python: 0 skips \"pip install -r requirements.txt\"
                              (docker_run_amd_mi300x.sh defaults this to 0 to avoid pip
                              replacing the image ROCm torch with a CUDA +cu wheel). 1 refreshes deps.
  AMDEXP_ALLOW_NO_GPU=1       Emergency only: skip GPU + ROCm-torch checks (not for real benchmarks)

Focused profiling knobs (Transformers full-model decode):
  PROFILE_SEQ_LEN   default 8192
  PROFILE_N_DECODE  default 8
  PROFILE_N_WARMUP  default 2
  PROFILE_BATCH       default 1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --setup-only) MODE="setup-only"; shift ;;
    --focused) MODE="focused"; shift ;;
    --full) MODE="full"; shift ;;
    --model) MODEL="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1"; usage; exit 1 ;;
  esac
done

echo "[install_and_run] root: $ROOT_DIR"
echo "[install_and_run] venv: $VENV_DIR"
echo "[install_and_run] mode: $MODE"
echo "[install_and_run] model: $MODEL"
echo "[install_and_run] use_system_python: ${AMDEXP_USE_SYSTEM_PYTHON:-0}"
echo "[install_and_run] pip_requirements: ${AMDEXP_PIP_REQUIREMENTS:-unset}"

# In AMD ROCm 7.2 MI300X containers (e.g. rocm/primus), PyTorch is preinstalled.
# Set AMDEXP_USE_SYSTEM_PYTHON=1 to skip venv and use container Python (recommended).
if [[ "${AMDEXP_USE_SYSTEM_PYTHON:-0}" == "1" ]]; then
  if [[ -z "${AMDEXP_PYTHON:-}" ]] && [[ -x /opt/venv/bin/python3 ]]; then
    AMDEXP_PYTHON="/opt/venv/bin/python3"
    echo "[install_and_run] defaulting AMDEXP_PYTHON to $AMDEXP_PYTHON"
  fi
  VENV_PY="${AMDEXP_PYTHON:-python3}"
  if ! command -v "$VENV_PY" >/dev/null 2>&1; then
    echo "ERROR: interpreter not found: $VENV_PY" >&2
    exit 2
  fi
  echo "[install_and_run] using system/container Python: $(command -v "$VENV_PY")"
else
  if [[ ! -d "$VENV_DIR" ]]; then
    python3 -m venv "$VENV_DIR"
  fi

  VENV_PY="$VENV_DIR/bin/python3"
  if [[ ! -x "$VENV_PY" ]]; then
    echo "ERROR: venv python not found at $VENV_PY" >&2
    exit 2
  fi

  if ! "$VENV_PY" -m pip --version >/dev/null 2>&1; then
    echo "[install_and_run] pip missing in venv, bootstrapping..."
    if ! "$VENV_PY" -m ensurepip --upgrade >/dev/null 2>&1; then
      TMP_GETPIP="$(mktemp)"
      curl -fsSL https://bootstrap.pypa.io/get-pip.py -o "$TMP_GETPIP"
      "$VENV_PY" "$TMP_GETPIP"
      rm -f "$TMP_GETPIP"
    fi
  fi

  "$VENV_PY" -m pip install --upgrade pip wheel setuptools
fi

# System / container Python: do not aggressively upgrade setuptools — vLLM pins
# setuptools<81 on Python 3.12, and pip upgrades can also pull a CUDA torch if
# requirements resolve badly. ROCm Primus images ship a working torch; preserve it.
if [[ "${AMDEXP_USE_SYSTEM_PYTHON:-0}" == "1" ]]; then
  "$VENV_PY" -m pip install --upgrade pip wheel
  if "$VENV_PY" -c "import vllm" 2>/dev/null; then
    "$VENV_PY" -m pip install "setuptools>=77.0.3,<81.0.0"
  else
    "$VENV_PY" -m pip install "setuptools>=77.0.3,<81.0.0" || true
  fi
fi

# Container-first policy: ROCm 7.2 MI300X images ship a matched PyTorch — preserve it
# (AMDEXP_USE_SYSTEM_PYTHON=1, AMDEXP_PIP_REQUIREMENTS=0 in docker_run_amd_mi300x.sh).

REQ=1
if [[ "${AMDEXP_USE_SYSTEM_PYTHON:-0}" == "1" ]] && [[ "${AMDEXP_PIP_REQUIREMENTS:-1}" == "0" ]]; then
  REQ=0
  echo "[install_and_run] skipping pip install -r requirements.txt (AMDEXP_PIP_REQUIREMENTS=0)"
fi
if [[ "$REQ" == "1" ]]; then
  "$VENV_PY" -m pip install --upgrade-strategy only-if-needed -r requirements.txt
fi

"$VENV_PY" - <<'PY'
import os
import sys
import torch

# ROCm note: PyTorch still exposes AMD GPUs through torch.cuda.* (HIP device
# stack). "cuda" in API names is historical; is_available() is the right check
# for MI300X + ROCm wheels too.
allow_no_gpu = os.environ.get("AMDEXP_ALLOW_NO_GPU", "") == "1"
print(f"torch={torch.__version__}")
hip = getattr(torch.version, "hip", None)
print(f"hip={hip}")
gpu_ok = torch.cuda.is_available()
print(f"gpu_visible_to_torch={gpu_ok}")
if hip is None and "+cu" in torch.__version__:
    print(
        "ERROR: PyTorch looks like a CUDA wheel (+cu*) but torch.version.hip is None. "
        "Primus must use a ROCm torch build. Hint: AMDEXP_PIP_REQUIREMENTS=0 to skip "
        "pip install -r requirements.txt so the image vendor torch is preserved, or "
        "set AMDEXP_PYTHON to a ROCm venv interpreter.",
        file=sys.stderr,
    )
    sys.exit(3)
if not gpu_ok and not allow_no_gpu:
    print(
        "ERROR: No GPU visible to PyTorch (torch.cuda.is_available() is False). "
        "On AMD ROCm this is still the HIP path: check rocm-smi, HIP_VISIBLE_DEVICES, "
        "and that the container/host exposes /dev/kfd and /dev/dri to the process. "
        "To bypass (not recommended): AMDEXP_ALLOW_NO_GPU=1",
        file=sys.stderr,
    )
    sys.exit(2)
if gpu_ok:
    print(f"device={torch.cuda.get_device_name(0)}")
if hip is None and not allow_no_gpu:
    print(
        "ERROR: This PyTorch build is not ROCm (torch.version.hip is None). "
        "Use an AMD ROCm 7.2 MI300X image or a ROCm torch wheel.",
        file=sys.stderr,
    )
    sys.exit(3)
if hip is not None and not str(hip).startswith("7.2"):
    print(f"WARNING: expected ROCm 7.2 in container-first workflow, got hip={hip}", file=sys.stderr)
PY

mkdir -p results/logs

if [[ "$MODE" == "setup-only" ]]; then
  echo "[install_and_run] Setup complete."
  exit 0
fi

if [[ "$MODE" == "focused" ]]; then
  export VLLM_TQ_DEBUG_PATHS=1
  export VLLM_TQ_SDPA_BACKEND=flash
  export VLLM_TQ_USE_FUSED_KERNEL=1

  "$VENV_PY" benchmarks/validate_triton_e2e.py 2>&1 | tee results/logs/validate_triton_e2e.log
  "$VENV_PY" benchmarks/bench_triton_attention.py 2>&1 | tee results/logs/bench_triton_attention.log
  "$VENV_PY" benchmarks/bench_flash_attn_check.py 2>&1 | tee results/logs/bench_flash_attn_check.log
  "$VENV_PY" benchmarks/profile_full_model_decode.py \
    --model "$MODEL" --seq-len "${PROFILE_SEQ_LEN:-8192}" --n-decode "${PROFILE_N_DECODE:-8}" \
    --n-warmup "${PROFILE_N_WARMUP:-2}" --batch-size "${PROFILE_BATCH:-1}" \
    --output results/profile_full_model_decode.json \
    2>&1 | tee results/logs/profile_full_model_decode.log
  "$VENV_PY" benchmarks/bench_vllm_turboquant_ab.py --model "$MODEL" --only-backend fp16 --enforce-eager \
    2>&1 | tee results/logs/bench_vllm_turboquant_ab_fp16.log
  "$VENV_PY" benchmarks/bench_vllm_turboquant_ab.py --model "$MODEL" --only-backend turboquant_fused --enforce-eager \
    2>&1 | tee results/logs/bench_vllm_turboquant_ab_fused.log
  "$VENV_PY" benchmarks/bench_batch_decode_v2.py --model "$MODEL" \
    2>&1 | tee results/logs/bench_batch_decode_v2.log
  echo "[install_and_run] Focused run complete."
  exit 0
fi

if [[ "$MODE" == "full" ]]; then
  bash run_all_benchmarks.sh "$MODEL"
  echo "[install_and_run] Full run complete."
  exit 0
fi

echo "Unexpected mode: $MODE" >&2
exit 1
