#!/usr/bin/env bash
# docker_run_amd_mi300x.sh — run amd-experiments inside AMD's ROCm 7.2 MI300X stack.
#
# Same device/security/volume pattern as AMD ROCm docs (Primus / PyTorch training container):
#   https://rocm.docs.amd.com/en/docs-7.2.1/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html
#
# Default image: rocm/primus:v26.2 (ROCm 7.2.0, gfx942 / MI300X class)
#
# Usage:
#   bash docker_run_amd_mi300x.sh --pull --setup-only
#   bash docker_run_amd_mi300x.sh --focused
#   bash docker_run_amd_mi300x.sh -- bash -lc 'python3 benchmarks/bench_flash_attn_check.py'
#
#   ROCM_IMAGE=rocm/primus:v26.2 bash docker_run_amd_mi300x.sh --setup-only
#
# GPU / ROCm: benchmarks require a real MI300X (or passed-through GPU). Run
#   bash scripts/verify_mi300x_stack.sh
# inside the container if torch.cuda.is_available() is False — see
# docs/mi300x_docker_runtime_requirements.md
#
# vLLM (canonical): install with scripts/install_rocm72_uv_torch_vllm.sh into
#   $REPO/.benchmark_mi300_vllm_frozen (see docs/benchmark_mi300_locked_env.md,
#   docs/rocm72_uv_torch_vllm_venv.md). Then TurboQuant backend:
#   bash scripts/install_turboquant_vllm_backend.sh
#   export PYTHONPATH=/workspace/amd-experiments/kernels:$PYTHONPATH
# See docs/vllm_turboquant_wiring.md. Do not put repo root before site-packages when
# importing PyPI vLLM (tq_backends/ replaces the old vllm/ stub to avoid shadowing).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROCM_IMAGE="${ROCM_IMAGE:-rocm/primus:v26.2}"

TTY_ARGS=()
if [[ -t 0 ]] && [[ -t 1 ]]; then
  TTY_ARGS=(-it)
else
  TTY_ARGS=(-i)
fi

DO_PULL=0
PRESET=""

usage() {
  cat <<'EOF'
docker_run_amd_mi300x.sh [options] [--] [command ...]

Options:
  --pull              docker pull ROCM_IMAGE before run
  --setup-only        run install_and_run.sh --setup-only
  --focused           run install_and_run.sh --focused
  --full              run install_and_run.sh --full
  -h, --help          this help

With no preset and no arguments after -- : interactive login shell.

Environment:
  ROCM_IMAGE          Docker image (default: rocm/primus:v26.2)
  AMDEXP_EXTRA_ARGS   extra docker run arguments (space-separated, use with care)
  AMDEXP_PIP_REQUIREMENTS  Passed into the container (default here: 0). When 0 and
                      AMDEXP_USE_SYSTEM_PYTHON=1, install_and_run.sh skips
                      \"pip install -r requirements.txt\" so the image's ROCm PyTorch
                      is not replaced by a CUDA wheel. Set to 1 to refresh deps.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --pull) DO_PULL=1; shift ;;
    --setup-only) PRESET="setup-only"; shift ;;
    --focused) PRESET="focused"; shift ;;
    --full) PRESET="full"; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    *) break ;;
  esac
done

if [[ "$DO_PULL" == "1" ]]; then
  echo "[docker_run_amd_mi300x] pulling $ROCM_IMAGE"
  docker pull "$ROCM_IMAGE"
fi

echo "[docker_run_amd_mi300x] image: $ROCM_IMAGE"
echo "[docker_run_amd_mi300x] mount: $SCRIPT_DIR -> /workspace/amd-experiments"

# Build inner command: preset wins, else passthrough, else bash -l
if [[ -n "$PRESET" ]]; then
  INNER="bash install_and_run.sh --$PRESET"
elif [[ $# -gt 0 ]]; then
  INNER=$(printf '%q ' "$@")
else
  INNER="bash -l"
fi

echo "[docker_run_amd_mi300x] inner: $INNER"

# shellcheck disable=SC2206
EXTRA=( ${AMDEXP_EXTRA_ARGS:-} )

exec docker run "${TTY_ARGS[@]}" --rm \
  --device /dev/dri \
  --device /dev/kfd \
  --network host \
  --ipc host \
  --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --privileged \
  -v "$SCRIPT_DIR:/workspace/amd-experiments" \
  --shm-size 64G \
  -w /workspace/amd-experiments \
  -e AMDEXP_USE_SYSTEM_PYTHON=1 \
  -e "AMDEXP_PIP_REQUIREMENTS=${AMDEXP_PIP_REQUIREMENTS:-0}" \
  "${EXTRA[@]}" \
  "$ROCM_IMAGE" \
  bash -lc "cd /workspace/amd-experiments && export AMDEXP_USE_SYSTEM_PYTHON=1 && exec $INNER"
