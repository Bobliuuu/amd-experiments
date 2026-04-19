#!/usr/bin/env bash
# verify_mi300x_stack.sh — print MI300X / ROCm / PyTorch GPU diagnostics.
#
# Run on the host or inside docker (e.g. after docker_run_amd_mi300x.sh with bash -l).
# Picks the first Python that can `import torch`: AMDEXP_PYTHON (if set and valid),
# then /opt/venv/bin/python3, then repo .benchmark_mi300_vllm_frozen/.venv, then repo .venv, then python3 on PATH.
#
#   bash scripts/verify_mi300x_stack.sh
#   AMDEXP_PYTHON=/opt/venv/bin/python3 bash scripts/verify_mi300x_stack.sh

set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY=""
PY_HOW=""

if [[ -n "${AMDEXP_PYTHON:-}" ]]; then
  if ! "$AMDEXP_PYTHON" -c "import sys" >/dev/null 2>&1; then
    echo "ERROR: AMDEXP_PYTHON is not runnable: $AMDEXP_PYTHON" >&2
    exit 2
  fi
  if ! "$AMDEXP_PYTHON" -c "import torch" >/dev/null 2>&1; then
    echo "ERROR: AMDEXP_PYTHON has no importable torch: $AMDEXP_PYTHON" >&2
    echo "  Check: $AMDEXP_PYTHON -c 'import torch'" >&2
    exit 2
  fi
  PY="$AMDEXP_PYTHON"
  PY_HOW="AMDEXP_PYTHON"
else
  for cand in /opt/venv/bin/python3 "$ROOT/.benchmark_mi300_vllm_frozen/.venv/bin/python3" "$ROOT/.venv/bin/python3"; do
    if [[ -x "$cand" ]] && "$cand" -c "import torch" >/dev/null 2>&1; then
      PY="$cand"
      PY_HOW="auto"
      break
    fi
  done
  if [[ -z "$PY" ]]; then
    cand="$(command -v python3 2>/dev/null || true)"
    if [[ -n "$cand" ]] && "$cand" -c "import torch" >/dev/null 2>&1; then
      PY="$cand"
      PY_HOW="auto"
    fi
  fi
  if [[ -z "$PY" ]]; then
    echo "ERROR: No Python with PyTorch found." >&2
    echo "  Tried: /opt/venv/bin/python3, $ROOT/.benchmark_mi300_vllm_frozen/.venv/bin/python3, $ROOT/.venv/bin/python3, and \$(command -v python3)" >&2
    echo "  Fix: install torch (see docs/benchmark_mi300_locked_env.md), or set AMDEXP_PYTHON" >&2
    echo "  Example: AMDEXP_PYTHON=$ROOT/.benchmark_mi300_vllm_frozen/.venv/bin/python3 bash scripts/verify_mi300x_stack.sh" >&2
    exit 2
  fi
fi

echo "================================================================"
echo "MI300X / ROCm stack verification"
echo "time: $(date -Iseconds)"
echo "hostname: $(hostname)"
echo "python: $(command -v "$PY" 2>/dev/null || echo "$PY")"
echo "resolved_python: $PY ($PY_HOW)"
echo "================================================================"
echo

echo "--- /dev/kfd /dev/dri (GPU character devices) ---"
if [[ -e /dev/kfd ]]; then
  ls -l /dev/kfd
else
  echo "MISSING: /dev/kfd (ROCm compute; container/host must expose this)"
fi
if [[ -d /dev/dri ]]; then
  ls -l /dev/dri | head -30
else
  echo "MISSING: /dev/dri (DRI; container/host must expose this)"
fi
echo

echo "--- rocm-smi (if installed) ---"
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showproductname 2>/dev/null || rocm-smi -i 2>/dev/null || rocm-smi
else
  echo "rocm-smi not in PATH"
fi
echo

echo "--- user / groups (video/render matter for /dev/dri) ---"
id
echo

echo "--- PyTorch ---"
"$PY" <<'PY'
import os
import torch

print("torch.__version__:", torch.__version__)
hip = getattr(torch.version, "hip", None)
print("torch.version.hip:", hip)
gpu = torch.cuda.is_available()
print("torch.cuda.is_available():", gpu)
if gpu:
    n = torch.cuda.device_count()
    print("torch.cuda.device_count():", n)
    for i in range(n):
        print(f"  [{i}]", torch.cuda.get_device_name(i))
else:
    print("No GPU visible to PyTorch in this process.")
    print("HIP_VISIBLE_DEVICES:", repr(os.environ.get("HIP_VISIBLE_DEVICES")))
    print("CUDA_VISIBLE_DEVICES:", repr(os.environ.get("CUDA_VISIBLE_DEVICES")))

if hip is None and "+cu" in torch.__version__:
    print()
    print("WARNING: This looks like a CUDA PyTorch wheel (+cu*), not ROCm.")
    print("  On MI300X you need a ROCm build (torch.version.hip should be set).")
PY

echo
echo "================================================================"
echo "Done. If gpu=False, fix host/container device passthrough first."
echo "If hip=None with +cu*, fix PyTorch wheel (do not replace ROCm torch with CUDA)."
echo "See: docs/mi300x_docker_runtime_requirements.md"
echo "================================================================"
