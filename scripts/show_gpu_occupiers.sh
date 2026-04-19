#!/usr/bin/env bash
# show_gpu_occupiers.sh — who is using the AMD GPU (VRAM / KFD / DRI)?
#
# Run on the host (not inside a child container) for a truthful picture.
# Nothing here kills processes unless you copy the suggested commands yourself.
#
# Usage: bash scripts/show_gpu_occupiers.sh

set -euo pipefail

echo "========== ROCm VRAM =========="
if command -v rocm-smi >/dev/null 2>&1; then
  rocm-smi --showmeminfo vram 2>/dev/null || true
  echo
  echo "========== KFD / GPU processes (rocm-smi --showpids) =========="
  rocm-smi --showpids 2>/dev/null || true
else
  echo "rocm-smi not found"
fi

echo
echo "========== Docker containers (often the real VRAM hog) =========="
if command -v docker >/dev/null 2>&1; then
  docker ps --format 'table {{.ID}}\t{{.Names}}\t{{.Status}}\t{{.Image}}'
else
  echo "docker not found"
fi

echo
echo "========== Likely ML processes (truncated cmdline; review before kill) =========="
ps -eo pid,user,%mem,cmd --sort=-%mem 2>/dev/null \
  | grep -Ei 'python|vllm|torch|jupyter|ray|text-generation|triton' \
  | grep -v grep | grep -v unattended-upgrade | grep -v fail2ban \
  | while IFS= read -r line; do echo "${line:0:200}"; done || true

echo
echo "========== /dev/kfd users (if fuser exists) =========="
if command -v fuser >/dev/null 2>&1; then
  fuser -v /dev/kfd 2>/dev/null || true
else
  echo "fuser not installed"
fi

echo
echo "----- Suggested cleanup (YOU run manually after review) -----"
echo "  Stop a container:     docker stop <CONTAINER_ID>"
echo "  Stop Jupyter kernels:  close notebooks or: pkill -f 'jupyter-lab'   # careful on shared systems"
echo "  Kill a stray PID:      kill <PID>   # or kill -9 only if stuck"
echo "  Nuclear (host only):   sync workloads then reboot if driver is wedged"
echo
echo "Then rerun Mistral profile, e.g.:"
echo "  bash docker_run_amd_mi300x.sh -- bash -lc 'cd /workspace/amd-experiments && export AMDEXP_USE_SYSTEM_PYTHON=1 && python3 benchmarks/profile_full_model_decode.py --model mistralai/Mistral-7B-v0.1 --seq-len 4096 --n-decode 8'"
