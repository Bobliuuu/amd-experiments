#!/usr/bin/env bash
# Copy TurboQuant ROCm attention backend from the repo into the installed vLLM
# site-packages tree. Run after: pip install vllm (ROCm-matched wheel or source).
#
# Usage:
#   bash scripts/install_turboquant_vllm_backend.sh
#   PYTHON=python3 bash scripts/install_turboquant_vllm_backend.sh

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SRC="$ROOT/tq_backends/attention/backends/rocm_flash_attn.py"

if [[ ! -f "$SRC" ]]; then
  echo "ERROR: missing $SRC" >&2
  exit 2
fi

DST_DIR="$("$PYTHON" -c "import vllm, pathlib; print(pathlib.Path(vllm.__path__[0]) / 'attention' / 'backends')")"
DST="$DST_DIR/rocm_flash_attn.py"

echo "[install_turboquant_vllm_backend] src: $SRC"
echo "[install_turboquant_vllm_backend] dst: $DST"
mkdir -p "$DST_DIR"
cp -f "$SRC" "$DST"
echo "[install_turboquant_vllm_backend] OK"

"$PYTHON" "$ROOT/scripts/patch_vllm_cache_dtype_tq3.py" || {
  echo "[install_turboquant_vllm_backend] WARNING: tq3 CacheDType patch failed (non-fatal)" >&2
}

"$PYTHON" "$ROOT/scripts/patch_vllm_rocm_sliding_window_custom_paged.py" || {
  echo "[install_turboquant_vllm_backend] WARNING: ROCm sliding-window custom paged patch failed (non-fatal)" >&2
}
