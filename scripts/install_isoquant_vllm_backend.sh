#!/usr/bin/env bash
# Patch vLLM CacheDType for iq3 and optionally copy IsoQuant backend into site-packages.
# Run after: pip install vllm   (ROCm-matched wheel or source)
#
#   bash scripts/install_isoquant_vllm_backend.sh
#   PYTHON=python3 bash scripts/install_isoquant_vllm_backend.sh
#
# IsoQuant logic lives in-repo under tq_backends/; the V1 bridge is
# tq_backends/vllm_v1_isoquant_bridge.py. Copying the backend module into vLLM is
# optional (useful for tooling that expects vllm.attention.backends.isoquant_rocm_attn).

set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-python3}"
SRC="$ROOT/tq_backends/attention/backends/isoquant_rocm_attn.py"

echo "[install_isoquant_vllm_backend] patching CacheDType (iq3)"
"$PYTHON" "$ROOT/scripts/patch_vllm_cache_dtype_iq3.py"

DST_DIR="$("$PYTHON" -c "import vllm, pathlib; print(pathlib.Path(vllm.__path__[0]) / 'attention' / 'backends')")"
DST="$DST_DIR/isoquant_rocm_attn.py"
echo "[install_isoquant_vllm_backend] copy: $SRC -> $DST"
cp "$SRC" "$DST"
echo "[install_isoquant_vllm_backend] OK"
