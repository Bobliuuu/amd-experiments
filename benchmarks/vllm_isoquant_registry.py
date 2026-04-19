"""
Register IsoQuant V1 backend (``iq3`` KV) with an installed vLLM.

Call ``register_isoquant_rocm_backend()`` **before** ``from vllm import LLM``.

Uses ``AttentionBackendEnum.CUSTOM`` — **do not** also call
``register_turboquant_rocm_backend()`` in the same process.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_registered = False


def ensure_repo_on_path() -> None:
    r = str(_ROOT)
    k = str(_ROOT / "kernels")
    for p in (k, r):
        if p not in sys.path:
            sys.path.insert(0, p)


def register_isoquant_rocm_backend() -> bool:
    """Register IsoQuant on V1 CUSTOM slot. Returns True on success."""
    global _registered
    if _registered:
        return True
    ensure_repo_on_path()

    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )

        register_backend(
            AttentionBackendEnum.CUSTOM,
            "tq_backends.vllm_v1_isoquant_bridge.IsoQuantRocmV1Backend",
        )
        print(
            "  Registered IsoQuant as V1 AttentionBackendEnum.CUSTOM "
            "(tq_backends.vllm_v1_isoquant_bridge.IsoQuantRocmV1Backend)"
        )
        _registered = True
        return True
    except Exception as exc:
        print(f"  WARNING: IsoQuant V1 registration failed: {exc}")
        return False
