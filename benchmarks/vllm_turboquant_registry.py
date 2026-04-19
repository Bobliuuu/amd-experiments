"""
Register TurboQuantROCmAttentionBackend with an installed vLLM.

Import and call ``register_turboquant_rocm_backend()`` **before**
``from vllm import LLM`` when using TQ3 KV + TurboQuant kernels.

vLLM 0.19+ (V1 engine): registers ``AttentionBackendEnum.CUSTOM`` to point at
``tq_backends.vllm_v1_turboquant_bridge.TurboQuantRocmV1Backend``. Benchmarks must
pass ``LLM(..., attention_config={"backend": "CUSTOM"}, kv_cache_dtype="tq3")``.

Older builds: may still patch ``vllm.attention.selector`` when that module exists.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
_registered = False


def ensure_repo_on_path() -> None:
    r = str(_ROOT)
    if r not in sys.path:
        sys.path.insert(0, r)


def register_turboquant_rocm_backend() -> bool:
    """
    Register TurboQuant with vLLM (legacy selector and/or V1 enum path).

    Returns True if a patch was applied, False otherwise.
    """
    global _registered
    if _registered:
        return True
    ensure_repo_on_path()

    # --- vLLM 0.19+ V1: CUSTOM backend override ---
    try:
        from vllm.v1.attention.backends.registry import (
            AttentionBackendEnum,
            register_backend,
        )

        register_backend(
            AttentionBackendEnum.CUSTOM,
            "tq_backends.vllm_v1_turboquant_bridge.TurboQuantRocmV1Backend",
        )
        print(
            "  Registered TurboQuant as V1 AttentionBackendEnum.CUSTOM "
            "(tq_backends.vllm_v1_turboquant_bridge.TurboQuantRocmV1Backend)"
        )
        _registered = True
        return True
    except Exception as exc:
        print(f"  WARNING: V1 CUSTOM registration failed: {exc}")

    # --- Legacy selector (pre-V1 / older wheels) ---
    try:
        from tq_backends.attention.backends.rocm_flash_attn import (
            TurboQuantROCmAttentionBackend,
        )
    except ImportError as e:
        print(f"  WARNING: cannot import TurboQuant backend (tq_backends): {e}")
        return False

    try:
        import vllm.attention.selector as _sel

        reg = getattr(_sel, "_ATTENTION_BACKEND_REGISTRY", None)
        if isinstance(reg, dict):
            reg["TURBOQUANT_ROCM"] = TurboQuantROCmAttentionBackend
            print(
                "  Registered TURBOQUANT_ROCM via "
                "vllm.attention.selector._ATTENTION_BACKEND_REGISTRY"
            )
            _registered = True
            return True
    except Exception:
        pass

    try:
        import vllm.attention.selector as _sel

        _orig = _sel.get_attn_backend

        def _patched(*args, **kwargs):
            if os.environ.get("VLLM_ATTENTION_BACKEND") == "TURBOQUANT_ROCM":
                return TurboQuantROCmAttentionBackend
            return _orig(*args, **kwargs)

        _sel.get_attn_backend = _patched  # type: ignore[assignment]
        print("  Registered TURBOQUANT_ROCM via get_attn_backend() monkey-patch")
        _registered = True
        return True
    except Exception as exc:
        print(f"  WARNING: could not register TURBOQUANT_ROCM (legacy): {exc}")
        return False
