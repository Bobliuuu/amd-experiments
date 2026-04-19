#!/usr/bin/env python3
"""Insert ``iq3`` into vLLM's ``CacheDType`` Literal so ``kv_cache_dtype='iq3'`` validates.

Run after ``pip install vllm`` (same interpreter). Idempotent. Compatible whether or
not ``tq3`` was already patched by ``patch_vllm_cache_dtype_tq3.py``.
"""
from __future__ import annotations

import pathlib
import sys


def main() -> int:
    import vllm

    p = pathlib.Path(vllm.__path__[0]) / "config" / "cache.py"
    text = p.read_text()
    if '"iq3"' in text:
        print(f"[patch_vllm_cache_dtype_iq3] already patched: {p}")
        return 0

    if '"tq3"' in text:
        text = text.replace('"tq3",', '"tq3",\n    "iq3",', 1)
        p.write_text(text)
        print(f"[patch_vllm_cache_dtype_iq3] patched after tq3: {p}")
        return 0

    needle = '\n    "auto",\n    "float16",'
    if needle not in text:
        print(
            f"[patch_vllm_cache_dtype_iq3] ERROR: expected snippet not found in {p}",
            file=sys.stderr,
        )
        return 2

    text = text.replace(
        needle,
        '\n    "auto",\n    "iq3",\n    "float16",',
        1,
    )
    p.write_text(text)
    print(f"[patch_vllm_cache_dtype_iq3] patched (no tq3): {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
