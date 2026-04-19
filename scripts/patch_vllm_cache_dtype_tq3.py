#!/usr/bin/env python3
"""Insert ``tq3`` into vLLM's ``CacheDType`` Literal so ``kv_cache_dtype='tq3'`` validates.

Run after ``pip install vllm`` (same interpreter). Idempotent.
"""
from __future__ import annotations

import pathlib
import re
import sys


def main() -> int:
    import vllm

    p = pathlib.Path(vllm.__path__[0]) / "config" / "cache.py"
    text = p.read_text()
    if '"tq3"' in text:
        print(f"[patch_vllm_cache_dtype_tq3] already patched: {p}")
        return 0

    needle = '\n    "auto",\n    "float16",'
    if needle not in text:
        print(
            f"[patch_vllm_cache_dtype_tq3] ERROR: expected snippet not found in {p}",
            file=sys.stderr,
        )
        return 2

    text = text.replace(
        needle,
        '\n    "auto",\n    "tq3",\n    "float16",',
        1,
    )
    p.write_text(text)
    print(f"[patch_vllm_cache_dtype_tq3] patched: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
