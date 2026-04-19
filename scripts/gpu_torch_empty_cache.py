#!/usr/bin/env python3
"""Best-effort VRAM cleanup between vLLM benchmark invocations (same node, same user)."""
from __future__ import annotations

import gc
import sys


def main() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            print("[gpu_torch_empty_cache] no CUDA device; nothing to do", file=sys.stderr)
            return 0
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        free, total = torch.cuda.mem_get_info()
        print(
            f"[gpu_torch_empty_cache] ok — torch sees "
            f"{free / 2**30:.2f} / {total / 2**30:.2f} GiB free"
        )
    except Exception as e:
        print(f"[gpu_torch_empty_cache] skipped: {e}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
