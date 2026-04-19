#!/usr/bin/env python3
"""
Print a conservative --gpu-memory-utilization fraction for vLLM startup checks.

vLLM rejects startup when reported free VRAM is below (approximately) utilization × total.
Use: GPU_MEM=$(python3 scripts/estimate_vllm_safe_gpu_mem_frac.py)
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import torch

        if not torch.cuda.is_available():
            print("0.50", file=sys.stderr)
            print("0.50")
            return 0
        free_b, total_b = torch.cuda.mem_get_info()
        if total_b <= 0:
            print("0.50")
            return 0
        # Headroom: vLLM's probe can be stricter than torch; stay below free/total.
        frac = (free_b / float(total_b)) * 0.92
        frac = max(0.06, min(0.92, frac))
        print(f"{frac:.4f}")
    except Exception as e:
        print(f"[estimate_vllm_safe_gpu_mem_frac] fallback 0.50: {e}", file=sys.stderr)
        print("0.50")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
