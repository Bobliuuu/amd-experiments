#!/usr/bin/env python3
"""Print ROCm / PyTorch / hipBLASLt-related versions for GEMM-stack tracking.

Use before/after driver or ROCm upgrades and when comparing decode profiles.
No vLLM import required.
"""
from __future__ import annotations

import ctypes
import os
import subprocess
import sys


def _try_run(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, timeout=5).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def main() -> int:
    print("=== ROCm / GEMM stack probe ===")
    print(f"ROCM_PATH: {os.environ.get('ROCM_PATH', '')}")
    print(f"HIP_PATH: {os.environ.get('HIP_PATH', '')}")

    for cmd in (["rocminfo"],):
        out = _try_run(cmd)
        if out:
            # First ~15 lines usually contain agent name
            lines = out.splitlines()[:20]
            print("\n--- rocminfo (head) ---")
            print("\n".join(lines))
            break
    else:
        print("\n(rocminfo not available)")

    try:
        import torch

        print("\n--- torch ---")
        print(f"torch.__version__: {torch.__version__}")
        ver = getattr(torch.version, "hip", None)
        if ver:
            print(f"torch.version.hip: {ver}")
        cuda = getattr(torch.version, "cuda", None)
        if cuda:
            print(f"torch.version.cuda: {cuda}")
    except ImportError:
        print("\n(torch not installed in this interpreter)")

    # hipblaslt shared object path if loaded
    for name in ("libhipblaslt.so", "libhipblaslt.so.0"):
        try:
            ctypes.CDLL(name)
            print(f"\nctypes: loaded {name}")
            break
        except OSError:
            continue
    else:
        print("\nctypes: libhipblaslt not loaded in this bare process (normal without torch GEMM)")

    print("\n--- optional: hipconfig ---")
    hipc = _try_run(["hipconfig", "--version"])
    if hipc:
        print(hipc)
    else:
        print("(hipconfig not on PATH)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
