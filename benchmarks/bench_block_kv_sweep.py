#!/usr/bin/env python3
"""
Sweep BLOCK_KV for Split-K fused TQ3 attention at a fixed long seq_k.

Output: results/bench_block_kv_sweep.json
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))

import torch

from turboquant_mi300x import TurboQuantMI300X
from tq_triton import compress_kv_for_triton, turboquant_attention_fwd


def bench_ms(fn, warmup: int = 5, reps: int = 30) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


def main() -> None:
    if not torch.cuda.is_available():
        print(json.dumps({"error": "no GPU"}))
        sys.exit(1)

    device = torch.cuda.get_device_name(0)
    B, H, D, S_q = 1, 32, 128, 1
    seq_k = 32768
    sm = D**-0.5
    block_kvs = [1024, 2048, 4096, 8192]

    tq = TurboQuantMI300X(bits=3, device="cuda")
    torch.manual_seed(0)
    q = torch.randn(B, H, S_q, D, device="cuda", dtype=torch.float16)
    k_fp = torch.randn(B, H, seq_k, D, device="cuda", dtype=torch.float16)
    v_fp = torch.randn(B, H, seq_k, D, device="cuda", dtype=torch.float16)
    k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k_fp, v_fp, tq)
    q_rot = tq.rotate_queries(q.float())

    rows = []
    best_kv, best_ms = None, float("inf")
    for bkv in block_kvs:
        ms = bench_ms(
            lambda b=bkv: turboquant_attention_fwd(
                q_rot,
                k_planes,
                k_norms,
                v_planes,
                v_norms,
                rotation=tq.rotation,
                sm_scale=sm,
                use_split_k=True,
                BLOCK_KV=b,
            )
        )
        rows.append({"BLOCK_KV": bkv, "triton_ms": round(ms, 4)})
        if ms < best_ms:
            best_ms, best_kv = ms, bkv

    out = {
        "device": device,
        "seq_k": seq_k,
        "B": B,
        "H": H,
        "D": D,
        "S_q": S_q,
        "sweep": rows,
        "best_BLOCK_KV": best_kv,
        "best_triton_ms": round(best_ms, 4),
    }
    outp = Path(__file__).resolve().parents[1] / "results" / "bench_block_kv_sweep.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"Saved -> {outp}")


if __name__ == "__main__":
    main()
