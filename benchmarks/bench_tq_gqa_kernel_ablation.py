#!/usr/bin/env python3
"""
GQA kernel ablation: sweep (B, S_k, gqa_ratio) for the new GQA-aware Triton
kernel vs the existing expand+MHA fused kernel.

Builds synthetic compressed K/V directly in the layout the Triton kernels
consume — bypasses the paged-cache gather so we measure the kernel cost,
not gather Python-loop overhead.

  PYTHONPATH=./kernels:. python3 benchmarks/bench_tq_gqa_kernel_ablation.py \\
      --json-out bench_tq_gqa_kernel_ablation.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT))

import torch

from tq_triton import (
    compress_kv_for_triton,
    turboquant_attention_fwd,
    turboquant_gqa_attention_fwd,
)
from turboquant_mi300x import TurboQuantMI300X
from tq_backends.attention.backends.rocm_flash_attn import expand_tq_compressed_for_gqa


def bench_ms(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


def _bench_cell(B: int, S_k: int, gqa_ratio: int, H_kv: int,
                warmup: int, reps: int) -> dict:
    device = "cuda"
    H_q = H_kv * gqa_ratio
    D = 128
    S_q = 1
    torch.manual_seed(7)

    tq = TurboQuantMI300X(bits=3, device=device)

    q_fp16 = torch.randn(B, H_q, S_q, D, device=device, dtype=torch.float16)
    k_fp16 = torch.randn(B, H_kv, S_k, D, device=device, dtype=torch.float16)
    v_fp16 = torch.randn(B, H_kv, S_k, D, device=device, dtype=torch.float16)

    k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k_fp16, v_fp16, tq)
    q_rot = tq.rotate_queries(q_fp16.float())

    if gqa_ratio > 1:
        k_planes_exp, k_norms_exp, v_planes_exp, v_norms_exp = expand_tq_compressed_for_gqa(
            k_planes, k_norms, v_planes, v_norms, gqa_ratio
        )
    else:
        k_planes_exp, k_norms_exp = k_planes, k_norms
        v_planes_exp, v_norms_exp = v_planes, v_norms

    sm_scale = D ** -0.5

    def run_expand():
        return turboquant_attention_fwd(
            q_rot, k_planes_exp, k_norms_exp, v_planes_exp, v_norms_exp,
            rotation=tq.rotation, sm_scale=sm_scale, use_split_k=True,
        )

    def run_gqa():
        return turboquant_gqa_attention_fwd(
            q_rot, k_planes, k_norms, v_planes, v_norms,
            gqa_ratio=gqa_ratio,
            rotation=tq.rotation, sm_scale=sm_scale, use_split_k=True,
        )

    # Peak alloc delta: measure incremental allocation around one forward.
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.max_memory_allocated(device)
    _ = run_expand()
    torch.cuda.synchronize()
    expand_peak = torch.cuda.max_memory_allocated(device) - base_alloc

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    base_alloc = torch.cuda.max_memory_allocated(device)
    _ = run_gqa()
    torch.cuda.synchronize()
    gqa_peak = torch.cuda.max_memory_allocated(device) - base_alloc

    ms_expand = bench_ms(run_expand, warmup, reps)
    ms_gqa    = bench_ms(run_gqa,    warmup, reps)

    kv_bytes_expand = 2 * B * H_q  * S_k * 48
    kv_bytes_gqa    = 2 * B * H_kv * S_k * 48

    return {
        "B": B, "S_k": S_k, "H_kv": H_kv, "H_q": H_q, "gqa_ratio": gqa_ratio,
        "expand_ms":             round(ms_expand, 4),
        "gqa_ms":                round(ms_gqa,    4),
        "speedup":               round(ms_expand / ms_gqa, 3),
        "kv_bytes_expand":       kv_bytes_expand,
        "kv_bytes_gqa":          kv_bytes_gqa,
        "kv_bytes_ratio":        round(kv_bytes_expand / kv_bytes_gqa, 3),
        "peak_mem_expand_MB":    round(expand_peak / 1e6, 3),
        "peak_mem_gqa_MB":       round(gqa_peak    / 1e6, 3),
        "peak_mem_delta_MB":     round((expand_peak - gqa_peak) / 1e6, 3),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--batch",     type=str, default="1,4,16")
    p.add_argument("--seq-k",     type=str, default="4096,32768,131072")
    p.add_argument("--gqa-ratio", type=str, default="1,2,4,8")
    p.add_argument("--h-kv",      type=int, default=8)
    p.add_argument("--warmup",    type=int, default=5)
    p.add_argument("--reps",      type=int, default=20)
    p.add_argument("--json-out",  type=str, default="bench_tq_gqa_kernel_ablation.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no cuda")
        return

    batches  = [int(x) for x in args.batch.split(",")     if x.strip()]
    seq_ks   = [int(x) for x in args.seq_k.split(",")     if x.strip()]
    ratios   = [int(x) for x in args.gqa_ratio.split(",") if x.strip()]

    rows = []
    for B in batches:
        for S_k in seq_ks:
            for r in ratios:
                try:
                    row = _bench_cell(B, S_k, r, args.h_kv, args.warmup, args.reps)
                except torch.cuda.OutOfMemoryError as e:
                    print(f"OOM at B={B} S_k={S_k} gqa={r}: {e}")
                    continue
                rows.append(row)
                print(
                    f"B={B:>2} S_k={S_k:>6} H_kv={args.h_kv} gqa={r}  "
                    f"expand={row['expand_ms']:7.3f} ms  "
                    f"gqa={row['gqa_ms']:7.3f} ms  "
                    f"speedup={row['speedup']:.2f}x  "
                    f"bytes_ratio={row['kv_bytes_ratio']:.2f}x  "
                    f"peak_delta={row['peak_mem_delta_MB']:+.1f} MB"
                )

    outp = Path(args.json_out)
    if not outp.is_absolute():
        outp = ROOT / "results" / outp.name
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(
        {"device": torch.cuda.get_device_name(0), "rows": rows},
        indent=2,
    ))
    print(f"Wrote {outp}")


if __name__ == "__main__":
    main()
