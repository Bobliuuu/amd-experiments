#!/usr/bin/env python3
"""
Benchmark GQA decode: three paths on the same paged TQ3 cache.

  decompress_ms  : decompress to fp16, expand for GQA, SDPA
  expand_ms      : compressed cache, expand_tq_compressed_for_gqa, MHA Triton kernel
  gqa_ms         : compressed cache, GQA-aware Triton kernel (no expand)

Reports median ms per forward over `reps` after warmup, plus speedups
vs decompress baseline and vs expand+MHA fused (the meaningful comparison
for the GQA-structure win).

  PYTHONPATH=./kernels:. python3 benchmarks/bench_tq_gqa_decode_paths.py --seq-len 8192
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT))

import torch

from tq_backends.attention.backends.rocm_flash_attn import (
    TurboQuantROCmAttentionImpl,
    TurboQuantROCmAttentionMetadata,
    tq3_store_tokens,
)


def fill_cache(kv_cache, k_hist, v_hist, tq):
    seq_len = k_hist.shape[0]
    slot_mapping = torch.arange(seq_len, device=kv_cache.device, dtype=torch.int64)
    tq3_store_tokens(kv_cache, k_hist, v_hist, slot_mapping, tq)


def _bench_one(seq_len: int, warmup: int, reps: int) -> dict:
    device = "cuda"
    torch.manual_seed(1)
    num_q_heads, num_kv_heads, head_dim = 32, 8, 128
    block_size = 16
    n_blocks = (seq_len + block_size - 1) // block_size
    num_blocks_pool = max(n_blocks + 4, 16)

    kv_dec    = torch.zeros(2, num_blocks_pool, num_kv_heads, block_size, 52,
                            dtype=torch.uint8, device=device)
    kv_expand = kv_dec.clone()
    kv_gqa    = kv_dec.clone()

    k_hist = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v_hist = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    impl = TurboQuantROCmAttentionImpl(
        num_heads=num_q_heads, head_size=head_dim, scale=head_dim ** -0.5,
        num_kv_heads=num_kv_heads, alibi_slopes=None, sliding_window=None,
        kv_cache_dtype="tq3",
    )
    tq = impl._tq(device)
    for cache in (kv_dec, kv_expand, kv_gqa):
        fill_cache(cache, k_hist, v_hist, tq)

    bt = torch.zeros(num_blocks_pool, dtype=torch.int32, device=device)
    for b in range(n_blocks):
        bt[b] = b
    block_tables = bt.unsqueeze(0)
    meta = TurboQuantROCmAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=None,
        block_tables=block_tables,
        seq_lens_decode=[seq_len],
        max_decode_seq_len=seq_len,
    )
    q = torch.randn(1, num_q_heads, head_dim, device=device, dtype=torch.float16)

    os.environ["VLLM_TQ_USE_FUSED_KERNEL"] = "1"
    os.environ["VLLM_TQ_USE_GQA_KERNEL"]   = "0"
    ms_expand = bench_ms(lambda: impl._forward_decode_fused(q, kv_expand, meta), warmup, reps)

    os.environ["VLLM_TQ_USE_GQA_KERNEL"] = "1"
    ms_gqa = bench_ms(lambda: impl._forward_decode_fused(q, kv_gqa, meta), warmup, reps)

    os.environ["VLLM_TQ_USE_FUSED_KERNEL"] = "0"
    os.environ["VLLM_TQ_USE_GQA_KERNEL"]   = "0"
    ms_dec = bench_ms(lambda: impl._forward_decode_decompress(q, kv_dec, meta), warmup, reps)

    return {
        "seq_len": seq_len,
        "decompress_ms":  round(ms_dec,    4),
        "expand_ms":      round(ms_expand, 4),
        "gqa_ms":         round(ms_gqa,    4),
        "speedup_vs_decompress": round(ms_dec    / ms_gqa, 3),
        "speedup_vs_expand":     round(ms_expand / ms_gqa, 3),
    }


def bench_ms(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / reps


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--reps", type=int, default=20)
    p.add_argument(
        "--sweep", type=str, default="",
        help="Comma-separated seq_lens; if set, runs sweep and writes --json-out",
    )
    p.add_argument(
        "--json-out", type=str, default="",
        help="Write results/<name>.json (use with --sweep)",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no cuda")
        return

    if args.sweep:
        seqs = [int(x.strip()) for x in args.sweep.split(",") if x.strip()]
        rows = []
        for seq_len in seqs:
            row = _bench_one(seq_len, args.warmup, args.reps)
            rows.append(row)
            print(
                f"seq_len={seq_len:>6}  "
                f"decompress={row['decompress_ms']:7.3f} ms  "
                f"expand={row['expand_ms']:7.3f} ms  "
                f"gqa={row['gqa_ms']:7.3f} ms  "
                f"vs_dec={row['speedup_vs_decompress']:.2f}x  "
                f"vs_expand={row['speedup_vs_expand']:.2f}x"
            )
        if args.json_out:
            outp = Path(args.json_out)
            if not outp.is_absolute():
                outp = ROOT / "results" / outp.name
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(json.dumps(
                {"device": torch.cuda.get_device_name(0), "rows": rows},
                indent=2,
            ))
            print(f"Wrote {outp}")
        return

    row = _bench_one(args.seq_len, args.warmup, args.reps)
    print(
        f"seq_len={row['seq_len']}  GQA 32/8  "
        f"decompress={row['decompress_ms']:.3f} ms  "
        f"expand={row['expand_ms']:.3f} ms  "
        f"gqa={row['gqa_ms']:.3f} ms  "
        f"vs_decompress={row['speedup_vs_decompress']:.2f}x  "
        f"vs_expand={row['speedup_vs_expand']:.2f}x"
    )


if __name__ == "__main__":
    main()
