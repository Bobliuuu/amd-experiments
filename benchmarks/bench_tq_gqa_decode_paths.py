#!/usr/bin/env python3
"""
Benchmark GQA decode: fused (compressed head expand + Triton) vs decompress+SDPA.

Uses the same synthetic paged cache as validate_tq_gqa_fused_decode.py.
Prints median ms per forward over `reps` after warmup.

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

    kv_f = torch.zeros(
        2, num_blocks_pool, num_kv_heads, block_size, 52, dtype=torch.uint8, device=device
    )
    kv_d = kv_f.clone()

    k_hist = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)
    v_hist = torch.randn(seq_len, num_kv_heads, head_dim, device=device, dtype=torch.float16)

    impl = TurboQuantROCmAttentionImpl(
        num_heads=num_q_heads,
        head_size=head_dim,
        scale=head_dim**-0.5,
        num_kv_heads=num_kv_heads,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="tq3",
    )
    tq = impl._tq(device)
    fill_cache(kv_f, k_hist, v_hist, tq)
    fill_cache(kv_d, k_hist, v_hist, tq)

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
    ms_fused = bench_ms(lambda: impl._forward_decode_fused(q, kv_f, meta), warmup, reps)
    os.environ["VLLM_TQ_USE_FUSED_KERNEL"] = "0"
    ms_dec = bench_ms(lambda: impl._forward_decode_decompress(q, kv_d, meta), warmup, reps)
    return {
        "seq_len": seq_len,
        "fused_ms": round(ms_fused, 4),
        "decompress_ms": round(ms_dec, 4),
        "speedup": round(ms_dec / ms_fused, 3),
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
        "--sweep",
        type=str,
        default="",
        help="Comma-separated seq_lens; if set, runs sweep and writes --json-out",
    )
    p.add_argument(
        "--json-out",
        type=str,
        default="",
        help="Write results/bench_tq_gqa_decode_sweep.json (use with --sweep)",
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
                f"seq_len={seq_len}  fused_ms={row['fused_ms']:.3f}  "
                f"decompress_ms={row['decompress_ms']:.3f}  speedup={row['speedup']:.2f}x"
            )
        if args.json_out:
            outp = Path(args.json_out)
            if not outp.is_absolute():
                outp = ROOT / "results" / outp.name
            outp.parent.mkdir(parents=True, exist_ok=True)
            outp.write_text(
                json.dumps(
                    {"device": torch.cuda.get_device_name(0), "rows": rows},
                    indent=2,
                )
            )
            print(f"Wrote {outp}")
        return

    row = _bench_one(args.seq_len, args.warmup, args.reps)
    print(
        f"seq_len={row['seq_len']}  GQA 32/8  "
        f"fused_ms={row['fused_ms']:.3f}  decompress_sdpa_ms={row['decompress_ms']:.3f}  "
        f"speedup={row['speedup']:.2f}x"
    )


if __name__ == "__main__":
    main()
