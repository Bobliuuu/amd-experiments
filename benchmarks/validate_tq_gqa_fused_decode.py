#!/usr/bin/env python3
"""
Validate GQA fused TQ3 decode against decompress+SDPA (golden) and the
expand+MHA fused path (existing).

Three paths compared, all on the same paged TQ3 kv_cache:
  1. decompress+SDPA      — TurboQuantROCmAttentionImpl._forward_decode_decompress
  2. expand+MHA fused     — _forward_decode_fused with VLLM_TQ_USE_GQA_KERNEL=0
  3. GQA fused (new)      — _forward_decode_fused with VLLM_TQ_USE_GQA_KERNEL=1

Cosine similarity targets:
  cos(expand_fused, decompress) ≥ 0.92  (existing fused path against golden)
  cos(gqa_fused,    decompress) ≥ 0.92  (new GQA path against golden)
  cos(gqa_fused,    expand_fused) ≥ 0.99  (same compressed bytes, just better
                                          access pattern; any drift is reduce-order
                                          floating-point noise)

Run from repo root with kernels on PYTHONPATH:
  PYTHONPATH=./kernels:. python3 benchmarks/validate_tq_gqa_fused_decode.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT))

import torch

from tq_backends.attention.backends.rocm_flash_attn import (  # noqa: E402
    TurboQuantROCmAttentionImpl,
    TurboQuantROCmAttentionMetadata,
    tq3_store_tokens,
)


def _fill_paged_tq_cache(kv_cache, k_hist, v_hist, tq) -> None:
    seq_len = k_hist.shape[0]
    slot_mapping = torch.arange(seq_len, device=kv_cache.device, dtype=torch.int64)
    tq3_store_tokens(kv_cache, k_hist, v_hist, slot_mapping, tq)


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    return torch.nn.functional.cosine_similarity(
        a.flatten().float().unsqueeze(0),
        b.flatten().float().unsqueeze(0),
    ).item()


def _max_abs(a: torch.Tensor, b: torch.Tensor) -> float:
    return (a.float() - b.float()).abs().max().item()


def _run_one(num_q: int, num_kv: int, seq_len: int, device: str = "cuda") -> dict:
    torch.manual_seed(0)
    head_dim = 128
    gqa_ratio = num_q // num_kv
    assert num_q == num_kv * gqa_ratio

    block_size = 16
    n_blocks_needed = (seq_len + block_size - 1) // block_size
    num_blocks_pool = max(n_blocks_needed + 2, 8)

    kv_cache = torch.zeros(
        2, num_blocks_pool, num_kv, block_size, 52,
        dtype=torch.uint8, device=device,
    )
    k_hist = torch.randn(seq_len, num_kv, head_dim, device=device, dtype=torch.float16)
    v_hist = torch.randn(seq_len, num_kv, head_dim, device=device, dtype=torch.float16)

    impl = TurboQuantROCmAttentionImpl(
        num_heads=num_q,
        head_size=head_dim,
        scale=head_dim ** -0.5,
        num_kv_heads=num_kv,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="tq3",
    )
    tq = impl._tq(device)
    _fill_paged_tq_cache(kv_cache, k_hist, v_hist, tq)

    block_table = torch.zeros(num_blocks_pool, dtype=torch.int32, device=device)
    for b in range(n_blocks_needed):
        block_table[b] = b
    block_tables = block_table.unsqueeze(0)

    meta = TurboQuantROCmAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decode_tokens=1,
        slot_mapping=None,
        block_tables=block_tables,
        seq_lens_decode=[seq_len],
        max_decode_seq_len=seq_len,
    )

    q = torch.randn(1, num_q, head_dim, device=device, dtype=torch.float16)

    with torch.no_grad():
        out_dec = impl._forward_decode_decompress(q, kv_cache, meta)

        os.environ["VLLM_TQ_USE_FUSED_KERNEL"] = "1"
        os.environ["VLLM_TQ_USE_GQA_KERNEL"]   = "0"
        out_expand = impl._forward_decode_fused(q, kv_cache, meta)

        os.environ["VLLM_TQ_USE_GQA_KERNEL"] = "1"
        out_gqa = impl._forward_decode_fused(q, kv_cache, meta)

    return {
        "num_q": num_q, "num_kv": num_kv, "gqa_ratio": gqa_ratio, "seq_len": seq_len,
        "cos_expand_vs_dec":   _cos(out_expand, out_dec),
        "cos_gqa_vs_dec":      _cos(out_gqa,    out_dec),
        "cos_gqa_vs_expand":   _cos(out_gqa,    out_expand),
        "maxabs_gqa_vs_dec":      _max_abs(out_gqa,    out_dec),
        "maxabs_gqa_vs_expand":   _max_abs(out_gqa,    out_expand),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--num-q",    type=int, default=None)
    p.add_argument("--num-kv",   type=int, default=None)
    p.add_argument("--seq-len",  type=int, default=None)
    p.add_argument(
        "--cos-floor", type=float, default=0.92,
        help="Min cosine sim vs decompress+SDPA (default 0.92).",
    )
    p.add_argument(
        "--cos-floor-gqa-vs-expand", type=float, default=0.99,
        help="Min cosine sim between GQA fused and expand+MHA fused (default 0.99).",
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("no cuda")
        sys.exit(0)

    if args.num_q is not None and args.num_kv is not None and args.seq_len is not None:
        cases = [(args.num_q, args.num_kv, args.seq_len)]
    else:
        head_configs = [(32, 32), (32, 16), (32, 8), (64, 8)]
        seq_lens = [1024, 8192, 65536]
        cases = [(q, kv, s) for (q, kv) in head_configs for s in seq_lens]

    fails = 0
    for num_q, num_kv, seq_len in cases:
        r = _run_one(num_q, num_kv, seq_len)
        ok_expand = r["cos_expand_vs_dec"] >= args.cos_floor
        ok_gqa    = r["cos_gqa_vs_dec"]    >= args.cos_floor
        ok_drift  = r["cos_gqa_vs_expand"] >= args.cos_floor_gqa_vs_expand
        verdict = "PASS" if (ok_expand and ok_gqa and ok_drift) else "FAIL"
        if verdict == "FAIL":
            fails += 1
        print(
            f"{verdict}  num_q={r['num_q']:>3} num_kv={r['num_kv']:>3} "
            f"gqa={r['gqa_ratio']} seq={r['seq_len']:>6}  "
            f"cos(expand,dec)={r['cos_expand_vs_dec']:.4f}  "
            f"cos(gqa,dec)={r['cos_gqa_vs_dec']:.4f}  "
            f"cos(gqa,expand)={r['cos_gqa_vs_expand']:.6f}  "
            f"maxabs(gqa,dec)={r['maxabs_gqa_vs_dec']:.4f}"
        )

    if fails:
        print(f"\n{fails}/{len(cases)} cases FAILED")
        sys.exit(1)
    print(f"\nALL {len(cases)} cases PASS")


if __name__ == "__main__":
    main()
