#!/usr/bin/env python3
"""
Validate GQA fused TQ3 decode vs decompress+SDPA (vLLM-style backend, no full vLLM).

Fills a paged TQ3 kv_cache from FP16 K/V, then compares:
  TurboQuantROCmAttentionImpl._forward_decode_fused
  TurboQuantROCmAttentionImpl._forward_decode_decompress

Run from repo root with kernels on PYTHONPATH:

  PYTHONPATH=./kernels:. python3 benchmarks/validate_tq_gqa_fused_decode.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "kernels"))
sys.path.insert(0, str(ROOT))

import torch

os.environ["VLLM_TQ_USE_FUSED_KERNEL"] = "1"

from tq_backends.attention.backends.rocm_flash_attn import (  # noqa: E402
    TurboQuantROCmAttentionImpl,
    TurboQuantROCmAttentionMetadata,
    tq3_store_tokens,
)


def _fill_paged_tq_cache(
    kv_cache: torch.Tensor,
    k_hist: torch.Tensor,
    v_hist: torch.Tensor,
    tq,
) -> None:
    """k_hist, v_hist: (seq_len, num_kv_heads, D) fp16 — write via TQ3 store."""
    seq_len, num_kv, D = k_hist.shape
    block_size = kv_cache.shape[3]
    slot_mapping = torch.arange(seq_len, device=kv_cache.device, dtype=torch.int64)
    tq3_store_tokens(kv_cache, k_hist, v_hist, slot_mapping, tq)


def main() -> None:
    device = "cuda"
    torch.manual_seed(0)

    num_q_heads, num_kv_heads = 32, 8
    head_dim = 128
    gqa_ratio = num_q_heads // num_kv_heads
    assert gqa_ratio == 4

    seq_len = 2048
    block_size = 16
    n_blocks_needed = (seq_len + block_size - 1) // block_size
    num_blocks_pool = max(n_blocks_needed + 2, 8)

    kv_cache = torch.zeros(
        2, num_blocks_pool, num_kv_heads, block_size, 52,
        dtype=torch.uint8,
        device=device,
    )
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

    q = torch.randn(1, num_q_heads, head_dim, device=device, dtype=torch.float16)

    with torch.no_grad():
        out_fused = impl._forward_decode_fused(q, kv_cache, meta)
        out_dec = impl._forward_decode_decompress(q, kv_cache, meta)

    cos = torch.nn.functional.cosine_similarity(
        out_fused.flatten().float().unsqueeze(0),
        out_dec.flatten().float().unsqueeze(0),
    ).item()
    max_abs = (out_fused.float() - out_dec.float()).abs().max().item()
    print(f"seq_len={seq_len} num_q={num_q_heads} num_kv={num_kv_heads} gqa_ratio={gqa_ratio}")
    print(f"cosine_sim(fused, decompress_sdpa)={cos:.6f}  max_abs_err={max_abs:.6f}")
    if cos < 0.92:
        print("FAIL: cosine similarity below 0.92")
        sys.exit(1)
    print("PASS")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("no cuda")
        sys.exit(0)
    main()
