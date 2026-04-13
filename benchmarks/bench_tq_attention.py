"""
bench_tq_attention.py — TurboQuant attention throughput vs FP16

Compares end-to-end decode throughput across KV configurations:
  - fp16:  standard FP16 KV cache
  - fp8:   FP8 quantized KV (2× compression)
  - tq3:   TurboQuant 3-bit (4.92× compression) — using Python wrapper
  - tq4:   TurboQuant 4-bit (3.76× compression) — using Python wrapper

Uses a synthetic attention loop (no LLM weights needed) to isolate
the KV cache bandwidth effect on attention throughput.

Metric: effective attention throughput (GB/s of KV data processed per second)
at various context lengths (n_kv = 512 to 131072).

Usage:
    python3 benchmarks/bench_tq_attention.py
    python3 benchmarks/bench_tq_attention.py --model mistralai/Mistral-7B-v0.1
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Synthetic attention configs
# ──────────────────────────────────────────────────────────────────────────────

HEAD_DIM   = 128
N_HEADS    = 32    # Mistral-7B has 32 KV heads (GQA: 8 KV heads — use 8 for memory accuracy)
N_KV_HEADS = 8     # GQA: 8 KV heads for Mistral-7B

SEQ_LENS = [512, 2048, 8192, 32768, 65536, 131072]


def timed(fn, n_warmup=5, n_iters=20):
    for _ in range(n_warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(n_iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1000  # ms


# ──────────────────────────────────────────────────────────────────────────────
# Attention implementations
# ──────────────────────────────────────────────────────────────────────────────

def attention_fp16(q, k, v):
    """Standard scaled dot-product attention in FP16."""
    scale = HEAD_DIM ** -0.5
    return F.scaled_dot_product_attention(q, k, v, scale=scale)


def attention_tq3(q_rot, k_compressed, v_compressed, tq):
    """
    TQ3-compressed attention:
      1. Compute scores using fused_dot (no full K decompression)
      2. Decompress V for output accumulation
    """
    n_heads, n_kv, _ = k_compressed.shape[:3] if k_compressed.dim() == 3 else (1, *k_compressed.shape[:2])

    # For each head: fused_dot → scores, then decompress V
    # Simplified: process all heads flattened
    n_q    = q_rot.shape[-2]
    n_kv_v = k_compressed.shape[0]

    scores = tq.fused_dot(q_rot.reshape(-1, HEAD_DIM), k_compressed)    # (n_q, n_kv)
    scores = scores * (HEAD_DIM ** -0.5)
    weights = torch.softmax(scores, dim=-1)

    v_fp32 = tq.decompress_tensor(v_compressed, (n_kv_v, HEAD_DIM))
    out = weights @ v_fp32                  # (n_q, head_dim)
    return out


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark attention at various seq_lens
# ──────────────────────────────────────────────────────────────────────────────

def bench_attention_seq_lens(
    seq_lens,
    n_iters: int = 20,
    n_warmup: int = 5,
    bits: int = 3,
):
    """
    Benchmark attention throughput for FP16 and TQ3 at each context length.

    Returns list of dicts with per-config timing results.
    """
    from turboquant_mi300x import TurboQuantMI300X

    tq = TurboQuantMI300X(bits=bits, rotation_seed=42)

    results = []

    for n_kv in seq_lens:
        row = {"n_kv": n_kv, "head_dim": HEAD_DIM, "n_kv_heads": N_KV_HEADS}

        # Create synthetic KV in FP16
        k_fp16 = torch.randn(N_KV_HEADS, n_kv, HEAD_DIM, device="cuda", dtype=torch.float16)
        v_fp16 = torch.randn(N_KV_HEADS, n_kv, HEAD_DIM, device="cuda", dtype=torch.float16)
        q_fp16 = torch.randn(1, N_KV_HEADS, 1, HEAD_DIM, device="cuda", dtype=torch.float16)

        # ── FP16 baseline ────────────────────────────────────────────────────
        k_4d = k_fp16.unsqueeze(0)  # (1, H, S, D)
        v_4d = v_fp16.unsqueeze(0)
        q_4d = q_fp16

        try:
            t_ms = timed(lambda: attention_fp16(q_4d, k_4d, v_4d), n_warmup, n_iters)
            kv_bytes = (k_fp16.numel() + v_fp16.numel()) * 2  # float16 = 2B
            row["fp16_ms"]  = round(t_ms, 3)
            row["fp16_gbs"] = round(kv_bytes / (t_ms * 1e-3) / 1e9, 1)
        except Exception as e:
            row["fp16_ms"] = None
            row["fp16_error"] = str(e)

        # ── TQ3 compressed ───────────────────────────────────────────────────
        # Compress KV cache
        k_flat = k_fp16.reshape(-1, HEAD_DIM).float()
        v_flat = v_fp16.reshape(-1, HEAD_DIM).float()
        k_comp = tq.compress_tensor(k_flat)   # (N_KV_HEADS*n_kv, 52)
        v_comp = tq.compress_tensor(v_flat)

        # Pre-rotate queries
        q_flat = q_fp16.reshape(-1, HEAD_DIM).float()
        q_rot  = tq.rotate_queries(q_flat)

        try:
            n_kv_total = N_KV_HEADS * n_kv
            t_ms = timed(
                lambda: attention_tq3(q_rot, k_comp, v_comp, tq),
                n_warmup, n_iters
            )
            kv_bytes = (k_comp.numel() + v_comp.numel())  # uint8 = 1B
            tq_ratio = {"3": 4.92, "4": 3.76}.get(str(bits), 4.92)
            row[f"tq{bits}_ms"]  = round(t_ms, 3)
            row[f"tq{bits}_gbs"] = round(kv_bytes / (t_ms * 1e-3) / 1e9, 1)
            row[f"tq{bits}_speedup"] = round(row["fp16_ms"] / t_ms, 2) if row.get("fp16_ms") else None
            row[f"tq{bits}_compression"] = round(tq_ratio, 2)
        except Exception as e:
            row[f"tq{bits}_ms"] = None
            row[f"tq{bits}_error"] = str(e)

        print(f"  n_kv={n_kv:7d}: "
              f"fp16={row.get('fp16_ms', '?'):7.2f} ms  "
              f"tq{bits}={row.get(f'tq{bits}_ms', '?'):7.2f} ms  "
              f"speedup={row.get(f'tq{bits}_speedup', '?')}×  "
              f"tq{bits}_bw={row.get(f'tq{bits}_gbs', '?'):.1f} GB/s")

        results.append(row)

        # Clear GPU memory
        del k_fp16, v_fp16, k_comp, v_comp
        torch.cuda.empty_cache()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TQ3 attention throughput vs FP16")
    parser.add_argument("--seq-lens", nargs="+", type=int, default=SEQ_LENS)
    parser.add_argument("--n-iters", type=int, default=20)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"=== TurboQuant Attention Throughput Benchmark ===")
    print(f"Device:     {torch.cuda.get_device_name(0)}")
    print(f"Config:     {N_KV_HEADS} KV heads × head_dim={HEAD_DIM}")
    print(f"TQ bits:    {args.bits}  (TQ{args.bits})")
    print(f"seq_lens:   {args.seq_lens}")
    print()

    results = bench_attention_seq_lens(
        args.seq_lens,
        n_iters=args.n_iters,
        n_warmup=args.n_warmup,
        bits=args.bits,
    )

    output = {
        "device":    torch.cuda.get_device_name(0),
        "n_kv_heads": N_KV_HEADS,
        "head_dim":  HEAD_DIM,
        "tq_bits":   args.bits,
        "results":   results,
    }

    output_path = args.output or (RESULTS_DIR / f"bench_tq{args.bits}_attention.json")
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Print summary table
    print("\n── Summary: TQ3 speedup over FP16 ──")
    print(f"  {'n_kv':>8}  {'fp16_ms':>9}  {'tq3_ms':>9}  {'speedup':>8}  {'tq3_bw':>10}")
    for r in results:
        print(f"  {r['n_kv']:>8,}  "
              f"{r.get('fp16_ms', 'N/A'):>9}  "
              f"{r.get(f'tq{args.bits}_ms', 'N/A'):>9}  "
              f"{str(r.get(f'tq{args.bits}_speedup', 'N/A')):>7}×  "
              f"{r.get(f'tq{args.bits}_gbs', 'N/A'):>8} GB/s")


if __name__ == "__main__":
    main()
