"""
bench_batch_attention.py — TQ3 vs FP16 SDPA at multiple batch sizes.

Sweeps batch × seq_k to show the crossover from compute-bound (batch=1,
dequant overhead > BW savings) to bandwidth-bound (batch≥4, KV-cache BW
dominates and 4.9× fewer bytes → ~4× more tokens/sec).

Approach: compress a single-element KV once and expand along the batch dim.
This correctly scales HBM traffic (each batch element reads its own KV slice)
while avoiding the OOM that would come from compressing full (batch, H, S, D)
tensors via the Python wrapper.

Usage:
    python3 benchmarks/bench_batch_attention.py
    python3 benchmarks/bench_batch_attention.py --seq-lens 8192 32768 131072
    python3 benchmarks/bench_batch_attention.py --batch-sizes 1 2 4 8 16 32
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.nn.functional import scaled_dot_product_attention

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR.mkdir(exist_ok=True)


def _median_ms(fn, warmup: int, reps: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(reps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return times[len(times) // 2]


def run(
    batch_sizes=(1, 2, 4, 8, 16, 32),
    seq_lens=(8192, 32768, 131072),
    warmup: int = 5,
    reps: int = 30,
    head_dim: int = 128,
    n_heads: int = 32,
):
    from turboquant_mi300x import TurboQuantMI300X
    from tq_triton import (
        turboquant_attention_fwd,
        turboquant_nibble_attention_fwd,
        compress_kv_for_triton,
        compress_kv_nibble,
    )

    tq = TurboQuantMI300X(bits=3, device="cuda")
    sm_scale = head_dim ** -0.5

    print("=" * 100)
    print("Batch-decode attention: FP16 SDPA  vs  TQ3 bit-plane (v2)  vs  TQ3 nibble (v2)")
    print("=" * 100)
    print()
    print("Physics of batch=1 vs batch>1:")
    print("  Each decode step reads:  model weights (~14 GB, fixed)  +  KV cache (~0.13 GB/seq-32K/step)")
    print("  At batch=1, weights dominate → TQ3's smaller KV doesn't help; dequant adds overhead.")
    print("  At batch≥4 (seq=32K), KV BW ≥ weight BW → 4.9× fewer bytes read ≈ 4× more tokens/sec.")
    print("  This benchmark isolates the *attention kernel only* to show the pure KV-BW effect.")
    print()

    all_results = []

    for seq_k in seq_lens:
        # Bytes read per decode step per batch element (K + V, all heads)
        bw_fp16 = n_heads * seq_k * head_dim * 2 * 2   # FP16 K+V
        bw_bp   = n_heads * seq_k * 52          * 2   # bit-plane K+V (52B norm+planes per token)
        bw_nb   = n_heads * seq_k * 68          * 2   # nibble K+V (64B nibbles + 4B norm per token)

        print(f"\n── seq_k = {seq_k:,}  "
              f"(FP16 KV: {bw_fp16/1e6:.0f} MB  "
              f"TQ3-BP KV: {bw_bp/1e6:.0f} MB  "
              f"TQ3-Nb KV: {bw_nb/1e6:.0f} MB  per step per batch-elem) ──")

        # Compress once for a single batch element, then expand
        torch.cuda.empty_cache()
        k1 = torch.randn(1, n_heads, seq_k, head_dim, device="cuda", dtype=torch.float16)
        v1 = torch.randn(1, n_heads, seq_k, head_dim, device="cuda", dtype=torch.float16)
        kp1, kn1, vp1, vn1 = compress_kv_for_triton(k1, v1, tq)
        knb1, kn21, vnb1, vn21 = compress_kv_nibble(k1, v1, tq)
        # Keep k1/v1 for batch expansion (FP16 SDPA needs contiguous copies)
        k1_keep = k1
        v1_keep = v1

        col = (f"  {'batch':>5}  {'FP16 ms':>9}  {'TQ3-BP ms':>10}  {'ratio':>7}"
               f"  {'TQ3-Nb ms':>10}  {'ratio':>7}"
               f"  {'tok/s FP16':>11}  {'tok/s BP':>10}  {'tok/s Nb':>10}")
        print(col)
        print("  " + "─" * (len(col) - 2))

        seq_results = []

        for batch in batch_sizes:
            torch.cuda.empty_cache()
            try:
                q_fp16 = torch.randn(batch, n_heads, 1, head_dim,
                                     device="cuda", dtype=torch.float16)
                q_rot  = tq.rotate_queries(q_fp16.float()).half()

                # FP16: each batch element gets its own KV copy (contiguous for SDPA)
                k_b = k1_keep.expand(batch, -1, -1, -1).contiguous()
                v_b = v1_keep.expand(batch, -1, -1, -1).contiguous()

                # TQ3: expand compressed buffers
                kp  = kp1.expand(batch, -1, -1, -1).contiguous()
                kn  = kn1.expand(batch, -1, -1).contiguous()
                vp  = vp1.expand(batch, -1, -1, -1).contiguous()
                vn  = vn1.expand(batch, -1, -1).contiguous()
                knb = knb1.expand(batch, -1, -1, -1).contiguous()
                kn2 = kn21.expand(batch, -1, -1).contiguous()
                vnb = vnb1.expand(batch, -1, -1, -1).contiguous()
                vn2 = vn21.expand(batch, -1, -1).contiguous()

                ms_fp16 = _median_ms(
                    lambda: scaled_dot_product_attention(q_fp16, k_b, v_b, scale=sm_scale),
                    warmup, reps)
                ms_bp = _median_ms(
                    lambda: turboquant_attention_fwd(
                        q_rot, kp, kn, vp, vn,
                        rotation=tq.rotation, sm_scale=sm_scale),
                    warmup, reps)
                ms_nb = _median_ms(
                    lambda: turboquant_nibble_attention_fwd(
                        q_rot, knb, kn2, vnb, vn2,
                        rotation=tq.rotation, sm_scale=sm_scale),
                    warmup, reps)

            except torch.cuda.OutOfMemoryError:
                print(f"  {batch:>5}  OOM")
                torch.cuda.empty_cache()
                continue

            ratio_bp = ms_fp16 / ms_bp
            ratio_nb = ms_fp16 / ms_nb
            tps_fp16 = batch / (ms_fp16 / 1000)
            tps_bp   = batch / (ms_bp   / 1000)
            tps_nb   = batch / (ms_nb   / 1000)

            flag_bp = " ◀ FASTER" if ratio_bp >= 1.0 else ""
            flag_nb = " ◀ FASTER" if ratio_nb >= 1.0 else ""

            print(f"  {batch:>5}  {ms_fp16:>9.3f}  {ms_bp:>10.3f}  {ratio_bp:>6.2f}×{flag_bp:<10}"
                  f"  {ms_nb:>10.3f}  {ratio_nb:>6.2f}×{flag_nb:<10}"
                  f"  {tps_fp16:>11.1f}  {tps_bp:>10.1f}  {tps_nb:>10.1f}")

            seq_results.append({
                "seq_k": seq_k, "batch": batch,
                "fp16_ms": ms_fp16, "bp_ms": ms_bp, "nb_ms": ms_nb,
                "ratio_bp": ratio_bp, "ratio_nb": ratio_nb,
                "tps_fp16": tps_fp16, "tps_bp": tps_bp, "tps_nb": tps_nb,
                "kv_bw_fp16_GBs": bw_fp16 * batch / (ms_fp16 / 1000) / 1e9,
                "kv_bw_bp_GBs":   bw_bp   * batch / (ms_bp   / 1000) / 1e9,
                "kv_bw_nb_GBs":   bw_nb   * batch / (ms_nb   / 1000) / 1e9,
            })

            del q_fp16, q_rot, k_b, v_b, kp, kn, vp, vn, knb, kn2, vnb, vn2
            torch.cuda.empty_cache()

        del k1, v1, k1_keep, v1_keep, kp1, kn1, vp1, vn1, knb1, kn21, vnb1, vn21
        torch.cuda.empty_cache()
        all_results.extend(seq_results)

    return all_results


def _print_summary(results):
    print()
    print("=" * 80)
    print("Crossover summary — first batch where TQ3 outperforms FP16:")
    print("=" * 80)
    by_seq = {}
    for r in results:
        by_seq.setdefault(r["seq_k"], []).append(r)

    for seq_k, rows in sorted(by_seq.items()):
        cross_bp = next((r["batch"] for r in rows if r["ratio_bp"] >= 1.0), None)
        cross_nb = next((r["batch"] for r in rows if r["ratio_nb"] >= 1.0), None)
        peak_bp  = max(r["ratio_bp"] for r in rows)
        peak_nb  = max(r["ratio_nb"] for r in rows)
        b1_bp    = next((r["ratio_bp"] for r in rows if r["batch"] == 1), None)
        b1_nb    = next((r["ratio_nb"] for r in rows if r["batch"] == 1), None)
        print(f"  seq={seq_k:>7,}: "
              f"TQ3-BP: batch=1 → {b1_bp:.2f}×  crossover={cross_bp or '(beyond range)'}  "
              f"peak={peak_bp:.2f}×")
        print(f"           "
              f"TQ3-Nb: batch=1 → {b1_nb:.2f}×  crossover={cross_nb or '(beyond range)'}  "
              f"peak={peak_nb:.2f}×")
    print()
    print("TQ3 throughput gain at max tested batch (tokens/sec ratio):")
    for seq_k, rows in sorted(by_seq.items()):
        last = rows[-1]
        print(f"  seq={seq_k:>7,} batch={last['batch']:>2}:  "
              f"FP16={last['tps_fp16']:.0f} tok/s  "
              f"BP={last['tps_bp']:.0f} tok/s ({last['ratio_bp']:.2f}×)  "
              f"Nb={last['tps_nb']:.0f} tok/s ({last['ratio_nb']:.2f}×)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch-sizes", nargs="+", type=int,
                    default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--seq-lens", nargs="+", type=int,
                    default=[8192, 32768, 131072])
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--reps",   type=int, default=30)
    ap.add_argument("--out", default=str(RESULTS_DIR / "bench_batch_attention.json"))
    args = ap.parse_args()

    results = run(
        batch_sizes=args.batch_sizes,
        seq_lens=args.seq_lens,
        warmup=args.warmup,
        reps=args.reps,
    )
    _print_summary(results)

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults → {args.out}")


if __name__ == "__main__":
    main()
