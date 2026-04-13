"""
bench_compress_decompress.py — Compress/Decompress Microbenchmark for All Methods

Measures standalone compress and decompress throughput (GB/s) for:
  TurboQuant (TQ3, TQ4)  — existing HIP + PyTorch wrapper
  IsoQuant (iso3, iso4)  — Triton (ROCm port)
  PlanarQuant (planar3, planar4) — Triton (ROCm port)
  RotorQuant (rotor3, rotor4)   — Triton (ROCm port)

This is the cleanest apples-to-apples comparison: same input size, same device,
same head_dim=128. FMA counts and bandwidth efficiency show why Planar/Iso win.

Usage:
    python3 benchmarks/bench_compress_decompress.py
    python3 benchmarks/bench_compress_decompress.py --n-vectors 65536 --n-iters 100
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR.mkdir(exist_ok=True)


def sync():
    torch.cuda.synchronize()


def timed_fn(fn, n_warmup: int = 20, n_iters: int = 100) -> float:
    """Return median latency in microseconds."""
    for _ in range(n_warmup):
        fn()
    sync()
    times = []
    for _ in range(n_iters):
        sync()
        t0 = time.perf_counter()
        fn()
        sync()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)) * 1e6  # µs


def bytes_to_gbs(n_bytes: int, time_us: float) -> float:
    return n_bytes / (time_us * 1e-6) / 1e9


def bench_method(method: str, bits: int, x: torch.Tensor,
                 n_iters: int, n_warmup: int) -> dict:
    """Benchmark compress + decompress for one (method, bits) configuration."""
    from block_quant_rocm import make_quantizer, BYTES_PER_VEC, FMAS_PER_VEC, FP16_BYTES

    N, D = x.shape
    fp16_bytes_in = N * D * 2   # input FP16 bytes
    compressed_bytes = N * BYTES_PER_VEC[bits]   # output bytes

    q = make_quantizer(method, bits=bits, head_dim=D, device=str(x.device))

    # Warmup (triggers Triton JIT compile on first call)
    print(f"    Warming up {method}{bits}...", end="", flush=True)
    for _ in range(max(n_warmup, 5)):
        comp = q.compress(x)
        _ = q.decompress(comp, x.shape)
    sync()
    print(" done")

    # Benchmark compress
    compress_us = timed_fn(lambda: q.compress(x), n_warmup=n_warmup, n_iters=n_iters)
    # Benchmark decompress (pre-compress so we measure decompress only)
    comp = q.compress(x)
    sync()
    decompress_us = timed_fn(lambda: q.decompress(comp, x.shape), n_warmup=n_warmup, n_iters=n_iters)

    compress_bw = bytes_to_gbs(fp16_bytes_in + compressed_bytes, compress_us)
    decompress_bw = bytes_to_gbs(compressed_bytes + fp16_bytes_in, decompress_us)

    return {
        "method": method,
        "bits": bits,
        "n_vectors": N,
        "head_dim": D,
        "compress_us": compress_us,
        "decompress_us": decompress_us,
        "compress_bw_gbs": compress_bw,
        "decompress_bw_gbs": decompress_bw,
        "fp16_bytes_in": fp16_bytes_in,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": FP16_BYTES / BYTES_PER_VEC[bits],
        "fmas_per_vec": FMAS_PER_VEC.get(method, 0),
    }


def bench_turboquant(bits: int, x: torch.Tensor,
                     n_iters: int, n_warmup: int) -> dict:
    """Benchmark TurboQuant using the existing Python wrapper."""
    from turboquant_mi300x import (TurboQuantMI300X, COMPRESSION_RATIO,
                                    TQ2_BLOCK_BYTES, TQ3_BLOCK_BYTES, TQ4_BLOCK_BYTES)
    TQ_BPV = {2: TQ2_BLOCK_BYTES, 3: TQ3_BLOCK_BYTES, 4: TQ4_BLOCK_BYTES}

    N, D = x.shape
    fp16_bytes_in = N * D * 2
    bpv = TQ_BPV.get(bits, 52)
    compressed_bytes = N * bpv

    tq = TurboQuantMI300X(bits=bits, rotation_seed=42)
    x_fp32 = x.float()

    print(f"    Warming up turbo{bits}...", end="", flush=True)
    for _ in range(max(n_warmup, 5)):
        comp = tq.compress_tensor(x_fp32)
        _ = tq.decompress_tensor(comp, x_fp32.shape)
    sync()
    print(" done")

    compress_us = timed_fn(lambda: tq.compress_tensor(x_fp32), n_warmup=n_warmup, n_iters=n_iters)
    comp = tq.compress_tensor(x_fp32)
    sync()
    decompress_us = timed_fn(lambda: tq.decompress_tensor(comp, x_fp32.shape), n_warmup=n_warmup, n_iters=n_iters)

    compress_bw = bytes_to_gbs(fp16_bytes_in + compressed_bytes, compress_us)
    decompress_bw = bytes_to_gbs(compressed_bytes + fp16_bytes_in, decompress_us)

    return {
        "method": "turbo",
        "bits": bits,
        "n_vectors": N,
        "head_dim": D,
        "compress_us": compress_us,
        "decompress_us": decompress_us,
        "compress_bw_gbs": compress_bw,
        "decompress_bw_gbs": decompress_bw,
        "fp16_bytes_in": fp16_bytes_in,
        "compressed_bytes": compressed_bytes,
        "compression_ratio": COMPRESSION_RATIO.get(bits, 4.92),
        "fmas_per_vec": 16384,
    }


def main():
    parser = argparse.ArgumentParser(description="Compress/decompress microbenchmark")
    parser.add_argument("--n-vectors", type=int, default=65536,
                        help="Number of KV vectors to benchmark")
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--bits", type=int, nargs="+", default=[3, 4])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["planar", "iso", "rotor", "turbo"])
    parser.add_argument("--n-iters", type=int, default=100)
    parser.add_argument("--n-warmup", type=int, default=20)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = "cuda"
    print(f"\n{'='*70}")
    print(f"Compress/Decompress Microbenchmark")
    print(f"n_vectors={args.n_vectors}, head_dim={args.head_dim}, device={device}")
    print(f"{'='*70}")

    rng = np.random.default_rng(42)
    x = torch.from_numpy(rng.standard_normal(
        (args.n_vectors, args.head_dim)).astype(np.float32)
    ).to(device)

    all_results = []

    # FP16 reference
    ref_compress_us = timed_fn(lambda: x.half(), n_warmup=10, n_iters=50)
    print(f"\nFP16 reference (no-op cast): {ref_compress_us:.1f} µs "
          f"({bytes_to_gbs(args.n_vectors * args.head_dim * 2, ref_compress_us):.1f} GB/s)")

    print(f"\n{'Method':<12} {'Bits':<5} {'Compress µs':>12} {'Compress GB/s':>14} "
          f"{'Decomp µs':>11} {'Decomp GB/s':>13} {'Ratio':>7} {'FMAs/vec':>10}")
    print("-" * 90)

    for method in args.methods:
        for bits in args.bits:
            print(f"  Benchmarking {method}{bits}...")
            try:
                if method == "turbo":
                    result = bench_turboquant(bits, x, args.n_iters, args.n_warmup)
                else:
                    result = bench_method(method, bits, x, args.n_iters, args.n_warmup)

                all_results.append(result)
                label = f"{method}{bits}"
                print(f"  {label:<12} {bits:<5} "
                      f"{result['compress_us']:>12.1f} {result['compress_bw_gbs']:>14.1f} "
                      f"{result['decompress_us']:>11.1f} {result['decompress_bw_gbs']:>13.1f} "
                      f"{result['compression_ratio']:>6.2f}× {result['fmas_per_vec']:>10,}")
            except Exception as e:
                print(f"  ERROR {method}{bits}: {e}")

    print("=" * 90)

    # Relative comparison table
    if all_results:
        print(f"\n{'Relative Performance vs TurboQuant3 (lower is faster, higher decomp BW is better)'}")
        tq3 = next((r for r in all_results if r["method"] == "turbo" and r["bits"] == 3), None)
        if tq3:
            print(f"{'Method':<12} {'Compress speedup':>17} {'Decomp speedup':>15}")
            print("-" * 46)
            for r in all_results:
                label = f"{r['method']}{r['bits']}"
                cs = tq3["compress_us"] / r["compress_us"]
                ds = tq3["decompress_us"] / r["decompress_us"]
                print(f"{label:<12} {cs:>16.2f}× {ds:>14.2f}×")

    # Save results
    out_path = args.output or str(RESULTS_DIR / "bench_compress_decompress.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
