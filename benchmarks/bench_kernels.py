"""
bench_kernels.py — TurboQuant kernel throughput microbenchmark

Measures raw kernel performance (GB/s, us/call) for:
  1. TQ3 compress (quantize)        — using Python wrapper (torch.matmul → MFMA)
  2. TQ3 decompress (dequantize)    — using Python wrapper
  3. TQ3 fused dot product          — using Python wrapper
  4. Standalone binary throughput   — runs tq_bench_mi300x for C-level numbers
  5. Reference comparison vs FP16   — torch.matmul equivalent

For context: the standalone binary measures pure HIP kernel throughput
(without Python overhead). The Python wrapper uses torch.matmul → rocBLAS → MFMA.
Both are benchmarked to understand the overhead of the Python abstraction.

Usage:
    python3 benchmarks/bench_kernels.py
    python3 benchmarks/bench_kernels.py --n-vectors 65536 --n-iters 200
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

# Add kernels dir to path
KERNELS_DIR = Path(__file__).parent.parent / "kernels"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

def timed_kernel(fn, n_warmup: int = 10, n_iters: int = 100):
    """Time a GPU kernel with warmup. Returns (median_us, throughput_gbs_fn)."""
    # Warmup
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

    return float(np.median(times)) * 1e6  # median microseconds


def bytes_to_gbs(n_bytes: int, time_us: float) -> float:
    return n_bytes / (time_us * 1e-6) / 1e9


# ──────────────────────────────────────────────────────────────────────────────
# Python-wrapper benchmarks (torch.matmul → rocBLAS → MFMA)
# ──────────────────────────────────────────────────────────────────────────────

def bench_python_wrapper(n_vectors: int, n_iters: int = 100, n_warmup: int = 20):
    """Benchmark TurboQuantMI300X Python wrapper operations."""
    from turboquant_mi300x import TurboQuantMI300X

    tq = TurboQuantMI300X(bits=3, rotation_seed=42)
    HEAD_DIM = 128
    TQ3_BYTES = 52

    # Generate test data (use different seed from rotation)
    rng = np.random.default_rng(0)
    x = torch.from_numpy(
        rng.standard_normal((n_vectors, HEAD_DIM)).astype(np.float32)
    ).cuda()
    x = (x / x.norm(dim=-1, keepdim=True)).contiguous()

    # Pre-compress for decompress and fused-dot benchmarks
    compressed = tq.compress_tensor(x)

    # Pre-rotated queries for fused dot
    n_queries = min(64, n_vectors)
    q_raw = torch.randn(n_queries, HEAD_DIM, device="cuda")
    q_rot = tq.rotate_queries(q_raw)

    results = {}

    # 1. Compress (quantize)
    t_us = timed_kernel(lambda: tq.compress_tensor(x), n_warmup, n_iters)
    in_bytes  = n_vectors * HEAD_DIM * 4   # float32 input
    out_bytes = n_vectors * TQ3_BYTES       # uint8 output
    io_bytes  = in_bytes + out_bytes
    results["compress"] = {
        "op":           "tq3_compress",
        "n_vectors":    n_vectors,
        "median_us":    round(t_us, 2),
        "throughput_gbs": round(bytes_to_gbs(io_bytes, t_us), 1),
        "vectors_per_us": round(n_vectors / t_us, 2),
        "note": "rotation (GEMM→MFMA) + centroid lookup + bitpack",
    }

    # 2. Decompress (dequantize)
    t_us = timed_kernel(
        lambda: tq.decompress_tensor(compressed, (n_vectors, HEAD_DIM)),
        n_warmup, n_iters
    )
    in_bytes  = n_vectors * TQ3_BYTES
    out_bytes = n_vectors * HEAD_DIM * 4
    io_bytes  = in_bytes + out_bytes
    results["decompress"] = {
        "op":             "tq3_decompress",
        "n_vectors":      n_vectors,
        "median_us":      round(t_us, 2),
        "throughput_gbs": round(bytes_to_gbs(io_bytes, t_us), 1),
        "vectors_per_us": round(n_vectors / t_us, 2),
        "note": "bitunpack + centroid lookup + inverse rotation (GEMM→MFMA)",
    }

    # 3. Fused dot product
    t_us = timed_kernel(
        lambda: tq.fused_dot(q_rot, compressed),
        n_warmup, n_iters
    )
    in_bytes  = n_queries * HEAD_DIM * 4 + n_vectors * TQ3_BYTES
    out_bytes = n_queries * n_vectors * 4
    io_bytes  = in_bytes + out_bytes
    results["fused_dot"] = {
        "op":             "tq3_fused_dot",
        "n_queries":      n_queries,
        "n_kv":           n_vectors,
        "median_us":      round(t_us, 2),
        "throughput_gbs": round(bytes_to_gbs(io_bytes, t_us), 1),
        "note": "bitunpack + centroid lookup + dot product (GEMM→MFMA)",
    }

    # 4. Reference: plain FP16 matrix multiply (no compression)
    q_fp16 = q_raw.half()
    kv_fp16 = x[:n_vectors].half()
    t_us_ref = timed_kernel(
        lambda: torch.matmul(q_fp16, kv_fp16.T),
        n_warmup, n_iters
    )
    ref_bytes = q_fp16.numel() * 2 + kv_fp16.numel() * 2 + n_queries * n_vectors * 2
    results["fp16_matmul"] = {
        "op":             "fp16_matmul (reference)",
        "n_queries":      n_queries,
        "n_kv":           n_vectors,
        "median_us":      round(t_us_ref, 2),
        "throughput_gbs": round(bytes_to_gbs(ref_bytes, t_us_ref), 1),
    }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Standalone C binary benchmark
# ──────────────────────────────────────────────────────────────────────────────

def bench_standalone_binary(n_vectors: int, n_iters: int = 50):
    """Run tq_bench_mi300x binary and parse output."""
    binary = KERNELS_DIR / "hip" / "tq_bench_mi300x"
    if not binary.exists():
        return {"error": f"{binary} not found. Run: cd kernels/hip && bash build_mi300x.sh bench"}

    try:
        result = subprocess.run(
            [str(binary), str(n_vectors), str(n_iters)],
            capture_output=True, text=True, timeout=120
        )
        lines = result.stdout.strip().split("\n")
        parsed = {}
        for line in lines:
            # Parse lines like: "quantize:   65536 vectors  200 iters  9.8 GB/s  ..."
            if "GB/s" in line or "us/call" in line:
                parts = line.split()
                if parts:
                    op = parts[0].rstrip(":")
                    for i, p in enumerate(parts):
                        if "GB/s" in p and i > 0:
                            try:
                                parsed[op] = {
                                    "throughput_gbs": float(parts[i - 1]),
                                    "source": "standalone_binary",
                                }
                            except (ValueError, IndexError):
                                pass
        return {"binary_results": parsed, "raw_output": result.stdout[:2000]}
    except subprocess.TimeoutExpired:
        return {"error": "tq_bench_mi300x timed out"}
    except Exception as e:
        return {"error": str(e)}


# ──────────────────────────────────────────────────────────────────────────────
# Memory bandwidth analysis
# ──────────────────────────────────────────────────────────────────────────────

def analyze_memory_savings(n_vectors: int):
    """Compute theoretical memory savings for different KV configurations."""
    HEAD_DIM = 128
    fp16_bytes  = n_vectors * HEAD_DIM * 2    # float16
    tq3_bytes   = n_vectors * 52              # TQ3: 4 + 48 bytes
    tq4_bytes   = n_vectors * 68             # TQ4: 4 + 64 bytes
    fp8_bytes   = n_vectors * HEAD_DIM * 1   # FP8 (1 byte per element)
    int4_bytes  = n_vectors * HEAD_DIM // 2  # INT4 nibble-packed

    base = fp16_bytes
    return {
        "n_vectors":    n_vectors,
        "fp16_bytes":   fp16_bytes,
        "fp8_bytes":    fp8_bytes,
        "int4_bytes":   int4_bytes,
        "tq3_bytes":    tq3_bytes,
        "tq4_bytes":    tq4_bytes,
        "fp8_ratio":    round(base / fp8_bytes, 2),
        "int4_ratio":   round(base / int4_bytes, 2),
        "tq3_ratio":    round(base / tq3_bytes, 2),
        "tq4_ratio":    round(base / tq4_bytes, 2),
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TurboQuant kernel throughput benchmark")
    parser.add_argument("--n-vectors", type=int, default=65536,
                        help="Number of KV vectors to benchmark")
    parser.add_argument("--n-iters", type=int, default=100,
                        help="Timing iterations (median reported)")
    parser.add_argument("--n-warmup", type=int, default=20,
                        help="Warmup iterations (discarded)")
    parser.add_argument("--skip-binary", action="store_true",
                        help="Skip standalone binary benchmark")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"=== TurboQuant Kernel Throughput Benchmark ===")
    print(f"Device:    {torch.cuda.get_device_name(0)}")
    print(f"Vectors:   {args.n_vectors:,}  (head_dim=128, {args.n_vectors * 128 * 4 / 1e6:.1f} MB FP32)")
    print(f"Iters:     {args.n_iters} (median)")
    print()

    all_results = {
        "device":        torch.cuda.get_device_name(0),
        "n_vectors":     args.n_vectors,
        "head_dim":      128,
        "python_wrapper": {},
        "standalone_binary": {},
        "memory_analysis": {},
    }

    # Python wrapper benchmarks
    print("── Python wrapper (torch.matmul → rocBLAS → MFMA) ──")
    py_results = bench_python_wrapper(args.n_vectors, args.n_iters, args.n_warmup)
    all_results["python_wrapper"] = py_results

    for op, r in py_results.items():
        n = r.get("n_vectors") or r.get("n_kv", args.n_vectors)
        print(f"  {r['op']:35s}  {r['median_us']:8.1f} us  "
              f"{r['throughput_gbs']:6.1f} GB/s")

    print()

    # Standalone binary
    if not args.skip_binary:
        print("── Standalone binary (tq_bench_mi300x, ROCm 7.2 HIP kernels) ──")
        binary_results = bench_standalone_binary(args.n_vectors, n_iters=50)
        all_results["standalone_binary"] = binary_results
        if "error" in binary_results:
            print(f"  ERROR: {binary_results['error']}")
        elif "binary_results" in binary_results:
            for op, r in binary_results["binary_results"].items():
                print(f"  {op:35s}  {r.get('throughput_gbs', '?'):6.1f} GB/s")
            if binary_results.get("raw_output"):
                print()
                print("  Raw output:")
                for line in binary_results["raw_output"].strip().split("\n"):
                    print(f"    {line}")
        print()

    # Memory analysis
    print("── Memory savings analysis ──")
    mem = analyze_memory_savings(args.n_vectors)
    all_results["memory_analysis"] = mem
    print(f"  {'Config':<15} {'Bytes/vec':>12} {'Total MB':>10} {'vs FP16':>10}")
    print(f"  {'-'*50}")
    for cfg, ratio_key, bytes_key in [
        ("FP16 (base)",   "fp16_ratio", "fp16_bytes"),
        ("FP8",           "fp8_ratio",  "fp8_bytes"),
        ("INT4",          "int4_ratio", "int4_bytes"),
        ("TQ3 (ours)",    "tq3_ratio",  "tq3_bytes"),
        ("TQ4 (ours)",    "tq4_ratio",  "tq4_bytes"),
    ]:
        total_mb = mem[bytes_key] / 1e6
        bpv = mem[bytes_key] / args.n_vectors
        ratio = mem.get(ratio_key, 1.0)
        ratio_str = f"{ratio:.2f}×" if ratio != 1.0 else "1.00× (ref)"
        print(f"  {cfg:<15} {bpv:>12.1f} B/vec  {total_mb:>8.1f} MB  {ratio_str:>10}")

    # Save
    output_path = args.output or (RESULTS_DIR / "bench_kernels.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
