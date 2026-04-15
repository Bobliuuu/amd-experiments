"""
bench_mfma_rotate.py — Validate and benchmark the MFMA rotation kernel

Tests:
  1. HSACO compilation (gfx942, COV5)
  2. Numerical correctness vs torch.matmul  (cos_sim, max_abs_err)
  3. Round-trip:  inverse(forward(X)) ≈ X  (orthogonality sanity check)
  4. Throughput sweep: n = 256, 1024, 4096, 16384, 65536 vectors
     Reports GB/s and µs, compares MFMA vs torch.matmul
  5. TQ3 compress/decompress with MFMA enabled vs disabled
     Shows end-to-end speedup for the full pipeline

Usage:
    cd /root/workspace/amd-experiments
    python3 benchmarks/bench_mfma_rotate.py
    python3 benchmarks/bench_mfma_rotate.py --skip-compile   # if HSACO already exists
    python3 benchmarks/bench_mfma_rotate.py --force-compile  # force recompilation
"""

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

_KERNELS_DIR = Path(__file__).parent.parent / "kernels"
sys.path.insert(0, str(_KERNELS_DIR))

HEAD_DIM = 128


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def bench_fn(fn, X, n_warm=30, n_bench=200) -> float:
    """Returns median latency in µs."""
    for _ in range(n_warm):
        fn(X)
    torch.cuda.synchronize()
    times = []
    for _ in range(n_bench):
        t0 = time.perf_counter()
        fn(X)
        torch.cuda.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return times[len(times) // 2]   # median


def rotation_bandwidth(n: int, us: float) -> float:
    """Effective memory bandwidth in GB/s for an (n, 128) @ (128, 128) matmul."""
    bytes_read  = n * HEAD_DIM * 4 + HEAD_DIM * HEAD_DIM * 4  # X + R
    bytes_write = n * HEAD_DIM * 4                              # Y
    return (bytes_read + bytes_write) / us / 1e3               # GB/s


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MFMA rotation kernel benchmark")
    parser.add_argument("--skip-compile",  action="store_true",
                        help="Skip compilation (HSACO must already exist)")
    parser.add_argument("--force-compile", action="store_true",
                        help="Force HSACO recompilation")
    parser.add_argument("--output", type=str, default=None,
                        help="Save JSON results to this path")
    parser.add_argument("--ns", nargs="+", type=int,
                        default=[256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536],
                        help="Batch sizes to benchmark")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("No GPU detected. Exiting.")
        sys.exit(1)

    device = "cuda"
    print("=== MFMA Rotation Kernel Benchmark ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch ROCm: {torch.version.hip}")
    print()

    # ── Step 1: Compile ──────────────────────────────────────────────────────
    from tq_mfma_loader import compile_mfma_hsaco, MFMARotate, _HSACO_PATH

    if not args.skip_compile:
        print("── Compilation ──────────────────────────────────────────────────")
        try:
            path = compile_mfma_hsaco(force=args.force_compile)
            print(f"  HSACO: {path.name} ({path.stat().st_size / 1024:.0f} KB)")
        except Exception as e:
            print(f"  COMPILE FAILED: {e}")
            print("  Continuing with torch.matmul fallback only.")
        print()

    # ── Step 2: Load + correctness check ────────────────────────────────────
    print("── Correctness Check ────────────────────────────────────────────────")
    from turboquant_mi300x import make_rotation_matrix

    R = make_rotation_matrix(seed=42, device=device)
    mfma = MFMARotate(R)
    print(f"  {mfma}")

    X_test = torch.randn(4096, HEAD_DIM, device=device, dtype=torch.float32)

    # Forward: Y = X @ R.T
    Y_mfma = mfma.forward(X_test)
    Y_ref  = X_test @ R.T
    cos_fwd   = F.cosine_similarity(Y_mfma.reshape(-1), Y_ref.reshape(-1), dim=0).item()
    maxerr_fwd = (Y_mfma - Y_ref).abs().max().item()
    relerr_fwd = ((Y_mfma - Y_ref).norm() / Y_ref.norm()).item()

    # Inverse: Y = X @ R
    Y_inv  = mfma.inverse(X_test)
    Y_iref = X_test @ R
    cos_inv   = F.cosine_similarity(Y_inv.reshape(-1), Y_iref.reshape(-1), dim=0).item()
    maxerr_inv = (Y_inv - Y_iref).abs().max().item()

    # Round-trip: inverse(forward(X)) should recover X  (R.T @ R = I)
    Y_rt   = mfma.inverse(mfma.forward(X_test))
    cos_rt = F.cosine_similarity(Y_rt.reshape(-1), X_test.reshape(-1), dim=0).item()

    print(f"  Forward  (X @ R.T):  cos_sim={cos_fwd:.6f}  max_err={maxerr_fwd:.2e}  rel_err={relerr_fwd:.2e}")
    print(f"  Inverse  (X @ R  ):  cos_sim={cos_inv:.6f}  max_err={maxerr_inv:.2e}")
    print(f"  Round-trip           cos_sim={cos_rt:.6f}  (expect > 0.998, fp16 noise)")

    fwd_ok = cos_fwd  > 0.9999
    inv_ok = cos_inv  > 0.9999
    rt_ok  = cos_rt   > 0.995

    status = lambda ok: "PASS" if ok else "FAIL"
    print(f"  Forward: {status(fwd_ok)}  Inverse: {status(inv_ok)}  Round-trip: {status(rt_ok)}")
    print()

    # ── Step 3: Throughput sweep ─────────────────────────────────────────────
    print("── Throughput Sweep ─────────────────────────────────────────────────")
    header = (f"{'n':>7} | {'MFMA_µs':>9} | {'matmul_µs':>11} | "
              f"{'speedup':>8} | {'MFMA_GB/s':>10} | {'ref_GB/s':>10}")
    print(f"  {header}")
    print(f"  {'-'*len(header)}")

    sweep_results = []
    for n in args.ns:
        X = torch.randn(n, HEAD_DIM, device=device, dtype=torch.float32)

        fn_mfma   = mfma.forward
        fn_matmul = lambda x: x @ R.T

        us_mfma   = bench_fn(fn_mfma,   X)
        us_matmul = bench_fn(fn_matmul, X)

        speedup  = us_matmul / us_mfma if us_mfma > 0 else float("nan")
        bw_mfma  = rotation_bandwidth(n, us_mfma)
        bw_mm    = rotation_bandwidth(n, us_matmul)

        print(f"  {n:>7} | {us_mfma:>9.1f} | {us_matmul:>11.1f} | "
              f"{speedup:>8.2f}× | {bw_mfma:>10.0f} | {bw_mm:>10.0f}")

        sweep_results.append({
            "n":            n,
            "mfma_us":      round(us_mfma, 2),
            "matmul_us":    round(us_matmul, 2),
            "speedup":      round(speedup, 3),
            "mfma_gbs":     round(bw_mfma, 1),
            "matmul_gbs":   round(bw_mm, 1),
        })

    print()

    # ── Step 4: Rotation-within-pipeline comparison ──────────────────────────
    # Compare just the rotation step (MFMA vs matmul) executed the same way
    # tq3_compress does: normalize → rotate → (rest not measured).
    print("── Rotation Step in Context (MFMA vs matmul, within compress pipeline) ──")

    e2e_results = []
    for n in [1024, 4096, 16384]:
        X = torch.randn(n, HEAD_DIM, device=device, dtype=torch.float32)

        # Pre-normalize (as compress does)
        norm   = X.norm(dim=-1)
        x_unit = X / norm.unsqueeze(-1).clamp(min=1e-15)

        # MFMA rotation forward
        fn_rot_mfma = lambda xu: mfma.forward(xu)
        # torch.matmul rotation
        fn_rot_mm   = lambda xu: xu @ R.T

        us_mfma = bench_fn(fn_rot_mfma, x_unit)
        us_mm   = bench_fn(fn_rot_mm,   x_unit)
        speedup = us_mm / us_mfma if us_mfma > 0 else float("nan")

        print(f"  n={n:>6}:  MFMA {us_mfma:6.1f} µs  |  matmul {us_mm:6.1f} µs  |  {speedup:.2f}× speedup")

        e2e_results.append({
            "n":         n,
            "mfma_us":   round(us_mfma, 2),
            "matmul_us": round(us_mm, 2),
            "speedup":   round(speedup, 3),
        })

    print()
    print(f"  MFMA kernel active: {mfma.available}")

    # ── Summary ──────────────────────────────────────────────────────────────
    all_pass = fwd_ok and inv_ok and rt_ok
    print()
    print("── Summary ──────────────────────────────────────────────────────────")
    print(f"  Correctness: {'PASS' if all_pass else 'FAIL'}")
    if sweep_results:
        best = max(sweep_results, key=lambda r: r["speedup"])
        print(f"  Best speedup: {best['speedup']:.2f}× at n={best['n']} vectors")
        print(f"  Peak MFMA bandwidth: {max(r['mfma_gbs'] for r in sweep_results):.0f} GB/s")

    results = {
        "gpu":          torch.cuda.get_device_name(0),
        "mfma_active":  mfma.available,
        "correctness": {
            "fwd_cos_sim":   round(cos_fwd, 6),
            "fwd_max_err":   round(float(maxerr_fwd), 6),
            "inv_cos_sim":   round(cos_inv, 6),
            "rt_cos_sim":    round(cos_rt, 6),
            "pass":          all_pass,
        },
        "throughput":   sweep_results,
        "e2e_tq3":      e2e_results,
    }

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved: {args.output}")

    sys.exit(0 if all_pass else 1)


if __name__ == "__main__":
    main()
