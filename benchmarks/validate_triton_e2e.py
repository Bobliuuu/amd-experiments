"""
validate_triton_e2e.py — End-to-end correctness and throughput validation for the
Triton fused TQ3 attention kernel vs FP16 SDPA.

Two passes:
  1. validate_correctness  — numerically compares Triton fused output with FP16 SDPA
     reference at several context lengths.
  2. measure_throughput    — benchmarks Triton fused attention against the Python
     decompress+SDPA wrapper (mirrors bench_triton_attention.py but uses the
     correctness-validated config).

Output: results/validate_triton_e2e.json

Usage:
    python3 benchmarks/validate_triton_e2e.py
    python3 benchmarks/validate_triton_e2e.py --no-throughput
"""

import argparse
import json
import sys
import time
from pathlib import Path

# ── Import kernels from sibling directory ─────────────────────────────────────
KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))

import torch
import torch.nn.functional as F

from turboquant_mi300x import TurboQuantMI300X
from tq_triton import turboquant_attention_fwd, compress_kv_for_triton


# ──────────────────────────────────────────────────────────────────────────────
# Timing helper
# ──────────────────────────────────────────────────────────────────────────────

def bench(fn, warmup: int = 5, reps: int = 50) -> float:
    """Return median wall-time in milliseconds over `reps` repetitions."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps  # ms


# ──────────────────────────────────────────────────────────────────────────────
# 1. Correctness validation
# ──────────────────────────────────────────────────────────────────────────────

def validate_correctness(
    B: int = 1,
    H: int = 32,
    D: int = 128,
    seq_ks: list[int] | None = None,
) -> dict:
    """
    Compare Triton fused TQ3 attention against FP16 SDPA for decode (S_q=1).

    For each seq_k:
      - Generates random q, k, v in float16.
      - Computes FP16 reference via F.scaled_dot_product_attention.
      - Computes Triton output: compress → turboquant_attention_fwd → apply R.
      - Reports max absolute error, mean cosine similarity, and PASS/FAIL
        (pass threshold: max_abs_err < 0.1).

    Returns a dict keyed by seq_k.
    """
    if seq_ks is None:
        seq_ks = [512, 2048, 8192]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required for validate_correctness")

    tq = TurboQuantMI300X(bits=3, rotation_seed=42)
    sm_scale = D ** -0.5

    print("=" * 70)
    print("Correctness validation: Triton fused TQ3 vs FP16 SDPA")
    print(f"Config: B={B}, H={H}, D={D}, S_q=1")
    print(f"Pass threshold: max_abs_err < 0.10")
    print("-" * 70)
    print(f"{'seq_k':>8}  {'max_abs_err':>12}  {'cosine_sim':>11}  {'result':>6}")
    print("-" * 70)

    results = {}

    for seq_k in seq_ks:
        torch.manual_seed(0)
        q = torch.randn(B, H, 1, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, seq_k, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, seq_k, D, device="cuda", dtype=torch.float16)

        # ── FP16 reference ────────────────────────────────────────────────────
        ref = F.scaled_dot_product_attention(q, k, v, scale=sm_scale)
        # ref: (B, H, 1, D) float16

        # ── Triton fused TQ3 ─────────────────────────────────────────────────
        triton_ok = True
        triton_err_msg = None
        try:
            k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k, v, tq)
            # Pre-rotate queries: q_rot = q @ R^T  (rotate_queries handles this)
            q_rot = tq.rotate_queries(q.float())  # (B, H, 1, D) float32 → float16 after kernel

            # turboquant_attention_fwd applies inverse rotation (@R) internally
            # when rotation= is passed — output is already in original space.
            out_triton = turboquant_attention_fwd(
                q_rot, k_planes, k_norms, v_planes, v_norms,
                rotation=tq.rotation, sm_scale=sm_scale,
            )

        except Exception as exc:
            triton_ok = False
            triton_err_msg = str(exc)

        if not triton_ok:
            print(f"{seq_k:>8}  {'ERROR':>12}  {'ERROR':>11}  FAIL  [{triton_err_msg[:60]}]")
            results[seq_k] = {
                "seq_k": seq_k,
                "passed": False,
                "error": triton_err_msg,
            }
            continue

        # ── Metrics ───────────────────────────────────────────────────────────
        ref_f   = ref.float()            # (B, H, 1, D)
        tri_f   = out_triton.float()     # (B, H, 1, D)

        max_abs_err = (ref_f - tri_f).abs().max().item()

        # Cosine similarity over the head-dim axis, averaged over all (B, H, 1) positions
        ref_flat = ref_f.reshape(-1, D)   # (B*H, D)
        tri_flat = tri_f.reshape(-1, D)
        cos_sim  = F.cosine_similarity(ref_flat, tri_flat, dim=-1).mean().item()

        passed = max_abs_err < 0.1
        tag    = "PASS" if passed else "FAIL"

        print(f"{seq_k:>8}  {max_abs_err:>12.4f}  {cos_sim:>11.4f}  {tag:>6}")

        results[seq_k] = {
            "seq_k":        seq_k,
            "max_abs_err":  round(max_abs_err, 6),
            "cosine_sim":   round(cos_sim,    6),
            "passed":       passed,
        }

    print("-" * 70)

    any_failed = any(not r.get("passed", False) for r in results.values())
    if any_failed:
        print("WARNING: one or more seq_k configurations FAILED correctness check.")
    else:
        print("All configurations PASSED.")
    print()

    return results


# ──────────────────────────────────────────────────────────────────────────────
# 2. Throughput measurement
# ──────────────────────────────────────────────────────────────────────────────

def measure_throughput(
    seq_ks: list[int] | None = None,
    warmup: int = 5,
    reps:   int = 50,
) -> list[dict]:
    """
    Benchmark Triton fused TQ3 attention vs Python decompress+SDPA wrapper
    at several context lengths (decode step: S_q=1).

    Reports per seq_k:
      ms_pywrap    — Python decompress + SDPA latency
      ms_triton    — Triton fused kernel latency
      speedup_vs_pywrap
      eff_bw_gbs   — effective HBM bandwidth of Triton kernel reading TQ3 data

    NOTE: comparison is against the Python wrapper (not FP16 SDPA) because
    the Triton kernel reads TQ3-compressed data, making it a different operation
    from uncompressed FP16 attention.
    """
    if seq_ks is None:
        seq_ks = [1024, 4096, 16384, 32768, 65536]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required for measure_throughput")

    B, H, D, S_q = 1, 32, 128, 1
    sm_scale = D ** -0.5

    # TQ3: 52 bytes/token (4-byte norm + 3×16-byte bit-planes)
    TQ3_BYTES_PER_TOKEN = 52

    tq = TurboQuantMI300X(bits=3, rotation_seed=42)

    print("=" * 80)
    print("Throughput: Triton fused TQ3 vs Python decompress+SDPA wrapper")
    print(f"Config: B={B}, H={H}, D={D}, S_q={S_q}  (decode step)")
    print(f"Timing: warmup={warmup}, reps={reps}")
    print("-" * 80)
    print(
        f"{'seq_k':>8}  {'pywrap_ms':>10}  {'triton_ms':>10}  "
        f"{'speedup':>9}  {'eff_bw_gbs':>12}"
    )
    print("-" * 80)

    results = []

    for seq_k in seq_ks:
        torch.manual_seed(42)
        q    = torch.randn(B, H, S_q, D, device="cuda", dtype=torch.float16)
        k_fp = torch.randn(B, H, seq_k, D, device="cuda", dtype=torch.float16)
        v_fp = torch.randn(B, H, seq_k, D, device="cuda", dtype=torch.float16)

        # ── Compress for both paths ───────────────────────────────────────────
        k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k_fp, v_fp, tq)
        q_rot = tq.rotate_queries(q.float())

        # Python wrapper: compress tensors for decompress+SDPA path
        k_comp = tq.compress_tensor(k_fp.float().reshape(-1, D))
        v_comp = tq.compress_tensor(v_fp.float().reshape(-1, D))

        # ── Python decompress + SDPA wrapper ─────────────────────────────────
        def py_wrapper():
            k_d = tq.decompress_tensor(k_comp, (B, H, seq_k, D)).half()
            v_d = tq.decompress_tensor(v_comp, (B, H, seq_k, D)).half()
            return F.scaled_dot_product_attention(q, k_d, v_d, scale=sm_scale)

        ms_pywrap = bench(py_wrapper, warmup=warmup, reps=reps)

        # ── Triton fused TQ3 ─────────────────────────────────────────────────
        def triton_fused():
            return turboquant_attention_fwd(
                q_rot, k_planes, k_norms, v_planes, v_norms,
                rotation=tq.rotation, sm_scale=sm_scale,
            )

        triton_ok = True
        try:
            ms_triton = bench(triton_fused, warmup=warmup, reps=reps)
        except Exception as exc:
            print(f"  Triton error at seq_k={seq_k}: {exc}", flush=True)
            ms_triton = float("nan")
            triton_ok = False

        if triton_ok:
            speedup   = ms_pywrap / ms_triton
            # Effective BW: each token costs TQ3_BYTES_PER_TOKEN bytes for K and V → ×2
            kv_tq3_bytes = 2 * H * seq_k * TQ3_BYTES_PER_TOKEN
            eff_bw_gbs   = (kv_tq3_bytes / 1e9) / (ms_triton / 1e3)
        else:
            speedup    = float("nan")
            eff_bw_gbs = 0.0

        spd_str = f"{speedup:.2f}x" if triton_ok else "  N/A"
        bw_str  = f"{eff_bw_gbs:.1f}" if triton_ok else "  N/A"

        print(
            f"{seq_k:>8}  {ms_pywrap:>10.3f}  "
            f"{ms_triton if triton_ok else float('nan'):>10.3f}  "
            f"{spd_str:>9}  {bw_str:>10} GB/s"
        )

        results.append({
            "seq_k":              seq_k,
            "pywrap_ms":          round(ms_pywrap,   4),
            "triton_ms":          round(ms_triton,   4) if triton_ok else None,
            "speedup_vs_pywrap":  round(speedup,     3) if triton_ok else None,
            "eff_bw_gbs":         round(eff_bw_gbs,  1) if triton_ok else None,
        })

        del k_fp, v_fp, k_planes, k_norms, v_planes, v_norms, k_comp, v_comp
        torch.cuda.empty_cache()

    print("-" * 80)
    print()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 3. Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate Triton TQ3 attention correctness and throughput."
    )
    parser.add_argument(
        "--no-throughput",
        action="store_true",
        help="Skip throughput measurement; only run correctness validation.",
    )
    parser.add_argument(
        "--seq-ks-val",
        nargs="+",
        type=int,
        default=[512, 2048, 8192],
        metavar="N",
        help="Context lengths for correctness validation (default: 512 2048 8192).",
    )
    parser.add_argument(
        "--seq-ks-bw",
        nargs="+",
        type=int,
        default=[1024, 4096, 16384, 32768, 65536],
        metavar="N",
        help="Context lengths for throughput benchmark.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/ROCm device available.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}\n")

    # ── Correctness ───────────────────────────────────────────────────────────
    val_results = validate_correctness(seq_ks=args.seq_ks_val)
    any_failed  = any(not r.get("passed", False) for r in val_results.values())
    if any_failed:
        print("WARNING: correctness check failed for one or more seq_k values.")
        print("         Continuing to throughput measurement anyway.\n")

    # ── Throughput ────────────────────────────────────────────────────────────
    tp_results: list[dict] = []
    if not args.no_throughput:
        tp_results = measure_throughput(seq_ks=args.seq_ks_bw)
    else:
        print("Skipping throughput measurement (--no-throughput).")

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "validate_triton_e2e.json"

    output = {
        "device":           device_name,
        "B": 1, "H": 32, "D": 128, "S_q": 1,
        "tq_bits":          3,
        "correctness":      {str(k): v for k, v in val_results.items()},
        "any_correctness_failed": any_failed,
        "throughput":       tp_results,
    }

    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
