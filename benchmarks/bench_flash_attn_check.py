"""
bench_flash_attn_check.py — Investigate SDPA dispatch and effective bandwidth on ROCm/MI300X.

Determines whether F.scaled_dot_product_attention is using CK-based Flash Attention
on ROCm, and measures effective bandwidth to evaluate whether the expected speedup is
achieved.

Observation heuristic:
  If achieved SDPA bandwidth < 20 % of MI300X peak (5300 GB/s), ROCm CK flash
  attention is likely NOT being used; the fallback math/naive path is running instead.

Functions:
  check_sdpa_dispatch()           — probe available backends and env vars
  benchmark_sdpa_vs_bmm()         — measure SDPA and manual BMM bandwidth
  benchmark_causal_vs_noncausal() — compare causal vs non-causal SDPA

Output: results/bench_flash_attn_check.json

Usage:
    python3 benchmarks/bench_flash_attn_check.py
    python3 benchmarks/bench_flash_attn_check.py --skip-dispatch-check
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

# ── sys.path: not needed for this script (no kernels/ imports), but kept
# consistent with other benchmarks in this repo.
KERNELS_DIR = Path(__file__).resolve().parent.parent / "kernels"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))

import torch
import torch.nn.functional as F

# MI300X peak HBM3 bandwidth (GB/s)
MI300X_PEAK_GBS = 5300.0


# ──────────────────────────────────────────────────────────────────────────────
# Timing helper
# ──────────────────────────────────────────────────────────────────────────────

def bench(fn, warmup: int = 5, reps: int = 50) -> float:
    """Return average wall-time in milliseconds over `reps` repetitions."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps  # ms


# ──────────────────────────────────────────────────────────────────────────────
# 1. Dispatch check
# ──────────────────────────────────────────────────────────────────────────────

def check_sdpa_dispatch() -> dict:
    """
    Probe the PyTorch / ROCm environment to determine which SDPA backends are
    active, whether the standalone flash_attn package is installed, and which
    ROCm-specific environment variables are set.

    Prints a human-readable summary and returns a dict of findings.
    """
    print("=" * 70)
    print("SDPA dispatch check")
    print("=" * 70)

    findings: dict = {}

    # ── PyTorch / ROCm version ─────────────────────────────────────────────
    pt_version = torch.__version__
    is_rocm    = hasattr(torch.version, "hip") and torch.version.hip is not None
    rocm_ver   = getattr(torch.version, "hip", None)
    print(f"PyTorch version : {pt_version}")
    print(f"ROCm build      : {'yes — ' + str(rocm_ver) if is_rocm else 'no (CUDA build)'}")
    findings["pytorch_version"] = pt_version
    findings["rocm_build"]      = is_rocm
    findings["rocm_version"]    = rocm_ver

    # ── flash_attn package ────────────────────────────────────────────────
    try:
        import flash_attn
        fa_ver = getattr(flash_attn, "__version__", "unknown")
        print(f"flash_attn pkg  : installed, version={fa_ver}")
        findings["flash_attn_installed"] = True
        findings["flash_attn_version"]   = fa_ver
    except ImportError:
        print("flash_attn pkg  : NOT installed")
        findings["flash_attn_installed"] = False
        findings["flash_attn_version"]   = None

    # ── Flash SDP enabled (PyTorch ≥ 2.0 CUDA API; may not exist on ROCm) ─
    try:
        flash_enabled = torch.backends.cuda.flash_sdp_enabled()
        print(f"flash_sdp_enabled(): {flash_enabled}")
        findings["flash_sdp_enabled"] = flash_enabled
    except AttributeError:
        print("flash_sdp_enabled(): AttributeError — not available on this build")
        findings["flash_sdp_enabled"] = None

    # ── Math SDP enabled ──────────────────────────────────────────────────
    try:
        math_enabled = torch.backends.cuda.math_sdp_enabled()
        print(f"math_sdp_enabled()  : {math_enabled}")
        findings["math_sdp_enabled"] = math_enabled
    except AttributeError:
        print("math_sdp_enabled()  : AttributeError — not available")
        findings["math_sdp_enabled"] = None

    # ── Memory-efficient SDP enabled ──────────────────────────────────────
    try:
        mem_eff_enabled = torch.backends.cuda.mem_efficient_sdp_enabled()
        print(f"mem_efficient_sdp_enabled(): {mem_eff_enabled}")
        findings["mem_efficient_sdp_enabled"] = mem_eff_enabled
    except AttributeError:
        print("mem_efficient_sdp_enabled(): AttributeError — not available")
        findings["mem_efficient_sdp_enabled"] = None

    # ── cudnn SDP enabled ─────────────────────────────────────────────────
    try:
        cudnn_enabled = torch.backends.cuda.cudnn_sdp_enabled()
        print(f"cudnn_sdp_enabled()         : {cudnn_enabled}")
        findings["cudnn_sdp_enabled"] = cudnn_enabled
    except AttributeError:
        print("cudnn_sdp_enabled()         : AttributeError — not available")
        findings["cudnn_sdp_enabled"] = None

    # ── ROCm-specific environment variables ───────────────────────────────
    rocm_env_vars = [
        "ROCM_USE_FLASH_ATTN_V2",
        "PYTORCH_ROCM_FLASH_ATTN_VERSION",
        "ROCBLAS_LAYER",
        "HIP_VISIBLE_DEVICES",
        "ROCR_VISIBLE_DEVICES",
        "GPU_MAX_HW_QUEUES",
        "PYTORCH_NO_HIPBLASLT",
    ]
    env_findings: dict[str, str | None] = {}
    print()
    print("ROCm-relevant environment variables:")
    for var in rocm_env_vars:
        val = os.environ.get(var)
        env_findings[var] = val
        print(f"  {var:<38} = {val!r}")
    findings["env_vars"] = env_findings

    # ── Summarise ─────────────────────────────────────────────────────────
    print()
    if is_rocm:
        print("Note: On ROCm, SDPA may dispatch to CK (Composable Kernel) Flash "
              "Attention if built in.\n"
              "      Use rocprofv2 --sys-trace or infer from bandwidth (see below).\n"
              "      Peak MI300X HBM3: 5300 GB/s; CK FA achieves ~1-2 TB/s at batch=1 decode.")
    else:
        print("Note: Running on CUDA (not ROCm); CK paths are irrelevant.")
    print()

    return findings


# ──────────────────────────────────────────────────────────────────────────────
# 2. SDPA vs manual BMM bandwidth
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_sdpa_vs_bmm(
    seq_lens: list[int] | None = None,
    warmup:   int = 5,
    reps:     int = 50,
) -> list[dict]:
    """
    Benchmark F.scaled_dot_product_attention vs manual BMM-based attention
    at various context lengths for a single decode step (B=1, H=32, S_q=1).

    Computes effective bandwidth for each approach and the fraction of MI300X
    peak (5300 GB/s).

    FP16 SDPA bandwidth utilization:
      If <20% peak → likely NOT using CK Flash Attention (math/naive fallback).

    Bandwidth denominator: reading K and V from HBM.
      bytes = 2 (K + V) × H × seq_len × D × 2 (bytes per fp16 element)
    """
    if seq_lens is None:
        seq_lens = [512, 1024, 4096, 8192, 32768, 65536, 131072]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required for benchmark_sdpa_vs_bmm")

    B, H, D, S_q = 1, 32, 128, 1
    sm_scale = D ** -0.5

    # bytes_per_seq_len: K + V, both FP16 (2 bytes), shape (H, seq_len, D)
    def kv_bytes(seq_len: int) -> int:
        return 2 * H * seq_len * D * 2  # factor-of-2 for K and V

    print("FP16 SDPA bandwidth utilization — "
          "if <20% peak, not using CK flash attention")
    print("=" * 90)
    print(
        f"{'seq_len':>8}  {'sdpa_ms':>9}  {'sdpa_gbs':>10}  {'pct_peak':>10}  "
        f"{'bmm_ms':>8}  {'bmm_gbs':>9}  {'sdpa_speedup':>13}"
    )
    print("-" * 90)

    results = []

    for seq_len in seq_lens:
        torch.manual_seed(0)
        # decode-step tensors: S_q=1
        q = torch.randn(B, H, S_q, D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float16)

        bw_denom_bytes = kv_bytes(seq_len)

        # ── F.scaled_dot_product_attention ───────────────────────────────────
        sdpa_ok = True
        try:
            ms_sdpa = bench(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=sm_scale),
                warmup=warmup, reps=reps,
            )
            sdpa_gbs  = (bw_denom_bytes / 1e9) / (ms_sdpa / 1e3)
            pct_peak  = sdpa_gbs / MI300X_PEAK_GBS * 100.0
        except Exception as exc:
            print(f"  sdpa error at seq_len={seq_len}: {exc}")
            sdpa_ok   = False
            ms_sdpa   = float("nan")
            sdpa_gbs  = 0.0
            pct_peak  = 0.0

        # ── Manual BMM attention: softmax(Q·K^T * scale) · V ─────────────────
        #    Use torch.bmm which operates on 3D tensors: (batch, M, K)
        #    Reshape: (B*H, S_q, D) for Q, (B*H, D, seq_len) for K^T, etc.
        bmm_ok = True
        try:
            # Precompute reshaped views once; the lambda closes over them.
            q_3d  = q.reshape(B * H, S_q, D)         # (32, 1, 128)
            kt_3d = k.reshape(B * H, seq_len, D).transpose(-2, -1)  # (32, 128, seq_len)
            v_3d  = v.reshape(B * H, seq_len, D)      # (32, seq_len, 128)

            def manual_bmm():
                scores  = torch.bmm(q_3d, kt_3d) * sm_scale  # (32, 1, seq_len)
                weights = torch.softmax(scores.float(), dim=-1).half()
                return torch.bmm(weights, v_3d)               # (32, 1, 128)

            ms_bmm  = bench(manual_bmm, warmup=warmup, reps=reps)
            bmm_gbs = (bw_denom_bytes / 1e9) / (ms_bmm / 1e3)
        except Exception as exc:
            print(f"  bmm error at seq_len={seq_len}: {exc}")
            bmm_ok  = False
            ms_bmm  = float("nan")
            bmm_gbs = 0.0

        sdpa_speedup = (ms_bmm / ms_sdpa) if (sdpa_ok and bmm_ok) else float("nan")

        pct_str  = f"{pct_peak:8.2f}%" if sdpa_ok else "     N/A"
        spd_str  = f"{sdpa_speedup:.3f}x" if (sdpa_ok and bmm_ok) else "       N/A"
        sgbs_str = f"{sdpa_gbs:.1f}"       if sdpa_ok else "    N/A"
        bgbs_str = f"{bmm_gbs:.1f}"        if bmm_ok  else "   N/A"

        print(
            f"{seq_len:>8}  {ms_sdpa if sdpa_ok else float('nan'):>9.3f}  "
            f"{sgbs_str:>10}  {pct_str:>10}  "
            f"{ms_bmm if bmm_ok else float('nan'):>8.3f}  {bgbs_str:>9}  {spd_str:>13}"
        )

        results.append({
            "seq_len":       seq_len,
            "sdpa_ms":       round(ms_sdpa,      4) if sdpa_ok else None,
            "sdpa_gbs":      round(sdpa_gbs,     1) if sdpa_ok else None,
            "sdpa_pct_peak": round(pct_peak,     2) if sdpa_ok else None,
            "bmm_ms":        round(ms_bmm,       4) if bmm_ok  else None,
            "bmm_gbs":       round(bmm_gbs,      1) if bmm_ok  else None,
            "sdpa_speedup":  round(sdpa_speedup, 3) if (sdpa_ok and bmm_ok) else None,
        })

        del q, k, v
        torch.cuda.empty_cache()

    print("-" * 90)
    print()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 3. Causal vs non-causal SDPA
# ──────────────────────────────────────────────────────────────────────────────

def benchmark_causal_vs_noncausal(
    seq_lens: list[int] | None = None,
    warmup:   int = 5,
    reps:     int = 50,
) -> list[dict]:
    """
    Compare causal (is_causal=True) vs non-causal (is_causal=False) SDPA at
    the decode step (S_q=1).

    At batch=1 decode, the causal mask only masks tokens with position > current;
    with S_q=1 pointing at the last token, all K/V positions are visible regardless
    of causal masking.  Therefore causal and non-causal should produce identical
    timing (and output) — any meaningful difference reveals an unexpected code path.
    """
    if seq_lens is None:
        seq_lens = [4096, 32768]

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA/ROCm device required for benchmark_causal_vs_noncausal")

    B, H, D, S_q = 1, 32, 128, 1
    sm_scale = D ** -0.5

    print("Causal vs non-causal SDPA at decode step (S_q=1)")
    print("=" * 65)
    print("Note: at S_q=1, causal should equal non-causal (no tokens masked).")
    print("-" * 65)
    print(f"{'seq_len':>8}  {'causal_ms':>10}  {'noncausal_ms':>13}  {'ratio':>7}  {'diff?':>6}")
    print("-" * 65)

    results = []

    for seq_len in seq_lens:
        torch.manual_seed(0)
        q = torch.randn(B, H, S_q,     D, device="cuda", dtype=torch.float16)
        k = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float16)
        v = torch.randn(B, H, seq_len, D, device="cuda", dtype=torch.float16)

        # Non-causal
        ok_nc = True
        try:
            ms_nc = bench(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=False, scale=sm_scale),
                warmup=warmup, reps=reps,
            )
        except Exception as exc:
            print(f"  non-causal error at seq_len={seq_len}: {exc}")
            ok_nc = False
            ms_nc = float("nan")

        # Causal — at S_q=1, requires seq_k == seq_q for the causal mask
        # to be valid; PyTorch will error if seq_k > seq_q and is_causal=True
        # unless using the "lower-right" (alibi-style) convention.  We catch
        # any such error gracefully.
        ok_c = True
        try:
            ms_c = bench(
                lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True, scale=sm_scale),
                warmup=warmup, reps=reps,
            )
        except Exception as exc:
            print(f"  causal error at seq_len={seq_len}: {exc}")
            ok_c   = False
            ms_c   = float("nan")

        if ok_nc and ok_c:
            ratio    = ms_c / ms_nc
            # Treat as "different" if >10% difference in latency
            is_diff  = abs(ratio - 1.0) > 0.10
            diff_str = "YES" if is_diff else "no"
        else:
            ratio    = float("nan")
            is_diff  = None
            diff_str = "N/A"

        print(
            f"{seq_len:>8}  {ms_c if ok_c else float('nan'):>10.3f}  "
            f"{ms_nc if ok_nc else float('nan'):>13.3f}  "
            f"{ratio if (ok_nc and ok_c) else float('nan'):>7.3f}  {diff_str:>6}"
        )

        results.append({
            "seq_len":         seq_len,
            "causal_ms":       round(ms_c,   4) if ok_c  else None,
            "noncausal_ms":    round(ms_nc,  4) if ok_nc else None,
            "ratio_c_over_nc": round(ratio,  4) if (ok_nc and ok_c) else None,
            "timing_differs":  is_diff,
        })

        del q, k, v
        torch.cuda.empty_cache()

    print("-" * 65)
    print()
    return results


# ──────────────────────────────────────────────────────────────────────────────
# 4. Main
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check SDPA dispatch and bandwidth on ROCm/MI300X."
    )
    parser.add_argument(
        "--skip-dispatch-check",
        action="store_true",
        help="Skip the SDPA dispatch / environment variable probe.",
    )
    parser.add_argument(
        "--seq-lens",
        nargs="+",
        type=int,
        default=[512, 1024, 4096, 8192, 32768, 65536, 131072],
        metavar="N",
        help="Context lengths for SDPA vs BMM benchmark.",
    )
    parser.add_argument(
        "--causal-seq-lens",
        nargs="+",
        type=int,
        default=[4096, 32768],
        metavar="N",
        help="Context lengths for causal vs non-causal comparison.",
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no CUDA/ROCm device available.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    print(f"Device: {device_name}\n")

    # ── Dispatch check ────────────────────────────────────────────────────────
    dispatch_findings: dict = {}
    if not args.skip_dispatch_check:
        dispatch_findings = check_sdpa_dispatch()
    else:
        print("Skipping dispatch check (--skip-dispatch-check).\n")

    # ── SDPA vs BMM bandwidth ─────────────────────────────────────────────────
    bw_results = benchmark_sdpa_vs_bmm(seq_lens=args.seq_lens)

    # ── Causal vs non-causal ──────────────────────────────────────────────────
    causal_results = benchmark_causal_vs_noncausal(seq_lens=args.causal_seq_lens)

    # ── Save results ──────────────────────────────────────────────────────────
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "bench_flash_attn_check.json"

    output = {
        "device":              device_name,
        "mi300x_peak_gbs":     MI300X_PEAK_GBS,
        "B": 1, "H": 32, "D": 128, "S_q": 1,
        "dispatch_check":      dispatch_findings,
        "sdpa_vs_bmm":         bw_results,
        "causal_vs_noncausal": causal_results,
    }

    out_path.write_text(json.dumps(output, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
