"""
bench_triton_attention.py — Compare FP16, Python TQ3, and Triton fused TQ3 attention.

Output: results/bench_triton_attention.json
"""

import argparse
import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))

import torch
import torch.nn.functional as F

from turboquant_mi300x import TurboQuantMI300X
from tq_triton import turboquant_attention_fwd, compress_kv_for_triton
from cache_utils import add_swa_args, clamp_seq_to_window, print_swa_status


def bench(fn, warmup=5, reps=50):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000 / reps  # ms


def run(swa: str = "off", window: int = 0):
    if not torch.cuda.is_available():
        print(json.dumps({"error": "no GPU"}))
        return

    device = torch.cuda.get_device_name(0)
    print(f"Device: {device}", flush=True)

    tq = TurboQuantMI300X(bits=3, device="cuda")

    B   = 1
    H   = 32   # 32 KV heads (Mistral-7B-style: 8 heads × 4 query groups = 32 effective)
    D   = 128
    S_q = 1    # decode step
    sm  = D ** -0.5
    HEAD_BYTES_FP16 = H * D * 2  # bytes per KV token for FP16 (both K and V)
    HEAD_BYTES_TQ3  = H * 52 * 2  # bytes per KV token for TQ3 (K + V)

    results = []
    seq_ks = [1024, 4096, 16384, 32768, 65536, 131072]

    print(f"{'seq_k':>8}  {'cache_k':>8}  {'FP16ms':>8}  {'PyTQ3ms':>9}  {'Tritonms':>9}  "
          f"{'vs_fp16':>8}  {'vs_pytq3':>9}  {'triton_gbs':>11}", flush=True)

    for seq_k in seq_ks:
        cache_seq_k = clamp_seq_to_window(seq_k, swa, window)
        torch.manual_seed(42)
        q    = torch.randn(B, H, S_q, D, device="cuda", dtype=torch.float16)
        k_fp = torch.randn(B, H, cache_seq_k, D, device="cuda", dtype=torch.float16)
        v_fp = torch.randn(B, H, cache_seq_k, D, device="cuda", dtype=torch.float16)

        # Compress KV for Triton
        k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k_fp, v_fp, tq)
        q_rot = tq.rotate_queries(q.float())

        # Compress for Python wrapper
        k_comp = tq.compress_tensor(k_fp.float().reshape(-1, D))
        v_comp = tq.compress_tensor(v_fp.float().reshape(-1, D))
        # Use cache_seq_k for shape ops below (decompression target shape)

        # ── FP16 ─────────────────────────────────────────────────────────────
        ms_fp16 = bench(lambda: F.scaled_dot_product_attention(q, k_fp, v_fp, scale=sm))

        # ── Python TQ3 wrapper (decompress + SDPA) ───────────────────────────
        def py_tq3():
            k_d = tq.decompress_tensor(k_comp, (B, H, cache_seq_k, D)).half()
            v_d = tq.decompress_tensor(v_comp, (B, H, cache_seq_k, D)).half()
            return F.scaled_dot_product_attention(q, k_d, v_d, scale=sm)
        ms_pywrap = bench(py_tq3)

        # ── Triton fused TQ3 ─────────────────────────────────────────────────
        def triton_tq3():
            return turboquant_attention_fwd(
                q_rot, k_planes, k_norms, v_planes, v_norms,
                rotation=tq.rotation, sm_scale=sm,
            )
        try:
            ms_triton = bench(triton_tq3)
            triton_ok = True
        except Exception as e:
            print(f"  Triton error at seq_k={seq_k}: {e}", flush=True)
            ms_triton = float("nan")
            triton_ok = False

        spd_fp16   = ms_fp16   / ms_triton if triton_ok else float("nan")
        spd_pywrap = ms_pywrap / ms_triton if triton_ok else float("nan")
        # Effective bandwidth for Triton: reads TQ3 data from HBM (uses actual cache size)
        bw_triton  = (HEAD_BYTES_TQ3 * cache_seq_k / 1e9) / (ms_triton / 1e3) if triton_ok else 0

        print(f"{seq_k:>8}  {cache_seq_k:>8}  {ms_fp16:>8.3f}  {ms_pywrap:>9.3f}  {ms_triton:>9.3f}  "
              f"{spd_fp16:>7.2f}×  {spd_pywrap:>8.2f}×  {bw_triton:>10.1f} GB/s", flush=True)

        results.append({
            "seq_k":             seq_k,
            "cache_seq_k":       cache_seq_k,
            "fp16_ms":           round(ms_fp16, 4),
            "pywrap_ms":         round(ms_pywrap, 4),
            "triton_ms":         round(ms_triton, 4) if triton_ok else None,
            "speedup_vs_fp16":   round(spd_fp16,  3) if triton_ok else None,
            "speedup_vs_pywrap": round(spd_pywrap, 3) if triton_ok else None,
            "triton_eff_bw_gbs": round(bw_triton, 1) if triton_ok else None,
            "swa_window":        window if swa == "on" else None,
        })

    out = {
        "device":    device,
        "B": B, "H": H, "D": D, "S_q": S_q,
        "tq_bits":   3,
        "results":   results,
    }

    out_path = Path(__file__).parent.parent / "results" / "bench_triton_attention.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nSaved → {out_path}", flush=True)
    return out


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FP16 vs Python TQ3 vs Triton TQ3 attention")
    add_swa_args(parser)
    args = parser.parse_args()
    print_swa_status(args.swa, args.window if args.swa == "on" else None)
    run(swa=args.swa, window=args.window)
