"""
bench_all_methods_decode.py — End-to-End Decode Benchmark for All Methods

Measures decode throughput (tok/s) and latency (ms/token) for:
  FP16, FP8, INT4, TurboQuant (TQ3/TQ4),
  IsoQuant (iso3/iso4), PlanarQuant (planar3/planar4), RotorQuant (rotor3/rotor4)

All methods run via the Python/Triton path for fair comparison.
TurboQuant HIP results (from bench_kernels.py) are noted separately.

Key design decisions:
  - ALL methods use Python/Triton path (no mixing HIP-TQ vs Triton-Planar)
  - Decode-only (prefill done separately, not counted)
  - batch=1 by default (bandwidth-bound behavior at batch>1 in bench_batch_decode_v2.py)

Usage:
    python3 benchmarks/bench_all_methods_decode.py --model mistralai/Mistral-7B-v0.1
    python3 benchmarks/bench_all_methods_decode.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --seq-lens 512 2048 8192 32768 \\
        --methods fp16 planar3 iso3 rotor3 turbo3
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import torch
import torch.nn.functional as F

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR.mkdir(exist_ok=True)


def sync():
    torch.cuda.synchronize()


def build_fake_kv_cache(
    n_layers: int, n_kv_heads: int, seq_len: int, head_dim: int,
    method: str, bits: int, device: str
) -> Tuple[object, object]:
    """
    Build a compressed KV cache for one sequence.
    Returns (cache_object, compress_fn_or_None) depending on method.

    For fair decode-only benchmarking: prefill is simulated by compressing
    a random KV tensor. We then measure only the decode step cost.
    """
    K_raw = torch.randn(n_layers, n_kv_heads, seq_len, head_dim, device=device)
    V_raw = torch.randn(n_layers, n_kv_heads, seq_len, head_dim, device=device)

    if method == "fp16":
        return (K_raw, V_raw), None

    if method == "turbo":
        from turboquant_mi300x import TurboQuantMI300X
        tq = TurboQuantMI300X(bits=bits, rotation_seed=42)
        K_comp = [tq.compress_tensor(K_raw[l].reshape(-1, head_dim).float())
                  for l in range(n_layers)]
        V_comp = [tq.compress_tensor(V_raw[l].reshape(-1, head_dim).float())
                  for l in range(n_layers)]
        return (K_comp, V_comp, K_raw.shape), tq

    from block_quant_rocm import make_quantizer
    q = make_quantizer(method, bits=bits, head_dim=head_dim, device=device)
    K_comp = [q.compress(K_raw[l].reshape(-1, head_dim))
              for l in range(n_layers)]
    V_comp = [q.compress(V_raw[l].reshape(-1, head_dim))
              for l in range(n_layers)]
    return (K_comp, V_comp, K_raw.shape), q


def decode_step_cost(
    cache, compress_obj, method: str, bits: int,
    n_layers: int, n_kv_heads: int, seq_len: int, head_dim: int,
    device: str, sm_scale: float
) -> float:
    """
    Simulate one decode step: decompress K/V, compute attention for one query token.
    Returns wall time in seconds.

    This isolates the KV compression overhead from the model's FFN/weight-load cost,
    giving the pure attention+decompress time per decode step.
    """
    q_token = torch.randn(n_kv_heads, 1, head_dim, device=device)  # (H, 1, D)

    sync()
    t0 = time.perf_counter()

    if method == "fp16":
        K, V = cache
        for l in range(n_layers):
            k = K[l]  # (H, S, D)
            v = V[l]  # (H, S, D)
            # Attention: (H, 1, D) × (H, D, S) → (H, 1, S)
            scores = torch.bmm(q_token, k.transpose(-2, -1)) * sm_scale
            weights = torch.softmax(scores, dim=-1)
            _ = torch.bmm(weights, v)
    elif method == "turbo":
        K_comp, V_comp, kv_shape = cache
        tq = compress_obj
        q_rot = tq.rotate_queries(q_token.reshape(-1, head_dim))
        for l in range(n_layers):
            k_fp = tq.decompress_tensor(K_comp[l], (n_kv_heads * seq_len, head_dim))
            v_fp = tq.decompress_tensor(V_comp[l], (n_kv_heads * seq_len, head_dim))
            k_fp = k_fp.reshape(n_kv_heads, seq_len, head_dim)
            v_fp = v_fp.reshape(n_kv_heads, seq_len, head_dim)
            scores = torch.bmm(q_token, k_fp.transpose(-2, -1)) * sm_scale
            weights = torch.softmax(scores, dim=-1)
            _ = torch.bmm(weights, v_fp)
    else:
        K_comp, V_comp, kv_shape = cache
        q = compress_obj
        for l in range(n_layers):
            k_fp = q.decompress(K_comp[l], (n_kv_heads * seq_len, head_dim))
            v_fp = q.decompress(V_comp[l], (n_kv_heads * seq_len, head_dim))
            k_fp = k_fp.reshape(n_kv_heads, seq_len, head_dim)
            v_fp = v_fp.reshape(n_kv_heads, seq_len, head_dim)
            scores = torch.bmm(q_token, k_fp.transpose(-2, -1)) * sm_scale
            weights = torch.softmax(scores, dim=-1)
            _ = torch.bmm(weights, v_fp)

    sync()
    return time.perf_counter() - t0


def bench_one_config(
    method_spec: str, seq_len: int, args, device: str
) -> Dict:
    """Benchmark one (method, bitwidth, seq_len) configuration."""
    if method_spec == "fp16":
        method, bits = "fp16", 0
    elif method_spec in ("fp8", "int4"):
        method, bits = method_spec, 0
    else:
        method = method_spec[:-1]
        bits = int(method_spec[-1])

    sm_scale = 1.0 / (args.head_dim ** 0.5)

    print(f"  Building cache for {method_spec} seq={seq_len}...", end="", flush=True)
    try:
        if method in ("fp8", "int4"):
            # These use simple per-element cast, no rotation
            K_raw = torch.randn(args.n_layers, args.n_kv_heads, seq_len, args.head_dim, device=device)
            V_raw = torch.randn(args.n_layers, args.n_kv_heads, seq_len, args.head_dim, device=device)
            if method == "fp8":
                K_comp = K_raw.to(torch.float8_e4m3fnuz)
                V_comp = V_raw.to(torch.float8_e4m3fnuz)
            else:  # int4: fake via clamp+round
                K_comp = (K_raw * 7).round().clamp(-8, 7).to(torch.int8)
                V_comp = (V_raw * 7).round().clamp(-8, 7).to(torch.int8)
            cache = (K_comp, V_comp, K_raw.shape)
            compress_obj = None
        else:
            cache, compress_obj = build_fake_kv_cache(
                args.n_layers, args.n_kv_heads, seq_len,
                args.head_dim, method, bits, device
            )
    except Exception as e:
        print(f" FAILED: {e}")
        return {"method": method, "bits": bits, "seq_len": seq_len,
                "error": str(e), "tok_per_sec": None}
    print(" done")

    # Warmup
    for _ in range(args.n_warmup):
        if method in ("fp8", "int4"):
            K_comp_f, V_comp_f = cache[0].float(), cache[1].float()
            q_t = torch.randn(args.n_kv_heads, 1, args.head_dim, device=device)
            scores = torch.bmm(q_t, K_comp_f[0].transpose(-2, -1)) * sm_scale
            _ = torch.bmm(torch.softmax(scores, -1), V_comp_f[0])
        else:
            decode_step_cost(cache, compress_obj, method, bits,
                             args.n_layers, args.n_kv_heads, seq_len,
                             args.head_dim, device, sm_scale)
    sync()

    # Timed runs
    step_times = []
    for _ in range(args.n_decode):
        if method in ("fp8", "int4"):
            sync()
            t0 = time.perf_counter()
            K_f = cache[0].float()
            V_f = cache[1].float()
            q_t = torch.randn(args.n_kv_heads, 1, args.head_dim, device=device)
            for l in range(args.n_layers):
                scores = torch.bmm(q_t, K_f[l].transpose(-2, -1)) * sm_scale
                weights = torch.softmax(scores, -1)
                _ = torch.bmm(weights, V_f[l])
            sync()
            step_times.append(time.perf_counter() - t0)
        else:
            t = decode_step_cost(cache, compress_obj, method, bits,
                                 args.n_layers, args.n_kv_heads, seq_len,
                                 args.head_dim, device, sm_scale)
            step_times.append(t)

    median_s = float(np.median(step_times))
    p25_s = float(np.percentile(step_times, 25))
    p75_s = float(np.percentile(step_times, 75))
    tok_per_sec = 1.0 / median_s  # batch=1 → 1 token per step

    # Compression ratio
    if method == "fp16":
        comp_ratio = 1.0
    elif method == "fp8":
        comp_ratio = 2.0
    elif method == "int4":
        comp_ratio = 4.0
    elif method == "turbo":
        from turboquant_mi300x import COMPRESSION_RATIO
        comp_ratio = COMPRESSION_RATIO.get(bits, 4.92)
    else:
        from block_quant_rocm import COMPRESSION_RATIO
        comp_ratio = COMPRESSION_RATIO.get(bits, 4.92)

    return {
        "method": method,
        "bits": bits,
        "seq_len": seq_len,
        "tok_per_sec": tok_per_sec,
        "latency_ms": median_s * 1e3,
        "latency_p25_ms": p25_s * 1e3,
        "latency_p75_ms": p75_s * 1e3,
        "compression_ratio": comp_ratio,
        "n_layers": args.n_layers,
        "n_kv_heads": args.n_kv_heads,
        "head_dim": args.head_dim,
        "n_decode": args.n_decode,
    }


def main():
    parser = argparse.ArgumentParser(description="All-methods decode benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[512, 2048, 8192, 32768, 65536, 131072])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["fp16", "fp8", "int4",
                                 "planar3", "iso3", "rotor3", "turbo3",
                                 "planar4", "iso4", "rotor4", "turbo4"])
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n-decode", type=int, default=50,
                        help="Decode steps per (method, seq_len) config")
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = "cuda"
    all_results = []

    print(f"\n{'='*70}")
    print("All-Methods Decode Benchmark (Python/Triton path, fair comparison)")
    print(f"n_layers={args.n_layers}, n_kv_heads={args.n_kv_heads}, head_dim={args.head_dim}")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'SeqLen':>8} {'Tok/s':>10} {'Lat ms':>10} {'CompRatio':>10}")
    print("-" * 55)

    for method_spec in args.methods:
        for seq_len in args.seq_lens:
            try:
                result = bench_one_config(method_spec, seq_len, args, device)
                all_results.append(result)
                label = method_spec
                if result.get("tok_per_sec"):
                    print(f"  {label:<12} {seq_len:>8} "
                          f"{result['tok_per_sec']:>10.1f} "
                          f"{result['latency_ms']:>10.1f} "
                          f"{result['compression_ratio']:>9.2f}×")
            except Exception as e:
                print(f"  ERROR {method_spec} seq={seq_len}: {e}")
                all_results.append({
                    "method": method_spec, "seq_len": seq_len, "error": str(e)})

    print("=" * 55)

    # Relative speedup table vs fp16
    fp16_results = {r["seq_len"]: r for r in all_results
                    if r.get("method") == "fp16" and r.get("tok_per_sec")}
    if fp16_results and len(all_results) > len(fp16_results):
        print(f"\nSpeedup vs FP16 at seq_len={args.seq_lens[-1]}:")
        ref_seq = args.seq_lens[-1]
        ref_tps = fp16_results.get(ref_seq, {}).get("tok_per_sec", None)
        if ref_tps:
            seen = set()
            for r in all_results:
                if r["seq_len"] == ref_seq and r.get("tok_per_sec") and r["method"] != "fp16":
                    label = f"{r['method']}{r['bits']}" if r.get("bits") else r["method"]
                    if label not in seen:
                        seen.add(label)
                        speedup = r["tok_per_sec"] / ref_tps
                        print(f"  {label:<12}: {speedup:.2f}× vs FP16")

    # Save results
    model_slug = args.model.replace("/", "_").replace("-", "_")
    out_path = args.output or str(RESULTS_DIR / f"bench_all_methods_decode_{model_slug}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
