"""
bench_batch_decode_v2.py — Measured Batch Decode Benchmark for All Methods

Extends bench_batch_decode.py (which only had TurboQuant + theoretical model)
to include IsoQuant, PlanarQuant, and RotorQuant with MEASURED throughput.

The key question: does IsoQuant/PlanarQuant's faster rotation actually translate
to better throughput at the batch sizes where KV bandwidth dominates?

Bandwidth crossover model:
  batch* = W_bytes / (K_bytes_per_seq)
  Below batch*: compute-bound (model weights dominate), compression doesn't help
  Above batch*: bandwidth-bound (KV cache dominates), compression gives speedup

For Mistral-7B at seq=32K:
  W = 14 GB weights
  K = 2 × 32 × 8 × 32768 × 128 × 2B = 4.29 GB/seq
  batch* ≈ 14/4.29 ≈ 3.3 → meaningful gain starts at batch ≥ 4

FP16 attention variants (for ROCm CK / SDPA dispatch comparison):

- ``fp16`` — ``torch.bmm`` matmul path (legacy microbench; often memory-suboptimal).
- ``fp16_sdpa`` — decode-shaped **non-causal** ``scaled_dot_product_attention`` (one
  query token vs full KV); on ROCm this often stays on the **math** dispatcher.
- ``fp16_sdpa_causal`` — full-sequence **causal** SDPA (prefill-shaped, ``is_causal=True``),
  useful as an upper bound when the stack routes to **CK FlashAttention**.

Usage:
    python3 benchmarks/bench_batch_decode_v2.py
    python3 benchmarks/bench_batch_decode_v2.py \\
        --batch-sizes 1 4 8 16 32 64 \\
        --seq-lens 8192 32768 \\
        --methods fp16 fp16_sdpa fp16_sdpa_causal planar3 iso3 rotor3 turbo3
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

from cache_utils import add_swa_args, clamp_seq_to_window, print_swa_status


def parse_method_spec(method_spec: str) -> tuple[str, int]:
    """Return (method_key, bits) for e.g. ``planar3`` → (``planar``, 3)."""
    if method_spec == "fp16":
        return "fp16", 0
    if method_spec == "fp16_sdpa":
        return "fp16_sdpa", 0
    if method_spec == "fp16_sdpa_causal":
        return "fp16_sdpa_causal", 0
    if len(method_spec) < 2 or not method_spec[-1].isdigit():
        raise ValueError(
            f"Unknown method {method_spec!r}; expected fp16, fp16_sdpa, fp16_sdpa_causal, "
            "or a name ending in bit width (e.g. planar3)."
        )
    return method_spec[:-1], int(method_spec[-1])


# Model parameters (Mistral-7B defaults)
MODEL_PARAMS = {
    "mistralai/Mistral-7B-v0.1": {
        "n_params": 7e9,
        "n_layers": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "weight_bytes": 14e9,
    },
    "meta-llama/Meta-Llama-3-8B": {
        "n_params": 8e9,
        "n_layers": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "weight_bytes": 16e9,
    },
}


def sync():
    torch.cuda.synchronize()


def compute_theoretical_speedup(
    compression_ratio: float, weight_bytes: float,
    n_layers: int, n_kv_heads: int, seq_len: int, head_dim: int,
    batch_size: int
) -> float:
    """Amdahl's law speedup from KV compression."""
    kv_bytes_per_seq = 2 * n_layers * n_kv_heads * seq_len * head_dim * 2  # FP16
    total_kv = kv_bytes_per_seq * batch_size
    total_weight = weight_bytes
    total_fp16 = total_weight + total_kv

    weight_fraction = total_weight / total_fp16
    kv_fraction = total_kv / total_fp16

    # After compression: KV bytes reduced by compression_ratio
    effective_time_ratio = weight_fraction + kv_fraction / compression_ratio
    return 1.0 / effective_time_ratio


def simulate_batch_decode_step(
    cache_list: list, method: str, bits: int,
    batch_size: int, seq_len: int, n_layers: int,
    n_kv_heads: int, head_dim: int, device: str
) -> float:
    """
    Simulate one batch decode step (no model weights — pure KV bandwidth cost).
    Returns wall time in seconds.
    """
    sm_scale = 1.0 / (head_dim ** 0.5)

    # Query token for each batch item and head: (batch × H, 1, D)
    q = torch.randn(batch_size * n_kv_heads, 1, head_dim, device=device)

    sync()
    t0 = time.perf_counter()

    if method == "fp16":
        for l in range(n_layers):
            K, V = cache_list[l]  # (batch×H, S, D)
            scores = torch.bmm(q, K.transpose(-2, -1)) * sm_scale
            weights = torch.softmax(scores, dim=-1)
            _ = torch.bmm(weights, V)

    elif method == "fp16_sdpa":
        # (B*Hkv, 1, D) and (B*Hkv, S, D) → batched SDPA, non-causal (decode step shape).
        for l in range(n_layers):
            K, V = cache_list[l]
            q_b = q.view(batch_size, n_kv_heads, 1, head_dim).transpose(1, 2)
            k_b = K.view(batch_size, n_kv_heads, seq_len, head_dim).transpose(1, 2)
            v_b = V.view(batch_size, n_kv_heads, seq_len, head_dim).transpose(1, 2)
            _ = torch.nn.functional.scaled_dot_product_attention(
                q_b, k_b, v_b, is_causal=False, scale=sm_scale
            )

    elif method == "fp16_sdpa_causal":
        # Full causal attention over seq_len (prefill-shaped); often hits CK FA on ROCm.
        for l in range(n_layers):
            Q, K, V = cache_list[l]
            q_b = Q.transpose(1, 2)
            k_b = K.transpose(1, 2)
            v_b = V.transpose(1, 2)
            _ = torch.nn.functional.scaled_dot_product_attention(
                q_b, k_b, v_b, is_causal=True, scale=sm_scale
            )

    elif method == "turbo":
        tq = cache_list[-1]["tq"]
        for l in range(n_layers):
            K_comp = cache_list[l]["K"]
            V_comp = cache_list[l]["V"]
            K_fp = tq.decompress_tensor(K_comp, (batch_size * n_kv_heads * seq_len, head_dim))
            V_fp = tq.decompress_tensor(V_comp, (batch_size * n_kv_heads * seq_len, head_dim))
            K_fp = K_fp.reshape(batch_size * n_kv_heads, seq_len, head_dim)
            V_fp = V_fp.reshape(batch_size * n_kv_heads, seq_len, head_dim)
            scores = torch.bmm(q, K_fp.transpose(-2, -1)) * sm_scale
            weights = torch.softmax(scores, dim=-1)
            _ = torch.bmm(weights, V_fp)

    else:
        q_obj = cache_list[-1]["q"]
        for l in range(n_layers):
            K_comp = cache_list[l]["K"]
            V_comp = cache_list[l]["V"]
            K_fp = q_obj.decompress(K_comp, (batch_size * n_kv_heads * seq_len, head_dim))
            V_fp = q_obj.decompress(V_comp, (batch_size * n_kv_heads * seq_len, head_dim))
            K_fp = K_fp.reshape(batch_size * n_kv_heads, seq_len, head_dim)
            V_fp = V_fp.reshape(batch_size * n_kv_heads, seq_len, head_dim)
            scores = torch.bmm(q, K_fp.transpose(-2, -1)) * sm_scale
            weights = torch.softmax(scores, dim=-1)
            _ = torch.bmm(weights, V_fp)

    sync()
    return time.perf_counter() - t0


def build_cache(method: str, bits: int, batch_size: int, seq_len: int,
                n_layers: int, n_kv_heads: int, head_dim: int, device: str) -> list:
    """Build compressed KV cache for all layers."""
    cache = []

    if method == "fp16":
        for l in range(n_layers):
            K = torch.randn(batch_size * n_kv_heads, seq_len, head_dim, device=device)
            V = torch.randn(batch_size * n_kv_heads, seq_len, head_dim, device=device)
            cache.append((K, V))
        return cache

    if method == "fp16_sdpa":
        return build_cache("fp16", bits, batch_size, seq_len, n_layers, n_kv_heads, head_dim, device)

    if method == "fp16_sdpa_causal":
        for l in range(n_layers):
            Q = torch.randn(batch_size, n_kv_heads, seq_len, head_dim, device=device)
            K = torch.randn(batch_size, n_kv_heads, seq_len, head_dim, device=device)
            V = torch.randn(batch_size, n_kv_heads, seq_len, head_dim, device=device)
            cache.append((Q, K, V))
        return cache

    if method == "turbo":
        from turboquant_mi300x import TurboQuantMI300X
        tq = TurboQuantMI300X(bits=bits, rotation_seed=42)
        for l in range(n_layers):
            K_raw = torch.randn(batch_size * n_kv_heads * seq_len, head_dim, device=device)
            V_raw = torch.randn(batch_size * n_kv_heads * seq_len, head_dim, device=device)
            cache.append({
                "K": tq.compress_tensor(K_raw.float()),
                "V": tq.compress_tensor(V_raw.float()),
            })
        cache.append({"tq": tq})
        return cache

    from block_quant_rocm import make_quantizer
    q = make_quantizer(method, bits=bits, head_dim=head_dim, device=device)
    for l in range(n_layers):
        K_raw = torch.randn(batch_size * n_kv_heads * seq_len, head_dim, device=device)
        V_raw = torch.randn(batch_size * n_kv_heads * seq_len, head_dim, device=device)
        cache.append({
            "K": q.compress(K_raw),
            "V": q.compress(V_raw),
        })
    cache.append({"q": q})
    return cache


def main():
    parser = argparse.ArgumentParser(description="Measured batch decode benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seq-lens", type=int, nargs="+", default=[8192, 32768])
    parser.add_argument("--batch-sizes", type=int, nargs="+",
                        default=[1, 2, 4, 8, 16, 32, 64])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["fp16", "planar3", "iso3", "rotor3", "turbo3"])
    parser.add_argument("--n-measure", type=int, default=30)
    parser.add_argument("--n-warmup", type=int, default=5)
    parser.add_argument("--n-layers", type=int, default=32)
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--output", type=str, default="")
    add_swa_args(parser)
    args = parser.parse_args()
    print_swa_status(args.swa, args.window if args.swa == "on" else None)

    device = "cuda"
    model_info = MODEL_PARAMS.get(args.model, {
        "n_layers": args.n_layers, "n_kv_heads": args.n_kv_heads,
        "head_dim": args.head_dim, "weight_bytes": 14e9,
    })
    n_layers = model_info["n_layers"]
    n_kv_heads = model_info["n_kv_heads"]
    head_dim = model_info["head_dim"]
    weight_bytes = model_info["weight_bytes"]

    print(f"\n{'='*75}")
    print(f"Measured Batch Decode Benchmark — all methods")
    print(f"Model: {args.model}, layers={n_layers}, kv_heads={n_kv_heads}, head_dim={head_dim}")
    print(f"{'='*75}")

    all_results = []

    for seq_len in args.seq_lens:
        print(f"\n--- seq_len = {seq_len} ---")
        print(f"{'Method':<12} {'Batch':>6} {'Tok/s':>10} {'Lat ms':>10} "
              f"{'KV BW TB/s':>11} {'Theoret speedup':>16} {'vs FP16':>9}")
        print("-" * 80)

        # Track FP16 results for speedup computation
        fp16_tps = {}

        for method_spec in args.methods:
            try:
                method, bits = parse_method_spec(method_spec)
            except ValueError as e:
                print(f"  SKIP {method_spec}: {e}")
                continue

            # Get compression ratio
            if method in ("fp16", "fp16_sdpa", "fp16_sdpa_causal"):
                comp_ratio = 1.0
            elif method == "turbo":
                from turboquant_mi300x import COMPRESSION_RATIO
                comp_ratio = COMPRESSION_RATIO.get(bits, 4.92)
            else:
                from block_quant_rocm import COMPRESSION_RATIO
                comp_ratio = COMPRESSION_RATIO.get(bits, 4.92)

            for batch_size in args.batch_sizes:
                cache_seq = clamp_seq_to_window(seq_len, args.swa, args.window)
                # Skip if OOM risk: batch × cache_seq × layers × kv_heads × head_dim × 2B
                kv_bytes = batch_size * cache_seq * n_layers * n_kv_heads * head_dim * 2
                if kv_bytes > 150e9:  # > 150 GB FP16 KV — definitely OOM
                    print(f"  {method_spec:<12} {batch_size:>6} SKIP (OOM risk: {kv_bytes/1e9:.0f} GB KV)")
                    continue

                try:
                    cache = build_cache(method, bits, batch_size, cache_seq,
                                        n_layers, n_kv_heads, head_dim, device)

                    # Warmup
                    for _ in range(args.n_warmup):
                        simulate_batch_decode_step(
                            cache, method, bits, batch_size, cache_seq,
                            n_layers, n_kv_heads, head_dim, device)
                    sync()

                    # Measure
                    step_times = []
                    for _ in range(args.n_measure):
                        t = simulate_batch_decode_step(
                            cache, method, bits, batch_size, cache_seq,
                            n_layers, n_kv_heads, head_dim, device)
                        step_times.append(t)

                    median_s = float(np.median(step_times))
                    tps = batch_size / median_s  # total tokens/sec across batch
                    lat_ms = median_s * 1e3

                    # KV bandwidth (actual bytes read)
                    kv_bytes_compressed = (kv_bytes / comp_ratio) if comp_ratio > 0 else kv_bytes
                    kv_bw_tbs = kv_bytes_compressed / median_s / 1e12

                    # Theoretical speedup from bandwidth model
                    theoret = compute_theoretical_speedup(
                        comp_ratio, weight_bytes, n_layers, n_kv_heads,
                        cache_seq, head_dim, batch_size)

                    if method == "fp16":
                        fp16_tps[batch_size] = tps

                    fp16_ref = fp16_tps.get(batch_size, None)
                    speedup_vs_fp16 = tps / fp16_ref if fp16_ref else float("nan")

                    result = {
                        "method": method,
                        "bits": bits,
                        "seq_len": seq_len,
                        "cache_seq_len": cache_seq,
                        "batch_size": batch_size,
                        "tok_per_sec": tps,
                        "latency_ms": lat_ms,
                        "kv_bandwidth_tbs": kv_bw_tbs,
                        "theoretical_speedup": theoret,
                        "measured_speedup_vs_fp16": speedup_vs_fp16,
                        "compression_ratio": comp_ratio,
                        "swa_window": args.window if args.swa == "on" else None,
                    }
                    all_results.append(result)

                    speedup_str = f"{speedup_vs_fp16:.2f}×" if not np.isnan(speedup_vs_fp16) else "—"
                    print(f"  {method_spec:<12} {batch_size:>6} {tps:>10.0f} "
                          f"{lat_ms:>10.1f} {kv_bw_tbs:>11.3f} "
                          f"{theoret:>15.2f}× {speedup_str:>9}")

                    # Clean up cache to free VRAM
                    del cache
                    torch.cuda.empty_cache()

                except torch.cuda.OutOfMemoryError:
                    print(f"  {method_spec:<12} {batch_size:>6} OOM")
                    torch.cuda.empty_cache()
                except Exception as e:
                    print(f"  {method_spec:<12} {batch_size:>6} ERROR: {e}")

        print("")

    # Print crossover batch size analysis
    print(f"\n{'Bandwidth crossover analysis (batch* = W/K_per_seq):'}")
    for seq_len in args.seq_lens:
        kv_per_seq = 2 * n_layers * n_kv_heads * seq_len * head_dim * 2
        batch_star = weight_bytes / kv_per_seq
        print(f"  seq={seq_len:>7}: batch* ≈ {batch_star:.1f} "
              f"(KV bandwidth dominates above this batch size)")

    out_path = args.output or str(RESULTS_DIR / "bench_batch_decode_v2.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
