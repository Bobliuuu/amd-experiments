"""
bench_prefill.py — Prefill Throughput Benchmark for All KV Compression Methods

Measures time-to-first-token (TTFT) and prefill tokens/sec for:
  FP16 (baseline), TurboQuant, IsoQuant, PlanarQuant, RotorQuant

Key metric: The RotorQuant README claims a 5.3× prefill speed difference between
PlanarQuant and TurboQuant. This benchmark verifies that claim on AMD ROCm.

Why prefill matters:
  During prefill, ALL input tokens are processed and their KV pairs stored.
  Each token requires one compress() call per layer. For TurboQuant, this means
  16,384 FMAs/vector × 32 layers × n_tokens in the WHT rotation step alone.
  For PlanarQuant, it's only 256 FMAs/vector — 64× fewer. The difference
  manifests directly as prefill latency.

Usage:
    python3 benchmarks/bench_prefill.py --model mistralai/Mistral-7B-v0.1
    python3 benchmarks/bench_prefill.py --model mistralai/Mistral-7B-v0.1 \\
        --seq-lens 512 2048 8192 32768 \\
        --methods fp16 planar3 iso3 rotor3 turbo3
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


def get_method_compress_fn(method: str, bits: int, n_layers: int,
                            n_kv_heads: int, head_dim: int, device: str):
    """Return a (compress_kv_fn, cache_meta) for one method."""
    if method == "fp16":
        return None, None

    if method == "turbo":
        from turboquant_mi300x import TurboQuantMI300X
        tq = TurboQuantMI300X(bits=bits, rotation_seed=42)
        def compress_kv(k, v):
            ck = tq.compress_tensor(k.float())
            cv = tq.compress_tensor(v.float())
            return ck, cv
        return compress_kv, {"tq": tq}

    from block_quant_rocm import make_quantizer
    q = make_quantizer(method, bits=bits, head_dim=head_dim, device=device)

    def compress_kv(k, v):
        # k, v: (n_kv_heads, seq_len, head_dim)
        n_kv, seq, hd = k.shape
        k_flat = k.reshape(n_kv * seq, hd)
        v_flat = v.reshape(n_kv * seq, hd)
        ck = q.compress(k_flat)
        cv = q.compress(v_flat)
        return ck, cv

    return compress_kv, {"q": q}


def simulate_prefill_compress(
    n_tokens: int,
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    compress_fn,
    device: str = "cuda",
) -> float:
    """
    Simulate the KV compression cost during prefill.
    Returns total wall time in seconds for compressing all tokens' KV pairs.

    This isolates the compression overhead without model forward pass time,
    giving a direct measure of how much prefill is slowed by compression.
    """
    # Simulate K and V tensors for one layer (n_kv_heads, n_tokens, head_dim)
    k = torch.randn(n_kv_heads, n_tokens, head_dim, device=device)
    v = torch.randn(n_kv_heads, n_tokens, head_dim, device=device)

    sync()
    t0 = time.perf_counter()
    for _ in range(n_layers):
        compress_fn(k, v)
    sync()
    return time.perf_counter() - t0


def bench_prefill_standalone(args, device: str):
    """Benchmark standalone KV compress cost for varying sequence lengths."""
    results = []

    print(f"\n{'='*70}")
    print("Prefill KV Compression Cost (standalone, no model forward pass)")
    print(f"n_layers={args.n_layers}, n_kv_heads={args.n_kv_heads}, head_dim={args.head_dim}")
    print(f"{'='*70}")
    print(f"{'Method':<12} {'SeqLen':>8} {'CompTime ms':>12} {'Compress tok/s':>15}")
    print("-" * 55)

    for method_spec in args.methods:
        if method_spec == "fp16":
            method, bits = "fp16", 0
        else:
            method = method_spec[:-1]  # "planar3" → "planar"
            bits = int(method_spec[-1])

        if method == "fp16":
            for seq_len in args.seq_lens:
                results.append({
                    "method": "fp16", "bits": 0, "seq_len": seq_len,
                    "compress_time_ms": 0.0, "compress_toks_per_sec": float("inf"),
                    "note": "No KV compression in FP16"
                })
                print(f"{'fp16':<12} {seq_len:>8} {'—':>12} {'∞ (no compress)':>15}")
            continue

        compress_fn, _ = get_method_compress_fn(
            method, bits, args.n_layers, args.n_kv_heads, args.head_dim, device)

        # Warmup (trigger JIT)
        print(f"  Warming up {method_spec}...", end="", flush=True)
        for _ in range(3):
            simulate_prefill_compress(
                64, args.n_layers, args.n_kv_heads, args.head_dim, compress_fn, device)
        print(" done")

        for seq_len in args.seq_lens:
            times = []
            for _ in range(args.n_runs):
                t = simulate_prefill_compress(
                    seq_len, args.n_layers, args.n_kv_heads,
                    args.head_dim, compress_fn, device)
                times.append(t)
            median_ms = float(np.median(times)) * 1e3
            toks_per_sec = seq_len / (float(np.median(times)))

            result = {
                "method": method,
                "bits": bits,
                "seq_len": seq_len,
                "compress_time_ms": median_ms,
                "compress_toks_per_sec": toks_per_sec,
                "n_layers": args.n_layers,
                "n_kv_heads": args.n_kv_heads,
                "head_dim": args.head_dim,
            }
            results.append(result)
            print(f"  {method_spec:<10} {seq_len:>8} {median_ms:>12.1f} {toks_per_sec:>15.0f}")

    return results


def bench_prefill_e2e(args, device: str):
    """
    End-to-end prefill benchmark including model forward pass.
    Measures TTFT (time to first token) for each method.
    """
    print(f"\n{'='*70}")
    print(f"End-to-End Prefill (model forward + KV compress) — {args.model}")
    print(f"{'='*70}")

    try:
        from transformers import AutoTokenizer, AutoModelForCausalLM
    except ImportError:
        print("transformers not available — skipping E2E prefill benchmark")
        return []

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16, device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded. Starting prefill benchmarks...")

    results = []
    # Use a fixed prompt padded to each seq_len
    base_text = "The quick brown fox jumped over the lazy dog. " * 4000

    for method_spec in args.methods:
        if method_spec == "fp16":
            method, bits = "fp16", 0
        else:
            method = method_spec[:-1]
            bits = int(method_spec[-1])

        if method != "fp16":
            compress_fn, _ = get_method_compress_fn(
                method, bits, 32, 8, 128, device)
            # Warmup compress
            dummy_k = torch.randn(8, 64, 128, device=device)
            for _ in range(3):
                compress_fn(dummy_k, dummy_k)
            sync()

        for seq_len in args.seq_lens:
            tokens = tokenizer(base_text, return_tensors="pt",
                               truncation=True, max_length=seq_len)
            input_ids = tokens["input_ids"].to(device)
            actual_len = input_ids.shape[1]

            # Warmup
            with torch.no_grad():
                _ = model(input_ids, use_cache=False)
            sync()

            times = []
            for _ in range(args.n_runs):
                sync()
                t0 = time.perf_counter()
                with torch.no_grad():
                    out = model(input_ids, use_cache=True)
                sync()
                t1 = time.perf_counter()

                # If method != fp16, compress the returned KV cache
                if method != "fp16" and hasattr(out, "past_key_values") and out.past_key_values:
                    sync()
                    t_compress_start = time.perf_counter()
                    for layer_kv in out.past_key_values:
                        k, v = layer_kv[0], layer_kv[1]
                        compress_fn(k.squeeze(0), v.squeeze(0))
                    sync()
                    t1 = time.perf_counter()

                times.append(t1 - t0)

            median_s = float(np.median(times))
            ttft_ms = median_s * 1e3
            tps = actual_len / median_s

            result = {
                "method": method,
                "bits": bits,
                "seq_len": actual_len,
                "ttft_ms": ttft_ms,
                "prefill_toks_per_sec": tps,
                "model": args.model,
            }
            results.append(result)
            print(f"  {method_spec:<12} seq={actual_len:>6} "
                  f"TTFT={ttft_ms:>8.1f}ms  {tps:>8.0f} tok/s")

    del model
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser(description="Prefill throughput benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seq-lens", type=int, nargs="+",
                        default=[512, 2048, 8192, 32768])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["fp16", "planar3", "iso3", "rotor3", "turbo3",
                                 "planar4", "iso4", "rotor4", "turbo4"])
    parser.add_argument("--n-layers", type=int, default=32,
                        help="Number of transformer layers (for standalone benchmark)")
    parser.add_argument("--n-kv-heads", type=int, default=8)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--standalone-only", action="store_true",
                        help="Only run standalone compress benchmark (no model needed)")
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    device = "cuda"
    all_results = {"standalone": [], "e2e": []}

    standalone = bench_prefill_standalone(args, device)
    all_results["standalone"] = standalone

    if not args.standalone_only:
        e2e = bench_prefill_e2e(args, device)
        all_results["e2e"] = e2e

    # Print comparison table
    if standalone:
        print(f"\n{'Speedup vs TurboQuant3 (prefill compress — standalone)'}")
        tq3_base = next((r for r in standalone
                         if r.get("method") == "turbo" and r.get("bits") == 3
                         and r["seq_len"] == args.seq_lens[-1]), None)
        if tq3_base:
            tq3_tps = tq3_base["compress_toks_per_sec"]
            for r in standalone:
                if r.get("compress_toks_per_sec", 0) > 0:
                    speedup = r["compress_toks_per_sec"] / tq3_tps
                    label = f"{r['method']}{r['bits']}" if r.get("bits") else "fp16"
                    if r["seq_len"] == args.seq_lens[-1]:
                        print(f"  {label:<12}: {speedup:.2f}× faster at seq={r['seq_len']}")

    out_path = args.output or str(RESULTS_DIR / "bench_prefill_all_methods.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
