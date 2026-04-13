"""
bench_tq3_decode.py — End-to-end TQ3 KV cache decode benchmark

Measures tok/s for decode with TQ3-compressed KV cache vs FP16 baseline.
Uses the pure-PyTorch TurboQuant wrapper (rotation via torch.matmul → MFMA).

Compression levels:
  FP16   — baseline (16 bits/element, no compression)
  TQ3    — 3-bit TurboQuant (4.92× compression vs FP16)
  TQ2    — 2-bit TurboQuant (7.11× compression vs FP16, experimental)

Protocol (matching paper's decode setup):
  1. Prefill at seq_len tokens in FP16 to warm up KV cache
  2. Compress KV cache to TQ format (all layers)
  3. Decode n_decode tokens, dequantizing KV on each step
  4. Measure steady-state tok/s (median over n_runs)

Usage:
    python3 benchmarks/bench_tq3_decode.py --model mistralai/Mistral-7B-v0.1
    python3 benchmarks/bench_tq3_decode.py --seq-lens 512 8192 65536 131072
    python3 benchmarks/bench_tq3_decode.py --bits 3 4 2
"""

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
if _hf_token := os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR.mkdir(exist_ok=True)

os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt_ids(tokenizer, seq_len: int) -> torch.Tensor:
    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((1, seq_len), pad_id, dtype=torch.long, device="cuda")
    ids[0, 0] = tokenizer.bos_token_id or 1
    return ids


def apply_kv_roundtrip(cache, compress_fn) -> None:
    """Apply compress_fn(k, v) -> (k_hat, v_hat) in-place to all layers.

    Uses cache.layers (transformers 5.x DynamicCache internal structure).
    Each layer exposes .keys and .values as directly settable tensors.
    """
    for layer in cache.layers:
        k_hat, v_hat = compress_fn(layer.keys, layer.values)
        layer.keys   = k_hat
        layer.values = v_hat


def kv_memory_bytes_from_cache(cache) -> int:
    """Total FP16 bytes of KV cache from cache.layers."""
    return sum(
        layer.keys.numel() * layer.keys.element_size()
        + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    )


# ──────────────────────────────────────────────────────────────────────────────
# TQ KV cache management
# ──────────────────────────────────────────────────────────────────────────────

def compress_kv_cache(
    past_kv: list,
    tq,
) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Compress all KV pairs to TQ format.

    Returns list of (k_comp, v_comp, k_orig_shape, v_orig_shape) per layer.
    k_comp, v_comp: (n_vectors, block_bytes) uint8
    """
    compressed = []
    for k, v in past_kv:
        k_shape, v_shape = k.shape, v.shape
        k_fp32 = k.reshape(-1, k.shape[-1]).float()
        v_fp32 = v.reshape(-1, v.shape[-1]).float()
        k_comp = tq.compress_tensor(k_fp32)
        v_comp = tq.compress_tensor(v_fp32)
        compressed.append((k_comp, v_comp, k_shape, v_shape))
    return compressed


def decompress_kv_cache(
    compressed: list,
    tq,
    dtype: torch.dtype = torch.float16,
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """Decompress all TQ-compressed KV pairs back to fp16."""
    result = []
    for k_comp, v_comp, k_shape, v_shape in compressed:
        k_hat = tq.decompress_tensor(k_comp, k_shape).to(dtype)
        v_hat = tq.decompress_tensor(v_comp, v_shape).to(dtype)
        result.append((k_hat, v_hat))
    return result


def kv_memory_bytes(compressed: list) -> int:
    """Total bytes used by compressed KV cache."""
    return sum(k_c.numel() + v_c.numel() for k_c, v_c, _, _ in compressed)


def kv_fp16_bytes(kv_pairs: list) -> int:
    """Total bytes of FP16 KV cache."""
    return sum(k.numel() * 2 + v.numel() * 2 for k, v in kv_pairs)


# ──────────────────────────────────────────────────────────────────────────────
# FP16 baseline benchmark
# ──────────────────────────────────────────────────────────────────────────────

def bench_fp16_decode(
    model, tokenizer,
    seq_len: int,
    n_decode: int = 30,
    n_runs: int = 3,
) -> Dict:
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    prompt_ids = make_prompt_ids(tokenizer, seq_len)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    cache = prefill_out.past_key_values
    fp16_kv_bytes = kv_memory_bytes_from_cache(cache)
    del prefill_out
    torch.cuda.empty_cache()

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    times = []

    for _ in range(n_runs):
        next_token = torch.tensor([[tokenizer.eos_token_id or 1]], device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_decode):
            with torch.no_grad():
                out = model(next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    elapsed = float(np.median(times))
    return {
        "mode":             "fp16",
        "bits":             16,
        "seq_len":          seq_len,
        "tokens_per_sec":   round(n_decode / elapsed, 2),
        "latency_ms":       round(elapsed / n_decode * 1000, 3),
        "vram_peak_gb":     round(peak_vram, 2),
        "prefill_ms":       round(prefill_ms, 1),
        "kv_bytes":         fp16_kv_bytes,
        "compression_ratio": 1.0,
        "n_decode":         n_decode,
        "n_runs":           n_runs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# TQ decode benchmark
# ──────────────────────────────────────────────────────────────────────────────

def bench_tq_decode(
    model, tokenizer,
    tq,
    seq_len: int,
    bits: int = 3,
    n_decode: int = 30,
    n_runs: int = 3,
) -> Dict:
    """
    Benchmark TQ KV cache decode.

    Prefill → apply TQ round-trip to cache in-place → decode loop.
    Each step: model reads (dequantized) KV, appends new token, we recompress.
    """
    from turboquant_mi300x import COMPRESSION_RATIO, TQ3_BLOCK_BYTES, TQ4_BLOCK_BYTES, TQ2_BLOCK_BYTES
    BLOCK_BYTES = {2: TQ2_BLOCK_BYTES, 3: TQ3_BLOCK_BYTES, 4: TQ4_BLOCK_BYTES}

    def tq_roundtrip(k: torch.Tensor, v: torch.Tensor):
        """Compress and immediately decompress K,V to simulate TQ storage."""
        head_dim = k.shape[-1]
        k_comp = tq.compress_tensor(k.reshape(-1, head_dim).float())
        v_comp = tq.compress_tensor(v.reshape(-1, head_dim).float())
        k_hat = tq.decompress_tensor(k_comp, k.shape).to(k.dtype)
        v_hat = tq.decompress_tensor(v_comp, v.shape).to(v.dtype)
        return k_hat, v_hat

    model.eval()
    torch.cuda.reset_peak_memory_stats()

    prompt_ids = make_prompt_ids(tokenizer, seq_len)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    cache = prefill_out.past_key_values
    fp16_bytes = kv_memory_bytes_from_cache(cache)

    # Measure TQ compressed size from one representative layer
    sample_k = cache.layers[0].keys  # (1, n_kv_heads, seq_len, head_dim)
    n_vecs_per_layer = sample_k.numel() // sample_k.shape[-1]
    tq_bytes_per_layer = n_vecs_per_layer * BLOCK_BYTES[bits]
    tq_bytes = tq_bytes_per_layer * len(cache.layers) * 2  # × 2 for K and V
    actual_ratio = fp16_bytes / tq_bytes

    # Initial TQ round-trip on prefill cache
    t_comp0 = time.perf_counter()
    apply_kv_roundtrip(cache, tq_roundtrip)
    torch.cuda.synchronize()
    compress_ms = (time.perf_counter() - t_comp0) * 1000

    del prefill_out
    torch.cuda.empty_cache()

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    times = []

    for _ in range(n_runs):
        next_token = torch.tensor([[tokenizer.eos_token_id or 1]], device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_decode):
            with torch.no_grad():
                out = model(next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            # Simulate TQ storage: compress + decompress the updated cache in-place
            apply_kv_roundtrip(cache, tq_roundtrip)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    elapsed = float(np.median(times))
    tq_name = f"tq{bits}"

    return {
        "mode":             tq_name,
        "bits":             bits,
        "seq_len":          seq_len,
        "tokens_per_sec":   round(n_decode / elapsed, 2),
        "latency_ms":       round(elapsed / n_decode * 1000, 3),
        "vram_peak_gb":     round(peak_vram, 2),
        "prefill_ms":       round(prefill_ms, 1),
        "initial_compress_ms": round(compress_ms, 1),
        "kv_bytes_fp16":    fp16_bytes,
        "kv_bytes_tq":      tq_bytes,
        "compression_ratio": round(actual_ratio, 3),
        "theoretical_ratio": round(COMPRESSION_RATIO[bits], 3),
        "n_decode":         n_decode,
        "n_runs":           n_runs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="End-to-end TQ3 KV cache decode benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[512, 2048, 8192, 32768, 65536, 131072])
    parser.add_argument("--bits", nargs="+", type=int, default=[3, 4],
                        help="TurboQuant bit widths to benchmark (2, 3, or 4)")
    parser.add_argument("--n-decode", type=int, default=30,
                        help="Decode steps per benchmark run")
    parser.add_argument("--n-runs", type=int, default=3,
                        help="Benchmark runs to median over")
    parser.add_argument("--skip-fp16", action="store_true",
                        help="Skip FP16 baseline (if already measured)")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=== TQ KV Cache End-to-End Decode Benchmark ===")
    print(f"Model:    {args.model}")
    print(f"Device:   {torch.cuda.get_device_name(0)}")
    print(f"Seq lens: {args.seq_lens}")
    print(f"TQ bits:  {args.bits}")
    print(f"Decode:   {args.n_decode} steps × {args.n_runs} runs (median)")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant_mi300x import TurboQuantMI300X

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"Model:    {n_params:.1f}B params, dtype={next(model.parameters()).dtype}")

    # Pre-build TQ engines for each bit width
    tq_engines = {}
    for bits in args.bits:
        tq_engines[bits] = TurboQuantMI300X(bits=bits, rotation_seed=42)
        print(f"TQ{bits} engine: {tq_engines[bits]}")
    print()

    all_results = {
        "model":   args.model,
        "device":  torch.cuda.get_device_name(0),
        "results": [],
    }

    header = f"{'seq_len':>8} | {'mode':>6} | {'tok/s':>8} | {'lat_ms':>8} | {'ratio':>7} | {'VRAM_GB':>8}"
    sep    = "-" * len(header)
    print(header)
    print(sep)

    for seq_len in sorted(args.seq_lens):
        # FP16 baseline
        if not args.skip_fp16:
            try:
                r = bench_fp16_decode(model, tokenizer, seq_len,
                                      n_decode=args.n_decode, n_runs=args.n_runs)
                all_results["results"].append(r)
                print(f"{seq_len:>8} | {'fp16':>6} | {r['tokens_per_sec']:>8.1f} | "
                      f"{r['latency_ms']:>8.1f} | {r['compression_ratio']:>7.2f}× | "
                      f"{r['vram_peak_gb']:>8.1f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>8} | {'fp16':>6} |      OOM")
                gc.collect(); torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"{seq_len:>8} | {'fp16':>6} | ERROR: {e}")
                gc.collect(); torch.cuda.empty_cache()
                continue
            finally:
                gc.collect(); torch.cuda.empty_cache()

        # TQ baselines
        for bits in args.bits:
            tq = tq_engines[bits]
            try:
                r = bench_tq_decode(model, tokenizer, tq, seq_len,
                                    bits=bits,
                                    n_decode=args.n_decode, n_runs=args.n_runs)
                all_results["results"].append(r)
                print(f"{seq_len:>8} | {r['mode']:>6} | {r['tokens_per_sec']:>8.1f} | "
                      f"{r['latency_ms']:>8.1f} | {r['compression_ratio']:>7.2f}× | "
                      f"{r['vram_peak_gb']:>8.1f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>8} | {f'tq{bits}':>6} |      OOM")
                break
            except Exception as e:
                print(f"{seq_len:>8} | {f'tq{bits}':>6} | ERROR: {e}")
            finally:
                gc.collect(); torch.cuda.empty_cache()

        print(sep)

    # Print compression summary
    fp16_rows = [r for r in all_results["results"] if r["mode"] == "fp16"]
    tq3_rows  = [r for r in all_results["results"] if r["mode"] == "tq3"]
    if fp16_rows and tq3_rows:
        print()
        print("── KV Compression Summary ──")
        for tq_r in tq3_rows:
            fp16_r = next((r for r in fp16_rows if r["seq_len"] == tq_r["seq_len"]), None)
            if fp16_r:
                speedup = tq_r["tokens_per_sec"] / fp16_r["tokens_per_sec"]
                kv_savings_pct = (1 - 1 / tq_r["compression_ratio"]) * 100
                print(f"  seq={tq_r['seq_len']:>7}  ratio={tq_r['compression_ratio']:.2f}×  "
                      f"KV_savings={kv_savings_pct:.0f}%  "
                      f"tok/s ratio={speedup:.2f}×")

    # Save
    model_slug = args.model.replace("/", "_")
    output_path = args.output or (RESULTS_DIR / f"bench_tq3_decode_{model_slug}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {output_path}")

    # #region agent log
    import json as _j, time as _t
    with open('/root/workspace/.cursor/debug-5ac54c.log', 'a') as _lf:
        summary = [{'seq_len': r['seq_len'], 'mode': r['mode'],
                    'tok_s': r['tokens_per_sec'], 'ratio': r.get('compression_ratio', 1.0)}
                   for r in all_results['results']]
        _lf.write(_j.dumps({'sessionId': '5ac54c', 'location': 'bench_tq3_decode.py:main',
                            'message': 'benchmark_complete', 'data': {'results_count': len(summary), 'summary': summary[:12]},
                            'timestamp': int(_t.time() * 1000), 'hypothesisId': 'D'}) + '\n')
    # #endregion


if __name__ == "__main__":
    main()
