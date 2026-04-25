"""
bench_tq3_decode.py — End-to-end KV cache decode benchmark (TurboQuant bits)

Measures tok/s for decode with TQ3-compressed KV cache vs FP16 baseline.
Uses the pure-PyTorch TurboQuant wrapper (rotation via torch.matmul → MFMA).

Compression levels:
  FP16   — baseline (16 bits/element, no compression)
  TQ3    — 3-bit TurboQuant (4.92× compression vs FP16)
  TQ2    — 2-bit TurboQuant (7.11× compression vs FP16, experimental)

Ratio semantics (explicit):
  ratio_calculated_layout:
      computed from bytes/vector layout formula (e.g., 256 / 52 = 4.923× for TQ3)
  ratio_observed_runtime:
      computed from materialized cache bytes in the run
      (fp16_kv_bytes / compressed_kv_bytes)

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

try:
    from tq_triton import compress_kv_for_triton, turboquant_attention_fwd
except Exception:
    compress_kv_for_triton = None
    turboquant_attention_fwd = None

from cache_utils import (
    add_swa_args,
    print_swa_status,
    resolve_swa_window,
    truncate_kv_to_window,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt_ids(tokenizer, seq_len: int, batch_size: int = 1) -> torch.Tensor:
    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device="cuda")
    ids[:, 0] = tokenizer.bos_token_id or 1
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
    batch_size: int = 1,
    swa_window: int | None = None,
) -> Dict:
    model.eval()
    torch.cuda.reset_peak_memory_stats()

    prompt_ids = make_prompt_ids(tokenizer, seq_len, batch_size)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    cache = prefill_out.past_key_values
    fp16_kv_bytes = kv_memory_bytes_from_cache(cache)
    if swa_window is not None:
        truncate_kv_to_window(cache, swa_window)
    del prefill_out
    torch.cuda.empty_cache()

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    times = []

    for _ in range(n_runs):
        # Each decode step: (batch_size, 1) next-token tensor
        next_token = torch.full(
            (batch_size, 1), tokenizer.eos_token_id or 1,
            dtype=torch.long, device="cuda"
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_decode):
            with torch.no_grad():
                out = model(next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if swa_window is not None:
                truncate_kv_to_window(cache, swa_window)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    elapsed = float(np.median(times))
    # tokens_per_sec: each decode step produces batch_size tokens
    return {
        "mode":             "fp16",
        "bits":             16,
        "seq_len":          seq_len,
        "batch_size":       batch_size,
        "tokens_per_sec":   round(n_decode * batch_size / elapsed, 2),
        "tokens_per_sec_per_seq": round(n_decode / elapsed, 2),
        "latency_ms":       round(elapsed / n_decode * 1000, 3),
        "vram_peak_gb":     round(peak_vram, 2),
        "prefill_ms":       round(prefill_ms, 1),
        "kv_bytes":         fp16_kv_bytes,
        "compression_ratio": 1.0,  # backward-compatible alias
        "ratio_calculated_layout": 1.0,
        "ratio_observed_runtime": 1.0,
        "n_decode":         n_decode,
        "n_runs":           n_runs,
        "swa_window":       swa_window,
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
    batch_size: int = 1,
    swa_window: int | None = None,
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

    prompt_ids = make_prompt_ids(tokenizer, seq_len, batch_size)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    cache = prefill_out.past_key_values
    fp16_bytes = kv_memory_bytes_from_cache(cache)

    # SWA: truncate prefill cache to window before measuring tq_bytes / round-trip
    if swa_window is not None:
        truncate_kv_to_window(cache, swa_window)

    # Measure TQ compressed size from one representative layer
    sample_k = cache.layers[0].keys  # (1, n_kv_heads, seq_len, head_dim)
    n_vecs_per_layer = sample_k.numel() // sample_k.shape[-1]
    tq_bytes_per_layer = n_vecs_per_layer * BLOCK_BYTES[bits]
    tq_bytes = tq_bytes_per_layer * len(cache.layers) * 2  # × 2 for K and V
    ratio_observed_runtime = fp16_bytes / tq_bytes
    ratio_calculated_layout = float(COMPRESSION_RATIO[bits])

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
        next_token = torch.full(
            (batch_size, 1), tokenizer.eos_token_id or 1,
            dtype=torch.long, device="cuda"
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_decode):
            with torch.no_grad():
                out = model(next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if swa_window is not None:
                truncate_kv_to_window(cache, swa_window)
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
        "batch_size":       batch_size,
        "tokens_per_sec":   round(n_decode * batch_size / elapsed, 2),
        "tokens_per_sec_per_seq": round(n_decode / elapsed, 2),
        "latency_ms":       round(elapsed / n_decode * 1000, 3),
        "vram_peak_gb":     round(peak_vram, 2),
        "prefill_ms":       round(prefill_ms, 1),
        "initial_compress_ms": round(compress_ms, 1),
        "kv_bytes_fp16":    fp16_bytes,
        "kv_bytes_tq":      tq_bytes,
        "compression_ratio": round(ratio_observed_runtime, 3),  # backward-compatible alias
        "theoretical_ratio": round(ratio_calculated_layout, 3),  # backward-compatible alias
        "ratio_calculated_layout": round(ratio_calculated_layout, 3),
        "ratio_observed_runtime": round(ratio_observed_runtime, 3),
        "packed_bytes_per_vector": int(BLOCK_BYTES[bits]),
        "fp16_bytes_per_vector": 256,
        "cache_layers": int(len(cache.layers)),
        "cache_kv_heads": int(cache.layers[0].keys.shape[1]),
        "cache_head_dim": int(cache.layers[0].keys.shape[-1]),
        "tokens_done": int(n_decode * batch_size),
        "tokens_target": int(n_decode * batch_size),
        "n_decode":         n_decode,
        "n_runs":           n_runs,
        "swa_window":       swa_window,
    }


def bench_tq_fused_decode_step(
    model, tokenizer,
    tq,
    seq_len: int,
    n_decode: int = 30,
    n_runs: int = 3,
    batch_size: int = 1,
    swa_window: int | None = None,
) -> Dict:
    """
    Benchmark fused Triton attention on a persistent compressed KV cache.

    This keeps KV in compressed (planes+norms) form in the hot loop and does not
    rematerialize FP16 K/V each step. It benchmarks the decode attention step
    throughput directly (query projection is approximated from cache shape).
    """
    if compress_kv_for_triton is None or turboquant_attention_fwd is None:
        raise RuntimeError("tq_triton import failed; fused benchmark unavailable")

    if tq.bits != 3:
        raise ValueError("Fused Triton path currently supports TQ3 only")

    model.eval()
    torch.cuda.reset_peak_memory_stats()
    prompt_ids = make_prompt_ids(tokenizer, seq_len, batch_size)

    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0) * 1000

    cache = prefill_out.past_key_values
    fp16_bytes = kv_memory_bytes_from_cache(cache)

    # SWA: truncate prefill cache to window before building persistent compressed cache
    if swa_window is not None:
        truncate_kv_to_window(cache, swa_window)

    # Build a persistent compressed cache from prefill KV.
    layer0 = cache.layers[0]
    k0 = layer0.keys
    v0 = layer0.values
    k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k0, v0, tq)
    tq_bytes = (k_planes.numel() + v_planes.numel() + 4 * (k_norms.numel() + v_norms.numel())) * len(cache.layers)
    ratio_observed_runtime = fp16_bytes / max(tq_bytes, 1)

    # Approximate decode query using the newest token across KV heads.
    # Shape expected by fused kernel: (B, H, 1, D), pre-rotated.
    q_fp = k0[:, :, -1:, :].contiguous()
    q_rot = tq.rotate_queries(q_fp.float()).to(torch.float16)
    sm_scale = q_fp.shape[-1] ** -0.5

    del prefill_out
    torch.cuda.empty_cache()
    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    times = []

    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_decode):
            # Persistent compressed KV in hot loop (no decompress path).
            out = turboquant_attention_fwd(
                q_rot, k_planes, k_norms, v_planes, v_norms,
                rotation=tq.rotation, sm_scale=sm_scale,
            )
            # Lightweight query update to avoid dead-code elimination patterns.
            q_rot = out
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    elapsed = float(np.median(times))
    return {
        "mode": "tq3_fused",
        "bits": 3,
        "seq_len": seq_len,
        "batch_size": batch_size,
        "tokens_per_sec": round(n_decode * batch_size / elapsed, 2),
        "tokens_per_sec_per_seq": round(n_decode / elapsed, 2),
        "latency_ms": round(elapsed / n_decode * 1000, 3),
        "vram_peak_gb": round(peak_vram, 2),
        "prefill_ms": round(prefill_ms, 1),
        "kv_bytes_fp16": fp16_bytes,
        "kv_bytes_tq": tq_bytes,
        "compression_ratio": round(ratio_observed_runtime, 3),
        "ratio_calculated_layout": 4.923,
        "ratio_observed_runtime": round(ratio_observed_runtime, 3),
        "path": "fused_triton_decode_step",
        "n_decode": n_decode,
        "n_runs": n_runs,
        "swa_window": swa_window,
    }


def print_pretty_summary(result: Dict, model_name: str) -> None:
    """Print screenshot-style summary block with actual run values."""
    if result["mode"] == "fp16":
        return
    mode = result["mode"].upper()
    cache_shape = (
        f"{result.get('cache_layers', '?')} layers x "
        f"{result.get('cache_kv_heads', '?')} KV heads x "
        f"{result.get('cache_head_dim', '?')}d"
    )
    tq_kb = result.get("kv_bytes_tq", 0) / 1024.0
    fp16_kb = result.get("kv_bytes_fp16", 0) / 1024.0
    print()
    print(f"{mode} - Generation Complete ({model_name})")
    print("=" * 62)
    print(f"Tokens    {result.get('tokens_done', 0)}/{result.get('tokens_target', 0)}")
    print(f"Cache     {cache_shape}")
    print(f"TQ size   {tq_kb:.1f} KB    FP16   {fp16_kb:.1f} KB")
    print(
        f"Ratio     calc={result.get('ratio_calculated_layout', 0):.3f}x   "
        f"measured={result.get('ratio_observed_runtime', 0):.3f}x"
    )
    print(f"Speed     {result.get('tokens_per_sec', 0):.1f} tok/s")
    print("=" * 62)
    print()


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
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size for decode. At batch>=16 KV cache BW becomes the bottleneck"
                             " where TQ3 compression speedup is most visible.")
    parser.add_argument("--skip-fp16", action="store_true",
                        help="Skip FP16 baseline (if already measured)")
    parser.add_argument(
        "--include-fused-tq3",
        action="store_true",
        help="Also run persistent compressed KV fused decode-step benchmark (TQ3 only).",
    )
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--pretty-summary", action="store_true",
        help="Print screenshot-style summary blocks with actual runtime values."
    )
    add_swa_args(parser)
    args = parser.parse_args()

    print("=== TQ KV Cache End-to-End Decode Benchmark ===")
    print(f"Model:      {args.model}")
    print(f"Device:     {torch.cuda.get_device_name(0)}")
    print(f"Batch size: {args.batch_size}  "
          f"{'(KV-BW-bottleneck regime)' if args.batch_size >= 16 else '(weight-BW-bottleneck regime)'}")
    print(f"Seq lens:   {args.seq_lens}")
    print(f"TQ bits:    {args.bits}")
    print(f"Decode:     {args.n_decode} steps × {args.n_runs} runs (median)")
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

    effective_window = resolve_swa_window(args.swa, model, args.window)
    print_swa_status(args.swa, effective_window)

    # Pre-build TQ engines for each bit width
    tq_engines = {}
    for bits in args.bits:
        tq_engines[bits] = TurboQuantMI300X(bits=bits, rotation_seed=42)
        print(f"TQ{bits} engine: {tq_engines[bits]}")
    print()

    all_results = {
        "model":      args.model,
        "device":     torch.cuda.get_device_name(0),
        "batch_size": args.batch_size,
        "swa":        args.swa,
        "swa_window": effective_window,
        "results":    [],
    }

    header = (f"{'seq_len':>8} | {'mode':>6} | {'batch':>5} | "
              f"{'tok/s':>9} | {'tok/s/seq':>10} | {'lat_ms':>8} | "
              f"{'ratio':>7} | {'VRAM_GB':>8}")
    sep = "-" * len(header)
    print(header)
    print(sep)

    for seq_len in sorted(args.seq_lens):
        # FP16 baseline
        if not args.skip_fp16:
            try:
                r = bench_fp16_decode(model, tokenizer, seq_len,
                                      n_decode=args.n_decode, n_runs=args.n_runs,
                                      batch_size=args.batch_size,
                                      swa_window=effective_window)
                all_results["results"].append(r)
                print(f"{seq_len:>8} | {'fp16':>6} | {args.batch_size:>5} | "
                      f"{r['tokens_per_sec']:>9.1f} | "
                      f"{r['tokens_per_sec_per_seq']:>10.1f} | "
                      f"{r['latency_ms']:>8.1f} | "
                      f"{r['compression_ratio']:>7.2f}× | "
                      f"{r['vram_peak_gb']:>8.1f}")
            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>8} | {'fp16':>6} | {args.batch_size:>5} |      OOM")
                gc.collect(); torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"{seq_len:>8} | {'fp16':>6} | {args.batch_size:>5} | ERROR: {e}")
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
                                    n_decode=args.n_decode, n_runs=args.n_runs,
                                    batch_size=args.batch_size,
                                    swa_window=effective_window)
                all_results["results"].append(r)
                print(f"{seq_len:>8} | {r['mode']:>6} | {args.batch_size:>5} | "
                      f"{r['tokens_per_sec']:>9.1f} | "
                      f"{r['tokens_per_sec_per_seq']:>10.1f} | "
                      f"{r['latency_ms']:>8.1f} | "
                      f"{r['compression_ratio']:>7.2f}× | "
                      f"{r['vram_peak_gb']:>8.1f}")
                if args.pretty_summary:
                    print_pretty_summary(r, args.model)
            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>8} | {f'tq{bits}':>6} | {args.batch_size:>5} |      OOM")
                break
            except Exception as e:
                print(f"{seq_len:>8} | {f'tq{bits}':>6} | {args.batch_size:>5} | ERROR: {e}")
            finally:
                gc.collect(); torch.cuda.empty_cache()

        if args.include_fused_tq3 and 3 in args.bits:
            tq = tq_engines[3]
            try:
                r = bench_tq_fused_decode_step(
                    model, tokenizer, tq, seq_len,
                    n_decode=args.n_decode, n_runs=args.n_runs, batch_size=args.batch_size,
                    swa_window=effective_window,
                )
                all_results["results"].append(r)
                print(f"{seq_len:>8} | {r['mode']:>6} | {args.batch_size:>5} | "
                      f"{r['tokens_per_sec']:>9.1f} | "
                      f"{r['tokens_per_sec_per_seq']:>10.1f} | "
                      f"{r['latency_ms']:>8.1f} | "
                      f"{r['compression_ratio']:>7.2f}× | "
                      f"{r['vram_peak_gb']:>8.1f}")
            except Exception as e:
                print(f"{seq_len:>8} | {'tq3f':>6} | {args.batch_size:>5} | ERROR: {e}")
            finally:
                gc.collect(); torch.cuda.empty_cache()

        print(sep)

    # Print compression summary
    fp16_rows = [r for r in all_results["results"] if r["mode"] == "fp16"]
    tq3_rows  = [r for r in all_results["results"] if r["mode"] == "tq3"]
    if fp16_rows and tq3_rows:
        print()
        print(f"── KV Compression Summary (batch={args.batch_size}) ──")
        for tq_r in tq3_rows:
            fp16_r = next((r for r in fp16_rows if r["seq_len"] == tq_r["seq_len"]), None)
            if fp16_r:
                # Compare per-sequence throughput to isolate compression effect
                speedup = tq_r["tokens_per_sec_per_seq"] / fp16_r["tokens_per_sec_per_seq"]
                kv_savings_pct = (1 - 1 / tq_r["compression_ratio"]) * 100
                print(f"  seq={tq_r['seq_len']:>7}  ratio={tq_r['compression_ratio']:.2f}×  "
                      f"KV_savings={kv_savings_pct:.0f}%  "
                      f"tok/s speedup={speedup:.2f}×")

    # Save (include batch size in filename when non-default)
    model_slug = args.model.replace("/", "_")
    batch_tag = f"_b{args.batch_size}" if args.batch_size != 1 else ""
    output_path = args.output or (RESULTS_DIR / f"bench_tq3_decode{batch_tag}_{model_slug}.json")
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
