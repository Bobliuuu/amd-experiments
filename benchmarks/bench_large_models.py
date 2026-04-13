"""
bench_large_models.py — Large Model Benchmarks (Mistral-70B / Llama-3-70B on MI300X)

The AMD Instinct MI300X has 192 GB HBM3 — enough to run Llama-3-70B in FP16 on a
single GPU.  With TQ3 KV compression, the available KV cache budget is multiplied
by 4.92×, enabling:

  Llama-3-70B  (FP16 weights ~140 GB):
    FP16 KV budget: 192 - 140 = 52 GB → max context ~160K tokens
    TQ3 KV budget: 52 × 4.92 ≈ 256 GB effective → max context ~787K tokens

  Mistral-7B   (FP16 weights ~14 GB):
    FP16 KV budget: 192 - 14 = 178 GB → max context ~1.4M tokens
    TQ3 KV budget: 178 × 4.92 ≈ 876 GB effective → max context ~6.9M tokens (theoretical)

Model architecture reference:
  Llama-3-70B:  80 layers, 64 Q-heads, 8 KV-heads (GQA), head_dim=128
  Llama-3-8B:   32 layers, 32 Q-heads, 8 KV-heads (GQA), head_dim=128
  Mistral-7B:   32 layers, 32 Q-heads, 8 KV-heads (GQA/sliding), head_dim=128

This benchmark:
  1. Prints a capacity analysis table for all model variants
  2. If the model fits in GPU memory, runs a decode benchmark at long context
  3. Measures tok/s, latency, VRAM usage at context lengths up to capacity

Usage:
    # Capacity analysis only (always works, no model download needed)
    python3 benchmarks/bench_large_models.py --analysis-only

    # Full benchmark (requires model download; ~140 GB for 70B)
    python3 benchmarks/bench_large_models.py --model meta-llama/Meta-Llama-3-70B

    # 7B as a proxy if 70B doesn't fit (still demonstrates GQA path)
    python3 benchmarks/bench_large_models.py --model meta-llama/Meta-Llama-3-8B

    # Quick smoke test at shorter context
    python3 benchmarks/bench_large_models.py \\
        --model meta-llama/Meta-Llama-3-8B \\
        --seq-lens 8192 65536 131072
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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
# Model architecture catalogue
# ──────────────────────────────────────────────────────────────────────────────

# (model_name, num_layers, num_q_heads, num_kv_heads, head_dim, est_weights_gb)
KNOWN_MODELS: List[Tuple[str, int, int, int, int, float]] = [
    ("mistralai/Mistral-7B-v0.1",            32, 32,  8, 128, 14.0),
    ("mistralai/Mistral-7B-Instruct-v0.2",   32, 32,  8, 128, 14.0),
    ("meta-llama/Meta-Llama-3-8B",           32, 32,  8, 128, 16.0),
    ("meta-llama/Meta-Llama-3.1-8B-Instruct",32, 32,  8, 128, 16.0),
    ("meta-llama/Meta-Llama-3-70B",          80, 64,  8, 128, 140.0),
    ("meta-llama/Meta-Llama-3.1-70B",        80, 64,  8, 128, 140.0),
    ("meta-llama/Meta-Llama-3.1-70B-Instruct",80,64,  8, 128, 140.0),
    ("meta-llama/Llama-2-70b-hf",            80, 64,  8, 128, 140.0),
    ("meta-llama/Meta-Llama-3-405B",        126, 128, 8, 128, 810.0),
]


def lookup_model_arch(model_name: str) -> Optional[Tuple[int, int, int, int, float]]:
    """Return (n_layers, n_q_heads, n_kv_heads, head_dim, est_gb) or None."""
    for name, nl, nq, nkv, hd, gb in KNOWN_MODELS:
        if name == model_name:
            return nl, nq, nkv, hd, gb
    return None


# ──────────────────────────────────────────────────────────────────────────────
# Capacity analysis (no GPU required)
# ──────────────────────────────────────────────────────────────────────────────

TQ3_BLOCK_BYTES = 52   # 4 (norm) + 48 (3 bit-planes × 16 bytes)
TQ4_BLOCK_BYTES = 68
TQ2_BLOCK_BYTES = 36


def max_context_analysis(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    weights_gb: float,
    total_vram_gb: float = 192.0,
    overhead_gb: float = 2.0,    # PyTorch framework + activation overhead
) -> Dict:
    """Compute maximum context length for FP16, TQ2, TQ3, TQ4."""
    avail_gb = total_vram_gb - weights_gb - overhead_gb

    # Bytes per token per layer (K + V, both directions)
    fp16_per_tok  = 2 * n_kv_heads * head_dim * 2    # FP16
    tq3_per_tok   = 2 * n_kv_heads * TQ3_BLOCK_BYTES
    tq4_per_tok   = 2 * n_kv_heads * TQ4_BLOCK_BYTES
    tq2_per_tok   = 2 * n_kv_heads * TQ2_BLOCK_BYTES
    fp8_per_tok   = 2 * n_kv_heads * head_dim * 1    # FP8 (1 byte)

    # Total bytes per token across all layers
    fp16_total = fp16_per_tok * n_layers
    tq3_total  = tq3_per_tok  * n_layers
    tq4_total  = tq4_per_tok  * n_layers
    tq2_total  = tq2_per_tok  * n_layers
    fp8_total  = fp8_per_tok  * n_layers

    avail_bytes = avail_gb * 1e9

    return {
        "weights_gb":      weights_gb,
        "available_gb":    avail_gb,
        "fp16_per_tok_b":  fp16_total,
        "tq3_per_tok_b":   tq3_total,
        "tq3_ratio":       fp16_total / tq3_total,
        "max_ctx_fp16":    int(avail_bytes // fp16_total),
        "max_ctx_fp8":     int(avail_bytes // fp8_total),
        "max_ctx_tq4":     int(avail_bytes // tq4_total),
        "max_ctx_tq3":     int(avail_bytes // tq3_total),
        "max_ctx_tq2":     int(avail_bytes // tq2_total),
    }


def print_capacity_table(total_vram_gb: float = 192.0) -> List[Dict]:
    """Print a formatted capacity table for all known models."""
    print(f"\n{'='*90}")
    print(f"KV Cache Capacity on AMD Instinct MI300X ({total_vram_gb:.0f} GB HBM3)")
    print(f"{'='*90}")

    header = (f"{'Model':<42}  {'Weights':>8}  "
              f"{'FP16 max':>10}  {'TQ3 max':>10}  "
              f"{'TQ3/FP16':>9}  {'KV savings':>10}")
    print(header)
    print("-" * len(header))

    rows = []
    for model_name, n_layers, n_q_heads, n_kv_heads, head_dim, weights_gb in KNOWN_MODELS:
        if weights_gb > total_vram_gb * 0.95:
            tag = " [multi-GPU]"
        else:
            tag = ""
        a = max_context_analysis(n_layers, n_kv_heads, head_dim, weights_gb, total_vram_gb)
        row = {
            "model": model_name,
            "weights_gb": weights_gb,
            "n_layers": n_layers,
            "n_kv_heads": n_kv_heads,
            "head_dim": head_dim,
            **a,
        }
        rows.append(row)

        name_display = model_name.split("/")[-1] + tag
        print(
            f"{name_display:<42}  {weights_gb:>7.0f}G  "
            f"{a['max_ctx_fp16']:>10,}  {a['max_ctx_tq3']:>10,}  "
            f"{a['tq3_ratio']:>8.2f}×  "
            f"{(1-1/a['tq3_ratio'])*100:>9.1f}%"
        )

    print(f"\n  TQ3 = 3-bit TurboQuant: 4-byte norm + 48-byte bit-planes = 52 bytes/vector")
    print(f"  TQ3 cosine similarity vs FP16: 0.9831 (measured, Mistral-7B-v0.1)")
    print(f"  FP16 vector: {128*2} bytes; TQ3 vector: {TQ3_BLOCK_BYTES} bytes → "
          f"{128*2/TQ3_BLOCK_BYTES:.2f}× compression")
    return rows


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark helpers (reuse protocol from bench_tq3_decode.py)
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt_ids(tokenizer, seq_len: int, batch_size: int = 1) -> torch.Tensor:
    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device="cuda")
    ids[:, 0] = tokenizer.bos_token_id or 1
    return ids


def apply_kv_roundtrip(cache, compress_fn) -> None:
    for layer in cache.layers:
        k_hat, v_hat = compress_fn(layer.keys, layer.values)
        layer.keys   = k_hat
        layer.values = v_hat


def bench_one_context(
    model,
    tokenizer,
    tq,
    seq_len: int,
    mode: str,              # "fp16" or "tq3"
    n_decode: int = 20,
    n_runs: int = 3,
    batch_size: int = 1,
) -> Dict:
    """
    Benchmark decode throughput at a given context length.

    Same protocol as bench_tq3_decode.py for comparability.
    """
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    device = next(model.parameters()).device

    prompt_ids = make_prompt_ids(tokenizer, seq_len, batch_size)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t0) * 1000
    cache = prefill_out.past_key_values
    del prefill_out

    fp16_kv_bytes = sum(
        layer.keys.numel() * layer.keys.element_size()
        + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    )

    def tq_roundtrip(k, v):
        hd = k.shape[-1]
        k_comp = tq.compress_tensor(k.reshape(-1, hd).float())
        v_comp = tq.compress_tensor(v.reshape(-1, hd).float())
        return (tq.decompress_tensor(k_comp, k.shape).to(k.dtype),
                tq.decompress_tensor(v_comp, v.shape).to(v.dtype))

    if mode != "fp16":
        apply_kv_roundtrip(cache, tq_roundtrip)

    torch.cuda.empty_cache()
    peak_vram_prefill = torch.cuda.max_memory_allocated() / 1e9

    times = []
    for _ in range(n_runs):
        next_token = torch.full(
            (batch_size, 1), tokenizer.eos_token_id or 1,
            dtype=torch.long, device=device,
        )
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_decode):
            with torch.no_grad():
                out = model(next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if mode != "fp16":
                apply_kv_roundtrip(cache, tq_roundtrip)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    elapsed = float(np.median(times))
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    # TQ3 compressed bytes (estimate)
    tq3_kv_bytes = (fp16_kv_bytes // (128 * 2)) * TQ3_BLOCK_BYTES

    return {
        "mode":              mode,
        "seq_len":           seq_len,
        "batch_size":        batch_size,
        "tokens_per_sec":    round(n_decode * batch_size / elapsed, 2),
        "latency_ms":        round(elapsed / n_decode * 1000, 3),
        "prefill_ms":        round(prefill_ms, 1),
        "vram_peak_gb":      round(peak_vram, 2),
        "vram_prefill_gb":   round(peak_vram_prefill, 2),
        "fp16_kv_bytes":     fp16_kv_bytes,
        "tq3_kv_bytes":      tq3_kv_bytes,
        "kv_compression":    round(fp16_kv_bytes / tq3_kv_bytes, 3),
        "n_decode":          n_decode,
        "n_runs":            n_runs,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Large-model benchmark: Llama-3-70B / Mistral-70B on MI300X"
    )
    parser.add_argument("--model",
                        default="meta-llama/Meta-Llama-3-70B",
                        help="HuggingFace model name.  Use --analysis-only to skip download.")
    parser.add_argument("--analysis-only", action="store_true",
                        help="Print capacity analysis only (no GPU benchmark)")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[8192, 32768, 65536, 131072],
                        help="Context lengths to benchmark")
    parser.add_argument("--n-decode",   type=int, default=20)
    parser.add_argument("--n-runs",     type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--skip-fp16",  action="store_true")
    parser.add_argument("--output",     type=str, default=None)
    args = parser.parse_args()

    # ── Capacity analysis (always run) ────────────────────────────────────────
    total_vram_gb = 192.0
    if torch.cuda.is_available():
        total_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9

    capacity_rows = print_capacity_table(total_vram_gb)

    # Per-model detailed breakdown
    arch = lookup_model_arch(args.model)
    if arch is not None:
        n_layers, n_q_heads, n_kv_heads, head_dim, weights_gb = arch
        a = max_context_analysis(n_layers, n_kv_heads, head_dim, weights_gb, total_vram_gb)

        print(f"\n{'─'*60}")
        print(f"Detailed analysis: {args.model.split('/')[-1]}")
        print(f"{'─'*60}")
        print(f"  Layers / Q-heads / KV-heads / head_dim:  "
              f"{n_layers} / {n_q_heads} / {n_kv_heads} / {head_dim}")
        print(f"  Weights (FP16):               ~{weights_gb:.0f} GB")
        print(f"  Available for KV:             {a['available_gb']:.1f} GB")
        print(f"  FP16 bytes per token:         {a['fp16_per_tok_b'] / 1024:.1f} KB")
        print(f"  TQ3  bytes per token:         {a['tq3_per_tok_b']  / 1024:.1f} KB")
        print()
        print(f"  Max context FP16:  {a['max_ctx_fp16']:>10,} tokens  "
              f"({a['max_ctx_fp16']/1000:.0f}K)")
        print(f"  Max context FP8:   {a['max_ctx_fp8']:>10,} tokens  "
              f"({a['max_ctx_fp8']/1000:.0f}K)")
        print(f"  Max context TQ4:   {a['max_ctx_tq4']:>10,} tokens  "
              f"({a['max_ctx_tq4']/1000:.0f}K)")
        print(f"  Max context TQ3:   {a['max_ctx_tq3']:>10,} tokens  "
              f"({a['max_ctx_tq3']/1000:.0f}K)  ← 4.92× FP16")
        print(f"  Max context TQ2:   {a['max_ctx_tq2']:>10,} tokens  "
              f"({a['max_ctx_tq2']/1000:.0f}K)  ← 7.11× FP16 (experimental)")
        print()
        print(f"  GQA ratio: {n_q_heads}Q / {n_kv_heads}KV = {n_q_heads//n_kv_heads}× "
              f"(each KV token shared by {n_q_heads//n_kv_heads} query heads)")
        print(f"  KV cache compression does NOT affect model weight BW — "
              f"only KV cache BW and VRAM.")

        # KV cache size at target context lengths
        print(f"\n  KV cache size at various context lengths:")
        for ctx in [32768, 65536, 131072, 262144, 524288]:
            fp16_gb = ctx * a['fp16_per_tok_b'] / 1e9
            tq3_gb  = ctx * a['tq3_per_tok_b']  / 1e9
            fits_fp16 = "✓" if fp16_gb <= a["available_gb"] else "✗ OOM"
            fits_tq3  = "✓" if tq3_gb  <= a["available_gb"] else "✗ OOM"
            print(f"    ctx={ctx//1024:>4}K:  FP16={fp16_gb:6.1f} GB [{fits_fp16}]  "
                  f"TQ3={tq3_gb:5.1f} GB [{fits_tq3}]")

    if args.analysis_only:
        # Save analysis without running benchmark
        out_path = args.output or str(RESULTS_DIR / "large_model_capacity_analysis.json")
        with open(out_path, "w") as f:
            json.dump({"capacity_rows": capacity_rows, "vram_gb": total_vram_gb}, f, indent=2)
        print(f"\nAnalysis saved: {out_path}")
        return

    # ── GPU benchmark ─────────────────────────────────────────────────────────
    if not torch.cuda.is_available():
        print("\nNo GPU available — run with --analysis-only for capacity tables.")
        return

    print(f"\n{'='*60}")
    print(f"GPU Benchmark")
    print(f"  Model:  {args.model}")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  VRAM:   {total_vram_gb:.0f} GB")
    print(f"{'='*60}")

    # Check if model fits in VRAM
    if arch is not None:
        _, _, _, _, weights_gb = arch
        if weights_gb > total_vram_gb * 0.9:
            print(f"\nWARNING: {args.model} requires ~{weights_gb:.0f} GB FP16 weights.")
            print(f"  Available VRAM: {total_vram_gb:.0f} GB")
            print(f"  Model may not fit on a single GPU.")
            print(f"  Options:")
            print(f"    1. Use device_map='auto' with multiple GPUs")
            print(f"    2. Use --model meta-llama/Meta-Llama-3-8B as a proxy")
            print(f"    3. Run --analysis-only for capacity analysis")
            print()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant_mi300x import TurboQuantMI300X

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    try:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.float16,
            device_map="cuda",            # single GPU, OOM if doesn't fit
            attn_implementation="sdpa",
        )
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        print(f"\nOOM loading model: {e}")
        print("Try loading with device_map='auto' for multi-GPU support,")
        print("or use --model meta-llama/Meta-Llama-3-8B as a 70B proxy.")
        return

    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    cfg = model.config
    n_layers_actual   = cfg.num_hidden_layers
    n_kv_heads_actual = getattr(cfg, "num_key_value_heads",
                                  getattr(cfg, "num_attention_heads", 8))
    n_q_heads_actual  = getattr(cfg, "num_attention_heads", 64)
    head_dim_actual   = cfg.hidden_size // n_q_heads_actual

    print(f"Loaded: {n_params/1e9:.1f}B params, "
          f"{n_layers_actual}L × {n_q_heads_actual}Qh/{n_kv_heads_actual}KVh × "
          f"{head_dim_actual}d")
    print(f"GQA ratio: {n_q_heads_actual // n_kv_heads_actual}× "
          f"({'GQA' if n_kv_heads_actual < n_q_heads_actual else 'MHA'})")

    tq = TurboQuantMI300X(bits=3, rotation_seed=42)

    all_results = {
        "model":         args.model,
        "device":        torch.cuda.get_device_name(0),
        "n_params":      n_params,
        "n_layers":      n_layers_actual,
        "n_q_heads":     n_q_heads_actual,
        "n_kv_heads":    n_kv_heads_actual,
        "head_dim":      head_dim_actual,
        "vram_total_gb": total_vram_gb,
        "results":       [],
    }

    header = (f"{'seq_len':>10}  {'mode':>5}  {'tok/s':>9}  {'lat_ms':>9}  "
              f"{'pre_ms':>9}  {'VRAM_GB':>8}  {'KV_compr':>9}")
    sep = "-" * len(header)
    print(f"\n{header}")
    print(sep)

    for seq_len in sorted(args.seq_lens):
        # Check if this context even makes sense
        if arch is not None:
            n_layers_c, n_kv_heads_c, head_dim_c = n_layers_actual, n_kv_heads_actual, head_dim_actual
            fp16_kv_gb = seq_len * 2 * n_layers_c * n_kv_heads_c * head_dim_c * 2 / 1e9
            if arch is not None and fp16_kv_gb > total_vram_gb * 0.5:
                print(f"{'':>10}  {'NOTE':>5}  FP16 KV at {seq_len//1024}K = {fp16_kv_gb:.1f} GB — may OOM")

        if not args.skip_fp16:
            try:
                r = bench_one_context(
                    model, tokenizer, tq, seq_len, "fp16",
                    args.n_decode, args.n_runs, args.batch_size,
                )
                all_results["results"].append(r)
                print(f"{seq_len:>10,}  {'fp16':>5}  {r['tokens_per_sec']:>9.1f}  "
                      f"{r['latency_ms']:>9.1f}  {r['prefill_ms']:>9.1f}  "
                      f"{r['vram_peak_gb']:>8.2f}  {'1.00×':>9}")
            except torch.cuda.OutOfMemoryError:
                print(f"{seq_len:>10,}  {'fp16':>5}  OOM at {seq_len//1024}K context")
                gc.collect(); torch.cuda.empty_cache()
            except Exception as e:
                print(f"{seq_len:>10,}  {'fp16':>5}  ERROR: {e}")
                gc.collect(); torch.cuda.empty_cache()
            finally:
                gc.collect(); torch.cuda.empty_cache()

        try:
            r = bench_one_context(
                model, tokenizer, tq, seq_len, "tq3",
                args.n_decode, args.n_runs, args.batch_size,
            )
            all_results["results"].append(r)
            print(f"{seq_len:>10,}  {'tq3':>5}  {r['tokens_per_sec']:>9.1f}  "
                  f"{r['latency_ms']:>9.1f}  {r['prefill_ms']:>9.1f}  "
                  f"{r['vram_peak_gb']:>8.2f}  {r['kv_compression']:>8.2f}×")
        except torch.cuda.OutOfMemoryError:
            print(f"{seq_len:>10,}  {'tq3':>5}  OOM at {seq_len//1024}K context")
        except Exception as e:
            print(f"{seq_len:>10,}  {'tq3':>5}  ERROR: {e}")
        finally:
            gc.collect(); torch.cuda.empty_cache()

        print(sep)

    # Summary: longest context achieved
    fp16_rows = [r for r in all_results["results"] if r["mode"] == "fp16"]
    tq3_rows  = [r for r in all_results["results"] if r["mode"] == "tq3"]

    if fp16_rows:
        max_fp16 = max(r["seq_len"] for r in fp16_rows)
        print(f"\nMax context achieved (FP16): {max_fp16:,} tokens")
    if tq3_rows:
        max_tq3 = max(r["seq_len"] for r in tq3_rows)
        print(f"Max context achieved (TQ3):  {max_tq3:,} tokens")
        if fp16_rows:
            ctx_mult = max_tq3 / max(max_fp16, 1)
            print(f"Context length multiplier:   {ctx_mult:.2f}× (theoretical 4.92×)")

    # Throughput comparison at common context lengths
    print()
    for tq3_r in tq3_rows:
        fp16_r = next((r for r in fp16_rows if r["seq_len"] == tq3_r["seq_len"]), None)
        if fp16_r:
            sp = tq3_r["tokens_per_sec"] / max(fp16_r["tokens_per_sec"], 0.001)
            print(f"  seq={tq3_r['seq_len']:>7,}:  TQ3 speedup={sp:.2f}×  "
                  f"({tq3_r['tokens_per_sec']:.1f} vs {fp16_r['tokens_per_sec']:.1f} tok/s)")

    model_slug = args.model.replace("/", "_")
    out_path = args.output or str(RESULTS_DIR / f"bench_large_models_{model_slug}.json")
    with open(out_path, "w") as f:
        json.dump({
            "model":         args.model,
            "capacity_rows": capacity_rows,
            "results":       all_results,
        }, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
