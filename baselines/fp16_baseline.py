"""
fp16_baseline.py — FP16 KV cache baseline benchmark

Measures decode-phase tokens/sec at various context lengths using standard
FP16 KV cache (no compression). This is the reference baseline for all
TurboQuant comparisons.

Target hardware: AMD Instinct MI300X (gfx942), ROCm 7.2
Model: Mistral-7B-v0.1 (primary) and Llama-3.1-8B-Instruct (secondary)

Usage:
    python baselines/fp16_baseline.py --model mistralai/Mistral-7B-v0.1
    python baselines/fp16_baseline.py --model meta-llama/Llama-3.1-8B-Instruct
    python baselines/fp16_baseline.py --seq-lens 512 2048 8192 --n-decode 50
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
if _hf_token := os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
from cache_utils import (
    add_swa_args,
    print_swa_status,
    resolve_swa_window,
    truncate_kv_to_window,
)

# ──────────────────────────────────────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────────────────────────────────────

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Performance flags for MI300X
# NOTE: PYTORCH_TUNABLEOP_ENABLED=1 triggers offline autotuning on first run,
# adding 30-120s per unique shape. Disable for benchmark timing; enable separately
# for production (run once with tuning, then reuse tuning DB).
os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")
os.environ.setdefault("HIP_FORCE_DEV_KERNARG", "1")


def get_device_info() -> Dict:
    name = torch.cuda.get_device_name(0)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    free_gb  = torch.cuda.mem_get_info()[0] / 1e9
    return {"name": name, "total_gb": round(total_gb, 1), "free_gb": round(free_gb, 1)}


def vram_peak_gb() -> float:
    return torch.cuda.max_memory_allocated() / 1e9


def reset_vram_peak():
    torch.cuda.reset_peak_memory_stats()


# ──────────────────────────────────────────────────────────────────────────────
# Benchmarking
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt_ids(tokenizer, seq_len: int) -> torch.Tensor:
    """Generate a padded prompt of exactly seq_len tokens."""
    # Use EOS/PAD token to fill to seq_len
    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((1, seq_len), pad_id, dtype=torch.long, device="cuda")
    # Put a real token at the start
    ids[0, 0] = tokenizer.bos_token_id or 1
    return ids


def benchmark_fp16_decode(
    model,
    tokenizer,
    seq_len: int,
    n_decode: int = 100,
    n_runs: int = 5,
    kv_dtype: torch.dtype = torch.float16,
    swa_window: int | None = None,
) -> Dict:
    """
    Benchmark decode-phase throughput (tokens/sec) at given context length.

    Protocol:
      1. Prefill to seq_len with forward pass (no gradient)
      2. Decode n_decode tokens, measure wall time
      3. Repeat n_runs times, report median

    Returns dict with tokens_per_sec, latency_ms, vram_peak_gb, prefill_ms.
    """
    model.eval()
    reset_vram_peak()

    # ── Prefill ───────────────────────────────────────────────────────────────
    prompt_ids = make_prompt_ids(tokenizer, seq_len)
    with torch.no_grad():
        torch.cuda.synchronize()
        t_prefill_start = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
        t_prefill_end = time.perf_counter()
    prefill_ms = (t_prefill_end - t_prefill_start) * 1000

    # Convert KV cache to target dtype if needed
    past_kv = prefill_out.past_key_values
    if kv_dtype != torch.float16:
        past_kv = tuple(
            (k.to(kv_dtype), v.to(kv_dtype)) for k, v in past_kv
        )

    # SWA: truncate prefill cache to window before decode begins
    if swa_window is not None:
        truncate_kv_to_window(past_kv, swa_window)

    # ── Decode loop ────────────────────────────────────────────────────────────
    peak_vram = vram_peak_gb()
    times = []

    for run in range(n_runs):
        next_token = torch.tensor([[tokenizer.eos_token_id or 1]], device="cuda")
        kv = past_kv  # reuse prefill KV each run

        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_decode):
            with torch.no_grad():
                out = model(next_token, past_key_values=kv, use_cache=True)
            kv = out.past_key_values
            if swa_window is not None:
                truncate_kv_to_window(kv, swa_window)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        times.append(t1 - t0)

    elapsed_s = float(np.median(times))
    tokens_per_sec = n_decode / elapsed_s
    latency_ms     = elapsed_s / n_decode * 1000

    return {
        "tokens_per_sec": round(tokens_per_sec, 2),
        "latency_ms":     round(latency_ms, 3),
        "vram_peak_gb":   round(peak_vram, 2),
        "prefill_ms":     round(prefill_ms, 1),
        "seq_len":        seq_len,
        "n_decode":       n_decode,
        "kv_dtype":       str(kv_dtype),
        "n_runs":         n_runs,
        "swa_window":     swa_window,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FP16 KV cache baseline benchmark (MI300X)")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model ID")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[512, 2048, 8192, 32768, 65536, 131072],
                        help="Context lengths to benchmark")
    parser.add_argument("--n-decode", type=int, default=100,
                        help="Decode tokens per measurement")
    parser.add_argument("--n-runs", type=int, default=5,
                        help="Repetitions (median reported)")
    parser.add_argument("--max-seq-len", type=int, default=None,
                        help="Skip seq_lens larger than this (VRAM limit)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: results/fp16_baseline_{model}.json)")
    add_swa_args(parser)
    args = parser.parse_args()

    print(f"=== FP16 Baseline Benchmark ===")
    print(f"Model:   {args.model}")
    print(f"Device:  {torch.cuda.get_device_name(0)}")
    free_gb = torch.cuda.mem_get_info()[0] / 1e9
    print(f"VRAM:    {free_gb:.1f} GB free / {torch.cuda.get_device_properties(0).total_memory/1e9:.0f} GB total")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Use SDPA (memory-efficient attention) instead of eager to handle long contexts.
    # Eager attention allocates a full n×n attention matrix per layer (O(n²) memory),
    # which OOMs at seq_len ≥ 32768 even with 192 GB VRAM.
    # SDPA uses chunked/flash-style computation reducing peak to O(n) memory.
    attn_impl = "sdpa"
    print(f"Attention implementation: {attn_impl}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation=attn_impl,
    )
    model.eval()
    print(f"Model loaded. Parameters: {sum(p.numel() for p in model.parameters())/1e9:.1f}B")

    effective_window = resolve_swa_window(args.swa, model, args.window)
    print_swa_status(args.swa, effective_window)

    # Run benchmarks
    results = {
        "model":       args.model,
        "device":      get_device_info(),
        "kv_config":   "fp16",
        "swa":         args.swa,
        "swa_window":  effective_window,
        "benchmarks":  [],
    }

    for seq_len in sorted(args.seq_lens):
        if args.max_seq_len and seq_len > args.max_seq_len:
            print(f"  seq_len={seq_len:6d}: SKIPPED (> max_seq_len={args.max_seq_len})")
            continue

        print(f"  seq_len={seq_len:6d} ...", end="", flush=True)
        try:
            r = benchmark_fp16_decode(
                model, tokenizer,
                seq_len=seq_len,
                n_decode=args.n_decode,
                n_runs=args.n_runs,
                swa_window=effective_window,
            )
            results["benchmarks"].append(r)
            print(f" {r['tokens_per_sec']:7.1f} tok/s | "
                  f"latency {r['latency_ms']:.1f} ms | "
                  f"VRAM {r['vram_peak_gb']:.1f} GB | "
                  f"prefill {r['prefill_ms']:.0f} ms")
        except torch.cuda.OutOfMemoryError:
            print(f" OOM — stopping at seq_len={seq_len}")
            break
        except Exception as e:
            print(f" ERROR: {e}")
            break
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    # Save results
    model_slug = args.model.replace("/", "_")
    output_path = args.output or (RESULTS_DIR / f"fp16_baseline_{model_slug}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
