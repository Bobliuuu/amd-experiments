"""
bench_batch_decode.py — Batch Decode TQ3 Speedup (Bandwidth Bottleneck Regime)

Measures TQ3 speedup at batch sizes 1–64 to characterize the transition from
weight-bandwidth-bottleneck to KV-bandwidth-bottleneck.

Theoretical background
----------------------
For a decoder-only LLM at decode time, per-step compute has two components:

  Weight bandwidth (W):  read model weights once per forward pass
                          = num_params × 2 bytes (FP16)
                          scales with batch_size (amortized over batch)

  KV bandwidth (K):       read K and V cache for all past tokens
                          = 2 × num_layers × num_kv_heads × seq_len × head_dim × 2 bytes
                          scales with seq_len (constant overhead per step regardless of batch)

At small batch the weight read dominates; at large batch, weight cost is amortized
across batch_size tokens and KV traffic (which grows with seq_len) becomes the
bottleneck.  The crossover batch size is roughly:

    batch_crossover ≈ W_bytes / K_bytes_per_seq

For Mistral-7B at seq_len=32K:
    W_bytes ≈ 7e9 × 2 = 14 GB
    K_bytes = 2 × 32 × 8 × 32768 × 128 × 2 = 4.29 GB
    crossover ≈ 14 / 4.29 ≈ 3.3 → KV bottleneck starts at batch ≥ ~4

At seq_len=8K: crossover ≈ 14 / 1.07 ≈ 13 → meaningful TQ3 gain starts at batch ≥ 16

TQ3 advantage: KV bytes reduced 4.92×.  In the BW-bottleneck regime, TQ3 decode
throughput scales as:
    tokens_per_sec(TQ3) / tokens_per_sec(FP16) ≈ min(4.92×, batch/crossover)

Expected results on MI300X (5.3 TB/s HBM3):
  batch=1:  ~1× (weight BW dominates, TQ3 overhead from decompression)
  batch=4:  ~1.2-2× (transition zone)
  batch=16: ~2.5-4× (KV BW starting to dominate)
  batch=32: ~3.5-4.9× (near theoretical max)
  batch=64: ~4.9× if VRAM permits (KV BW fully dominant)

Usage:
    python3 benchmarks/bench_batch_decode.py --model mistralai/Mistral-7B-v0.1
    python3 benchmarks/bench_batch_decode.py --seq-len 8192 32768
    python3 benchmarks/bench_batch_decode.py --batch-sizes 1 4 8 16 32 64
"""

import argparse
import gc
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

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
# Theoretical bandwidth model
# ──────────────────────────────────────────────────────────────────────────────

def compute_kv_bytes_per_step(
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch_size: int,
    bits: int = 16,
) -> int:
    """
    KV cache bytes read per decode step.

    FP16 (bits=16): 2 × num_layers × num_kv_heads × seq_len × head_dim × 2 bytes
    TQ3 (bits=3):   2 × num_layers × num_kv_heads × seq_len × 52 bytes
    """
    if bits == 16:
        bytes_per_vec = head_dim * 2   # FP16
    elif bits == 3:
        bytes_per_vec = 52             # TQ3 block format
    elif bits == 4:
        bytes_per_vec = 68             # TQ4 block format
    elif bits == 8:
        bytes_per_vec = head_dim       # FP8
    else:
        bytes_per_vec = head_dim * 2   # default FP16

    return 2 * num_layers * num_kv_heads * seq_len * bytes_per_vec * batch_size


def compute_weight_bytes(num_params: int) -> int:
    """FP16 model weight bytes read per forward pass."""
    return num_params * 2


def compute_crossover_batch(
    num_params: int,
    num_layers: int,
    num_kv_heads: int,
    head_dim: int,
    seq_len: int,
) -> float:
    """
    Estimate batch size at which KV cache BW equals weight BW.
    Above this batch size, TQ3 compression yields speedup > 1×.
    """
    w_bytes = compute_weight_bytes(num_params)
    kv_bytes_per_seq = 2 * num_layers * num_kv_heads * seq_len * head_dim * 2
    return w_bytes / max(kv_bytes_per_seq, 1)


def theoretical_speedup(
    batch_size: int,
    crossover_batch: float,
    compression_ratio: float = 4.923,
) -> float:
    """
    Theoretical TQ3 vs FP16 speedup assuming perfect BW scaling.

    Returns a value in [1, compression_ratio].  At small batch (BW = weights),
    TQ3 provides no speedup. At large batch (BW = KV cache), speedup approaches
    the compression ratio.
    """
    if crossover_batch <= 0:
        return compression_ratio
    # Weight BW fraction at this batch_size
    weight_fraction = 1.0 / (1.0 + batch_size / crossover_batch)
    kv_fraction = 1.0 - weight_fraction
    # TQ3 only helps the KV fraction
    effective_ratio = 1.0 / (weight_fraction + kv_fraction / compression_ratio)
    return min(effective_ratio, compression_ratio)


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark helpers (same protocol as bench_tq3_decode.py)
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt_ids(tokenizer, seq_len: int, batch_size: int) -> torch.Tensor:
    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device="cuda")
    ids[:, 0] = tokenizer.bos_token_id or 1
    return ids


def apply_kv_roundtrip(cache, compress_fn) -> None:
    for layer in cache.layers:
        k_hat, v_hat = compress_fn(layer.keys, layer.values)
        layer.keys   = k_hat
        layer.values = v_hat


def run_decode_step(model, next_token, cache, n_steps: int) -> List[float]:
    """Run n_steps decode iterations, return per-step wall-clock times."""
    times = []
    for _ in range(n_steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(next_token, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000
        cache = out.past_key_values
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        times.append(elapsed_ms)
    return times, cache, next_token


def bench_one(
    model,
    tokenizer,
    tq,
    seq_len: int,
    batch_size: int,
    mode: str,          # "fp16" or "tq3"
    n_warmup: int = 3,
    n_measure: int = 20,
) -> Dict:
    """
    Benchmark one (seq_len, batch_size, mode) cell.

    Returns a dict with timing statistics and derived bandwidth metrics.
    """
    model.eval()
    torch.cuda.reset_peak_memory_stats()
    device = next(model.parameters()).device

    prompt_ids = make_prompt_ids(tokenizer, seq_len, batch_size)

    # Prefill
    with torch.no_grad():
        torch.cuda.synchronize()
        t_pre = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
    prefill_ms = (time.perf_counter() - t_pre) * 1000
    cache = prefill_out.past_key_values
    del prefill_out

    # Measure FP16 KV bytes (before compression)
    fp16_kv_bytes = sum(
        layer.keys.numel() * layer.keys.element_size()
        + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    )

    # Apply TQ3 round-trip to prefill cache
    if mode != "fp16":
        def tq_roundtrip(k, v):
            head_dim = k.shape[-1]
            k_comp = tq.compress_tensor(k.reshape(-1, head_dim).float())
            v_comp = tq.compress_tensor(v.reshape(-1, head_dim).float())
            return (tq.decompress_tensor(k_comp, k.shape).to(k.dtype),
                    tq.decompress_tensor(v_comp, v.shape).to(v.dtype))
        apply_kv_roundtrip(cache, tq_roundtrip)

    torch.cuda.empty_cache()
    peak_vram = torch.cuda.max_memory_allocated() / 1e9

    next_token = torch.full(
        (batch_size, 1), tokenizer.eos_token_id or 1,
        dtype=torch.long, device=device,
    )

    # Warmup (discard)
    def step_fn(nt, c):
        with torch.no_grad():
            out = model(nt, past_key_values=c, use_cache=True)
        c = out.past_key_values
        if mode != "fp16":
            apply_kv_roundtrip(c, tq_roundtrip)
        nt = out.logits[:, -1:, :].argmax(dim=-1)
        return nt, c

    for _ in range(n_warmup):
        next_token, cache = step_fn(next_token, cache)

    # Measurement
    torch.cuda.synchronize()
    step_times_ms = []
    t_total = time.perf_counter()
    for _ in range(n_measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            out = model(next_token, past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        if mode != "fp16":
            apply_kv_roundtrip(cache, tq_roundtrip)
        next_token = out.logits[:, -1:, :].argmax(dim=-1)
        torch.cuda.synchronize()
        step_times_ms.append((time.perf_counter() - t0) * 1000)

    total_elapsed = time.perf_counter() - t_total

    # Statistics
    step_ms = np.array(step_times_ms)
    median_ms = float(np.median(step_ms))
    p25_ms    = float(np.percentile(step_ms, 25))
    p75_ms    = float(np.percentile(step_ms, 75))

    tokens_per_sec = batch_size / (median_ms / 1000)

    # Effective bandwidth: KV cache bytes read / step time
    # (approx: only KV cache traffic, ignoring weight reads for illustration)
    if mode == "fp16":
        kv_bytes_per_step = fp16_kv_bytes
    else:
        from turboquant_mi300x import TQ3_BLOCK_BYTES as _TQ3_BB
        tq3_kv_bytes = (fp16_kv_bytes // (128 * 2)) * _TQ3_BB
        kv_bytes_per_step = tq3_kv_bytes

    kv_bandwidth_tbs = kv_bytes_per_step / (median_ms / 1000) / 1e12

    return {
        "mode":              mode,
        "seq_len":           seq_len,
        "batch_size":        batch_size,
        "tokens_per_sec":    round(tokens_per_sec, 2),
        "latency_ms":        round(median_ms, 3),
        "latency_p25_ms":    round(p25_ms, 3),
        "latency_p75_ms":    round(p75_ms, 3),
        "kv_bandwidth_tbs":  round(kv_bandwidth_tbs, 4),
        "fp16_kv_bytes":     fp16_kv_bytes,
        "kv_bytes_per_step": kv_bytes_per_step,
        "prefill_ms":        round(prefill_ms, 1),
        "vram_peak_gb":      round(peak_vram, 2),
        "n_measure":         n_measure,
        "n_warmup":          n_warmup,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Batch decode benchmark: TQ3 speedup vs FP16 in BW-bottleneck regime"
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[8192, 32768],
                        help="Context lengths to sweep (longer = stronger TQ3 benefit)")
    parser.add_argument("--batch-sizes", nargs="+", type=int,
                        default=[1, 2, 4, 8, 16, 32, 64],
                        help="Batch sizes to measure (batch≥16 is KV-BW-bottleneck)")
    parser.add_argument("--n-warmup",   type=int, default=3)
    parser.add_argument("--n-measure",  type=int, default=20)
    parser.add_argument("--skip-fp16",  action="store_true")
    parser.add_argument("--output",     type=str, default=None)
    args = parser.parse_args()

    print("=" * 74)
    print("Batch Decode Benchmark — TQ3 KV Cache Speedup (BW Bottleneck Regime)")
    print("=" * 74)
    print(f"Model:       {args.model}")
    print(f"Device:      {torch.cuda.get_device_name(0)}")
    print(f"Seq lens:    {args.seq_lens}")
    print(f"Batch sizes: {args.batch_sizes}")
    print(f"Steps:       {args.n_warmup} warmup + {args.n_measure} measured")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant_mi300x import TurboQuantMI300X

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:  {n_params / 1e9:.2f}B ({n_params * 2 / 1e9:.1f} GB FP16)")

    # Infer model config
    cfg = model.config
    n_layers    = cfg.num_hidden_layers
    n_kv_heads  = getattr(cfg, "num_key_value_heads",
                           getattr(cfg, "num_attention_heads", 32))
    n_q_heads   = getattr(cfg, "num_attention_heads", 32)
    head_dim    = cfg.hidden_size // n_q_heads
    print(f"Architecture: {n_layers}L × {n_q_heads}Qh/{n_kv_heads}KVh × {head_dim}d")

    tq = TurboQuantMI300X(bits=3, rotation_seed=42)

    all_results = {
        "model":        args.model,
        "device":       torch.cuda.get_device_name(0),
        "n_params":     n_params,
        "n_layers":     n_layers,
        "n_kv_heads":   n_kv_heads,
        "head_dim":     head_dim,
        "results":      [],
        "theory":       [],
    }

    for seq_len in args.seq_lens:
        # Theoretical crossover batch
        crossover = compute_crossover_batch(
            n_params, n_layers, n_kv_heads, head_dim, seq_len
        )
        print(f"\n{'─'*74}")
        print(f"seq_len = {seq_len:,}   |   KV-BW crossover at batch ≈ {crossover:.1f}")
        print(f"{'─'*74}")

        # Print theoretical speedup table
        theo_header = (f"  {'batch':>6}  {'Theoretical speedup':>20}  "
                       f"{'Regime':>25}")
        print(theo_header)
        for bs in args.batch_sizes:
            sp = theoretical_speedup(bs, crossover)
            regime = ("weight-BW dominant" if bs < crossover * 0.5
                      else "KV-BW dominant" if bs > crossover * 2
                      else "transition zone")
            theory_entry = {
                "seq_len": seq_len, "batch_size": bs,
                "crossover": round(crossover, 2),
                "theoretical_speedup": round(sp, 3), "regime": regime,
            }
            all_results["theory"].append(theory_entry)
            print(f"  {bs:>6}  {sp:>19.2f}×  {regime:>25}")
        print()

        # Measured results header
        meas_header = (f"  {'batch':>6}  {'mode':>6}  {'tok/s':>9}  {'lat_ms':>8}  "
                       f"{'KV_BW TB/s':>12}  {'VRAM_GB':>8}")
        print(meas_header)
        print("  " + "-" * (len(meas_header) - 2))

        fp16_results: Dict[int, Dict] = {}

        for bs in args.batch_sizes:
            # FP16 baseline
            if not args.skip_fp16:
                try:
                    r = bench_one(model, tokenizer, tq, seq_len, bs, "fp16",
                                  args.n_warmup, args.n_measure)
                    fp16_results[bs] = r
                    all_results["results"].append(r)
                    print(f"  {bs:>6}  {'fp16':>6}  {r['tokens_per_sec']:>9.1f}  "
                          f"{r['latency_ms']:>8.2f}  "
                          f"{r['kv_bandwidth_tbs']:>12.4f}  "
                          f"{r['vram_peak_gb']:>8.2f}")
                except torch.cuda.OutOfMemoryError:
                    print(f"  {bs:>6}  {'fp16':>6}  OOM")
                    gc.collect(); torch.cuda.empty_cache()
                    break
                except Exception as e:
                    print(f"  {bs:>6}  {'fp16':>6}  ERROR: {e}")
                finally:
                    gc.collect(); torch.cuda.empty_cache()

            # TQ3
            try:
                r = bench_one(model, tokenizer, tq, seq_len, bs, "tq3",
                              args.n_warmup, args.n_measure)
                all_results["results"].append(r)

                # Compute measured speedup vs FP16 baseline
                speedup_str = "N/A"
                if bs in fp16_results:
                    fp16_r = fp16_results[bs]
                    speedup = r["tokens_per_sec"] / max(fp16_r["tokens_per_sec"], 0.001)
                    r["speedup_vs_fp16"] = round(speedup, 3)
                    theo = theoretical_speedup(bs, crossover)
                    r["theoretical_speedup"] = round(theo, 3)
                    speedup_str = f"{speedup:.2f}× (theory {theo:.2f}×)"

                print(f"  {bs:>6}  {'tq3':>6}  {r['tokens_per_sec']:>9.1f}  "
                      f"{r['latency_ms']:>8.2f}  "
                      f"{r['kv_bandwidth_tbs']:>12.4f}  "
                      f"{r['vram_peak_gb']:>8.2f}  "
                      f"speedup={speedup_str}")
            except torch.cuda.OutOfMemoryError:
                print(f"  {bs:>6}  {'tq3':>6}  OOM")
                break
            except Exception as e:
                print(f"  {bs:>6}  {'tq3':>6}  ERROR: {e}")
            finally:
                gc.collect(); torch.cuda.empty_cache()

    # Summary: speedup at batch=16, 32, 64
    print()
    print("=" * 74)
    print("Summary — TQ3 Speedup in KV-BW-Bottleneck Regime")
    print("=" * 74)

    for seq_len in args.seq_lens:
        print(f"\nseq_len = {seq_len:,}")
        fp16_by_bs = {r["batch_size"]: r for r in all_results["results"]
                      if r["mode"] == "fp16" and r["seq_len"] == seq_len}
        tq3_by_bs  = {r["batch_size"]: r for r in all_results["results"]
                      if r["mode"] == "tq3"  and r["seq_len"] == seq_len}
        for bs in args.batch_sizes:
            if bs in fp16_by_bs and bs in tq3_by_bs:
                fp16_r = fp16_by_bs[bs]
                tq3_r  = tq3_by_bs[bs]
                sp = tq3_r["tokens_per_sec"] / max(fp16_r["tokens_per_sec"], 0.001)
                theo = theoretical_speedup(bs, compute_crossover_batch(
                    n_params, n_layers, n_kv_heads, head_dim, seq_len))
                print(f"  batch={bs:>3}:  {sp:.2f}× measured, {theo:.2f}× theoretical  "
                      f"({tq3_r['tokens_per_sec']:.1f} vs {fp16_r['tokens_per_sec']:.1f} tok/s)")

    # Save results
    model_slug = args.model.replace("/", "_")
    out_path = args.output or str(RESULTS_DIR / f"bench_batch_decode_{model_slug}.json")
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved: {out_path}")


if __name__ == "__main__":
    main()
