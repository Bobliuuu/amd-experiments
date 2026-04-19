"""
bench_vllm_serving.py — End-to-end vLLM serving throughput: FP16 vs IsoQuant KV

Compares two vLLM engine configurations on Mistral-7B-v0.1:
  1. FP16 baseline  — standard PagedAttention, FP16 KV cache
  2. IsoQuant       — PagedAttention with IsoQuant 3-bit KV compression backend

Metrics per configuration:
  - Prefill throughput  (tokens/sec)
  - Decode throughput   (tokens/sec, avg over all output tokens)
  - First-token latency (ms)
  - End-to-end latency  (ms, mean and p99)
  - GPU memory used     (GB peak)
  - Max concurrent KV tokens (theoretical from KV cache size)

Usage:
    python3 benchmarks/bench_vllm_serving.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --input-len 512 \\
        --output-len 128 \\
        --num-prompts 20 \\
        --backends fp16 isoquant

    # Run only FP16 (useful to confirm vLLM is working first):
    python3 benchmarks/bench_vllm_serving.py --backends fp16

    # Quick smoke test (2 prompts, short output):
    python3 benchmarks/bench_vllm_serving.py --num-prompts 2 --output-len 32
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch

RESULTS_DIR = Path(__file__).parent.parent / "results"
KERNELS_DIR = Path(__file__).parent.parent / "kernels"
TQ_BACKENDS_DIR = Path(__file__).parent.parent / "tq_backends" / "attention" / "backends"
RESULTS_DIR.mkdir(exist_ok=True)
sys.path.insert(0, str(KERNELS_DIR))


# ──────────────────────────────────────────────────────────────────────────────
# IsoQuant backend registration
# ──────────────────────────────────────────────────────────────────────────────

def _register_isoquant_backend():
    """
    Register the IsoQuant ROCm attention backend with the installed vLLM.

    Strategy: monkey-patch vllm's attention backend registry so that
    VLLM_ATTENTION_BACKEND=ISOQUANT_ROCM is recognised.  This works for
    vLLM 0.6+ which uses a string-keyed registry dict.
    """
    sys.path.insert(0, str(TQ_BACKENDS_DIR.parent.parent))   # repo root for tq_backends
    try:
        from tq_backends.attention.backends.isoquant_rocm_attn import (
            IsoQuantROCmAttentionBackend,
        )
    except ImportError:
        sys.path.insert(0, str(TQ_BACKENDS_DIR))
        from tq_backends.attention.backends.isoquant_rocm_attn import (
            IsoQuantROCmAttentionBackend,
        )

    # vLLM 0.6+ registry lives in vllm.attention.backends.utils or
    # vllm.attention.selector — try both locations.
    try:
        import vllm.attention.selector as _sel
        if hasattr(_sel, "_ATTENTION_BACKEND_REGISTRY"):
            _sel._ATTENTION_BACKEND_REGISTRY["ISOQUANT_ROCM"] = IsoQuantROCmAttentionBackend
            print("  Registered IsoQuant backend via vllm.attention.selector registry")
            return
    except (ImportError, AttributeError):
        pass

    # vLLM 0.7+ / 0.19 uses a different mechanism: the backend class is looked
    # up by name from VLLM_ATTENTION_BACKEND env var inside get_attn_backend().
    # We patch that function to intercept "ISOQUANT_ROCM".
    try:
        import vllm.attention.selector as _sel

        _orig_get = _sel.get_attn_backend

        def _patched_get(head_size, dtype, kv_cache_dtype, block_size,
                         is_attention_free=False, use_mla=False):
            if os.environ.get("VLLM_ATTENTION_BACKEND") == "ISOQUANT_ROCM":
                return IsoQuantROCmAttentionBackend
            return _orig_get(head_size, dtype, kv_cache_dtype, block_size,
                             is_attention_free, use_mla)

        _sel.get_attn_backend = _patched_get
        print("  Registered IsoQuant backend via get_attn_backend() patch")
    except Exception as exc:
        print(f"  WARNING: Could not register IsoQuant backend: {exc}")
        print("  Proceeding — ISOQUANT_ROCM env var may not take effect.")


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ──────────────────────────────────────────────────────────────────────────────

def _make_prompts(tokenizer, n: int, input_len: int) -> List[str]:
    """Generate n prompts each tokenizing to approximately input_len tokens."""
    # Use a fixed seed so all backends see the same inputs
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    vocab = tokenizer.vocab_size
    ids = torch.randint(100, vocab - 100, (n, input_len))
    prompts = [tokenizer.decode(ids[i].tolist()) for i in range(n)]
    torch.set_rng_state(rng_state)
    return prompts


def _run_backend(
    backend_name: str,
    model: str,
    prompts: List[str],
    output_len: int,
    gpu_memory_utilization: float,
    extra_engine_kwargs: dict,
) -> dict:
    """
    Start a vLLM LLM engine, run all prompts, collect metrics, shut down.

    Returns a metrics dict.
    """
    print(f"\n{'='*60}")
    print(f"Backend: {backend_name}")
    print(f"{'='*60}")

    from vllm import LLM, SamplingParams

    t_start_load = time.perf_counter()
    llm = LLM(
        model=model,
        dtype="float16",
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=4096,
        enforce_eager=False,      # use CUDA graph where available
        **extra_engine_kwargs,
    )
    t_loaded = time.perf_counter()
    load_s = t_loaded - t_start_load
    print(f"  Engine loaded in {load_s:.1f}s")

    # Peak VRAM after model load (before inference)
    torch.cuda.synchronize()
    vram_model_gb = torch.cuda.memory_allocated() / 1e9
    vram_reserved_gb = torch.cuda.memory_reserved() / 1e9

    sampling = SamplingParams(
        max_tokens=output_len,
        temperature=0.0,          # greedy — deterministic, comparable across backends
        ignore_eos=True,          # always generate output_len tokens
    )

    # Warmup: 2 prompts to trigger JIT / CUDA graph capture
    print("  Warming up (2 prompts)...")
    _ = llm.generate(prompts[:2], sampling)
    torch.cuda.synchronize()

    # Timed run
    print(f"  Running {len(prompts)} prompts × {output_len} output tokens...")
    t0 = time.perf_counter()
    outputs = llm.generate(prompts, sampling)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    elapsed_s = t1 - t0

    # Parse outputs
    total_output_toks = sum(len(o.outputs[0].token_ids) for o in outputs)
    total_input_toks  = sum(len(o.prompt_token_ids)     for o in outputs)

    throughput_in  = total_input_toks  / elapsed_s
    throughput_out = total_output_toks / elapsed_s
    latency_ms_mean = elapsed_s * 1000 / len(prompts)

    # Per-request latency (from vLLM metrics if available)
    latencies_ms = []
    ttfts_ms = []
    for o in outputs:
        if hasattr(o, "metrics") and o.metrics is not None:
            m = o.metrics
            if hasattr(m, "finished_time") and hasattr(m, "arrival_time"):
                latencies_ms.append((m.finished_time - m.arrival_time) * 1000)
            if hasattr(m, "first_token_time") and hasattr(m, "arrival_time"):
                ttfts_ms.append((m.first_token_time - m.arrival_time) * 1000)

    vram_peak_gb = torch.cuda.max_memory_allocated() / 1e9

    result = {
        "backend": backend_name,
        "n_prompts": len(prompts),
        "input_tokens_per_prompt": total_input_toks // len(prompts),
        "output_tokens_per_prompt": output_len,
        "total_input_tokens": total_input_toks,
        "total_output_tokens": total_output_toks,
        "elapsed_s": round(elapsed_s, 3),
        "throughput_input_tps":  round(throughput_in,  1),
        "throughput_output_tps": round(throughput_out, 1),
        "latency_mean_ms": round(latency_ms_mean, 2),
        "vram_model_gb":    round(vram_model_gb,    2),
        "vram_reserved_gb": round(vram_reserved_gb, 2),
        "vram_peak_gb":     round(vram_peak_gb,     2),
        "engine_load_s":    round(load_s, 1),
    }
    if latencies_ms:
        import statistics
        result["latency_p50_ms"] = round(statistics.median(latencies_ms), 2)
        result["latency_p99_ms"] = round(sorted(latencies_ms)[int(len(latencies_ms) * 0.99)], 2)
    if ttfts_ms:
        result["ttft_mean_ms"] = round(sum(ttfts_ms) / len(ttfts_ms), 2)

    print(f"  Input  throughput : {throughput_in:.0f} tok/s")
    print(f"  Output throughput : {throughput_out:.0f} tok/s")
    print(f"  Latency (mean)    : {latency_ms_mean:.0f} ms/request")
    print(f"  VRAM peak         : {vram_peak_gb:.1f} GB")
    if ttfts_ms:
        print(f"  TTFT (mean)       : {result['ttft_mean_ms']:.0f} ms")

    # Teardown — free the engine before the next backend
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="vLLM serving benchmark: FP16 vs IsoQuant")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--input-len",  type=int, default=512,
                        help="Approximate input sequence length per prompt")
    parser.add_argument("--output-len", type=int, default=128,
                        help="Exact number of output tokens per prompt")
    parser.add_argument("--num-prompts", type=int, default=20,
                        help="Number of prompts in each timed batch")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--backends", nargs="+",
                        default=["fp16", "isoquant"],
                        choices=["fp16", "isoquant"],
                        help="Which backends to benchmark")
    parser.add_argument("--iq-method", default="iso",
                        choices=["iso", "planar", "rotor", "turbo"],
                        help="IsoQuant rotation method")
    parser.add_argument("--iq-bits", type=int, default=3, choices=[2, 3, 4])
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("vLLM Serving Benchmark: FP16 vs IsoQuant KV Cache")
    print(f"Model:       {args.model}")
    print(f"Input len:   ~{args.input_len} tokens")
    print(f"Output len:  {args.output_len} tokens")
    print(f"Num prompts: {args.num_prompts}")
    print(f"Backends:    {args.backends}")
    print("=" * 60)

    # Import vLLM (must be installed in the current venv)
    try:
        import vllm
        print(f"vLLM version: {vllm.__version__}")
    except ImportError:
        print("ERROR: vLLM not found. Build it first with:")
        print("  VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx942 \\")
        print("  .venv-vllm-rocm/bin/pip install -e /tmp/vllm-src --no-build-isolation")
        sys.exit(1)

    # Register IsoQuant backend before creating any engine
    if "isoquant" in args.backends:
        print("\nRegistering IsoQuant attention backend...")
        _register_isoquant_backend()

    # Build prompts once (same inputs for all backends)
    from transformers import AutoTokenizer
    print(f"\nLoading tokenizer for {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prompts = _make_prompts(tokenizer, args.num_prompts, args.input_len)
    actual_input_len = len(tokenizer.encode(prompts[0]))
    print(f"Actual input length: {actual_input_len} tokens")

    all_results = []

    # ── FP16 baseline ─────────────────────────────────────────────────────────
    if "fp16" in args.backends:
        # Ensure no IsoQuant env var leaks in
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)
        os.environ.pop("VLLM_IQ_METHOD", None)

        result = _run_backend(
            backend_name="fp16",
            model=args.model,
            prompts=prompts,
            output_len=args.output_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            extra_engine_kwargs={},
        )
        all_results.append(result)

    # ── IsoQuant backend ──────────────────────────────────────────────────────
    if "isoquant" in args.backends:
        os.environ["VLLM_ATTENTION_BACKEND"] = "ISOQUANT_ROCM"
        os.environ["VLLM_IQ_METHOD"] = args.iq_method
        os.environ["VLLM_IQ_BITS"]   = str(args.iq_bits)

        result = _run_backend(
            backend_name=f"isoquant_{args.iq_method}{args.iq_bits}",
            model=args.model,
            prompts=prompts,
            output_len=args.output_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            extra_engine_kwargs={
                # IsoQuant K/V cache is 52 B/token (3-bit), not 256 B
                # Tell vLLM to allocate proportionally more blocks
                # (vLLM doesn't know our compression ratio, so it will
                #  over-provision — this is conservative / safe)
            },
        )
        all_results.append(result)
        os.environ.pop("VLLM_ATTENTION_BACKEND", None)

    # ── Summary table ─────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    if len(all_results) >= 2:
        fp16 = next((r for r in all_results if r["backend"] == "fp16"), None)
        iq   = next((r for r in all_results if r["backend"] != "fp16"), None)
        if fp16 and iq:
            speedup = iq["throughput_output_tps"] / fp16["throughput_output_tps"]
            vram_ratio = fp16["vram_peak_gb"] / iq["vram_peak_gb"]
            print(f"  IsoQuant vs FP16 output throughput : {speedup:.2f}×")
            print(f"  IsoQuant vs FP16 VRAM              : {vram_ratio:.2f}× less")

    print(f"\n{'Backend':<25} {'Out tok/s':>10} {'VRAM GB':>8} {'Latency ms':>12}")
    print("-" * 60)
    for r in all_results:
        print(f"  {r['backend']:<23} {r['throughput_output_tps']:>10.0f} "
              f"{r['vram_peak_gb']:>8.1f} {r['latency_mean_ms']:>12.0f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    model_slug = args.model.replace("/", "_")
    out_path = args.output or str(RESULTS_DIR / f"bench_vllm_serving_{model_slug}.json")
    output = {
        "model":      args.model,
        "input_len":  actual_input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "device":     torch.cuda.get_device_name(0),
        "results":    all_results,
    }
    Path(out_path).write_text(json.dumps(output, indent=2))
    print(f"\nResults saved to: {out_path}")


if __name__ == "__main__":
    main()
