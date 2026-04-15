"""
bench_vllm_turboquant_ab.py — vLLM TurboQuant backend A/B benchmark.

Compares three serving configurations on identical prompts:
  1) fp16 baseline
  2) turboquant_decompress (VLLM_TQ_USE_FUSED_KERNEL=0)
  3) turboquant_fused      (VLLM_TQ_USE_FUSED_KERNEL=1)
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import List

import torch

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _make_prompts(tokenizer, n: int, input_len: int) -> List[str]:
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    vocab = tokenizer.vocab_size
    ids = torch.randint(100, vocab - 100, (n, input_len))
    prompts = [tokenizer.decode(ids[i].tolist()) for i in range(n)]
    torch.set_rng_state(rng_state)
    return prompts


def _run_backend(
    name: str,
    model: str,
    prompts: List[str],
    output_len: int,
    gpu_mem: float,
    enforce_eager: bool = False,
):
    from vllm import LLM, SamplingParams
    try:
        from vllm.platforms import current_platform
        if not getattr(current_platform, "device_type", None):
            current_platform.device_type = "cuda"
    except Exception:
        pass

    t0 = time.perf_counter()
    llm = LLM(
        model=model,
        dtype="float16",
        gpu_memory_utilization=gpu_mem,
        max_model_len=4096,
        enforce_eager=enforce_eager,
    )
    load_s = time.perf_counter() - t0
    sampling = SamplingParams(max_tokens=output_len, temperature=0.0, ignore_eos=True)

    _ = llm.generate(prompts[:2], sampling)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sampling)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_out = sum(len(o.outputs[0].token_ids) for o in outs)
    out_tps = total_out / elapsed
    peak_gb = torch.cuda.max_memory_allocated() / 1e9

    result = {
        "backend": name,
        "elapsed_s": round(elapsed, 3),
        "throughput_output_tps": round(out_tps, 1),
        "vram_peak_gb": round(peak_gb, 2),
        "engine_load_s": round(load_s, 2),
        "enforce_eager": enforce_eager,
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main():
    parser = argparse.ArgumentParser(description="vLLM TurboQuant A/B fused toggle benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--only-backend",
        choices=["fp16", "turboquant_decompress", "turboquant_fused"],
        default="",
        help="Run only one backend mode (useful for profiling/timeline collection).",
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graph capture (recommended under rocprofv2 to avoid huge traces / launch failures).",
    )
    args = parser.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    prompts = _make_prompts(tok, args.num_prompts, args.input_len)

    results = []

    run_modes = (
        [args.only_backend]
        if args.only_backend
        else ["fp16", "turboquant_decompress", "turboquant_fused"]
    )
    for mode in run_modes:
        for env_key in (
            "VLLM_ATTENTION_BACKEND",
            "VLLM_KV_CACHE_DTYPE",
            "VLLM_TQ_USE_FUSED_KERNEL",
        ):
            os.environ.pop(env_key, None)
        if mode != "fp16":
            os.environ["VLLM_ATTENTION_BACKEND"] = "TURBOQUANT_ROCM"
            os.environ["VLLM_KV_CACHE_DTYPE"] = "tq3"
            os.environ["VLLM_TQ_USE_FUSED_KERNEL"] = "1" if mode == "turboquant_fused" else "0"
        results.append(
            _run_backend(
                mode,
                args.model,
                prompts,
                args.output_len,
                args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
            )
        )

    out = {
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "device": torch.cuda.get_device_name(0),
        "enforce_eager": args.enforce_eager,
        "results": results,
    }
    out_path = Path(args.output) if args.output else (RESULTS_DIR / f"bench_vllm_turboquant_ab_{args.model.replace('/', '_')}.json")
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
