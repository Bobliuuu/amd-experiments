#!/usr/bin/env python3
"""
bench_vllm_quant_quality_smoke.py — Greedy decode token match vs FP16 baseline (vLLM).

Runs two short vLLM.generate passes on the same prompt: FP16 reference vs
quantized checkpoint. Reports prefix length where token_ids diverge (0 = full match).

Requires vLLM + optional TURBOQUANT env (not used here; FP16 vs quant weights only).

  python3 benchmarks/bench_vllm_quant_quality_smoke.py \\
    --fp16-model mistralai/Mistral-7B-v0.1 \\
    --quant-model TheBloke/Mistral-7B-v0.1-AWQ \\
    --quantization awq --max-tokens 32
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
sys.path.insert(0, str(ROOT / "benchmarks"))


def _run_cell(model: str, quantization: str | None, prompt: str, max_tokens: int, gpu_mem: float) -> list[int]:
    from bench_vllm_turboquant_ab import _filter_llm_init_kwargs

    from vllm import LLM, SamplingParams

    try:
        from vllm.platforms import current_platform

        if not getattr(current_platform, "device_type", None):
            current_platform.device_type = "cuda"
    except Exception:
        pass

    kw = {
        "model": model,
        "dtype": "float16",
        "gpu_memory_utilization": gpu_mem,
        "max_model_len": 2048,
        "enforce_eager": True,
    }
    if quantization:
        kw["quantization"] = quantization
    kw, _ = _filter_llm_init_kwargs(LLM, kw)
    llm = LLM(**kw)
    sp = SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True)
    out = llm.generate([prompt], sp)
    ids = list(out[0].outputs[0].token_ids)
    del llm
    return ids


def _first_mismatch(a: list[int], b: list[int]) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return n if len(a) == len(b) else min(len(a), len(b))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--fp16-model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--quant-model", required=True)
    p.add_argument("--quantization", default="awq")
    p.add_argument("--prompt", default="The capital of France is")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = p.parse_args()

    ref_ids = _run_cell(args.fp16_model, None, args.prompt, args.max_tokens, args.gpu_memory_utilization)
    q_ids = _run_cell(args.quant_model, args.quantization, args.prompt, args.max_tokens, args.gpu_memory_utilization)
    mismatch_at = _first_mismatch(ref_ids, q_ids)

    out = {
        "fp16_model": args.fp16_model,
        "quant_model": args.quant_model,
        "quantization": args.quantization,
        "prompt": args.prompt,
        "max_tokens": args.max_tokens,
        "ref_len": len(ref_ids),
        "quant_len": len(q_ids),
        "first_mismatch_index": mismatch_at,
        "full_match": bool(
            mismatch_at == len(ref_ids) and len(ref_ids) == len(q_ids)
        ),
    }
    outp = RESULTS / "bench_vllm_quant_quality_smoke.json"
    outp.write_text(json.dumps(out, indent=2))
    print(outp.read_text())


if __name__ == "__main__":
    main()
