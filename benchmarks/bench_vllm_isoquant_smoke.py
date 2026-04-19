"""
bench_vllm_isoquant_smoke.py — Short vLLM decode with iq3 KV (IsoQuant packer).

Requires ROCm vLLM, ``bash scripts/install_isoquant_vllm_backend.sh``, and::

    PYTHONPATH=<repo>/kernels:<repo>

Run from repo root (so ``tq_backends`` resolves).
"""

from __future__ import annotations

import argparse
import gc
import json
import inspect
import sys
import time
from pathlib import Path
from typing import Any

import torch

_ROOT = Path(__file__).resolve().parents[1]


def _ensure_paths() -> None:
    for p in (_ROOT / "kernels", _ROOT):
        s = str(p)
        if s not in sys.path:
            sys.path.insert(0, s)


def _filter_llm_kwargs(llm_cls: Any, llm_kw: dict) -> tuple[dict, list[str]]:
    try:
        sig = inspect.signature(llm_cls.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
    except (TypeError, ValueError):
        return dict(llm_kw), []
    filtered = {k: v for k, v in llm_kw.items() if k in allowed and v is not None}
    dropped = sorted(k for k in llm_kw if k not in filtered)
    return filtered, dropped


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM IsoQuant iq3 smoke decode")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--input-len", type=int, default=64)
    parser.add_argument("--output-len", type=int, default=16)
    parser.add_argument("--max-model-len", type=int, default=512)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--output",
        type=Path,
        default=_ROOT / "results" / "bench_vllm_isoquant_smoke.json",
    )
    args = parser.parse_args()

    _ensure_paths()
    from vllm_isoquant_registry import register_isoquant_rocm_backend

    if not register_isoquant_rocm_backend():
        raise SystemExit("IsoQuant registration failed")

    from vllm import LLM, SamplingParams

    try:
        from vllm.platforms import current_platform

        if not getattr(current_platform, "device_type", None):
            current_platform.device_type = "cuda"
    except Exception:
        pass

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    torch.manual_seed(0)
    ids = torch.randint(100, tokenizer.vocab_size - 100, (args.input_len,))
    prompt = tokenizer.decode(ids.tolist())

    llm_kw: dict = {
        "model": args.model,
        "dtype": "float16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
        "kv_cache_dtype": "iq3",
        "attention_config": {"backend": "CUSTOM"},
    }
    llm_kw, dropped = _filter_llm_kwargs(LLM, llm_kw)

    t0 = time.perf_counter()
    llm = LLM(**llm_kw)
    load_s = time.perf_counter() - t0

    sampling = SamplingParams(max_tokens=args.output_len, temperature=0.0, ignore_eos=True)
    t1 = time.perf_counter()
    outs = llm.generate([prompt], sampling)
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t1

    n_tok = len(outs[0].outputs[0].token_ids)
    out = {
        "model": args.model,
        "kv_cache_dtype": "iq3",
        "engine_load_s": round(load_s, 2),
        "generate_s": round(gen_s, 4),
        "output_tokens": n_tok,
        "output_tok_per_s": round(n_tok / gen_s, 2) if gen_s > 0 else 0.0,
        "llm_kwargs_dropped": dropped,
        "enforce_eager": args.enforce_eager,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))

    del llm
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
