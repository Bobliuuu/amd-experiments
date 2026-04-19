"""
bench_vllm_turboquant_ab.py — vLLM TurboQuant backend A/B benchmark.

Compares serving configurations on identical prompts:

- ``fp16`` — FP16 weights, default attention / KV.
- ``turboquant_decompress`` / ``turboquant_fused`` — TQ3 KV + optional fused Triton decode.
- ``quant_*`` — AWQ/GPTQ weights (``--quant-model``) with FP16 or TQ3 KV.

Requires a ROCm-compatible vLLM install, backend copied via
``scripts/install_turboquant_vllm_backend.sh``, and ``kernels/`` on PYTHONPATH.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import inspect
from pathlib import Path
from typing import Any, List, Optional

import torch

RESULTS_DIR = Path(__file__).resolve().parents[1] / "results"
RESULTS_DIR.mkdir(exist_ok=True)
_ROOT = Path(__file__).resolve().parents[1]


def _ensure_import_paths() -> None:
    k = str(_ROOT / "kernels")
    r = str(_ROOT)
    for p in (k, r):
        if p not in sys.path:
            sys.path.insert(0, p)


def _make_prompts(tokenizer, n: int, input_len: int) -> List[str]:
    rng_state = torch.get_rng_state()
    torch.manual_seed(0)
    vocab = tokenizer.vocab_size
    ids = torch.randint(100, vocab - 100, (n, input_len))
    prompts = [tokenizer.decode(ids[i].tolist()) for i in range(n)]
    torch.set_rng_state(rng_state)
    return prompts


def _filter_llm_init_kwargs(llm_cls: Any, llm_kw: dict) -> tuple[dict, list[str]]:
    """Drop kwargs this vLLM build does not accept (version drift on ROCm)."""
    try:
        sig = inspect.signature(llm_cls.__init__)
        allowed = set(sig.parameters.keys()) - {"self"}
    except (TypeError, ValueError):
        return dict(llm_kw), []
    filtered = {k: v for k, v in llm_kw.items() if k in allowed and v is not None}
    dropped = sorted(k for k in llm_kw if k not in filtered)
    return filtered, dropped


def run_turboquant_vllm_cell(
    name: str,
    model: str,
    prompts: List[str],
    output_len: int,
    gpu_mem: float,
    enforce_eager: bool = False,
    max_model_len: int = 4096,
    use_turboquant: bool = False,
    fused: bool = False,
    quantization: Optional[str] = None,
    max_num_batched_tokens: Optional[int] = None,
    max_num_seqs: Optional[int] = None,
) -> dict:
    """
    One vLLM load + generate timing. Registers TURBOQUANT_ROCM when use_turboquant.

    Optional weight quantization (``quantization``) uses the same ``model`` HF id
    (caller should pass an AWQ/GPTQ checkpoint when using ``quantization='awq'``).
    """
    _ensure_import_paths()
    if use_turboquant:
        from vllm_turboquant_registry import register_turboquant_rocm_backend

        register_turboquant_rocm_backend()

    from vllm import LLM, SamplingParams

    try:
        from vllm.platforms import current_platform

        if not getattr(current_platform, "device_type", None):
            current_platform.device_type = "cuda"
    except Exception:
        pass

    llm_kw: dict = {
        "model": model,
        "dtype": "float16",
        "gpu_memory_utilization": gpu_mem,
        "max_model_len": max_model_len,
        "enforce_eager": enforce_eager,
    }
    if use_turboquant:
        llm_kw["kv_cache_dtype"] = "tq3"
        # V1: TurboQuant bridge registers AttentionBackendEnum.CUSTOM. Use in-process
        # EngineCore (VLLM_ENABLE_V1_MULTIPROCESSING=0) so workers see the registration.
        llm_kw["attention_config"] = {"backend": "CUSTOM"}
    if quantization:
        llm_kw["quantization"] = quantization
    if max_num_batched_tokens is not None:
        llm_kw["max_num_batched_tokens"] = max_num_batched_tokens
    if max_num_seqs is not None:
        llm_kw["max_num_seqs"] = max_num_seqs

    llm_kw, dropped_params = _filter_llm_init_kwargs(LLM, llm_kw)

    t0 = time.perf_counter()
    try:
        llm = LLM(**llm_kw)
    except TypeError:
        if use_turboquant:
            llm_kw.pop("kv_cache_dtype", None)
            llm_kw.pop("attention_config", None)
            llm_kw, dropped_params = _filter_llm_init_kwargs(LLM, llm_kw)
        llm = LLM(**llm_kw)
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
        "max_model_len": max_model_len,
        "num_prompts": len(prompts),
        "use_turboquant": use_turboquant,
        "fused": fused,
        "weight_quantization": quantization or "",
        "llm_kwargs_dropped": dropped_params,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
    }

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM TurboQuant A/B fused toggle benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-len", type=int, default=128)
    parser.add_argument("--num-prompts", type=int, default=20)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--output", type=str, default="")
    parser.add_argument(
        "--quant-model",
        default="",
        help="HF model id for weight-quantized cells (AWQ/GPTQ checkpoint).",
    )
    parser.add_argument(
        "--quantization",
        default="awq",
        help="vLLM weight quantization string when using --quant-model (e.g. awq, gptq).",
    )
    parser.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=0,
        help="If >0, pass max_num_batched_tokens to LLM() when supported.",
    )
    parser.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="If >0, pass max_num_seqs to LLM() when supported.",
    )
    parser.add_argument(
        "--only-backend",
        choices=[
            "fp16",
            "turboquant_decompress",
            "turboquant_fused",
            "quant_fp16_kv",
            "quant_turboquant_decompress",
            "quant_turboquant_fused",
        ],
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

    default_modes = ["fp16", "turboquant_decompress", "turboquant_fused"]
    if args.quant_model:
        default_modes += [
            "quant_fp16_kv",
            "quant_turboquant_decompress",
            "quant_turboquant_fused",
        ]
    run_modes = [args.only_backend] if args.only_backend else default_modes
    max_bt = args.max_num_batched_tokens if args.max_num_batched_tokens > 0 else None
    max_seqs = args.max_num_seqs if args.max_num_seqs > 0 else None

    for mode in run_modes:
        for env_key in ("VLLM_TQ_USE_FUSED_KERNEL",):
            os.environ.pop(env_key, None)

        use_tq = mode not in ("fp16", "quant_fp16_kv")
        fused = mode in ("turboquant_fused", "quant_turboquant_fused")
        quant = None
        eff_model = args.model

        if mode.startswith("quant_"):
            if not args.quant_model:
                results.append(
                    {
                        "backend": mode,
                        "error": "requires --quant-model (AWQ/GPTQ HF id)",
                        "use_turboquant": use_tq,
                        "fused": fused,
                    }
                )
                continue
            quant = args.quantization
            eff_model = args.quant_model

        if use_tq:
            os.environ["VLLM_TQ_USE_FUSED_KERNEL"] = "1" if fused else "0"

        results.append(
            run_turboquant_vllm_cell(
                mode,
                eff_model,
                prompts,
                args.output_len,
                args.gpu_memory_utilization,
                enforce_eager=args.enforce_eager,
                max_model_len=args.max_model_len,
                use_turboquant=use_tq,
                fused=fused,
                quantization=quant,
                max_num_batched_tokens=max_bt,
                max_num_seqs=max_seqs,
            )
        )

    out = {
        "model": args.model,
        "quant_model": args.quant_model or None,
        "quantization": args.quantization,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "max_model_len": args.max_model_len,
        "max_num_batched_tokens": max_bt,
        "max_num_seqs": max_seqs,
        "device": torch.cuda.get_device_name(0),
        "enforce_eager": args.enforce_eager,
        "results": results,
    }
    out_path = (
        Path(args.output)
        if args.output
        else (RESULTS_DIR / f"bench_vllm_turboquant_ab_{args.model.replace('/', '_')}.json")
    )
    out_path.write_text(json.dumps(out, indent=2))
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()
