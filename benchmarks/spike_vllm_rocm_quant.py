#!/usr/bin/env python3
"""
spike_vllm_rocm_quant.py — Probe which vLLM LLM() kwargs work on this ROCm + gfx942 stack.

Writes results/spike_vllm_rocm_quant.json with per-trial load_ok, error, and LLM.__init__ param names.

**Interpreter:** vLLM and torch live in this repo's ``.venv`` (not system ``python3``).
Use the venv explicitly:

  amd-experiments/.venv/bin/python benchmarks/spike_vllm_rocm_quant.py
  amd-experiments/.venv/bin/python benchmarks/spike_vllm_rocm_quant.py --inspect-only
  amd-experiments/.venv/bin/python benchmarks/spike_vllm_rocm_quant.py --quant-model TheBloke/Mistral-7B-v0.1-AWQ
"""

from __future__ import annotations

import argparse
import gc
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)
_VENV_PYTHON = ROOT / ".venv" / "bin" / "python"


def _suggested_python() -> str:
    if _VENV_PYTHON.is_file():
        return str(_VENV_PYTHON)
    return "<path-to-python-with-vllm>"


def _filter_only(name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Classify kwargs vs LLM.__init__ without constructing LLM."""
    try:
        from vllm import LLM
    except ImportError as e:
        return {"trial": name, "load_ok": False, "error": f"import vllm: {e}"}

    sig = inspect.signature(LLM.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    dropped = sorted(set(kwargs) - set(filtered))
    return {
        "trial": name,
        "load_ok": None,
        "inspect_only": True,
        "error": None,
        "kwargs_sent": filtered,
        "kwargs_dropped_unsupported": dropped,
    }


def _try_llm(name: str, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    try:
        from vllm import LLM
    except ImportError as e:
        return {"trial": name, "load_ok": False, "error": f"import vllm: {e}"}

    try:
        from vllm.platforms import current_platform

        if not getattr(current_platform, "device_type", None):
            current_platform.device_type = "cuda"
    except Exception:
        pass

    sig = inspect.signature(LLM.__init__)
    allowed = set(sig.parameters.keys()) - {"self"}
    filtered = {k: v for k, v in kwargs.items() if k in allowed and v is not None}
    dropped = sorted(set(kwargs) - set(filtered))

    err = None
    load_ok = False
    try:
        llm = LLM(**filtered)
        del llm
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        load_ok = True
    except Exception as e:
        err = repr(e)

    return {
        "trial": name,
        "load_ok": load_ok,
        "error": err,
        "kwargs_sent": filtered,
        "kwargs_dropped_unsupported": dropped,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inspect-only",
        action="store_true",
        help="Do not construct LLM(); only record filtered kwargs vs this vLLM build (fast).",
    )
    p.add_argument("--fp16-model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument(
        "--quant-model",
        default="",
        help="HF id for AWQ/GPTQ checkpoint; if empty, AWQ/GPTQ trials are skipped.",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    p.add_argument("--max-model-len", type=int, default=2048)
    p.add_argument("--enforce-eager", action="store_true")
    args = p.parse_args()

    try:
        from vllm import LLM

        llm_params = sorted(
            k for k in inspect.signature(LLM.__init__).parameters if k != "self"
        )
    except ImportError as e:
        llm_params = []
        print(
            f"[spike_vllm_rocm_quant] import vllm failed: {e}\n"
            f"  Use this repo's venv (has torch+vllm): {_suggested_python()} benchmarks/spike_vllm_rocm_quant.py",
            file=sys.stderr,
        )

    trials: List[Dict[str, Any]] = []
    try_fn = _filter_only if args.inspect_only else _try_llm

    base = {
        "dtype": "float16",
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "enforce_eager": args.enforce_eager,
    }

    trials.append(
        try_fn(
            "fp16_baseline",
            {"model": args.fp16_model, **base},
        )
    )

    for quant, qmodel, trial_suffix in (
        ("awq", args.quant_model, "awq"),
        ("gptq", args.quant_model, "gptq"),
    ):
        if not qmodel:
            trials.append(
                {
                    "trial": f"{trial_suffix}_skipped",
                    "load_ok": False,
                    "error": "no --quant-model",
                    "kwargs_sent": {},
                    "kwargs_dropped_unsupported": [],
                }
            )
            continue
        trials.append(
            try_fn(
                f"{trial_suffix}_weights",
                {"model": qmodel, "quantization": quant, **base},
            )
        )

    # FP8 KV / dtype trials depend on vLLM + checkpoint; optional probe
    trials.append(
        try_fn(
            "fp16_max_batched_tokens_4096",
            {
                "model": args.fp16_model,
                **base,
                "max_num_batched_tokens": 4096,
            },
        )
    )

    out = {
        "python_executable": sys.executable,
        "suggested_venv_python": str(_VENV_PYTHON) if _VENV_PYTHON.is_file() else None,
        "inspect_only": bool(args.inspect_only),
        "fp16_model": args.fp16_model,
        "quant_model": args.quant_model or None,
        "vllm_llm_init_params": llm_params,
        "torch_device": _device_name(),
        "trials": trials,
    }
    outp = RESULTS / "spike_vllm_rocm_quant.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(outp.read_text())


def _device_name() -> str:
    try:
        import torch

        if torch.cuda.is_available():
            return str(torch.cuda.get_device_name(0))
    except Exception:
        pass
    return ""


if __name__ == "__main__":
    main()
