#!/usr/bin/env python3
"""
Minimal vLLM decode benchmark to get *something* working for bottleneck discussion.

Tries each --model candidate in order (subprocess per attempt so CUDA memory resets).
Uses a tiny decode shape and scripts/estimate_vllm_safe_gpu_mem_frac.py by default.

  python3 benchmarks/decode_bottleneck_smoke.py
  python3 benchmarks/decode_bottleneck_smoke.py --models TinyLlama/TinyLlama-1.1B-Chat-v1.0

Output: results/decode_bottleneck_smoke.json (throughput + which model ran).
Kernel split still needs rocprof (see docs/decode_whole_step_amdahl_outcome.md).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ESTIMATE = ROOT / "scripts" / "estimate_vllm_safe_gpu_mem_frac.py"


def _default_python() -> Path:
    for p in (
        ROOT / ".benchmark_mi300_vllm_frozen" / ".venv" / "bin" / "python3",
        ROOT / ".venv" / "bin" / "python3",
    ):
        if p.is_file():
            return p
    return Path(sys.executable)


def _auto_gpu_mem(py: Path) -> float:
    try:
        out = subprocess.check_output([str(py), str(ESTIMATE)], text=True, timeout=60)
        return float(out.strip())
    except Exception:
        return 0.12


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--models",
        type=str,
        default="mistralai/Mistral-7B-v0.1,TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Comma-separated HF ids; first success wins.",
    )
    p.add_argument("--python", type=Path, default=_default_python())
    p.add_argument("--input-len", type=int, default=128)
    p.add_argument("--output-len", type=int, default=32)
    p.add_argument("--num-prompts", type=int, default=4)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=-1.0,
        help="If <=0, run scripts/estimate_vllm_safe_gpu_mem_frac.py in this venv.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=RESULTS / "decode_bottleneck_smoke.json",
    )
    args = p.parse_args()

    py = args.python
    if not py.is_file():
        print(f"[decode_bottleneck_smoke] python not found: {py}", file=sys.stderr)
        return 1

    gpu_mem = (
        args.gpu_memory_utilization
        if args.gpu_memory_utilization > 0
        else _auto_gpu_mem(py)
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'kernels'}:{ROOT}" + (
        f":{env['PYTHONPATH']}" if env.get("PYTHONPATH") else ""
    )
    env.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    ab = ROOT / "benchmarks" / "bench_vllm_turboquant_ab.py"
    attempts: list[dict] = []

    for model in models:
        tmp = RESULTS / f"_bottleneck_smoke_{model.replace('/', '_')}.json"
        cmd = [
            str(py),
            str(ab),
            "--model",
            model,
            "--input-len",
            str(args.input_len),
            "--output-len",
            str(args.output_len),
            "--num-prompts",
            str(args.num_prompts),
            "--max-model-len",
            str(args.max_model_len),
            "--gpu-memory-utilization",
            str(gpu_mem),
            "--enforce-eager",
            "--only-backend",
            "fp16",
            "--output",
            str(tmp),
        ]
        r = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True, text=True)
        row = {
            "model": model,
            "exit_code": r.returncode,
            "stderr_tail": (r.stderr or "")[-4000:],
        }
        attempts.append(row)
        if r.returncode == 0 and tmp.is_file():
            doc = json.loads(tmp.read_text())
            out = {
                "decode_bottleneck_smoke": True,
                "picked_model": model,
                "gpu_memory_utilization": gpu_mem,
                "input_len": args.input_len,
                "output_len": args.output_len,
                "num_prompts": args.num_prompts,
                "max_model_len": args.max_model_len,
                "attempts": attempts,
                "bench": doc,
                "note": (
                    "FP16-only micro-decode for stack sanity. For MI300X Mistral kv-heavy "
                    "GEMM vs attention split see results/decode_whole_step_rocprof_bucket_compare.json "
                    "and docs/decode_whole_step_amdahl_outcome.md."
                ),
            }
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(out, indent=2))
            print(f"Saved {args.output} (model={model})")
            try:
                tmp.unlink()
            except OSError:
                pass
            return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "decode_bottleneck_smoke": True,
                "picked_model": None,
                "gpu_memory_utilization": gpu_mem,
                "attempts": attempts,
                "error": "all model candidates failed",
            },
            indent=2,
        )
    )
    print("[decode_bottleneck_smoke] all candidates failed; see attempts in JSON", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
