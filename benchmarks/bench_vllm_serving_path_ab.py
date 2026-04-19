#!/usr/bin/env python3
"""
bench_vllm_serving_path_ab.py — Eager vs CUDA-graph (default) serving path on identical prompts.

Runs ``bench_vllm_turboquant_ab`` twice: with and without ``--enforce-eager``,
writes JSON comparing throughput and peak VRAM.

  python3 benchmarks/bench_vllm_serving_path_ab.py \\
    --model mistralai/Mistral-7B-v0.1 --only-backend fp16

On ROCm, graphs may be unstable under rocprof; use eager for profiling and
default (no flag) for max-throughput experiments when stable.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--only-backend", default="fp16")
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    p.add_argument("--quant-model", default="")
    p.add_argument("--quantization", default="awq")
    p.add_argument("--output", type=str, default="")
    args = p.parse_args()

    ab = ROOT / "benchmarks" / "bench_vllm_turboquant_ab.py"
    common = [
        sys.executable,
        str(ab),
        "--only-backend",
        args.only_backend,
        "--model",
        args.model,
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--num-prompts",
        str(args.num_prompts),
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
    ]
    if args.quant_model:
        common += ["--quant-model", args.quant_model, "--quantization", args.quantization]

    out_eager = RESULTS / "_serving_path_eager.json"
    out_graph = RESULTS / "_serving_path_graphs.json"

    eager_run = subprocess.run(
        common + ["--enforce-eager", "--output", str(out_eager)],
        check=False,
        cwd=str(ROOT),
    )
    if eager_run.returncode != 0 or not out_eager.is_file():
        raise SystemExit(
            f"eager bench_vllm_turboquant_ab failed (exit {eager_run.returncode}); "
            f"see stderr above. Expected {out_eager}"
        )

    eager_obj = json.loads(out_eager.read_text())
    graph_run = subprocess.run(
        common + ["--output", str(out_graph)],
        check=False,
        cwd=str(ROOT),
    )
    graph_row = None
    graph_error = None
    if graph_run.returncode == 0 and out_graph.is_file():
        graph_obj = json.loads(out_graph.read_text())
        graph_row = graph_obj.get("results", [{}])[0]
    else:
        graph_error = (
            f"graphs subprocess exit={graph_run.returncode}; "
            f"output_exists={out_graph.is_file()}"
        )

    out = {
        "model": args.model,
        "only_backend": args.only_backend,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "eager": eager_obj.get("results", [{}])[0],
        "cuda_graphs_default": graph_row,
        "cuda_graphs_error": graph_error,
    }
    outp = Path(args.output) if args.output else RESULTS / "bench_vllm_serving_path_ab.json"
    outp.write_text(json.dumps(out, indent=2))
    print(outp.read_text())


if __name__ == "__main__":
    main()
