#!/usr/bin/env python3
"""
Run two serving benchmarks (FP16 vs IsoQuant) and write summary outputs.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "kernels"))
from cache_utils import add_swa_args, print_swa_status, vllm_swa_warn


def _run_case(cmd: List[str], cwd: str) -> Dict:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\n{proc.stdout}\n{proc.stderr}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    joined = "\n".join(lines)
    start = joined.find("{")
    end = joined.rfind("}")
    if start < 0 or end < 0 or end <= start:
        raise RuntimeError(f"Could not parse JSON output:\n{proc.stdout}")
    return json.loads(joined[start : end + 1])


def _mk_cmd(args: argparse.Namespace, mode: str, out_json: str) -> List[str]:
    base = [
        sys.executable,
        "benchmarks/bench_vllm_serving_isoquant.py",
        "--model",
        args.model,
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--dtype",
        args.dtype,
        "--max-model-len",
        str(args.max_model_len),
        "--gpu-memory-utilization",
        str(args.gpu_memory_utilization),
        "--kernels-path",
        args.kernels_path,
        "--server-cwd",
        args.server_cwd,
        "--num-requests",
        str(args.num_requests),
        "--concurrency",
        str(args.concurrency),
        "--prompt-tokens",
        str(args.prompt_tokens),
        "--max-tokens",
        str(args.max_tokens),
        "--temperature",
        str(args.temperature),
        "--request-timeout-s",
        str(args.request_timeout_s),
        "--server-start-timeout-s",
        str(args.server_start_timeout_s),
        "--output-json",
        out_json,
        "--python-bin",
        args.python_bin,
        "--swa",
        args.swa,
        "--window",
        str(args.window),
    ]
    if mode == "fp16":
        base += ["--kv-cache-dtype", "auto", "--attention-backend", ""]
    elif mode == "isoquant":
        base += [
            "--kv-cache-dtype",
            args.kv_cache_dtype,
            "--attention-backend",
            "ISOQUANT_ROCM",
            "--iq-method",
            args.iq_method,
            "--iq-bits",
            str(args.iq_bits),
        ]
    else:
        raise ValueError(mode)
    return base


def _format_md(fp16: Dict, iso: Dict) -> str:
    fp16_tps = fp16.get("throughput_tokens_per_s", 0.0)
    iso_tps = iso.get("throughput_tokens_per_s", 0.0)
    speedup = (iso_tps / fp16_tps) if fp16_tps > 0 else 0.0
    return (
        "# vLLM Serving Comparison\n\n"
        "| Variant | Throughput (tok/s) | P50 Latency (s) | P95 Latency (s) | Success |\n"
        "|---|---:|---:|---:|---:|\n"
        f"| FP16 | {fp16_tps:.3f} | {fp16.get('latency_p50_s', 0.0):.3f} | "
        f"{fp16.get('latency_p95_s', 0.0):.3f} | "
        f"{fp16.get('successful_requests', 0)}/{fp16.get('num_requests', 0)} |\n"
        f"| IsoQuant ({iso.get('iq_method', 'iso')}{iso.get('iq_bits', 3)}) | {iso_tps:.3f} | "
        f"{iso.get('latency_p50_s', 0.0):.3f} | {iso.get('latency_p95_s', 0.0):.3f} | "
        f"{iso.get('successful_requests', 0)}/{iso.get('num_requests', 0)} |\n\n"
        f"- Throughput ratio (IsoQuant / FP16): **{speedup:.3f}x**\n"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--dtype", default="float16")
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--kv-cache-dtype", default="iso3")
    p.add_argument("--iq-method", choices=["iso", "planar"], default="iso")
    p.add_argument("--iq-bits", type=int, choices=[3, 4], default=3)
    p.add_argument("--kernels-path", default="/root/workspace/amd-experiments/kernels")
    p.add_argument("--server-cwd", default="/tmp")
    p.add_argument("--num-requests", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--prompt-tokens", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--request-timeout-s", type=float, default=180.0)
    p.add_argument("--server-start-timeout-s", type=float, default=240.0)
    p.add_argument("--output-prefix", default="results/vllm_compare")
    p.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python executable used by serving benchmark to start vLLM server.",
    )
    add_swa_args(p)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    print_swa_status(args.swa, args.window if args.swa == "on" else None)
    vllm_swa_warn(args.swa, args.max_model_len)
    out_prefix = Path(args.output_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    fp16_json = f"{args.output_prefix}_fp16.json"
    iso_json = f"{args.output_prefix}_isoquant.json"
    summary_json = f"{args.output_prefix}_summary.json"
    summary_md = f"{args.output_prefix}_summary.md"

    fp16_res = _run_case(_mk_cmd(args, "fp16", fp16_json), cwd="/root/workspace/amd-experiments")
    iso_res = _run_case(_mk_cmd(args, "isoquant", iso_json), cwd="/root/workspace/amd-experiments")

    combined = {"fp16": fp16_res, "isoquant": iso_res}
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(combined, f, indent=2)
    with open(summary_md, "w", encoding="utf-8") as f:
        f.write(_format_md(fp16_res, iso_res))

    print(json.dumps(combined, indent=2))
    print(f"Wrote: {fp16_json}, {iso_json}, {summary_json}, {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
