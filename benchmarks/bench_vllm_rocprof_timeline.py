"""
Collect per-mode ROCm kernel timeline summaries for vLLM serving.

Runs `bench_vllm_turboquant_ab.py --only-backend <mode>` under rocprofv2
and writes a compact JSON summary of top kernels by cumulative duration.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
from pathlib import Path


MODES = ("fp16", "turboquant_decompress", "turboquant_fused")


def _parse_kernel_durations(trace_file: Path, top_k: int | None) -> dict:
    kernel_time_us: dict[str, float] = {}
    total_us = 0.0
    with trace_file.open("r", encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        fieldmap = {k.lower(): k for k in (reader.fieldnames or [])}
        kernel_col = (
            fieldmap.get("kernel_name")
            or fieldmap.get("kernelname")
            or fieldmap.get("name")
        )
        start_col = fieldmap.get("start_timestamp")
        end_col = fieldmap.get("end_timestamp")
        dur_ns_col = (
            fieldmap.get("duration(ns)")
            or fieldmap.get("duration_ns")
            or fieldmap.get("duration")
        )

        for row in reader:
            kernel = ""
            dur_ns = 0.0
            if kernel_col:
                kernel = (row.get(kernel_col) or "").strip().strip('"')
            if start_col and end_col and kernel:
                try:
                    s = float(row.get(start_col, "0") or 0)
                    e = float(row.get(end_col, "0") or 0)
                    dur_ns = max(0.0, e - s)
                except ValueError:
                    continue
            elif dur_ns_col and kernel:
                try:
                    dur_ns = float(row.get(dur_ns_col, "0") or 0.0)
                except ValueError:
                    continue
            else:
                continue
            if not kernel:
                continue
            dur_us = dur_ns / 1000.0
            kernel_time_us[kernel] = kernel_time_us.get(kernel, 0.0) + dur_us
            total_us += dur_us

    limit = top_k if top_k is not None else len(kernel_time_us)
    ranked = sorted(kernel_time_us.items(), key=lambda x: x[1], reverse=True)[:limit]
    return {
        "trace_file": str(trace_file),
        "total_kernel_time_us": round(total_us, 3),
        "top_kernels": [
            {
                "kernel": k,
                "time_us": round(v, 3),
                "share_pct": round((100.0 * v / total_us), 2) if total_us else 0.0,
            }
            for k, v in ranked
        ],
    }


def _aggregate_results_csv(out_dir: Path, top_k: int) -> dict:
    """Merge all results_<pid>.csv (parent + EngineCore) into one ranked kernel list."""
    files = sorted(out_dir.glob("results_*.csv"))
    if not files:
        files = sorted(out_dir.glob("**/results_*.csv"))
    merged: dict[str, float] = {}
    for f in files:
        part = _parse_kernel_durations(f, top_k=None)
        for row in part["top_kernels"]:
            k = row["kernel"]
            merged[k] = merged.get(k, 0.0) + row["time_us"]
    total_us = sum(merged.values())
    ranked = sorted(merged.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return {
        "trace_files": [str(x) for x in files],
        "total_kernel_time_us": round(total_us, 3),
        "top_kernels": [
            {
                "kernel": k,
                "time_us": round(v, 3),
                "share_pct": round((100.0 * v / total_us), 2) if total_us else 0.0,
            }
            for k, v in ranked
        ],
    }


def _run_mode(args, mode: str) -> dict:
    out_dir = args.out_dir / f"rocprof_{mode}"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    out_json = args.out_dir / f"bench_vllm_turboquant_ab_{mode}.json"
    cmd = ["rocprofv2", "-d", str(out_dir)]
    if args.with_hip_trace:
        cmd.append("--hip-trace")
    cmd.append("--kernel-trace")
    cmd += [
        str(args.python),
        str(args.ab_script),
        "--only-backend",
        mode,
        "--model",
        args.model,
        "--input-len",
        str(args.input_len),
        "--output-len",
        str(args.output_len),
        "--num-prompts",
        str(args.num_prompts),
        "--output",
        str(out_json),
    ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    # Run from workspace root so `amd-experiments/vllm/` is not on sys.path (shadows installed vLLM).
    workspace_root = Path(__file__).resolve().parents[2]
    subprocess.run(cmd, check=True, cwd=str(workspace_root))
    parsed = _aggregate_results_csv(out_dir, top_k=args.top_k)
    if not parsed.get("trace_files"):
        return {"mode": mode, "error": "No rocprof results_*.csv found", "out_dir": str(out_dir)}
    parsed["mode"] = mode
    parsed["ab_result_json"] = str(out_json)
    return parsed


def main():
    p = argparse.ArgumentParser(description="Collect rocprof timeline per backend mode.")
    p.add_argument("--python", type=Path, default=Path("/root/workspace/amd-experiments/.venv-vllm-rocm/bin/python"))
    p.add_argument("--ab-script", type=Path, default=Path("/root/workspace/amd-experiments/benchmarks/bench_vllm_turboquant_ab.py"))
    p.add_argument("--out-dir", type=Path, default=Path("/root/workspace/amd-experiments/results"))
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--mode", choices=MODES, default="", help="Run one mode only")
    p.add_argument(
        "--with-hip-trace",
        action="store_true",
        help="Also record HIP API trace (heavy; can destabilize vLLM+rocprof on some stacks).",
    )
    p.add_argument(
        "--no-enforce-eager",
        action="store_false",
        dest="enforce_eager",
        help="Allow CUDA graphs while profiling (may fail or produce huge traces under rocprof).",
    )
    p.set_defaults(enforce_eager=True)
    args = p.parse_args()

    modes = [args.mode] if args.mode else list(MODES)
    all_results = [_run_mode(args, m) for m in modes]
    summary_path = args.out_dir / "bench_vllm_rocprof_timeline_summary.json"
    meta = {
        "rocprof_kernel_trace": True,
        "rocprof_hip_trace": bool(args.with_hip_trace),
        "enforce_eager": bool(args.enforce_eager),
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
    }
    summary_path.write_text(
        json.dumps({"meta": meta, "results": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved timeline summary -> {summary_path}")


if __name__ == "__main__":
    main()
