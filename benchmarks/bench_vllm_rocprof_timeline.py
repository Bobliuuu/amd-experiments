"""
Collect per-mode ROCm kernel timeline summaries for vLLM serving.

Runs `bench_vllm_turboquant_ab.py --only-backend <mode>` under rocprofv2
and writes a compact JSON summary of top kernels by cumulative duration.

Optional hardware counters (MI300X / gfx942): pass ``--pmc FETCH_SIZE,...`` to
append ``rocprofv2 --pmc ...`` alongside ``--kernel-trace``. On many VF
(virtual function) partitions SQ counter sets exceed the hardware limit and
rocprof fails with ``ROCPROFILER_STATUS_ERROR_PROFILE_EXCEEDS_HW_LIMIT`` — use
timeline-only (default) or a smaller counter set; see ``benchmarks/profile_rocprof.py``.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO_ROOT / "kernels"))
from cache_utils import add_swa_args, print_swa_status, vllm_swa_warn  # noqa: E402

_DEFAULT_VLLM_PYTHON = _REPO_ROOT / ".benchmark_mi300_vllm_frozen" / ".venv" / "bin" / "python"
if not _DEFAULT_VLLM_PYTHON.is_file():
    _DEFAULT_VLLM_PYTHON = _REPO_ROOT / ".venv" / "bin" / "python"
if not _DEFAULT_VLLM_PYTHON.is_file():
    _DEFAULT_VLLM_PYTHON = Path(sys.executable)


MODES = (
    "fp16",
    "turboquant_decompress",
    "turboquant_fused",
    "quant_fp16_kv",
    "quant_turboquant_decompress",
    "quant_turboquant_fused",
)


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
    if args.pmc.strip():
        cmd.extend(["--pmc", args.pmc.strip()])
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
    if mode.startswith("quant_"):
        if not args.quant_model:
            raise ValueError(f"Mode {mode} requires --quant-model (HF AWQ/GPTQ id)")
        cmd += [
            "--quant-model",
            args.quant_model,
            "--quantization",
            args.quantization,
        ]
    if args.enforce_eager:
        cmd.append("--enforce-eager")
    if args.max_model_len > 0:
        cmd += ["--max-model-len", str(args.max_model_len)]
    if args.gpu_memory_utilization > 0:
        cmd += ["--gpu-memory-utilization", str(args.gpu_memory_utilization)]
    if args.max_num_batched_tokens > 0:
        cmd += ["--max-num-batched-tokens", str(args.max_num_batched_tokens)]
    if args.max_num_seqs > 0:
        cmd += ["--max-num-seqs", str(args.max_num_seqs)]
    cmd += ["--swa", args.swa, "--window", str(args.window)]
    # Run from amd-experiments root; do not cwd to a path that puts a `vllm/` stub before site-packages.
    repo_root = Path(__file__).resolve().parents[1]
    subprocess.run(cmd, check=True, cwd=str(repo_root))
    parsed = _aggregate_results_csv(out_dir, top_k=args.top_k)
    if not parsed.get("trace_files"):
        return {"mode": mode, "error": "No rocprof results_*.csv found", "out_dir": str(out_dir)}
    parsed["mode"] = mode
    parsed["ab_result_json"] = str(out_json)
    return parsed


def main():
    p = argparse.ArgumentParser(description="Collect rocprof timeline per backend mode.")
    p.add_argument(
        "--python",
        type=Path,
        default=_DEFAULT_VLLM_PYTHON,
        help="Python with torch+vLLM (default: amd-experiments/.venv/bin/python if present).",
    )
    p.add_argument(
        "--ab-script",
        type=Path,
        default=_REPO_ROOT / "benchmarks" / "bench_vllm_turboquant_ab.py",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "results",
        help="Directory for per-mode rocprof dirs and bench_vllm_rocprof_timeline_summary.json",
    )
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument(
        "--quant-model",
        default="",
        help="HF id for AWQ/GPTQ weights; required for quant_* modes.",
    )
    p.add_argument("--quantization", default="awq")
    p.add_argument("--input-len", type=int, default=256)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--num-prompts", type=int, default=8)
    p.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=-1.0,
        help="If >0, pass --gpu-memory-utilization to bench_vllm_turboquant_ab.py (needed on VF / shared VRAM).",
    )
    p.add_argument(
        "--max-model-len",
        type=int,
        default=0,
        help="If >0, pass --max-model-len to bench_vllm_turboquant_ab.py.",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=0,
        help="If >0, forwarded to A/B bench when supported.",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="If >0, forwarded to A/B bench when supported.",
    )
    p.add_argument(
        "--kv-heavy-story2",
        action="store_true",
        help="Set input_len=1024, output_len=256, num_prompts=32 to match bench_vllm_turboquant_ab_sweep_kv_heavy.json.",
    )
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
    p.add_argument(
        "--pmc",
        default="",
        metavar="COUNTERS",
        help=(
            "Optional rocprofv2 hardware counters (comma-separated), e.g. "
            "FETCH_SIZE,WRITE_SIZE,SQ_INSTS_VALU. Appended after --kernel-trace. "
            "Often unavailable or limited on VF; see profile_rocprof.py / report/paper.md."
        ),
    )
    add_swa_args(p)
    args = p.parse_args()
    print_swa_status(args.swa, args.window if args.swa == "on" else None)
    vllm_swa_warn(args.swa, args.max_model_len if args.max_model_len > 0 else 4096)
    if args.kv_heavy_story2:
        args.input_len = 1024
        args.output_len = 256
        args.num_prompts = 32
    if args.max_model_len <= 0 and args.kv_heavy_story2:
        args.max_model_len = 8192

    modes = [args.mode] if args.mode else list(MODES)
    all_results = []
    for m in modes:
        if m.startswith("quant_") and not args.quant_model:
            all_results.append(
                {
                    "mode": m,
                    "error": "skipped: pass --quant-model for quant_* rocprof modes",
                }
            )
            continue
        all_results.append(_run_mode(args, m))
    summary_path = args.out_dir / "bench_vllm_rocprof_timeline_summary.json"
    meta = {
        "rocprof_kernel_trace": True,
        "rocprof_pmc": args.pmc.strip() or None,
        "rocprof_hip_trace": bool(args.with_hip_trace),
        "enforce_eager": bool(args.enforce_eager),
        "model": args.model,
        "quant_model": args.quant_model or None,
        "quantization": args.quantization,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "num_prompts": args.num_prompts,
        "max_model_len": args.max_model_len if args.max_model_len > 0 else None,
        "gpu_memory_utilization": args.gpu_memory_utilization
        if args.gpu_memory_utilization > 0
        else None,
        "max_num_batched_tokens": args.max_num_batched_tokens or None,
        "max_num_seqs": args.max_num_seqs or None,
        "kv_heavy_story2_reference": "results/bench_vllm_turboquant_ab_sweep_kv_heavy.json "
        "(input_len=1024, output_len=256, num_prompts=32)",
        "swa": args.swa,
        "swa_window": args.window if args.swa == "on" else None,
    }
    summary_path.write_text(
        json.dumps({"meta": meta, "results": all_results}, indent=2),
        encoding="utf-8",
    )
    print(f"Saved timeline summary -> {summary_path}")


if __name__ == "__main__":
    main()
