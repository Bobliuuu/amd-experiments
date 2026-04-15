#!/usr/bin/env python3
"""
consolidate_benchmarks.py — Merge results/*.json into one verbose report + optional JSON.

Usage:
  python3 scripts/consolidate_benchmarks.py --verbose
  python3 scripts/consolidate_benchmarks.py --verbose --write-json results/consolidated_benchmarks.json
  python3 scripts/consolidate_benchmarks.py --markdown > BENCHMARKS.md
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def load_json(path: Path) -> Any | None:
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def fmt_json(obj: Any, limit: int = 12000) -> str:
    s = json.dumps(obj, indent=2, default=str)
    if len(s) > limit:
        return s[:limit] + f"\n... ({len(s) - limit} chars truncated)\n"
    return s


def section(title: str) -> str:
    return f"\n{'=' * 72}\n{title}\n{'=' * 72}\n"


def summarize_compress(rows: list[dict]) -> str:
    if not rows:
        return "(no data)\n"
    lines = []
    for r in sorted(rows, key=lambda x: (x.get("method", ""), x.get("bits", 0))):
        m = f"{r.get('method','?')}{r.get('bits','')}"
        lines.append(
            f"  {m:10} compress {r.get('compress_bw_gbs', 0):6.2f} GB/s  "
            f"decompress {r.get('decompress_bw_gbs', 0):6.2f} GB/s  "
            f"ratio {r.get('compression_ratio', 0):.3f}x  "
            f"FMAs/vec {r.get('fmas_per_vec', 0)}"
        )
    return "\n".join(lines) + "\n"


def summarize_rocprof(obj: dict) -> str:
    lines = []
    for name, stats in sorted(obj.items(), key=lambda kv: -float(kv[1].get("total_ms", 0))):
        lines.append(
            f"  {name[:68]:68}  avg {stats.get('avg_us', 0):8.2f} µs  "
            f"total {stats.get('total_ms', 0):8.2f} ms  count {stats.get('count', 0)}"
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "results",
    )
    parser.add_argument("--verbose", action="store_true", help="Print full merged JSON blobs")
    parser.add_argument("--markdown", action="store_true", help="Markdown-ish output")
    parser.add_argument("--write-json", type=Path, default=None, help="Write merged summary JSON")
    args = parser.parse_args()
    rd: Path = args.results_dir

    merged: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "results_dir": str(rd),
        "files": {},
    }

    def take(name: str) -> Any | None:
        p = rd / name
        data = load_json(p)
        merged["files"][name] = {"path": str(p), "present": data is not None}
        return data

    # Priority datasets
    comp = take("bench_compress_decompress_recheck.json") or take("bench_compress_decompress.json")
    qual = take("bench_ppl_all_methods_quality_recheck.json") or take("bench_ppl_all_methods.json")
    roc = take("rocprof_kernel_timeline.json")
    mfma = take("bench_mfma_rotate.json")
    pope = take("pope_rotorquant_2026_claims.json")
    runtime_ratio = take("bench_runtime_ratio_all_methods.json")
    empirical_merged = take("bench_empirical_kv_validation.json")

    merged["summary"] = {}
    if isinstance(comp, list):
        merged["summary"]["compress_decompress"] = comp
    if isinstance(qual, list):
        merged["summary"]["kv_cosine_quality"] = qual
    if isinstance(roc, dict):
        merged["summary"]["rocprof_kernel_timeline"] = roc
    if isinstance(mfma, dict):
        merged["summary"]["mfma_rotate"] = mfma
    if isinstance(pope, dict):
        merged["summary"]["pope_author_claims"] = pope
    if isinstance(runtime_ratio, dict):
        merged["summary"]["runtime_ratio_all_methods"] = runtime_ratio
    if isinstance(empirical_merged, dict):
        merged["summary"]["empirical_kv_validation"] = empirical_merged

    out_lines: list[str] = []
    if args.markdown:
        out_lines.append(f"# Consolidated benchmarks\n\n_Generated: `{merged['generated_at']}`_\n")

    out_lines.append(section("Compress / decompress (GB/s)"))
    out_lines.append(summarize_compress(comp) if comp else "(missing bench_compress_decompress*.json)\n")

    out_lines.append(section("KV reconstruction cosine (quality-only runs)"))
    if isinstance(qual, list):
        for r in qual:
            m = f"{r.get('method','?')}{r.get('bits','')}"
            out_lines.append(
                f"  {m:10} mean_cos {r.get('cosine_sim_mean', 0):.6f}  "
                f"p5 {r.get('cosine_sim_p5', 0):.6f}  mse {r.get('mse', 0):.6f}\n"
            )
    else:
        out_lines.append("(missing bench_ppl_all_methods*.json)\n")

    out_lines.append(section("HIP kernel timeline / latency (rocprof summary JSON)"))
    out_lines.append(summarize_rocprof(roc) if isinstance(roc, dict) else "(missing rocprof_kernel_timeline.json)\n")

    out_lines.append(section("MFMA rotation microbench (TurboQuant path)"))
    if isinstance(mfma, dict) and "throughput" in mfma:
        for row in mfma["throughput"][:6]:
            out_lines.append(f"  n={row.get('n')}  mfma {row.get('mfma_us')} µs  matmul {row.get('matmul_us')} µs\n")
        if len(mfma["throughput"]) > 6:
            out_lines.append(f"  ... ({len(mfma['throughput'])} rows total)\n")
    else:
        out_lines.append("(missing bench_mfma_rotate.json)\n")

    out_lines.append(section("Author headline JSON (optional)"))
    out_lines.append(fmt_json(pope, limit=4000) if pope else "(missing pope_rotorquant_2026_claims.json)\n")

    out_lines.append(section("Runtime ratio benchmark (calculated vs measured)"))
    if isinstance(runtime_ratio, dict):
        for r in runtime_ratio.get("results", []):
            out_lines.append(
                f"  {r.get('method','?')}{r.get('bits','')}  "
                f"calc={r.get('ratio_calculated_layout', 0):.3f}x  "
                f"obs={r.get('ratio_observed_runtime', 0):.3f}x  "
                f"fp16={r.get('kv_bytes_fp16', 0)/1e6:.2f}MB  "
                f"comp={r.get('kv_bytes_compressed_materialized', 0)/1e6:.2f}MB\n"
            )
    else:
        out_lines.append("(missing bench_runtime_ratio_all_methods.json)\n")

    out_lines.append(section("Merged empirical validation artifact"))
    out_lines.append(
        fmt_json(empirical_merged, limit=8000)
        if empirical_merged
        else "(missing bench_empirical_kv_validation.json)\n"
    )

    text = "".join(out_lines)
    if args.verbose:
        text += section("Full merged object (JSON)")
        text += fmt_json(merged, limit=200000)

    print(text, end="")

    if args.write_json:
        args.write_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.write_json, "w") as f:
            json.dump(merged, f, indent=2, default=str)
        print(f"\nWrote merged summary → {args.write_json}", file=sys.stderr)


if __name__ == "__main__":
    main()
