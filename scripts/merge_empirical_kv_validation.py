#!/usr/bin/env python3
"""
merge_empirical_kv_validation.py — Join ratio + speed + quality into one artifact.

Inputs:
  - results/bench_runtime_ratio_all_methods.json
  - results/bench_compress_decompress_recheck.json (fallback: bench_compress_decompress.json)
  - results/bench_ppl_all_methods_quality_recheck.json (fallback: bench_ppl_all_methods.json)

Output:
  - results/bench_empirical_kv_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


def load_json(path: Path) -> Any:
    with open(path) as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(description="Merge empirical KV validation sources")
    parser.add_argument("--results-dir", default=str(Path(__file__).parent.parent / "results"))
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    rd = Path(args.results_dir)
    ratio_path = rd / "bench_runtime_ratio_all_methods.json"
    speed_path = rd / "bench_compress_decompress_recheck.json"
    qual_path = rd / "bench_ppl_all_methods_quality_recheck.json"
    if not speed_path.exists():
        speed_path = rd / "bench_compress_decompress.json"
    if not qual_path.exists():
        qual_path = rd / "bench_ppl_all_methods.json"

    ratio_obj = load_json(ratio_path)
    speed_rows = load_json(speed_path)
    qual_rows = load_json(qual_path)

    def pick(rows, method: str, bits: int) -> Dict | None:
        for r in rows:
            if r.get("method") == method and int(r.get("bits", -1)) == bits:
                return r
        return None

    bits = int(ratio_obj.get("bits", 3))
    turbo_speed = pick(speed_rows, "turbo", bits)
    if turbo_speed is None:
        raise RuntimeError("Turbo baseline row missing from speed benchmark JSON")

    merged_rows = []
    for r in ratio_obj.get("results", []):
        m = r["method"]
        s = pick(speed_rows, m, bits)
        q = pick(qual_rows, m, bits)
        row = dict(r)
        if s:
            row["compress_bw_gbs"] = s.get("compress_bw_gbs")
            row["decompress_bw_gbs"] = s.get("decompress_bw_gbs")
            row["compress_us"] = s.get("compress_us")
            row["decompress_us"] = s.get("decompress_us")
            row["fmas_per_vec"] = s.get("fmas_per_vec")
            row["compress_speedup_vs_turbo"] = (
                float(s.get("compress_bw_gbs", 0.0)) / float(turbo_speed.get("compress_bw_gbs", 1.0))
            )
            row["decompress_speedup_vs_turbo"] = (
                float(s.get("decompress_bw_gbs", 0.0)) / float(turbo_speed.get("decompress_bw_gbs", 1.0))
            )
        if q:
            row["cosine_sim_mean"] = q.get("cosine_sim_mean")
            row["cosine_sim_p5"] = q.get("cosine_sim_p5")
            row["mse"] = q.get("mse")
        merged_rows.append(row)

    out_obj = {
        "model": ratio_obj.get("model"),
        "device": ratio_obj.get("device"),
        "bits": bits,
        "sources": {
            "ratio": str(ratio_path),
            "speed": str(speed_path),
            "quality": str(qual_path),
        },
        "results": merged_rows,
    }

    out_path = Path(args.output) if args.output else (rd / "bench_empirical_kv_validation.json")
    with open(out_path, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"Saved merged artifact: {out_path}")


if __name__ == "__main__":
    main()
