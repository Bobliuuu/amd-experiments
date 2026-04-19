#!/usr/bin/env python3
"""
decode_whole_step_golden_driver.py — Package golden kv-heavy baseline + metadata.

If ``--from-existing`` points at a prior bench JSON (e.g. sweep_kv_heavy), copies
throughput rows into ``results/decode_whole_step_baseline_kv_heavy.json`` with
provenance. Otherwise prints the shell command to run on Primus.

  python3 benchmarks/decode_whole_step_golden_driver.py --from-existing results/bench_vllm_turboquant_ab_sweep_kv_heavy.json
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "decode_whole_step_baseline_kv_heavy.json"
SWEEP = ROOT / "results" / "bench_vllm_turboquant_ab_sweep_kv_heavy.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--from-existing",
        type=Path,
        default=None,
        help="Reuse results from a completed bench_vllm_turboquant_ab JSON",
    )
    ap.add_argument(
        "--run-subprocess",
        action="store_true",
        help="Invoke scripts/run_decode_whole_step_baseline_kv_heavy.sh (needs working vLLM)",
    )
    args = ap.parse_args()

    OUT.parent.mkdir(parents=True, exist_ok=True)

    if args.run_subprocess:
        sh = ROOT / "scripts" / "run_decode_whole_step_baseline_kv_heavy.sh"
        r = subprocess.run(["bash", str(sh)], cwd=str(ROOT))
        sys.exit(r.returncode)

    src = args.from_existing or SWEEP
    if not src.is_file():
        print(f"Missing {src}; run on MI300X:\n  bash scripts/run_decode_whole_step_baseline_kv_heavy.sh", file=sys.stderr)
        sys.exit(1)

    data = json.loads(src.read_text(encoding="utf-8"))
    wrapped = {
        "decode_whole_step_phase": "golden_baseline",
        "time_iso": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "provenance": {
            "source_file": str(src.relative_to(ROOT) if src.is_relative_to(ROOT) else src),
            "note": "Frozen kv-heavy recipe: input_len=1024, output_len=256, num_prompts=32, Mistral-7B-v0.1",
            "dispatch_log": "results/logs/decode_whole_step_dispatch_kv_heavy.log (after run_decode_whole_step_baseline_kv_heavy.sh)",
        },
        "model": data.get("model"),
        "input_len": data.get("input_len"),
        "output_len": data.get("output_len"),
        "num_prompts": data.get("num_prompts"),
        "device": data.get("device"),
        "results": data.get("results", []),
    }
    OUT.write_text(json.dumps(wrapped, indent=2), encoding="utf-8")
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
