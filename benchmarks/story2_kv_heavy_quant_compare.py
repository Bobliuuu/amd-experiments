#!/usr/bin/env python3
"""
Merge FP16/TQ-only kv-heavy sweep with optional quant+TQ run (Story 2 Phase 2).

Reads:
  - results/bench_vllm_turboquant_ab_sweep_kv_heavy.json (reference)
  - results/story2_quant_kv_heavy_ab.json (from scripts/run_story2_quant_kv_heavy.sh)

Writes results/story2_quant_kv_heavy_comparison.json (missing file → null entry).

  python3 benchmarks/story2_kv_heavy_quant_compare.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results" / "bench_vllm_turboquant_ab_sweep_kv_heavy.json"
QUANT = ROOT / "results" / "story2_quant_kv_heavy_ab.json"
OUT = ROOT / "results" / "story2_quant_kv_heavy_comparison.json"


def _load(p: Path) -> dict | None:
    if not p.is_file():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _tps(rows: list) -> dict:
    out = {}
    for r in rows:
        if "error" in r:
            continue
        b = r.get("backend")
        if b:
            out[b] = r.get("throughput_output_tps")
    return out


def main() -> None:
    ref = _load(REF)
    quant = _load(QUANT)
    ref_tps = _tps(ref.get("results", [])) if ref else {}
    quant_tps = _tps(quant.get("results", [])) if quant else {}

    comparison = {
        "reference_file": str(REF),
        "quant_run_file": str(QUANT) if quant else None,
        "reference_throughput_output_tps": ref_tps,
        "quant_run_throughput_output_tps": quant_tps,
        "notes": (
            "Compare quant_turboquant_fused vs fp16 in quant run; compare reference fp16 vs turboquant_fused "
            "for KV-only story. Quant weights should lower GEMM cost if AWQ kernels dominate."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(OUT.read_text())


if __name__ == "__main__":
    main()
