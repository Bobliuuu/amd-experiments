#!/usr/bin/env python3
"""
Bucket top rocprof kernels into coarse categories for Story 2 (GEMM vs attention-ish vs other).

Reads results/bench_vllm_rocprof_timeline_summary.json (from bench_vllm_rocprof_timeline.py)
and writes results/story2_rocprof_bucket_compare.json.

  python3 benchmarks/story2_rocprof_summarize.py
  python3 benchmarks/story2_rocprof_summarize.py --input results/custom_summary.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _bucket(kernel: str) -> str:
    k = kernel.lower()
    if "paged_attention" in k or "pageattention" in k:
        return "attention_named"
    if "scaled_dot" in k or "flash_attn" in k or "fused_attention" in k or "sdpa" in k:
        return "attention_named"
    if "attn" in k or "attention" in k or "mha" in k:
        return "attention_named"
    if "cijk_" in k and "alik" in k:
        return "gemm_hipblaslt"
    if any(x in k for x in ("gemm", "matmul", "wmma", "mfma", "blkgemm")):
        return "gemm_general"
    if "triton" in k or ("kernel" in k and "tq" in k):
        return "triton_or_custom"
    if any(x in k for x in ("silu", "swiglu", "gelu", "activation", "elementwise")):
        return "activation_elementwise"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "bench_vllm_rocprof_timeline_summary.json",
    )
    ap.add_argument(
        "--output",
        type=Path,
        default=ROOT / "results" / "story2_rocprof_bucket_compare.json",
    )
    args = ap.parse_args()

    data = json.loads(args.input.read_text(encoding="utf-8"))
    meta = data.get("meta", {})
    rows_out = []
    for block in data.get("results", []):
        mode = block.get("mode", "")
        if block.get("error") or "top_kernels" not in block:
            rows_out.append({"mode": mode, "error": block.get("error", "no_top_kernels")})
            continue
        buckets: dict[str, float] = {}
        for row in block["top_kernels"]:
            b = _bucket(row.get("kernel", ""))
            buckets[b] = buckets.get(b, 0.0) + float(row.get("share_pct", 0.0))
        rows_out.append(
            {
                "mode": mode,
                "bucket_share_pct_topk": {k: round(v, 2) for k, v in sorted(buckets.items())},
                "top_kernel": block["top_kernels"][0]["kernel"] if block["top_kernels"] else "",
            }
        )

    out = {
        "story2_phase": "rocprof_bucket_compare",
        "source_summary": str(args.input),
        "meta": meta,
        "interpretation": (
            "When E2E tok/s is flat across TQ paths, gemm_hipblaslt+gemm_general often dominates top-kernel time; "
            "attention_named share rises with long KV / fused path only if attention is on the critical path."
        ),
        "modes": rows_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(args.output.read_text())


if __name__ == "__main__":
    main()
