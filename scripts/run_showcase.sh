#!/usr/bin/env bash
set -euo pipefail

# One-command runner for screenshot-style experiment suite.
# Runs measured compression-grid + attention/latency/roofline/memory showcase.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "== TurboQuant showcase: measured compression grid =="
python3 benchmarks/bench_compression_ratio_grid.py \
  --bits 2 3 4 \
  --ctx 2048 4096 8192 16384 \
  --layers 28 \
  --kv-heads 2 \
  --head-dim 128 \
  --output results/bench_compression_ratio_grid.json

echo
echo "== TurboQuant showcase: quality/latency/roofline/memory =="
python3 benchmarks/bench_turboquant_showcase.py \
  --bits 3 \
  --quality-seq-k 2048 \
  --output results/bench_turboquant_showcase.json

echo
echo "== Regenerate report figures =="
python3 report/generate_figures_v2.py

echo
echo "Showcase complete."
echo "  - results/bench_compression_ratio_grid.json"
echo "  - results/bench_turboquant_showcase.json"
echo "  - report/figures_v2/fig27_tq_attention_quality_hist.png"
echo "  - report/figures_v2/fig28_mi300x_roofline_tq_attention.png"
echo "  - report/figures_v2/fig29_kv_cache_memory_curve.png"
echo "  - report/figures_v2/fig30_kv_component_breakdown.png"
