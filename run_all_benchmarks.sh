#!/usr/bin/env bash
# run_all_benchmarks.sh — Run the full TurboQuant benchmark suite
#
# Usage: bash run_all_benchmarks.sh [MODEL] [SEQ_LENS...]
# Example: bash run_all_benchmarks.sh mistralai/Mistral-7B-v0.1
#
# This runs all benchmarks in sequence (sharing GPU memory between them
# is not possible with separate Python processes, so they run serially).
#
# Estimated runtime: ~60-90 minutes for full suite on MI300X

set -euo pipefail

MODEL="${1:-mistralai/Mistral-7B-v0.1}"
SEQ_LENS="${2:-512 2048 8192 32768 65536 131072}"
N_DECODE=30
N_RUNS=3

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

LOG_DIR="$SCRIPT_DIR/results/logs"
mkdir -p "$LOG_DIR"

timestamp() { date '+%Y-%m-%d %H:%M:%S'; }

echo "================================================================"
echo "TurboQuant MI300X Full Benchmark Suite"
echo "Model:     $MODEL"
echo "Seq lens:  $SEQ_LENS"
echo "N_decode:  $N_DECODE  N_runs: $N_RUNS"
echo "Started:   $(timestamp)"
echo "================================================================"
echo

# 0. Kernel microbenchmarks (no model needed)
echo "[$(timestamp)] Step 0: Kernel throughput microbenchmark ..."
python3 benchmarks/bench_kernels.py \
    --n-vectors 65536 --n-iters 100 \
    2>&1 | tee "$LOG_DIR/bench_kernels.log"
echo

# 0b. Attention throughput vs context length (no model needed)
echo "[$(timestamp)] Step 0b: Attention throughput benchmark ..."
python3 benchmarks/bench_tq_attention.py \
    --seq-lens $SEQ_LENS --n-iters 20 \
    2>&1 | tee "$LOG_DIR/bench_tq_attention.log"
echo

# 1. FP16 baseline
echo "[$(timestamp)] Step 1: FP16 baseline ..."
python3 baselines/fp16_baseline.py \
    --model "$MODEL" \
    --seq-lens $SEQ_LENS \
    --n-decode "$N_DECODE" --n-runs "$N_RUNS" \
    2>&1 | tee "$LOG_DIR/fp16_baseline.log"
echo

# 2. FP8 baseline
echo "[$(timestamp)] Step 2: FP8 baseline ..."
python3 baselines/fp8_baseline.py \
    --model "$MODEL" \
    --seq-lens $SEQ_LENS \
    --n-decode "$N_DECODE" --n-runs "$N_RUNS" \
    2>&1 | tee "$LOG_DIR/fp8_baseline.log"
echo

# 3. INT4 baseline
echo "[$(timestamp)] Step 3: INT4 baseline ..."
python3 baselines/int4_baseline.py \
    --model "$MODEL" \
    --seq-lens $SEQ_LENS \
    --n-decode "$N_DECODE" --n-runs "$N_RUNS" \
    2>&1 | tee "$LOG_DIR/int4_baseline.log"
echo

# 4. Quality benchmark (perplexity)
echo "[$(timestamp)] Step 4: Quality benchmark (perplexity + KV reconstruction) ..."
python3 benchmarks/bench_quality.py \
    --model "$MODEL" \
    --n-tokens 4096 \
    --context-len 512 \
    2>&1 | tee "$LOG_DIR/bench_quality.log"
echo

# 5. Generate all plots
echo "[$(timestamp)] Step 5: Generating analysis plots ..."
python3 analysis/plot_results.py \
    2>&1 | tee "$LOG_DIR/plot_results.log"
echo

echo "================================================================"
echo "All benchmarks complete!"
echo "Results: $SCRIPT_DIR/results/"
echo "Figures: $SCRIPT_DIR/analysis/figures/"
echo "Report:  $SCRIPT_DIR/report/final_report.md"
echo "Finished: $(timestamp)"
echo "================================================================"
