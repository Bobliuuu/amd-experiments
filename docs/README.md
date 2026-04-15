# Documentation Guide and Consolidated Summaries

This project has multiple long-form docs. Use this guide to quickly find the right one and regenerate consolidated summaries.

## Documentation Map

- `README.md`: top-level quick start, benchmark catalogue, and architecture notes
- `summary.md`: compact table-oriented results across throughput, latency, quality
- `current.md`: execution status, blockers, and recent implementation notes
- `research.md`: algorithm/implementation deep-dive and design context
- `report/final_report_v2.md`: primary long-form analysis report
- `report/final_report.md`: earlier report version

## Consolidated Summary Workflow

Generate one merged benchmark artifact from all available JSON result files:

```bash
cd /root/workspace/amd-experiments
python3 scripts/consolidate_benchmarks.py --verbose --write-json results/consolidated_benchmarks.json
python3 scripts/consolidate_benchmarks.py --markdown > results/consolidated_benchmarks.md
```

Then review in this order:

1. `results/consolidated_benchmarks.md`
2. `summary.md`
3. `report/final_report_v2.md`

## Notebook-Assisted Consolidation

If you prefer Jupyter:

```bash
cd /root/workspace/amd-experiments
jupyter notebook notebooks/consolidate_benchmarks.ipynb
```

The notebook wraps the same consolidation script and prints merged key sections.

## Practical Reading Paths

- **Fast status check (5-10 min):**
  - `summary.md` -> `current.md`
- **Implementation/operator details (20-30 min):**
  - `README.md` -> `kernels/hip/README.md` -> `research.md`
- **Report/slide preparation:**
  - `results/consolidated_benchmarks.md` -> `report/final_report_v2.md` -> `report-ui/README.md`
