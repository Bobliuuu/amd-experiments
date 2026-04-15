# Notebooks (`notebooks`)

This folder contains interactive analysis notebooks for benchmark rollups.

## Available Notebook

- `consolidate_benchmarks.ipynb`: runs `scripts/consolidate_benchmarks.py` and inspects the merged JSON output.

## Prerequisites

Run benchmarks first so result JSON files exist in `results/`.

## Launch

From repository root:

```bash
cd /root/workspace/amd-experiments
jupyter notebook notebooks/consolidate_benchmarks.ipynb
```

The notebook auto-detects whether it was launched from repo root or from `notebooks/`.

## Non-Notebook Equivalent

Use the script directly when you want CI-friendly or terminal-only output:

```bash
cd /root/workspace/amd-experiments
python3 scripts/consolidate_benchmarks.py --verbose --write-json results/consolidated_benchmarks.json
python3 scripts/consolidate_benchmarks.py --markdown > results/consolidated_benchmarks.md
```

## Output Artifacts

- `results/consolidated_benchmarks.json`: machine-readable merged summary
- `results/consolidated_benchmarks.md`: markdown summary (optional)
