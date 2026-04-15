# Utility Scripts (`scripts`)

Support scripts for consolidating benchmark outputs and merging empirical KV validation data.

## Run Location

```bash
cd /root/workspace/amd-experiments
```

## Consolidate Benchmark JSON Files

```bash
# Write merged machine-readable artifact
python3 scripts/consolidate_benchmarks.py --verbose --write-json results/consolidated_benchmarks.json

# Emit markdown summary
python3 scripts/consolidate_benchmarks.py --markdown > results/consolidated_benchmarks.md
```

## Showcase Runner

```bash
# Executes the screenshot-style experiment set
bash scripts/run_showcase.sh
```

## Merge Empirical KV Validation

```bash
# See script help for expected arguments in your local run
python3 scripts/merge_empirical_kv_validation.py --help
```
