# Baselines (`baselines`)

Runnable baseline decode benchmarks for FP16, FP8, and INT4 KV cache formats.

## Run Location

```bash
cd /root/workspace/amd-experiments
```

## Commands

```bash
# FP16 baseline
python3 baselines/fp16_baseline.py --model mistralai/Mistral-7B-v0.1 --n-decode 30 --n-runs 3

# FP8 baseline
python3 baselines/fp8_baseline.py --model mistralai/Mistral-7B-v0.1 --n-decode 30 --n-runs 3

# INT4 baseline
python3 baselines/int4_baseline.py --model mistralai/Mistral-7B-v0.1 --n-decode 30 --n-runs 3
```

## Outputs

Each script writes result JSON under `results/` for later plotting and report generation.
