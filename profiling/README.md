# Profiling (`profiling`)

Hardware-counter and kernel-level profiling utilities for ROCm/MI300X runs.

## Run Location

```bash
cd /root/workspace/amd-experiments
```

## Kernel Profiling Utility

```bash
# Inspect available options first
python3 profiling/rocprof_kern_bench.py --help
```

This utility is intended to be used together with benchmark runs in `benchmarks/` when collecting low-level runtime evidence.
