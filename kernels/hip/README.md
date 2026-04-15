# HIP Kernel Implementation (`kernels/hip`)

Standalone HIP/C++ implementation of TurboQuant kernels for AMD Instinct MI300X (`gfx942:sramecc+:xnack-`).

## Contents

- `turboquant_mi300x.hip.cpp`: quantize, dequantize, fused-dot, QJL kernels
- `turboquant_mi300x.h`: C API, structs, block layouts, codebooks
- `turboquant_mi300x_test.cpp`: validation suite
- `tq_hip_benchmark_mi300x.cpp`: microbenchmark binary
- `build_mi300x.sh`: one-command build script

## Build

```bash
cd /root/workspace/amd-experiments/kernels/hip
bash build_mi300x.sh
```

Supported modes:

```bash
bash build_mi300x.sh lib
bash build_mi300x.sh test
bash build_mi300x.sh bench
bash build_mi300x.sh clean
```

## Validate and Benchmark

```bash
cd /root/workspace/amd-experiments/kernels/hip
./tq_validate_mi300x
./tq_bench_mi300x 4096 50
```

Expected validation result: all tests pass.

## Toolchain and Target

- Compiler: `/opt/rocm/bin/hipcc`
- Target arch: `gfx942:sramecc+:xnack-`
- Build flags are defined in `build_mi300x.sh`

## Important ABI Note

Do not `ctypes`-load `libturboquant_mi300x.so` in the same Python process as PyTorch in this environment.

- System ROCm: 7.2
- PyTorch bundled ROCm: 6.2

This mismatch can trigger `hipErrorNoBinaryForGpu (209)` when loading the shared object. For Python integration, use `kernels/turboquant_mi300x.py` (pure PyTorch path).

## Common Workflow

1. Build + validate HIP binaries for standalone correctness/perf checks.
2. Run Python benchmarks from repo root against `kernels/turboquant_mi300x.py`.
3. Consolidate outputs with:

```bash
python3 scripts/consolidate_benchmarks.py --verbose --write-json results/consolidated_benchmarks.json
```
