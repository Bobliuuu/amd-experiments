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

## Important ABI note

Avoid ``ctypes``-loading ``libturboquant_mi300x.so`` into the same Python process as PyTorch unless the **HIP / code-object** build matches the **runtime shipped with that PyTorch** — mismatches surface as **`hipErrorNoBinaryForGpu (209)`**.

**Canonical workflow:** build and run inside **ROCm 7.2 Primus** (`docker_run_amd_mi300x.sh`). For Python hot paths, use **`kernels/turboquant_mi300x.py`** (pure PyTorch).

## Common Workflow

1. Build + validate HIP binaries for standalone correctness/perf checks.
2. Run Python benchmarks from repo root against `kernels/turboquant_mi300x.py`.
3. Consolidate outputs with:

```bash
python3 scripts/consolidate_benchmarks.py --verbose --write-json results/consolidated_benchmarks.json
```
