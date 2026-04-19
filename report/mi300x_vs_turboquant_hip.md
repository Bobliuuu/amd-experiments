# MI300X Optimization Delta vs `domvox/turboquant-hip`

This document explains what was changed in this repo's MI300X port relative to the earlier HIP port (`domvox/turboquant-hip`, mirrored here as `kernels/ref/ggml_turboquant.hip.cpp`).

Scope: kernel implementation, build/toolchain targeting, ABI/runtime integration, and validation workflow.

---

## Baseline vs this repo

- **Reference HIP port** (`kernels/ref/ggml_turboquant.hip.cpp`)
  - Targeted RDNA3/gfx1100 conventions.
  - Uses warp-size-generic reduction and atomic-based bit packing.
  - No dedicated QJL keys kernel in that HIP file.
- **This repo's MI300X port** (`kernels/hip/turboquant_mi300x.hip.cpp`)
  - Targeted CDNA3/gfx942 (MI300X), Wave64-first.
  - Reworked pack/write/reduction for Wave64 behavior and occupancy.
  - Adds explicit QJL residual correction kernel for key vectors.

---

## Exact optimizations made for MI300X

## 1) Wave64 ballot bit-packing (replaces atomic-heavy pack)

In the MI300X kernel, bitplanes are packed with `__ballot()` masks directly from per-thread register bits (two 64-bit masks for 128 dims). This replaces the reference pattern that packs through shared memory and `atomicOr` updates.

Why this is MI300X-specific:
- MI300X runs Wave64; one wavefront naturally yields 64-bit ballot masks.
- Two wavefront leaders can write one 16-byte plane with minimal contention.

Net effect:
- Removes serialized LDS atomics from the hot quantize path.
- Converts packing into warp vote + leader store operations.

---

## 2) Register-resident indices (reduced LDS footprint)

Reference path stores quantized indices in shared memory (`s_indices`).
MI300X path keeps each thread's quantized index in a register (`my_idx`) and ballots directly from registers.

Net effect:
- Lower LDS bytes/block.
- Better occupancy headroom on CDNA3 CUs.
- Less LDS traffic before packing.

---

## 3) Fused plane writes by wavefront leaders

Reference flow includes a single-thread-style final write stage over packed bytes.
MI300X flow writes packed 64-bit masks directly to global memory from lane-0 of each wavefront per bitplane.

Net effect:
- Eliminates long serialized byte-copy loops in the writeback stage.
- Better write parallelism aligned to the ballot output format.

---

## 4) Wave64-native reductions (compile-time shape)

MI300X kernels use explicit Wave64 reduction steps (offsets 32->1) instead of runtime `warpSize` looping.

Net effect:
- Reduction path is fixed for gfx942 execution model.
- Lower control overhead and clearer codegen assumptions for CDNA3.

---

## 5) Added QJL keys kernel (Algorithm 2 path)

This MI300X implementation adds an explicit QJL keys kernel (`tqm_qjl_kernel`) and data path (`block_qjl_mi300x`) for key residual correction.

Reference HIP port in `kernels/ref/ggml_turboquant.hip.cpp` focuses on MSE-style quant/dequant/fused-dot and does not provide this full QJL key supplement path.

Net effect:
- Supports the intended TurboQuant paper-style 3-bit quality path (MSE quant + QJL correction for keys).

---

## 6) MI300X-targeted build and binary configuration

`kernels/hip/build_mi300x.sh` enforces MI300X-specific target and flags:

- `--offload-arch=gfx942:sramecc+:xnack-`
- `-mwavefrontsize64`
- `-DCDNA3`, `-DAMD_MFMA_AVAILABLE`, `-DTARGET_MI300X`

Why this matters:
- MI300X VF reports feature-qualified arch string (`sramecc+:xnack-`); mismatch can trigger `hipErrorNoBinaryForGpu (209)` when uploading constants or launching kernels.

---

## 7) Runtime integration adaptation for this machine

This repo handles **HIP / code-object** alignment explicitly:

- **Canonical path:** build and run inside **ROCm 7.2 Primus** (`docker_run_amd_mi300x.sh` / `rocm/primus:v26.2`) so system `hipcc`, image PyTorch, and drivers agree.
- **Python hot path:** `kernels/turboquant_mi300x.py` (pure PyTorch / rocBLAS / MFMA) avoids `ctypes`-loading a separately built `.so` into the PyTorch process when COV5/COV6 or runtime versions could disagree.

Outcome:
- Standalone HIP binaries (`tq_validate_mi300x`, `tq_bench_mi300x`) validate native kernels in the container toolchain.
- Python uses the PyTorch route above for stable end-to-end benchmarks.

This is a practical MI300X deployment optimization (stability + reproducibility), not just a kernel micro-optimization.

---

## 8) Expanded validation against reference and MI300X assumptions

`kernels/hip/turboquant_mi300x_test.cpp` adds MI300X-focused checks:

- Wave64 assumption warnings/checks.
- Codebook consistency against domvox reference values.
- QJL sign/residual sanity checks.
- End-to-end kernel correctness on the target architecture.

Net effect:
- Regression surface is broader than the reference HIP port.
- Better confidence that MI300X-specific optimizations preserve correctness.

---

## What did not change (important for interpretation)

- Core algorithmic family remains TurboQuant (rotation + scalar quantization, with optional QJL correction).
- Block geometry still uses `head_dim=128` and compact bitplane layouts.
- Compression-ratio arithmetic (for example, 52-byte TQ3 block) remains deterministic layout math.

---

## Practical summary

Compared with the earlier HIP port, this MI300X version is optimized around CDNA3 Wave64 execution, lower LDS/atomic pressure, fused writeback, and QJL key correction support, plus an ABI-safe runtime strategy for this exact VM setup.

One-line summary:

> `turboquant-hip` is the starting HIP translation; this repo's version is the MI300X production-tuned variant for gfx942 behavior, correctness validation, and reproducible benchmarking on this VM.
