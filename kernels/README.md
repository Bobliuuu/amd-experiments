# Kernels (`kernels`)

Kernel implementations and loaders used by benchmark scripts.

## Run Location

```bash
cd /root/workspace/amd-experiments
```

## Python-Accessible Components

- `turboquant_mi300x.py`: pure-PyTorch TurboQuant wrapper (recommended Python path)
- `block_quant_rocm.py`: Triton kernels for IsoQuant/PlanarQuant/RotorQuant
- `tq_triton.py`: fused TQ attention path
- `tq_hsaco_loader.py`, `tq_mfma_loader.py`: kernel loader utilities

These are generally imported by benchmark scripts in `benchmarks/` rather than executed directly.

## Standalone HIP Build and Binaries

For directly runnable C++/HIP binaries, use:

```bash
cd kernels/hip
bash build_mi300x.sh
./tq_validate_mi300x
./tq_bench_mi300x 4096 50
```

See `kernels/hip/README.md` for complete build options and ABI caveats.
