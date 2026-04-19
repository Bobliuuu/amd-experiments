# AMD MI300X — KV Cache Compression Benchmark

Four-way benchmark of **TurboQuant**, **IsoQuant**, **PlanarQuant**, and **RotorQuant** on AMD Instinct MI300X (gfx942). Triton/ROCm kernels, Mistral-7B-v0.1, compress/decompress bandwidth, prefill overhead, batch decode, and KV reconstruction quality.

**Hardware**: MI300X VF · gfx942:sramecc+:xnack- · 192 GB HBM3  
**Stack (benchmarks / vLLM):** locked **`uv`** venv at **`<repo>/.benchmark_mi300_vllm_frozen/`** — ROCm **7.2.0**, PyTorch **ROCm** line + **vLLM** `rocm721` pip stack (`stack_id` **`vllm_pip_rocm721_owned_torch`**). Exact commands, ACK env vars, archive snapshot, and reproducibility limits: **[`docs/benchmark_mi300_locked_env.md`](docs/benchmark_mi300_locked_env.md)**. Installer + gates: **[`docs/rocm72_uv_torch_vllm_venv.md`](docs/rocm72_uv_torch_vllm_venv.md)**. Primus Docker wrapper: **`docker_run_amd_mi300x.sh`**.

## Key Results

| Method | Compression | Prefill (32K tok/s) | Compress BW | Cosine sim |
|---|---|---|---|---|
| FP16 | 1× | — | — | 1.000 |
| TurboQuant (TQ3) | 4.923× | 42K | 2.9 GB/s | 0.983 |
| IsoQuant | 4.923× | 891K | **21.8 GB/s** | 0.983 |
| **PlanarQuant** ★ | 4.923× | **1,126K** | 18.7 GB/s | 0.983 |
| RotorQuant | 4.923× | 855K | 17.3 GB/s | 0.983 |

All 3-bit methods share identical 52-byte block layout and reconstruction quality. PlanarQuant is recommended: lowest FMA count (256 vs 16,384 for TurboQuant) and fastest prefill.

## Repository Structure

```
amd-experiments/
│
├── kernels/                   Python wrappers and Triton kernels
│   ├── turboquant_mi300x.py   TQ3/TQ4 compress/decompress (pure PyTorch, MFMA-backed)
│   ├── block_quant_rocm.py    IsoQuant / PlanarQuant / RotorQuant (Triton, ROCm)
│   ├── tq_triton.py           Fused TQ3 dequant-attention kernel (Flash Attn 2 style)
│   ├── tq_hsaco_loader.py     HSACO loader (bypasses HIP ABI mismatch)
│   ├── tq_mfma_loader.py      MFMA rotation kernel loader
│   ├── hip/                   HIP/C++ implementation (standalone binaries)
│   │   ├── turboquant_mi300x.hip.cpp   Quantize/dequantize/fused-dot kernels
│   │   ├── turboquant_mi300x.h         Public C API + structs + codebooks
│   │   ├── turboquant_mi300x_test.cpp  9-test validation suite
│   │   ├── tq_hip_benchmark_mi300x.cpp Kernel microbenchmark binary
│   │   ├── tq_mfma_rotate.hip.cpp      MFMA-accelerated rotation kernel
│   │   └── build_mi300x.sh             Build script (lib / test / bench / clean)
│   └── ref/                   Reference implementations (upstream / ggml)
│       ├── ggml_turboquant.c/h  ggml-compatible TurboQuant reference
│       └── turboquant.py        Python Lloyd-Max solver + codebook generator
│
├── benchmarks/                All benchmark scripts (see catalogue below)
├── baselines/                 FP16 / FP8 / INT4 end-to-end decode baselines
├── results/                   JSON outputs from all benchmark runs
├── report/
│   ├── final_report_v2.md     Full four-method analysis (current) + §14 repo closure / Figs 30–31
│   ├── final_report.md        Earlier TQ3-only study + §14.5 closure summary
│   ├── figures_v2/            v2 figures (incl. Figs 30–31 — rocprof buckets + deployment handoff)
│   └── generate_figures_v2.py Matplotlib figure generator
├── report-ui/                 React/Vite interactive report viewer
├── analysis/                  Plot scripts for v1 results
├── scripts/                   Consolidation and merge utilities
├── profiling/                 rocprof counter collection
├── tq_backends/               Drop-in TurboQuant / IsoQuant attention modules (not PyPI vLLM)
└── notebooks/                 Jupyter analysis notebook
```

## Quick Start

Use the **locked benchmark interpreter** when running GPU work (see **`docs/benchmark_mi300_locked_env.md`**):

```bash
# Example: after `source <repo>/.benchmark_mi300_vllm_frozen/.venv/bin/activate`
# Compress/decompress microbench (§4 of the report — ~30 seconds)
python3 benchmarks/bench_compress_decompress.py --n-vectors 4096 --n-iters 50

# Prefill throughput: verify 26.5× gap between PlanarQuant and TurboQuant
python3 benchmarks/bench_prefill.py --model mistralai/Mistral-7B-v0.1

# Full PPL comparison (3-bit and 4-bit, all methods — ~2 hours)
python3 benchmarks/bench_ppl_all_methods.py --model mistralai/Mistral-7B-v0.1

# Batch decode scaling (compute-bound vs bandwidth-bound regimes)
python3 benchmarks/bench_batch_decode_v2.py --model mistralai/Mistral-7B-v0.1

# Regenerate all v2 figures from existing results/
python3 report/generate_figures_v2.py
```

**vLLM kv-heavy decode — repository closure (what we shipped vs what is deployment-only):** [`docs/repo_decode_bottleneck_closure.md`](docs/repo_decode_bottleneck_closure.md) · ops checklist [`docs/bottleneck_improvement_mi300.md`](docs/bottleneck_improvement_mi300.md) · evidence trail [`docs/decode_whole_step_amdahl_outcome.md`](docs/decode_whole_step_amdahl_outcome.md).

## Building HIP Kernels

The HIP kernels provide standalone `C` binaries for isolated throughput benchmarks.
They cannot be loaded via ctypes in the same Python process as PyTorch (ROCm ABI
mismatch — see [ABI note](#rocm-abi-note)).

```bash
cd kernels/hip
bash build_mi300x.sh          # build all (lib + test + bench)
bash build_mi300x.sh lib       # shared library only
bash build_mi300x.sh test      # validation suite
bash build_mi300x.sh bench     # microbenchmark binary

./tq_validate_mi300x           # must print "9/9 tests passed"
./tq_bench_mi300x 4096 50      # throughput at N=4096 vectors, 50 iterations
```

**Compiler**: `/opt/rocm/bin/hipcc` · **Target**: `gfx942:sramecc+:xnack-`

## Full Benchmark Suite

```bash
bash run_all_benchmarks.sh mistralai/Mistral-7B-v0.1  # ~60–90 min
```

## Benchmark Catalogue

| Script | What it measures |
|---|---|
| `bench_compress_decompress.py` | Compress/decompress GB/s for all methods at multiple N |
| `bench_prefill.py` | Prefill tok/s (TTFT) per method per seq length |
| `bench_ppl_all_methods.py` | WikiText-2 PPL at 3-bit and 4-bit (roundtrip mode) |
| `bench_ppl_proper.py` | PPL via SDPA patching (alternative evaluation path) |
| `bench_batch_decode_v2.py` | Batch decode tok/s vs batch size and seq length |
| `bench_batch_decode.py` | Batch decode TQ3 speedup (bandwidth-bound regime) |
| `bench_all_methods_decode.py` | End-to-end decode tok/s, all methods |
| `bench_quality.py` | Cosine similarity and MSE across layers |
| `bench_konly_ppl.py` | K-only vs K+V compression PPL tradeoff |
| `bench_niah.py` | Needle-in-Haystack retrieval accuracy vs compression |
| `bench_kernels.py` | TurboQuant Triton kernel throughput progression |
| `bench_tq_attention.py` | TQ3 fused attention throughput vs FP16 SDPA |
| `bench_triton_attention.py` | FP16 vs Python-TQ3 vs Triton-fused-TQ3 attention |
| `bench_batch_attention.py` | SDPA at multiple batch sizes (TQ3 vs FP16) |
| `bench_mfma_rotate.py` | MFMA rotation kernel vs torch.matmul |
| `bench_runtime_ratio_all_methods.py` | Empirical compression ratio on live model KV cache |
| `bench_large_models.py` | Mistral-70B / Llama-3-70B scaling (MI300X multi-GPU) |
| `bench_flash_attn_check.py` | SDPA dispatch investigation, effective BW on ROCm |
| `bench_vllm_serving.py` | vLLM end-to-end serving throughput (FP16 vs IsoQuant) |
| `validate_triton_e2e.py` | Correctness + throughput: Triton TQ3 vs FP16 SDPA |
| `profile_rocprof.py` | ROCm hardware counter collection |

## ROCm / HIP ABI note

The **HIP runtime linked into the Python process** (via PyTorch) and a **system `hipcc`** build of standalone HIP may disagree on **code-object version** (COV5 vs COV6). Loading a mismatched `libturboquant_mi300x.so` via **ctypes** in the same process as PyTorch can fail with **HIP error 209** (`hipMemcpyToSymbol`, module load).

**For Python use**: `kernels/turboquant_mi300x.py` provides an equivalent
pure-PyTorch path; the rotation GEMM routes through
`torch.matmul → rocBLAS → MFMA`.

**For isolated C benchmarks**: build and run **`kernels/hip/`** binaries inside the **ROCm 7.2 Primus** workflow (`docker_run_amd_mi300x.sh`) so toolchain and runtime match.

## Attention Implementation Note

- `attn_implementation="sdpa"` — works up to 131K context (use this)
- `attn_implementation="eager"` — OOM at seq ≥ 32K (quadratic activations: 137 GB)
- `flash_attention_2` — requires `flash-attn` package (not installed on this system)

## One-command Showcase

To run all screenshot-style experiments (measured compression grid + attention quality + latency table + roofline + KV memory charts) in one go:

```bash
cd amd-experiments
./scripts/run_showcase.sh
```

Primary outputs:

- `results/bench_compression_ratio_grid.json`
- `results/bench_turboquant_showcase.json`
- `report/figures_v2/fig27_tq_attention_quality_hist.png`
- `report/figures_v2/fig28_mi300x_roofline_tq_attention.png`
- `report/figures_v2/fig29_kv_cache_memory_curve.png`
- `report/figures_v2/fig30_kv_component_breakdown.png`

## Reading Order

1. This `README.md` — layout and quick start
2. **`docs/benchmark_mi300_locked_env.md`** — canonical MI300X + vLLM benchmark Python path and lock procedure
3. `report/final_report_v2.md` — full four-method narrative with tables and figures
4. `current.md` — detailed phase-by-phase status, blockers, findings
5. `research.md` — TurboQuant algorithm deep-dive + MI300X porting notes
6. `kernels/hip/turboquant_mi300x.h` — HIP kernel architecture and block format

## Component READMEs

- **`docs/benchmark_mi300_locked_env.md`** — canonical locked vLLM + torch benchmark environment
- `benchmarks/README.md` — runnable benchmark entry points and quick commands
- `baselines/README.md` — FP16/FP8/INT4 baseline commands
- `scripts/README.md` — consolidation/showcase utility commands
- `profiling/README.md` — profiling utility usage
- `analysis/README.md` — plotting command reference
- `report/README.md` — report figure generation commands
- `kernels/README.md` — kernel component map and runnable HIP entry points
- `report-ui/README.md` — run/build the interactive report UI and refresh its content assets
- `kernels/hip/README.md` — build/test/benchmark HIP binaries on MI300X
- `notebooks/README.md` — use the consolidation notebook or equivalent script commands
- `docs/README.md` — consolidated documentation map and summary-generation workflow

## External References

- **TurboQuant** (Agarwal et al., Google 2024): [arXiv:2406.12820](https://arxiv.org/abs/2406.12820)
- **RotorQuant** (Pope 2026 — Clifford Cl(3,0) rotors): [scrya.com/rotorquant](https://www.scrya.com/rotorquant/)
- **RotorQuant repo** (IsoQuant / PlanarQuant / RotorQuant Triton kernels): [github.com/scrya-com/rotorquant](https://github.com/scrya-com/rotorquant)
- **TurboQuant first implementation** (kernel taxonomy table): [github.com/DevTechJr/turboquant_cutile](https://github.com/DevTechJr/turboquant_cutile)
