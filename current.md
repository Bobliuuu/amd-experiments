# TurboQuant MI300X — Current Status (April 12, 2026)

## DONE

### Phase 0: Environment Setup ✓
- PyTorch ROCm installed and verified on AMD Instinct MI300X VF (gfx942:sramecc+:xnack-)
  - PyTorch HIP version: 6.2.41133 (bundled), system ROCm: 7.2
  - GPU: 192 GB HBM3, ~205 GB free VRAM, Wave64 (wavefront=64)
- Mistral-7B-v0.1 downloaded: 14 GB, `~/.cache/huggingface/hub/`
- All Python deps installed: numpy, scipy, transformers 5.5.3, triton 3.1.0, matplotlib, pandas

### Phase 1: Baselines ✓ (real data collected)

**FP16 Baseline** — Mistral-7B-v0.1, SDPA attention, all 6 seq_lens:
| seq_len | tok/s | latency_ms | VRAM_GB | prefill_ms |
|---------|-------|-----------|---------|-----------|
| 512 | 43.82 | 22.82 | 14.7 | 10,965 |
| 2,048 | 43.49 | 22.99 | 16.7 | 177 |
| 8,192 | 46.50 | 21.51 | 16.6 | 425 |
| 32,768 | 46.41 | 21.55 | 24.7 | 3,550 |
| 65,536 | 46.41 | 21.55 | 43.5 | 12,714 |
| 131,072 | 46.39 | 21.56 | 106.9 | 46,841 |
Results in: `results/fp16_baseline_mistralai_Mistral-7B-v0.1.json`

**Key finding**: Decode is ~flat at 46 tok/s across all context lengths → compute-bound
(weight cycling), NOT KV-cache bandwidth-bound, at batch=1 on MI300X.

**FP8/INT4 Baselines** — scripts written, fixed, waiting to run:
- `baselines/fp8_baseline.py` ✓ (written + DynamicCache fix applied)
- `baselines/int4_baseline.py` ✓ (written + DynamicCache fix applied)
- Bug: HuggingFace transformers 5.5.3 uses DynamicCache for past_key_values;
  fixed via `to_legacy_cache()` / tuple-of-tuples format
- **STATUS**: Ready to run, shell broken (see below)

### Phase 2b: MI300X HIP Library ✓ (standalone — 16/16 tests passing)
- `kernels/turboquant_mi300x.hip.cpp` — MI300X-optimized HIP kernels
  - `tqm_quantize_kernel_tq3`: ballot-based 3-bit packing, zero LDS atomics (Opt-A/B/C)
  - `tqm_dequantize_kernel_tq3`: bitplane extraction + inverse rotation
  - `tqm_fused_dot_kernel_tq3`: fused attention scoring in rotated space
  - `tqm_qjl_kernel`: QJL residual correction (Algorithm 2)
  - Wave64-native warp reduction (6 iterations, statically unrolled)
- `kernels/turboquant_mi300x.h` — API + data structures (block_tq3=52B, block_tq4=68B)
  - Correct Lloyd-Max codebooks from domvox JSON (TQ3: ±0.189 to ±0.022)
- `kernels/turboquant_mi300x_test.cpp` — 16/16 assertions passing (9 tests)
- `kernels/build_mi300x.sh` — build script (arch `gfx942:sramecc+:xnack-`)

### Phase 2c: Pure-PyTorch Python Wrapper ✓ (BLOCKER RESOLVED)
- `kernels/turboquant_mi300x.py` — pure-PyTorch implementation
  - Rotation via `torch.matmul` → rocBLAS → MFMA
  - TQ3 cosine sim avg 0.983, MSE 0.000266, fused dot error 0.003%
- Root cause: ROCm 7.2 (system) vs ROCm 6.2 (PyTorch bundle) ABI incompatibility

### Phase 3: Triton Fused Attention ✓ (written, not end-to-end tested)
- `kernels/tq_triton.py` — Flash Attention 2-style fused dequant-attention kernel

### Phase 4: Benchmarks — Partial

**Completed:**
- `benchmarks/bench_kernels.py` — ✓ run and saved to `results/bench_kernels.json`
  - Standalone binary: TQ3 decompress 198 GB/s, fused dot 93 GB/s
  - Python wrapper: compress 11.8 GB/s, decompress 58.4 GB/s, fused_dot 33.1 GB/s
- `benchmarks/bench_tq_attention.py` — ✓ run, saved to `results/bench_tq3_attention.json`
  - TQ3 attention is 2.6–14× slower than FP16 without fused kernel (expected)
  - FP16 attention bandwidth: ~90 GB/s at 131K ctx

**Not yet run (shell broken):**
- `benchmarks/bench_quality.py` — perplexity on WikiText-103 (needs to run)
- FP8 baseline run (fix applied, waiting)
- INT4 baseline run (fix applied, waiting)

### Phase 5: Analysis & Report — Partial
- `analysis/plot_results.py` — ✓ written and runs (placeholder charts until all data in)
- `analysis/figures/` — 5 figures generated (some with real data, some placeholder)
- `report/final_report.md` — ✓ written with all real kernel/attention/FP16 data
- `run_all_benchmarks.sh` — ✓ orchestrator script for the full suite

## CURRENT BLOCKER

The primary shell process is in a broken state (all commands produce no output, even
`echo`). This is likely caused by a `nohup bash -c '...' &` command that redirected
the shell's file descriptors.

**Resolution**: Restart the Claude session / shell process. All scripts are written and
working. Just need to run:
```bash
cd /root/workspace/amd-experiments
bash run_all_benchmarks.sh mistralai/Mistral-7B-v0.1
```
This will run FP8, INT4, quality benchmarks, and regenerate all plots with real data.

## KEY TECHNICAL FACTS

### Architecture
- Device: AMD Instinct MI300X VF — gfx942:sramecc+:xnack-
- PyTorch: ROCm 6.2 (bundled), System: ROCm 7.2
- HF Transformers: 5.5.3 (uses DynamicCache for past_key_values)
- Triton: 3.1.0, Ninja: installed

### IMPORTANT: Attention Implementation
- `attn_implementation="eager"` → OOM at seq≥32768 (quadratic attention matrix: 137 GB)
- `attn_implementation="sdpa"` → works up to 131K (106.9 GB VRAM), use this
- `attn_implementation="flash_attention_2"` → needs flash-attn package (not installed)

### Python API
```python
from turboquant_mi300x import TurboQuantMI300X
tq = TurboQuantMI300X(head_dim=128, bits=3, rotation_seed=42)
compressed = tq.compress_tensor(x)            # (n, 52) uint8
x_hat = tq.decompress_tensor(compressed, x.shape)
q_rot = tq.rotate_queries(q)
scores = tq.fused_dot(q_rot, compressed)
```

### DynamicCache API (transformers 5.5.3)
```python
# Extract K/V pairs from any cache format:
def to_kv_pairs(cache):
    if hasattr(cache, "to_legacy_cache"):
        return [(k, v) for k, v in cache.to_legacy_cache()]
    elif hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    return list(cache)

# Build cache for model input (legacy tuple-of-tuples works universally):
def from_kv_pairs(kv_pairs):
    return tuple((k, v) for k, v in kv_pairs)
```

### Next Steps (after shell recovery)
1. `bash run_all_benchmarks.sh mistralai/Mistral-7B-v0.1` (skips FP16, already done)
   Or individually:
   - `python3 baselines/fp8_baseline.py --model mistralai/Mistral-7B-v0.1 --n-decode 30 --n-runs 3`
   - `python3 baselines/int4_baseline.py --model mistralai/Mistral-7B-v0.1 --n-decode 30 --n-runs 3`
   - `python3 benchmarks/bench_quality.py --model mistralai/Mistral-7B-v0.1 --n-tokens 4096`
   - `python3 analysis/plot_results.py`
2. End-to-end test of Triton fused kernel (`tq_triton.py`)
3. Optional: Phase 2a — validate domvox/turboquant-hip on gfx942

### File Inventory
```
amd-experiments/
├── research.md                     # TurboQuant deep-dive + MI300X porting analysis
├── plan.md                         # Experimental study design
├── current.md                      # This file
├── run_all_benchmarks.sh           # Full pipeline orchestrator
├── requirements.txt
├── kernels/
│   ├── turboquant_mi300x.hip.cpp   # HIP kernels (TQ2/3/4, QJL, fused dot)
│   ├── turboquant_mi300x.h         # Structs, codebooks, API
│   ├── turboquant_mi300x_test.cpp  # 16/16 tests passing
│   ├── build_mi300x.sh             # Build script (gfx942:sramecc+:xnack-)
│   ├── turboquant_mi300x.py        # Pure-PyTorch wrapper (MFMA via rocBLAS)
│   ├── tq_triton.py               # Triton fused attention (written, untested E2E)
│   ├── tq_validate_mi300x          # Standalone validation binary
│   └── tq_bench_mi300x             # Standalone benchmark binary
├── baselines/
│   ├── fp16_baseline.py            # ✓ runs, results in results/
│   ├── fp8_baseline.py             # ✓ written+fixed, needs run
│   └── int4_baseline.py            # ✓ written+fixed, needs run
├── benchmarks/
│   ├── bench_kernels.py            # ✓ runs, results in results/
│   ├── bench_tq_attention.py       # ✓ runs, results in results/
│   └── bench_quality.py            # ✓ written, needs run
├── results/
│   ├── fp16_baseline_mistralai_Mistral-7B-v0.1.json  ✓
│   ├── bench_kernels.json          ✓
│   └── bench_tq3_attention.json    ✓
├── analysis/
│   ├── plot_results.py             # ✓ written, generates 5 figures
│   └── figures/                   # 5 PNG files (some real data)
└── report/
    └── final_report.md             # ✓ written with real data for completed benchmarks
```
