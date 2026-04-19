# Engineering report: GQA fused TQ3 decode (MI300X / ROCm 7.2)

## 1. What was broken

- **`tq_backends/attention/backends/rocm_flash_attn.py`** (formerly under a `vllm/` stub): `_forward_decode_fused` used to return decompress whenever `gqa_ratio != 1`; **fixed** with `expand_tq_compressed_for_gqa` so fused Triton runs for Mistral-style GQA.
- **No dispatch visibility**: hard to prove in logs whether fused vs decompress ran.
- **Microbench vs serving gap**: isolated `turboquant_attention_fwd` wins did not apply to the vLLM-shaped decode entrypoint for GQA.

## 2. What changed

| Area | Change |
|------|--------|
| `tq_backends/attention/backends/rocm_flash_attn.py` | Added `expand_tq_compressed_for_gqa()` — `repeat_interleave` on head dim for `k_planes`, `k_norms`, `v_planes`, `v_norms` (same semantics as FP16 `_expand_kv_for_gqa`). |
| Same | Removed the `gqa_ratio != 1` fused block; fused decode runs whenever Triton import succeeds. |
| Same | `VLLM_TQ_LOG_DISPATCH=1` prints `[TQ_DISPATCH]` lines for fused vs decompress paths (`num_q`, `num_kv`, `gqa_ratio`, `seq_lens`). |
| Package layout | Renamed stub package to **`tq_backends/`** so PyPI `vllm` is not shadowed; benchmarks import `tq_backends.attention.backends.rocm_flash_attn`. |
| `benchmarks/validate_tq_gqa_fused_decode.py` | Paged-cache correctness: fused vs decompress+SDPA for GQA 32/8. |
| `benchmarks/bench_tq_gqa_decode_paths.py` | Timings + optional `--sweep` → `results/bench_tq_gqa_decode_sweep.json`. |

**Wiring for full vLLM measurement (follow-up in repo):**

- `docs/vllm_turboquant_wiring.md`, `scripts/install_turboquant_vllm_backend.sh`, `benchmarks/vllm_turboquant_registry.py`, `requirements-vllm-rocm.txt`, `bench_vllm_turboquant_e2e_sweep.py`.
- **HuggingFace** still does not call this backend without a custom attention swap.

## 3. Does fused GQA decode work?

**Yes**, with evidence:

- `validate_tq_gqa_fused_decode.py` on MI300X VF (Primus): **cosine similarity 1.000000** between `_forward_decode_fused` and `_forward_decode_decompress` outputs at `seq_len=2048`, **max abs error 7.6e-5** (FP16 quantization noise).

## 4. Before vs after (decode attention step only)

Synthetic **one-layer** paged TQ3 cache, **batch=1 decode token**, **32/8 GQA**, median ms over 10 reps (Primus, April 2026):

| seq_len | Fused ms | Decompress+SDPA ms | Speedup |
|--------:|---------:|-------------------:|--------:|
| 1024 | 5.18 | 35.58 | **6.9×** |
| 4096 | 19.19 | 142.73 | **7.4×** |
| 16384 | 77.43 | 576.10 | **7.4×** |
| 32768 | 155.21 | 1147.07 | **7.4×** |

**Before:** fused path was **unreachable** for GQA (always decompress). **After:** fused path runs and is **~6.7–7.5× faster** than decompress+SDPA on this harness.

Raw JSON: `results/bench_tq_gqa_decode_sweep.json`.

## 5. What dominates now

- **GQA fallback was the main blocker** for using fused Triton on the vLLM-style decode path for Mistral-like models. That is **removed** for the attention submodule tested here.
- **Within this attention step**, decompress+SDPA was **~7× slower** than fused at 4K–32K; the next relative lever for **whole-model** tok/s is still **FFN/GEMM / weight traffic** (unchanged by this diff — see existing `profile_full_model_decode.py` bucket splits on HF Mistral).

## 6. Final verdict

**Does KV compression help on MI300X once the implementation is wired?**  
For the **attention operator on the actual TQ3 paged decode path with GQA**: **yes** — fused decode is now used and is **several× faster** than decompress+SDPA in our measurements.

**What is the true remaining bottleneck for end-to-end serving tok/s?**  
This change does not shrink **MLP matmuls** or **weight bandwidth**. Until full **vLLM** (or HF) runs are benchmarked with TQ enabled on all layers, the conservative answer is: **attention was a false “solved everywhere” story for GQA; it is now unblocked on this path, and the ceiling for total decode latency is likely shared with GEMM/FFN** — re-profile full stack after wiring TQ into your chosen inference entrypoint.

---

### How to reproduce

```bash
cd /path/to/amd-experiments
export PYTHONPATH=./kernels:.
export VLLM_TQ_USE_FUSED_KERNEL=1   # for fused
python3 benchmarks/validate_tq_gqa_fused_decode.py
python3 benchmarks/bench_tq_gqa_decode_paths.py --sweep 1024,4096,8192,16384 --json-out bench_tq_gqa_decode_sweep.json
# Optional: prove dispatch
export VLLM_TQ_LOG_DISPATCH=1
```

Docker (Primus): `bash docker_run_amd_mi300x.sh -- bash -lc 'export AMDEXP_USE_SYSTEM_PYTHON=1 PYTHONPATH=/workspace/amd-experiments/kernels:/workspace/amd-experiments && cd /workspace/amd-experiments && python3 benchmarks/validate_tq_gqa_fused_decode.py'`

---

### Framing: memory story vs speed story

**Story 1 (memory):** smaller KV is a **production capacity** win (`max_model_len`, HBM headroom, scheduling) regardless of batch=1 tok/s.

**Story 2 (speed):** this note proves **GQA can use the fused attention path** and wins big **vs decompress in isolation**; **end-to-end** vLLM tok/s may still not separate until the **non-KV** decode path is lean enough — see **`report/final_report_v2.md` §13**, **`report/paper.md` §6.6–6.7**, and **`fig29_story_e2e_vs_isolated_attention_comparison.png`** from **`python3 report/generate_figures_v2.py`** (two charts; words in that section explain the gap).
