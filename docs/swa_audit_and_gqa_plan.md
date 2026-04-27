# SWA Audit, Bench State & GQA Kernel Plan

Working notes from a full read of every benchmark and baseline file in this repo. Captures (a) which benches honor sliding-window attention vs which ignore it, (b) two PPL benches that are broken in ways unrelated to SWA, (c) the vLLM SWA patch situation, (d) what the recently-landed `--swa` flag does, (e) the plan for a GQA-aware Triton kernel.

## 1. Project state recap

KV-cache compression benchmark suite for AMD MI300X. Compares **TurboQuant** (TQ3/TQ4), **IsoQuant**, **PlanarQuant**, **RotorQuant** on Mistral-7B-v0.1. Headline (`summary.md`): all four methods reach ~0.983 cosine sim at 4.92× compression; PlanarQuant fastest prefill (1,126K tok/s); WikiText-2 PPL ≈ 7.77 (TQ4) vs 7.82 (FP16). Hardware target: gfx942, 192 GB HBM3, ROCm 7.2 Primus + PyTorch 2.10 + vLLM 0.19 in `<repo>/.benchmark_mi300_vllm_frozen/.venv`.

Recent commit history shows the suite was finalized then "fix all benchmarks" / "add final nits". The code itself is in a shipped state. The audit below explains what those numbers do and don't measure.

## 2. SWA audit (Mistral-7B-v0.1 has `sliding_window: 4096`)

Two notions of "respects SWA" get conflated everywhere:

- **Attention math**: SDPA reads the model's mask, which applies the windowed-causal pattern. So output values reflect a windowed model.
- **Cache memory traffic**: HF `DynamicCache` is full-length regardless of window. Only `SlidingWindowCache` truncates. The TQ kernels are oblivious. So all reported VRAM, prefill ms, decode latency reflect O(seq) traffic that wouldn't happen in a properly-windowed serving stack.

### Audit verdict by group

#### Group A — Pure synthetic / kernel microbench (SWA inapplicable)

These have a `--model` flag for show but never call `AutoModelForCausalLM`. They build `torch.randn(B, H, S, D)` at the requested seq_len.

| Script | What it does |
|---|---|
| `bench_kernels.py` | TQ3 compress/decompress GB/s on synthetic `(N, 128)` tensors |
| `bench_compress_decompress.py` | Same, all four methods |
| `bench_mfma_rotate.py` | MFMA rotation kernel correctness + GB/s |
| `bench_compression_ratio_grid.py` | Ratio at varying ctx via random tensors |
| `bench_block_kv_sweep.py` | `BLOCK_KV` autotune for fused attn at `seq_k=32768` |
| `bench_tq_attention.py` | FP16 SDPA vs TQ3 attention, synthetic K/V up to 131K |
| `bench_triton_attention.py` | FP16 vs Python TQ3 vs Triton TQ3 fused, up to 131K |
| `bench_batch_attention.py` | TQ3 bit-plane vs nibble vs FP16 SDPA at batch 1–32 |
| `bench_flash_attn_check.py` | SDPA dispatch probe + bandwidth check |
| `bench_all_methods_decode.py` | "Decode" against full-length synthetic K/V |
| `bench_batch_decode_v2.py` | Synthetic batch decode + theoretical bandwidth model |
| `bench_turboquant_showcase.py` | Quality histograms + roofline + memory plots |
| `validate_triton_e2e.py` | Triton TQ3 attention correctness vs FP16 SDPA |

Honest as kernel numbers. **Misleading when cited as "compression matters at long context"** — a real SWA cache wouldn't be that big. The headline numbers in `summary.md` come from this group.

#### Group B — Real model loaded, SWA mask active, full-length cache in memory (IGNORES-SWA)

`AutoModelForCausalLM.from_pretrained(..., attn_implementation="sdpa")` with default config. Mistral's window=4096 applies to the mask, but `DynamicCache` stores all K/V.

| Script | Default seq_lens | Verified pattern |
|---|---|---|
| `baselines/fp16_baseline.py` | 512–131072 | Standard `model(prompt_ids, use_cache=True)` + decode loop. VRAM at 131K = 106.9 GB (full O(n) cache). |
| `baselines/fp8_baseline.py` | 512–131072 | Same protocol + one-time FP8 round-trip post-prefill. |
| `baselines/int4_baseline.py` | 512–131072 | INT4 nibble-pack round-trip per decode step. |
| `bench_prefill.py` | 512–32768 | E2E uses sdpa; standalone is synthetic. |
| `bench_quality.py` | ctx=512 | `attn_implementation="eager"`. See §3 — has a separate non-SWA bug. |
| `bench_ppl_all_methods.py` | n_tokens=2048 | Loaded with sdpa. See §3 — fundamentally broken. |
| `bench_ppl_proper.py` | ctx=512 | SDPA monkey-patching, `use_cache=False`. **Sound at default settings.** |
| `bench_konly_ppl.py` | ctx=512 | Same SDPA-patching pattern. **Sound.** |
| `bench_batch_decode.py` | seq=8192/32768, batch=1–64 | Theory section assumes O(seq) KV; "crossover batch ≈ W/K_per_seq" is wrong on a SWA model. |
| `bench_tq3_decode.py` | 512–131072 | Real model + per-step KV roundtrip. |
| `bench_runtime_ratio_all_methods.py` | seq=2048 | Within window — measurement is correct for what HF stores. |
| `bench_measured_cache_memory.py` | seq=512 | Same. **Correction:** `DynamicCache` does not auto-window — at seq>4096 this would be a full-length cache, not a windowed one. |
| `bench_hf_decode_e2e_sweep.py` | 4K–32K | Reuses `bench_batch_decode.bench_one`. Same caveats. |
| `bench_large_models.py` | 8K–131K | Capacity table claims "Mistral-7B TQ3 max context = 6.9M tokens" treating KV as O(n). On a real SWA-paged system this is conceptually wrong (cache wraps at 4K). |
| `decode_bottleneck_smoke.py` | input=128 | Within window. |
| `profile_full_model_decode.py` | seq=8192 | Profiles full O(n) attention path. |

#### Group C — SWA explicitly disabled (TWO scripts, only ones that mention `sliding_window` at all)

| Script | Mechanism |
|---|---|
| `bench_tq_gqa_decode_paths.py` | `sliding_window=None` to `TurboQuantROCmAttentionImpl` (line 61) |
| `validate_tq_gqa_fused_decode.py` | Same (line 76) |

These test the GQA fused decode kernel against a decompress+SDPA reference, with windowing turned off in the metadata. Kernel correctness/throughput, not SWA serving.

#### Group D — vLLM serving (PARTIAL, depends on a patch)

vLLM behavior owns the actual SWA path. Verified: **upstream vLLM disables the custom paged-attention path whenever `sliding_window != 0`**, forcing the Triton fallback. The patch script exists exactly for this:

```bash
"$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/python" \
    scripts/patch_vllm_rocm_sliding_window_custom_paged.py
```

Without it, **all vLLM benches on Mistral run on the Triton fallback**, including the headline result in `bench_vllm_turboquant_ab_sweep_kv_heavy.json` ("FP16 ≈ TQ decompress ≈ TQ fused"). That comparison may be fallback-vs-fallback.

| Script | `max_model_len` |
|---|---|
| `bench_vllm_serving.py` | 4096 (within window) |
| `bench_vllm_isoquant_smoke.py` | 512 |
| `bench_vllm_turboquant_ab.py` | 4096 default |
| `bench_vllm_turboquant_e2e_sweep.py` | sweeps 4096–8192 |
| `bench_vllm_serving_isoquant.py` | 8192 (past window) |
| `bench_vllm_serving_path_ab.py` | 4096 |
| `bench_vllm_quant_quality_smoke.py` | 2048 |

#### Group E — SEMANTICS-BROKEN under SWA

`bench_niah.py` (default mode) — Mistral with sdpa, needle at position 0.1 in 65K context = needle 58K tokens before the question. SWA=4096 means the model literally cannot attend to it. Any "PASS" reflects guessing or haystack repetition. `bench_niah.py --synthetic` is a pure tensor-rank test, no model — valid.

## 3. Two PPL benches that are broken (unrelated to SWA)

This is a bigger issue than SWA and worth flagging separately.

### `bench_ppl_all_methods.py` — does not actually compress

- The `CompressedKVHook` class is defined and **never instantiated**.
- `compute_ppl_compressed` runs an FP16 forward pass and returns FP16 PPL with no compression applied.
- `compute_ppl_with_kv_noise` adds `torch.randn(...) * sqrt(1 - cos²) * 0.1` to the **logits** and pretends that's a quantization effect.

Whatever's in `results/bench_ppl_all_methods.json` is not measuring compressed PPL. Headline numbers from this bench should not be trusted.

### `bench_quality.py` — quantization runs but isn't measured

- `evaluate_perplexity` does `out = model(input_ids, use_cache=True)` then `kv_wrapper.quantize_kv(out.past_key_values)`.
- But the loss is computed from `out.logits` which was produced **before** the quantize call.
- The `KVQuantWrapper` modifies the cache in-place; that cache is then discarded at the end of the loop iteration.

So the per-window loss reflects FP16 forward, not the quantized cache. Results are FP16 PPL relabeled as different schemes.

### What still works

`bench_ppl_proper.py` and `bench_konly_ppl.py` correctly intercept `F.scaled_dot_product_attention`, compress K/V, then call the original SDPA. The WikiText-2 PPL numbers in `summary.md` come from `bench_ppl_proper.py` — those are real.

## 4. Stray instrumentation

`baselines/fp8_baseline.py` and `benchmarks/bench_quality.py` contain leftover blocks that write JSON to `/root/workspace/.cursor/debug-5ac54c.log`. Cursor agent debugging artifacts. Doesn't affect results, should be removed.

## 5. SWA flag — what was just shipped

Commit `5cf705e` adds `--swa {on,off}` and `--window N` flags across the suite (default `--swa off`, backward-compatible).

### Helper module: `kernels/cache_utils.py`

| Function | Purpose |
|---|---|
| `truncate_kv_to_window(cache, window)` | In-place: keeps last `window` tokens of `cache.layers[i].keys/values`. Re-contiguous slice. |
| `get_swa_window(model)` | Returns `model.config.sliding_window` if positive, else None. |
| `resolve_swa_window(swa, model, window)` | Resolves CLI args. `off` → None. `on` → explicit `window` if >0, else config, else raises. |
| `clamp_seq_to_window(seq_len, swa, window)` | For synthetic benches: `min(seq_len, window)` when on. |
| `add_swa_args(parser)` | Registers `--swa`/`--window` on argparse parsers. |
| `print_swa_status(swa, window)` | One-line console status. |
| `vllm_swa_warn(swa, max_model_len)` | Reminds operator about the patch script and `max_model_len > window` requirement. |

### Coverage

- **HF-direct** (truncate cache after each decode step): 9 files — 3 baselines + `bench_tq3_decode`, `bench_batch_decode`, `bench_hf_decode_e2e_sweep`, `bench_large_models`, `bench_quality`, `profile_full_model_decode`.
- **Synthetic** (clamp seq_len to window when building synthetic K/V): 7 files — `bench_all_methods_decode`, `bench_batch_decode_v2`, `bench_tq_attention`, `bench_batch_attention`, `bench_triton_attention`, `bench_runtime_ratio_all_methods`, `bench_measured_cache_memory`.
- **vLLM** (record flag in JSON, warn about patch): 11 files — all `bench_vllm_*.py` + `decode_bottleneck_smoke`, `story2_env_gate`. Subprocess wrappers forward `--swa`/`--window` to inner scripts.
- **NIAH**: `_swa_safe_needle_position` pushes the needle into the last `window − 256` tokens when ctx > window, so retrieval is testable under SWA. Synthetic mode clamps `n_tokens`.

### Untouched (intentional)

`bench_compress_decompress.py`, `bench_kernels.py`, `bench_mfma_rotate.py`, `bench_compression_ratio_grid.py`, `bench_block_kv_sweep.py`, `bench_flash_attn_check.py`, `bench_tq_gqa_decode_paths.py`, `validate_triton_e2e.py`, `validate_tq_gqa_fused_decode.py`, `bench_ppl_proper.py`, `bench_konly_ppl.py`, `bench_ppl_all_methods.py`, `bench_turboquant_showcase.py`, `spike_vllm_rocm_quant.py`, `write_path_verification.py`, `decode_whole_step_golden_driver.py`, `story2_kv_heavy_quant_compare.py`, `story2_rocprof_summarize.py`, `profile_rocprof.py`. Pure kernel microbenches with no cache concept, post-processing scripts, or already-windowed defaults.

### Verification command

```bash
# Should drop ~8× when SWA on (32K → 4K cache)
python3 benchmarks/bench_measured_cache_memory.py --seq-len 32768 --swa off
python3 benchmarks/bench_measured_cache_memory.py --seq-len 32768 --swa on
```

## 6. GQA-aware Triton kernel — plan (NOT YET BUILT)

### Why the existing path doesn't count

`tq_backends/attention/backends/rocm_flash_attn.py:376` defines `expand_tq_compressed_for_gqa` which calls `k_planes.repeat_interleave(gqa_ratio, dim=1)` and feeds the existing **MHA-shaped** `turboquant_attention_fwd`. Its docstring (line 629) admits: *"This duplicates compressed reads per Q-head group (benchmark / prod tradeoff); a future kernel can index KV heads without materializing."*

So the reported "6–7.5× speedup" for GQA fused decode in `summary.md` is fused-vs-decompress comparison; both legs pay the `gqa_ratio×` HBM cost. The actual GQA hardware win — read each KV head's compressed bytes **once** and let all `gqa_ratio` Q heads in the group consume it — has not been measured.

### Kernel to write

Current signature (`kernels/tq_triton.py:853`):
```
q        : (B, H, S_q, D)         where H = num_q_heads (== num_kv_heads after expand)
k_planes : (B, H, S_k, 48) uint8
```

New:
```python
def turboquant_gqa_attention_fwd(
    q,                # (B, H_q, S_q, D)            float16, pre-rotated
    k_planes,         # (B, H_kv, S_k, 48)          uint8
    k_norms,          # (B, H_kv, S_k)              float32
    v_planes,         # (B, H_kv, S_k, 48)          uint8
    v_norms,          # (B, H_kv, S_k)              float32
    gqa_ratio: int,   # H_q == H_kv * gqa_ratio
    ...
)
```

Launch: `(triton.cdiv(S_q * gqa_ratio, BLOCK_M), B * H_kv)`. Each block serves all `gqa_ratio` Q heads in one KV head's group. Inside the kernel: load K plane bytes for one BLOCK_N slice once, reuse for all `gqa_ratio` Q rows, accumulate `gqa_ratio` softmax states.

### Files to add/modify

1. `kernels/tq_triton.py` — add `_tq3_gqa_attention_kernel`, `_tq3_gqa_splitk_partial_kernel`, `turboquant_gqa_attention_fwd` Python wrapper.
2. `tq_backends/attention/backends/rocm_flash_attn.py` — new dispatch in `_forward_decode_fused` gated by `VLLM_TQ_USE_GQA_KERNEL=1`. Old expand+MHA path remains as fallback.
3. `benchmarks/validate_tq_gqa_fused_decode.py` — three-way comparison: decompress+SDPA (golden) vs expand+MHA vs new GQA kernel.
4. `benchmarks/bench_tq_gqa_decode_paths.py` — add `gqa_kernel_ms` column.
5. `benchmarks/bench_tq_gqa_kernel_ablation.py` (new) — sweep `(B, S_k, gqa_ratio)`.

### Validation targets

- Cosine ≥ 0.92 vs SDPA reference at `seq ∈ {1024, 8192, 65536}`, `gqa_ratio ∈ {2, 4, 8}` (Mistral=4, Llama-3-70B=8).
- At `(B=1, H_kv=8, gqa_ratio=4, S_k=32768, S_q=1)`: ≥ 3× faster than expand+MHA path (ideal 4× minus VALU overhead).
- At `gqa_ratio=1` (MHA): within 5% of existing kernel.
- `torch.cuda.max_memory_allocated()` drops by ~`(gqa_ratio−1)/gqa_ratio` of K/V allocation.

### Risks

- **Block size**: BLOCK_M=4 (one Q row per group element at decode) is small; existing kernel autotunes to BLOCK_M=16. Needs separate autotune config space.
- **Register pressure**: holding `gqa_ratio` softmax states might exceed gfx942 VGPR budget at `gqa_ratio ≥ 8` (Llama-70B). Fallback: process group in sub-batches of 4.
- **Triton 3.1 on ROCm**: the existing v3 kernel was 10% slower than v2 because of 3D `tl.reshape` overhead. Try strided-pointer access first, only reshape if necessary.

### Estimated effort

3–5 days for kernel + Split-K variant + validation + bench scripts + autotune. +2 days for nibble-format variant.

## 7. The honest production number

After SWA-on AND GQA-aware kernel, the Mistral-7B TQ3-vs-FP16 decode tok/s ratio at `(B=1, ctx=32K, batch=1)` will be much closer to **1.0×** than the current `summary.md` ratio of ~7×:

- SWA shrinks the cache to 4K (regime where compression helps least).
- GQA-aware kernel removes the `gqa_ratio` waste FP16 was already paying; FP16 gets faster too.

That ratio is the production-deployment number, and neither the current suite nor the published reports actually measure it.
