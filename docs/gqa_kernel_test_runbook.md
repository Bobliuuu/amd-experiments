# GQA Kernel — Test & Tune Runbook

Runbook for validating and tuning the GQA-aware TurboQuant Triton kernel on AMD MI300X (gfx942). The kernel + scaffolding lives in:

- `kernels/tq_triton.py` — `_tq3_gqa_attention_kernel`, `_tq3_gqa_splitk_partial_kernel`, `turboquant_gqa_attention_fwd`
- `tq_backends/attention/backends/rocm_flash_attn.py` — dispatch behind `VLLM_TQ_USE_GQA_KERNEL=1`
- `benchmarks/validate_tq_gqa_fused_decode.py` — 3-way correctness check
- `benchmarks/bench_tq_gqa_decode_paths.py` — Mistral 32/8 decode sweep
- `benchmarks/bench_tq_gqa_kernel_ablation.py` — `(B, S_k, gqa_ratio)` sweep

## Setup

```bash
cd /path/to/amd-experiments
source .benchmark_mi300_vllm_frozen/.venv/bin/activate
export PYTHONPATH=$PWD/kernels:$PWD:$PYTHONPATH
```

## Step 1 — Correctness (must pass before anything else)

```bash
python3 benchmarks/validate_tq_gqa_fused_decode.py
```

**Pass:** prints `ALL 12 cases PASS`. Each row shows `cos(gqa,dec) ≥ 0.92` and `cos(gqa,expand) ≥ 0.99`.

**Fail modes:**

- `cos(gqa,dec) < 0.92` — kernel is numerically wrong. Stop and debug the kernel before proceeding. Likely culprits: wrong M-axis index math (check `g_off`/`s_q_off`/`q_head` derivation), wrong norm broadcasting, or wrong store mask.
- `cos(gqa,expand) < 0.99` but `cos(gqa,dec) ≥ 0.92` — likely Split-K reduce-order drift. Continue to Step 2; revisit if perf looks fine.
- Triton compile error — capture the full `triton.compiler.errors.CompilationError` stack trace. Most common cause: a `tl.constexpr` arg used in a way Triton can't fold. Check `gqa_ratio` and `BLOCK_M`/`BLOCK_N` usage.
- OOM at `seq_len=65536` — drop to `--seq-len 32768`. Note which shapes OOM'd; the kernel itself shouldn't allocate much, so OOM here means the FP16 reference SDPA's KV materialization can't fit. Lower the seq_len.

## Step 2 — Headline perf (Mistral 32/8 decode sweep)

```bash
python3 benchmarks/bench_tq_gqa_decode_paths.py \
    --sweep 1024,4096,8192,16384,32768,65536 \
    --json-out bench_tq_gqa_decode_sweep.json
```

**Targets at `seq_len=32768`:**

| `speedup_vs_expand` | Verdict |
|---|---|
| `≥ 3.0` | Ship |
| `[2.0, 3.0)` | Tune (Step 4) |
| `[1.0, 2.0)` | Suboptimal autotune; iterate Step 4 |
| `< 1.0` | GQA kernel slower than expand. Stop and investigate — likely culprits: wrong tile layout, register spill (Step 5), or a bug in Split-K reduce |

`speedup_vs_decompress` is informational (continuity with prior reports), not a target.

## Step 3 — Ablation across `(B, S_k, gqa_ratio)`

```bash
python3 benchmarks/bench_tq_gqa_kernel_ablation.py \
    --json-out bench_tq_gqa_kernel_ablation.json
```

**Sanity checks per row:**

- `gqa_ratio=1`: `gqa_ms` within 5% of `expand_ms`. If `gqa_ms > 1.05 × expand_ms`, the passthrough has overhead — check that `turboquant_gqa_attention_fwd` is actually short-circuiting via the early `if gqa_ratio == 1: return turboquant_attention_fwd(...)`.
- `gqa_ratio=2`: speedup in `[1.5, 2.0]×`.
- `gqa_ratio=4`: speedup in `[2.5, 4.0]×`.
- `gqa_ratio=8`: speedup in `[4.0, 8.0]×`. If any `gqa_ratio=8` cell crashes or speedup `< 2×`, suspect register spill — Step 5.
- `kv_bytes_ratio` should equal `gqa_ratio` exactly (it's a computed quantity, sanity that bookkeeping matches).
- `peak_mem_delta_MB` should be positive and scale with `gqa_ratio` and `S_k`. If negative or zero, the GQA kernel is allocating more than expand — check the partial-tensor shapes in the Split-K branch.

## Step 4 — Autotune (only if Step 2/3 misses targets)

Configs live in `kernels/tq_triton.py` in the `@triton.autotune` decorator above `_tq3_gqa_attention_kernel`.

**See what the autotuner picked:**

```bash
TRITON_PRINT_AUTOTUNING=1 python3 benchmarks/bench_tq_gqa_decode_paths.py --seq-len 32768
```

This dumps `(key) → chosen config` per kernel launch. Compare across `gqa_ratio` to spot patterns (e.g., "always picks BLOCK_N=128" → drop BLOCK_N=64 configs).

**Things to try, one at a time, re-running Step 2 between each:**

1. **Add wider BLOCK_N:**
   ```python
   triton.Config({"BLOCK_M":  4, "BLOCK_N": 256}, num_warps=8, num_stages=2),
   triton.Config({"BLOCK_M":  8, "BLOCK_N": 256}, num_warps=8, num_stages=2),
   ```
2. **Add `num_stages=1` variants** for existing entries (less pipeline depth = lower register pressure = better occupancy at small batch).
3. **Drop the `BLOCK_M=32` config** if autotune keeps picking it but perf is mediocre — too large for decode.
4. **Add `BLOCK_M=2`** specifically for `gqa_ratio=2`.

**Split-K params** are hardcoded in `turboquant_gqa_attention_fwd`:

```python
split_block_n = 64
split_block_m = max(gqa_ratio, 4)
split_kv      = max(2048, split_block_n)
```

If wall time at `S_k ≥ 32768` looks bad, A/B Split-K by setting `use_split_k=False` on the call inside the bench script (in `bench_tq_gqa_decode_paths.py`'s `_bench_one`, find the `gqa_fwd(..., use_split_k=True)` call and flip). If it's faster without Split-K, the threshold is wrong; raise the `S_k >= 4096` heuristic or change `BLOCK_KV`:

- `split_kv = 4096` — fewer splits, less reduce overhead.
- `split_kv = 1024` — more splits, better occupancy at small `B`.

Re-run Step 2 after each change.

## Step 5 — Register spill at `gqa_ratio=8`

**Symptom:** `gqa_ratio=8` rows have low speedup (`< 2×`) or crash with register-related errors.

**Diagnose:**

```bash
TRITON_PRINT_KERNEL_INFO=1 python3 benchmarks/bench_tq_gqa_kernel_ablation.py \
    --gqa-ratio 8 --batch 1 --seq-k 32768 2>&1 | grep -E "(VGPR|SPILL|register)"
```

If `VGPR_USAGE > 256` per wavefront or any line contains "spill", you're spilling.

**Fixes (in order of effort):**

1. Drop the `BLOCK_M=8` configs from autotune for the `gqa_ratio=8` case — force `BLOCK_M=4` so the M-tile holds half the group at a time. The kernel will be invoked twice per (B, H_kv) with different M-tile offsets; Split-K already covers this naturally.
2. Lower `BLOCK_N` to `32` in the autotune config space — smaller K-tile = smaller `k_centroids` register footprint.
3. If still spilling, implement sub-batched fallback: in the wrapper, when `gqa_ratio == 8`, call the kernel twice with `gqa_ratio=4` halves of Q and concat outputs. Costs 2× kernel launches but each fits in registers.

## Step 6 — HBM traffic verification (optional)

Confirms the kernel is actually reading less from HBM, not just running faster from cache effects.

```bash
cat > /tmp/tq_pmc.txt <<EOF
pmc: FETCH_SIZE
EOF

rocprof --pmc FETCH_SIZE -i /tmp/tq_pmc.txt -o gqa_rocprof.csv \
    python3 benchmarks/bench_tq_gqa_decode_paths.py --seq-len 32768
```

Look at the row for `_tq3_gqa_attention_kernel` (or `_tq3_gqa_splitk_partial_kernel`) vs the row for `_tq3_attention_kernel` (or `_tq3_splitk_partial_kernel`). `FETCH_SIZE` ratio should be ≈ `1/gqa_ratio` = `0.25` for Mistral. If it's not, the kernel is reading more than it should — likely a bug in the K/V pointer arithmetic (check that `k_base = batch_idx * stride_kb + head_kv_idx * stride_kh` uses `head_kv_idx`, not `q_head`).

## Decision tree TL;DR

| Symptom | Meaning | Action |
|---|---|---|
| Validate fails cos | Kernel numerical bug | Stop, debug kernel |
| `gqa=1` row regresses > 5% | Passthrough overhead | Verify the early-return in the wrapper |
| `gqa=4 @ S_k=32K` speedup ≥ 3× | Done | Ship |
| `gqa=4 @ S_k=32K` speedup 2–3× | Suboptimal autotune | Step 4 |
| `gqa=4 @ S_k=32K` speedup < 2× | Wrong tile layout or bug | Investigate; check Step 5; check rocprof Step 6 |
| `gqa=8` crash or speedup < 2× | Register spill | Step 5 |
| `peak_mem_delta ≤ 0` | Allocation bug | Check Split-K partial tensor shapes |
| Triton compile error | constexpr/typing issue | Read the trace; usually one specific tensor op |

## Files to update on success

Once perf hits target, update these to reflect the real numbers:

- `summary.md` — replace the "GQA fused decode 6–7×" line with the new `speedup_vs_expand` numbers from `bench_tq_gqa_decode_sweep.json`. Add a Mistral B=1 ctx=32K SWA-on TQ3-vs-FP16 decode tok/s ratio (re-run `bench_tq3_decode.py --swa on` with `VLLM_TQ_USE_GQA_KERNEL=1` against `baselines/fp16_baseline.py --swa on`).
- `docs/swa_audit_and_gqa_plan.md` — append a "Results" section to §6 with cosines from `validate_tq_gqa_fused_decode.py` and the ablation JSON.
