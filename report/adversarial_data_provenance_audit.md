# Adversarial audit: “too exact” numbers and what is actually experimental on MI300X

This note adopts a deliberately skeptical stance toward headline figures (especially **4.923×**-style ratios) and maps each class of number to its real source: **VM/GPU-reported timing and memory**, **deterministic layout arithmetic**, or **author-chosen literals / stubs**. Nothing here claims results were fabricated; the point is to separate **what must repeat exactly** from **what ought to show run-to-run spread**.

---

## 1. Why **4.923076923…** looks “fake” but usually is not

**Observation:** Many JSON artifacts repeat `4.923076923076923` (or rounded `4.923`) across rows, contexts, and even distinct benchmarks.

**Explanation:** For this repo’s TQ3 packing, “compression vs FP16” for a **single head vector** is not a noisy physical measurement. It is fixed by the **on-disk byte layout**:

- FP16 storage for one `head_dim=128` vector: `128 × sizeof(float16) = 256` bytes (see benchmarks reporting `fp16_bytes_per_vector: 256`).
- TQ3 packed block: **52 bytes** (`TQ3_BLOCK_BYTES` in `kernels/turboquant_mi300x.py`), from the documented layout: 4-byte norm + bitplanes for 128×3 bits.

Therefore

\[
\frac{256}{52} = \frac{64}{13} = 4.\overline{923076}
\]

That is **rational-number exactness**, not GPU telemetry precision. Any benchmark that:

1. counts FP16 bytes as `n_vectors × 256`, and  
2. counts compressed bytes as `n_vectors × 52`,  

must recover **identically** the same ratio for every run, every context length, and every model with the same head geometry—until you change the format.

**Experimental content here:** the run still **executes** `compress_tensor` on real tensors (so allocator behavior, kernel correctness, and tensor shapes are exercised—see `benchmarks/bench_compression_ratio_grid.py`), but the **ratio** is dominated by **integer byte accounting**, not statistical sampling. Expect **zero variance** in `compression_ratio_measured` for TQ3 under that definition.

---

## 2. Taxonomy: where each kind of number comes from

| Kind of metric | Typical source on the MI300X VM | Should it jitter between runs? |
|----------------|----------------------------------|----------------------------------|
| `tokens_per_sec`, `latency_ms`, `prefill_ms` | `time.perf_counter()` around CUDA work + `torch.cuda.synchronize()`; often **median** over `n_runs` | **Yes** (small spread; protocol-dependent). |
| `vram_peak_gb` | `torch.cuda.max_memory_allocated()` / `mem_get_info()` | **Some** stepwise behavior; not arbitrary floats. |
| `device.name`, `total_gb` | `torch.cuda.get_device_name(0)`, `get_device_properties` | **Stable** for a given VM SKU. |
| TQ3/TQ2/TQ4 **storage compression** vs FP16 | **Definition**: `256 / TQ*_BLOCK_BYTES` or equivalent byte counts | **No** (exact repeating decimals). |
| PPL, cosine similarity, kernel GB/s from microbenches | Forward passes / timers on GPU | **Yes** (usually more digits; still protocol-bound). |

So: **throughput and latency are the primary “experimental” quantities** in the sense of wall-clock measurement on the accelerator. **4.923× is primarily a specification checksum.**

---

## 3. Actually suspicious or sloppy patterns (real issues for a skeptic)

### 3.1 One hard-coded literal instead of the shared constant

In `benchmarks/bench_tq3_decode.py`, the fused Triton decode path returns:

- `"ratio_calculated_layout": 4.923` as a **numeric literal**, while the round-trip path uses `float(COMPRESSION_RATIO[bits])` from `turboquant_mi300x.py`.

That is **not** evidence of invented data, but it **is** inconsistent engineering: the “calculated layout” field should always be **`256 / TQ3_BLOCK_BYTES`** (or `COMPRESSION_RATIO[3]`) so a reader never has to wonder if someone typed the headline number by hand.

### 3.2 Figure-generation stubs are explicitly non-experimental

`report/generate_figures_v2.py` contains functions such as `stub_ppl_data()`, `stub_decode_data()`, `stub_batch_decode_data()`, etc., with **embedded floats** (including rounded `4.92` and full-precision `4.923076923076923` in stub tables). Those paths exist for **plots when real JSON is missing or incomplete**.

**Takeaway:** a chart caption that says “MI300X” is only as honest as the **input file** passed to the generator. Always trace the figure back to **`results/*.json`** produced by a benchmark script on this machine.

### 3.3 README / report tables mixing roles

Text that says “confirmed” next to **4.923×** can overstate epistemology: the **layout** is confirmed by arithmetic; **end-to-end serving** still needs separate timing runs (which *do* exist in `results/`, with messier numbers like **43.82 vs 46.5 tok/s** for FP16).

---

## 4. What “all experimental from scratch” can honestly mean

Reasonable strict definitions:

1. **Scratch tensors:** Benchmarks synthesize or load weights and run forwards on GPU; they do not read a pre-printed “answer key” JSON for compression ratio—the ratio falls out of **byte shapes** after real `compress_tensor` calls (`bench_compression_ratio_grid.py`).
2. **Scratch timing:** Decode scripts measure their own loops; nothing in PyTorch “reports 4.923 tok/s” as a built-in—the VM reports **elapsed time**, code divides by token count.
3. **Not scratch:** Deriving **256/52** is mathematics from the **format spec**, not an independent empirical estimate of compression quality (quality is separate: PPL, cosine, etc.).

So the strongest honest sentence is:

> **KV storage compression vs FP16 for this TQ3 layout is a deterministic consequence of the packed format; timing and memory numbers are experimental measurements under the stated protocol.**

---

## 5. How to re-verify without trusting this document

1. Re-run a byte-counting benchmark, e.g. `python3 benchmarks/bench_compression_ratio_grid.py`, and open the emitted JSON: TQ3 rows should still show **exactly** `256/52` in floating point.
2. Re-run decode baselines with more `n_runs` and longer `n_decode`: **tok/s** should move slightly; **4.923076923** for layout ratio should not (unless the layout changes).
3. For any figure, grep the generator for `stub_` and confirm the code path loads **`results/<file>.json`** rather than stub arrays.

---

## 6. Bottom line for the adversarial reader

- **4.923× repeating everywhere is expected** for a fixed **52 B vs 256 B** definition; it is **not** evidence of high-precision fake sampling.
- **True experimental outputs** in this repo are mainly **timings**, **allocator peaks**, and **quality metrics**—those are the numbers that should be interrogated for variance, warmup, batch size, and `n_runs`.
- **Red flags to chase** are **literal constants** where a formula should live, and **figure stubs** mistaken for measured JSON.

This audit is itself **descriptive** of the codebase and result files as they exist in the workspace; it does not replace re-running benchmarks on your own MI300X VM.
