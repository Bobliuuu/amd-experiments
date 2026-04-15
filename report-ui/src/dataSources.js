/**
 * dataSources.js — Figure metadata and report content loaders
 *
 * Defines all benchmark figures (V1: fig1–fig10, V2: fig11–fig25) with
 * titles and descriptions sourced from final_report_v2.md.
 *
 * Images are served from /public/content/ (Vite static assets).
 * The /@fs route is intentionally not used — Vite restricts filesystem
 * access outside the project root, returning 403 for paths under /report/.
 */
// Legacy /@fs path kept for reference only — not used for serving
const FS_ROOT = "/@fs/root/workspace/amd-experiments/report";

function makeFigure(base, name, desc) {
  return {
    name,
    label: name.replace(/\.png$/, "").replaceAll("_", " ").replace(/^fig\d+_/, ""),
    desc: desc || "",
    src: `/content/${base}/${name}`,
    // Keep srcCandidates for legacy FigureCard compat
    srcCandidates: [`/content/${base}/${name}`],
  };
}

const V2_FIGS = [
  ["fig11_ppl_vs_compression.png", "PPL vs Compression Tradeoff",
   "Perplexity (lower = better) plotted against compression ratio for all four methods at 3-bit and 4-bit. All 3-bit methods cluster at 4.923×. PlanarQuant achieves the best published PPL (10.62) among 3-bit block methods. TurboQuant's 7.07 is measured with deferred K quantization (K stored FP16 during prefill), making it not directly comparable to strict roundtrip block method numbers."],
  ["fig12_decode_all_methods.png", "Batch Decode Throughput — All Methods",
   "Tokens/sec vs batch size for all four block methods at seq=4096 and seq=16384. All compressed methods plateau at constant throughput regardless of batch size — confirming they are compute-limited by the decompression kernel, not by KV memory bandwidth. FP16 scales with batch (more weight-cycling parallelism). At seq=16384, FP16 achieves 593 tok/s at batch=32 while PlanarQuant3 reaches only 116 tok/s."],
  ["fig13_batch_decode_crossover.png", "Batch Decode Crossover Analysis",
   "Crossover point where KV bandwidth savings from compression begin to outweigh decompression overhead. At seq=4096, FP16 KV only wins at batch > 26. At seq=16384, crossover is at batch ≈ 6.5 — meaning at longer contexts, FP16 is preferable even at moderate batch sizes. Current results use PyTorch SDPA (not Flash Attention); with CK-based ROCm Flash Attention, the FP16 baseline would be 3–5× faster, shifting crossover higher."],
  ["fig14_prefill_comparison.png", "Prefill KV Compression Speed",
   "Tokens/sec for compressing all KV vectors during a prefill pass (no model forward pass). At seq=32768: PlanarQuant achieves 1,126K tok/s (26.5× faster than TurboQuant), IsoQuant 891K tok/s (21.0×), RotorQuant 855K tok/s (20.1×), TurboQuant 42K tok/s (1× baseline). The 26.5× gap directly translates to Time-To-First-Token (TTFT) savings at long contexts. At seq=32768, TurboQuant adds 772ms of overhead vs PlanarQuant's 29ms."],
  ["fig15_compress_decompress_bw.png", "Compress / Decompress Kernel Bandwidth",
   "Compress and decompress kernel throughput (GB/s) measured on 4,096 random float32 vectors (head_dim=128), 50 iterations median. IsoQuant leads at 21.8 GB/s compress / 38.3 GB/s decompress despite having 2× the FMAs of PlanarQuant — because 4D quaternion groups align with CDNA3's SIMD-4 lanes. TurboQuant is 6–9× slower at 2.9/4.4 GB/s due to the O(d²) WHT full-matrix rotation."],
  ["fig16_max_context.png", "Maximum Context Tokens on 192 GB HBM3",
   "Maximum context window size storable in MI300X's 192 GB HBM3 at each bit width. FP16 (256 B/vector): 1.4M tokens. 4-bit (68 B/vector): 5.3M tokens. 3-bit (52 B/vector, all block methods + TQ3): 6.9M tokens. This 4.92× context expansion is the primary production motivation for KV compression on MI300X — more important than decode throughput at batch=1."],
  ["fig17_speed_vs_quality.png", "Speed vs Quality Tradeoff",
   "Scatter plot of prefill compression speed (tokens/sec) vs KV reconstruction cosine similarity. PlanarQuant occupies the optimal top-right corner: highest prefill speed (1,126K tok/s) with competitive quality (cosine 0.9829). RotorQuant is dominated on both axes — slower and no better quality than PlanarQuant. All 3-bit methods cluster tightly on the quality axis (±0.0003), making speed the only real differentiator."],
  ["fig18_roofline.png", "Roofline Model — MI300X gfx942",
   "Roofline analysis for compress/decompress kernels at N=4,096 vectors. All kernels operate at < 1% of MI300X's 5.3 TB/s theoretical peak memory bandwidth. The bottleneck at N=4K is kernel launch overhead and instruction latency, not HBM3 bandwidth. At serving scale (N > 100K vectors), efficiency improves significantly as fixed launch overhead amortizes over more work."],
  ["fig19_fmas_comparison.png", "FMA Count Comparison",
   "Floating-point multiply-add operations required to rotate one 128-dim KV head vector: PlanarQuant (256), IsoQuant (512), RotorQuant (1,176), TurboQuant (16,384). The 64× spread from PlanarQuant to TurboQuant fully explains the kernel throughput gap. RotorQuant uses 4.6× more FMAs than PlanarQuant with no quality benefit, due to Cl(3,0) rotor operations on 3D (non-power-of-2) groups."],
  ["fig20_k_only_ablation.png", "K-Only Compression Ablation",
   "Ablation comparing K-only compression (K at 3-bit, V at FP16) vs full K+V compression. K-only yields better perplexity because the V cache contributes directly to output values with high fidelity requirements, while K cache is used only for attention score computation where some noise is tolerable. PlanarQuant K-only (planar3/f16) achieves ~FP16 PPL at 5.1× compression — the best quality-compression tradeoff in the benchmark."],
  ["fig21_headline_compression_comparison.png", "Headline Compression — Upstream Reference",
   "Upstream llama.cpp / RotorQuant comparison table for Llama 3.1 8B Instruct Q4_K_M on a consumer NVIDIA GPU (RTX 5090). NOT measured on MI300X — included as cross-ecosystem context only. All symmetric 3-bit K+V rows achieve 10.3× in that upstream layout (different byte format than this repo's 52-byte TQ3). PlanarQuant leads on prefill (3,822 tok/s), IsoQuant on PPL (6.91)."],
  ["fig22_cache_compression_mi300x.png", "KV Cache Compression — MI300X Bandwidth",
   "Two-panel summary of MI300X compress/decompress kernel performance. Left panel: compress bandwidth (IsoQuant best at 21.8 GB/s). Right panel: decompress bandwidth (IsoQuant best at 38.3 GB/s). TurboQuant shown as the baseline (2.9/4.4 GB/s). IsoQuant's 4D quaternion structure maps cleanly to CDNA3 SIMD-4 lanes, achieving higher throughput than PlanarQuant despite 2× more FMAs."],
  ["fig23_kv_cache_compression_comparison.png", "KV Cache Compression — Full Comparison",
   "Side-by-side comparison of all methods across storage ratio, compress bandwidth, and decompress bandwidth. All 3-bit methods share identical 4.923× storage compression. The throughput columns show IsoQuant leading, with TurboQuant trailing by 7.5–8.7×. This figure makes clear that all 3-bit block methods achieve the same memory benefit — the choice between them is purely about kernel efficiency."],
  ["fig24_pope_rotorquant_2026_claims.png", "Pope (2026) RotorQuant — Author Claims",
   "Author-reported claims from John D. Pope's RotorQuant publication (CUDA/Metal, not MI300X): speedup over TurboQuant (10–19×), parameter reduction (44× fewer rotation parameters), attention fidelity (99% cosine similarity), and storage compression. These CUDA numbers are compared directly against MI300X measurements in Fig. 25."],
  ["fig25_mi300x_vs_author_claims.png", "MI300X vs Author Claims — Direct Comparison",
   "Direct comparison of Pope's CUDA-reported RotorQuant claims against AMD MI300X measurements from this benchmark. MI300X results: speedup 6.07–8.48× (vs 10–19× claimed on CUDA), parameter reduction 95× (vs 44× — AMD result is actually larger due to higher TurboQuant rotation cost), fidelity 98.30% cosine (vs 99% CUDA). The speedup gap is expected: CUDA WMMA vs ROCm MFMA differ in small-matrix throughput."],
];

const V1_FIGS = [
  ["fig1_throughput_vs_context.png", "Decode Throughput vs Context Length",
   "FP16 baseline achieves ~43–46 tok/s flat across all context lengths (512 to 131K) at batch=1 — confirming MI300X decode is compute-bound (model weight cycling), not KV-bandwidth-bound. TQ3 reaches 6.3 tok/s at 8K context, bottlenecked by Python-level compress/decompress overhead across 32 layers × 8 KV heads. FP8 matches FP16 closely (46.0 tok/s at 8K). INT4 drops to ~26 tok/s due to Python dequantization overhead."],
  ["fig2_latency_vs_context.png", "Per-Token Latency vs Context Length",
   "Per-token decode latency (ms) vs context length. FP16 stable at ~21.5ms across all context lengths — confirming compute-bound behavior. TQ3 reaches ~160ms at 8K context (the compress/decompress loop adds ~138ms overhead per token across 32 layers). FP8 adds only ~0.3ms overhead at 8K. Context length has minimal effect on latency in the compute-bound regime."],
  ["fig3_memory_analysis.png", "VRAM Usage vs Context Length",
   "Peak VRAM consumption vs context length for FP16, FP8, INT4, TQ3, TQ4. At 131K context: FP16 peaks at 106.9 GB (inflated by prefill activation buffers). Steady-state KV cache is ~16 GB for FP16 vs ~3.3 GB for TQ3 at 131K tokens. TQ3 enables fitting 131K context in ~17 GB total VRAM vs 107 GB for FP16 prefill, dramatically expanding practical context window."],
  ["fig4_kv_quality.png", "KV Reconstruction Quality",
   "KV vector reconstruction quality for TQ3 and TQ4 measured across all 32 layers × 8 KV heads of Mistral-7B-v0.1. TQ3: mean cosine similarity 0.9831, mean MSE 0.0355. TQ4: cosine 0.9954, MSE 0.0097. Both exceed typical quality thresholds for production use. TQ4 offers ~5× better MSE at the cost of 3.765× compression (vs TQ3's 4.923×)."],
  ["fig5_kernel_throughput.png", "Triton Kernel Throughput Progression",
   "Kernel throughput (GB/s) across development iterations of the TurboQuant Triton kernel. The fused nibble-pack v2 achieves 2.56× speedup over the Python baseline. The bit-plane variant achieves 1.87×. Both use Wave64 ballot() for 2× more efficient bitpacking on MI300X than CUDA Wave32 would achieve."],
  ["fig6_vram_vs_context.png", "VRAM vs Context — Compression Impact",
   "Steady-state VRAM for KV cache only (excluding model weights) vs context length. Linear growth in both cases: FP16 grows at 256 B/token·head, TQ3 at 52 B/token·head. At 65K context, FP16 KV cache uses ~24.7 GB total VRAM vs ~5 GB for TQ3 — a 4.92× difference that directly determines maximum context window size."],
  ["fig7_attention_speedup.png", "Attention Speedup — TQ3 Triton Kernel",
   "Attention kernel speedup from TQ3 Triton fused dequant-attention vs PyTorch SDPA baseline across sequence lengths 512–131072. The fused kernel avoids materializing the full decompressed KV tensor, instead decompressing K/V blocks on-the-fly during the attention dot-product computation. Speedup grows with sequence length as the KV bandwidth savings compound."],
  ["fig8_dashboard.png", "Summary Dashboard — All Results",
   "Four-panel summary dashboard. Top-left: decode throughput (tok/s) vs context — FP16/FP8 flat at ~46, TQ3 at ~6. Top-right: VRAM vs context — TQ3 grows 4.92× slower. Bottom-left: KV cosine similarity across layers — TQ3 stable at 0.983. Bottom-right: compression ratio summary — TQ3 at 4.923×, TQ4 at 3.76×, FP8 at 2×, INT4 at 4×."],
  ["fig9_max_context.png", "Maximum Context Tokens on MI300X",
   "Maximum storable context tokens at each compression scheme. FP16: ~1.35M tokens in 192 GB. TQ3 (4.923×): ~6.66M tokens — 4.92× more context capacity. This projection assumes steady-state KV cache only (no activation buffers). The 6.9M token figure is the primary production argument for KV compression on MI300X, independent of decode throughput considerations."],
  ["fig10_triton_speedup.png", "Triton Kernel Development Speedup",
   "Speedup progression across Triton kernel versions for TurboQuant on MI300X. V1 (scalar Python): 1×. V2 bit-plane Triton: 1.87×. V2 nibble Triton: 2.56×. The Wave64 ballot() advantage on CDNA3 (64-bit mask vs CUDA's 32-bit) gives 2× better bitpacking efficiency per wavefront. Final kernel gap vs FP16 throughput: ~3× (bit-plane) / ~2.3× (nibble), down from ~6× in V1."],
];

export function getFigureUrlsAll() {
  const hidden = new Set([
    "fig20_k_only_ablation.png",
    "fig21_headline_compression_comparison.png",
    "fig24_pope_rotorquant_2026_claims.png",
    "fig25_mi300x_vs_author_claims.png",
  ]);
  return [
    ...V2_FIGS.filter(([n]) => !hidden.has(n)).map(([n, , d]) => makeFigure("figures_v2", n, d)),
    ...V1_FIGS.map(([n, , d]) => makeFigure("figures", n, d)),
  ];
}

export function getFigureUrls(kind) {
  const figs = kind === "v2" ? V2_FIGS : V1_FIGS;
  const base = kind === "v2" ? "figures_v2" : "figures";
  return figs.map(([n, , d]) => makeFigure(base, n, d));
}

async function fetchFirst(urls) {
  for (const url of urls) {
    try {
      const resp = await fetch(url);
      if (!resp.ok) continue;
      const contentType = (resp.headers.get("content-type") || "").toLowerCase();
      const text = await resp.text();
      if (contentType.includes("text/html") || /^\s*<!doctype html>/i.test(text)) continue;
      return { text, sourceUrl: url };
    } catch {
      // continue
    }
  }
  throw new Error(`Failed to load any source from: ${urls.join(", ")}`);
}

export async function loadReportMarkdown(kind) {
  const fileName = kind === "v2" ? "final_report_v2.md" : "final_report.md";
  return fetchFirst([`/content/${fileName}`, `${FS_ROOT}/${fileName}`]);
}

export function resolveImageSrc(sourceUrl, imagePath) {
  if (/^https?:\/\//.test(imagePath)) return imagePath;
  if (imagePath.startsWith("/")) return imagePath;
  if (sourceUrl.includes("/@fs")) {
    const sourceDir = sourceUrl.slice(0, sourceUrl.lastIndexOf("/") + 1);
    return `${sourceDir}${imagePath}`.replace("/./", "/");
  }
  return `/content/${imagePath.replace(/^\.?\//, "")}`;
}
