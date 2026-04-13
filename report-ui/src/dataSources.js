const FS_ROOT = "/@fs/root/workspace/amd-experiments/report";

function makeFigure(base, name) {
  return {
    name,
    label: name.replace(/\.png$/, "").replaceAll("_", " ").replace(/^fig\d+_/, ""),
    srcCandidates: [
      `/content/${base}/${name}`,
      `${FS_ROOT}/${base}/${name}`,
    ],
  };
}

const V2_FIGS = [
  "fig11_ppl_vs_compression.png",
  "fig12_decode_all_methods.png",
  "fig13_batch_decode_crossover.png",
  "fig14_prefill_comparison.png",
  "fig15_compress_decompress_bw.png",
  "fig16_max_context.png",
  "fig17_speed_vs_quality.png",
  "fig18_roofline.png",
  "fig19_fmas_comparison.png",
  "fig20_k_only_ablation.png",
  "fig21_headline_compression_comparison.png",
  "fig22_cache_compression_mi300x.png",
  "fig23_kv_cache_compression_comparison.png",
  "fig24_pope_rotorquant_2026_claims.png",
  "fig25_mi300x_vs_author_claims.png",
];

const V1_FIGS = [
  "fig1_throughput_vs_context.png",
  "fig2_latency_vs_context.png",
  "fig3_memory_analysis.png",
  "fig4_kv_quality.png",
  "fig5_kernel_throughput.png",
  "fig6_vram_vs_context.png",
  "fig7_attention_speedup.png",
  "fig8_dashboard.png",
  "fig9_max_context.png",
  "fig10_triton_speedup.png",
];

export function getFigureUrlsAll() {
  return [
    ...V2_FIGS.map((n) => makeFigure("figures_v2", n)),
    ...V1_FIGS.map((n) => makeFigure("figures", n)),
  ];
}

// Legacy exports kept for compat
export function getFigureUrls(kind) {
  const names = kind === "v2" ? V2_FIGS : V1_FIGS;
  const base = kind === "v2" ? "figures_v2" : "figures";
  return names.map((n) => makeFigure(base, n));
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
  return fetchFirst([
    `/content/${fileName}`,
    `${FS_ROOT}/${fileName}`,
  ]);
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
