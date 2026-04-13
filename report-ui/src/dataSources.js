const ABS_ROOT = "/root/workspace/amd-experiments";

async function fetchFirst(urls) {
  for (const url of urls) {
    try {
      const resp = await fetch(url);
      if (resp.ok) {
        return { text: await resp.text(), sourceUrl: url };
      }
    } catch {
      // Continue through fallback list.
    }
  }
  throw new Error(`Failed to load any source from: ${urls.join(", ")}`);
}

export async function loadReportMarkdown(kind) {
  const fileName = kind === "v2" ? "final_report_v2.md" : "final_report.md";
  const sources = [
    `/@fs${ABS_ROOT}/report/${fileName}`,
    `/content/${fileName}`,
  ];
  return fetchFirst(sources);
}

export function getFigureUrls(kind) {
  const base = kind === "v2" ? "figures_v2" : "figures";
  const names =
    kind === "v2"
      ? [
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
        ]
      : [
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

  return names.map((name) => ({
    name,
    label: name.replace(/\.png$/, "").replaceAll("_", " "),
    srcCandidates: [
      `/@fs${ABS_ROOT}/report/${base}/${name}`,
      `/content/${base}/${name}`,
    ],
  }));
}

export function resolveImageSrc(sourceUrl, imagePath) {
  if (/^https?:\/\//.test(imagePath)) {
    return imagePath;
  }
  if (imagePath.startsWith("/")) {
    return imagePath;
  }

  if (sourceUrl.includes("/@fs")) {
    const sourceDir = sourceUrl.slice(0, sourceUrl.lastIndexOf("/") + 1);
    return `${sourceDir}${imagePath}`.replace("/./", "/");
  }

  return `/content/${imagePath.replace(/^\.?\//, "")}`;
}
