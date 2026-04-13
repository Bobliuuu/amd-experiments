"""
plot_results.py — Generate all analysis plots from benchmark results

Produces 5 figures saved to analysis/figures/:
  1. throughput_vs_context.png  — tokens/sec vs context length (fp16/fp8/int4/tq3/tq4)
  2. memory_vs_compression.png  — VRAM usage vs compression level
  3. quality_vs_compression.png — perplexity delta vs compression ratio
  4. kernel_throughput.png      — raw kernel GB/s comparison
  5. kv_reconstruction.png      — KV cosine similarity per compression scheme

Usage:
    python3 analysis/plot_results.py
    python3 analysis/plot_results.py --results-dir results/
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless rendering
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

FIGURES_DIR = Path(__file__).parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parent.parent / "results"

# Color scheme (AMD-inspired)
COLORS = {
    "fp16":  "#E31837",   # AMD red (baseline)
    "fp8":   "#F5821F",   # orange
    "int4":  "#FFD100",   # yellow
    "tq3":   "#1F7A8C",   # teal (our contribution)
    "tq4":   "#2E4057",   # dark blue
}
LABELS = {
    "fp16":  "FP16 (baseline)",
    "fp8":   "FP8 E4M3 (2×)",
    "int4":  "INT4 sym. (4×)",
    "tq3":   "TQ3 ours (4.92×)",
    "tq4":   "TQ4 ours (3.76×)",
}
MARKERS = {"fp16": "o", "fp8": "s", "int4": "^", "tq3": "D", "tq4": "P"}

plt.rcParams.update({
    "font.family":  "DejaVu Sans",
    "font.size":    11,
    "axes.grid":    True,
    "grid.alpha":   0.3,
    "figure.dpi":   150,
})


def load_json(path):
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


# ──────────────────────────────────────────────────────────────────────────────
# Plot 1: Tokens/sec vs context length
# ──────────────────────────────────────────────────────────────────────────────

def plot_throughput(results_dir: Path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    configs = [
        ("fp16",  results_dir / "fp16_baseline_mistralai_Mistral-7B-v0.1.json"),
        ("fp8",   results_dir / "fp8_baseline_mistralai_Mistral-7B-v0.1.json"),
        ("int4",  results_dir / "int4_baseline_mistralai_Mistral-7B-v0.1.json"),
    ]

    # Left: absolute tokens/sec
    for cfg, path in configs:
        data = load_json(path)
        if data is None:
            continue
        benchmarks = data.get("benchmarks", [])
        xs = [b["seq_len"] for b in benchmarks]
        ys = [b["tokens_per_sec"] for b in benchmarks]
        if xs:
            ax1.plot(xs, ys, marker=MARKERS.get(cfg, "o"),
                     color=COLORS.get(cfg, "gray"),
                     label=LABELS.get(cfg, cfg), linewidth=2, markersize=7)

    # Also add TQ3 attention benchmark if available
    tq_data = load_json(results_dir / "bench_tq3_attention.json")
    if tq_data:
        results = tq_data.get("results", [])
        xs = [r["n_kv"] for r in results if r.get("tq3_ms")]
        # Convert ms/token to tok/s — this is attention latency, not full decode
        # Mark as dashed estimate
        if xs:
            ys = [1000.0 / r["tq3_ms"] * 100 for r in results if r.get("tq3_ms")]
            ax1.plot(xs, ys, marker=MARKERS["tq3"], color=COLORS["tq3"],
                     label=LABELS["tq3"] + " (attn)", linewidth=2, markersize=7,
                     linestyle="--")

    ax1.set_xscale("log", base=2)
    ax1.set_xlabel("Context Length (tokens)")
    ax1.set_ylabel("Decode Throughput (tokens/sec)")
    ax1.set_title("Decode Throughput vs Context Length\n(Mistral-7B, batch=1)")
    ax1.legend(framealpha=0.9, fontsize=9)
    ax1.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    # Right: VRAM usage
    for cfg, path in configs:
        data = load_json(path)
        if data is None:
            continue
        benchmarks = data.get("benchmarks", [])
        xs = [b["seq_len"] for b in benchmarks]
        ys = [b.get("vram_peak_gb", 0) for b in benchmarks]
        if xs:
            ax2.plot(xs, ys, marker=MARKERS.get(cfg, "o"),
                     color=COLORS.get(cfg, "gray"),
                     label=LABELS.get(cfg, cfg), linewidth=2, markersize=7)

    ax2.set_xscale("log", base=2)
    ax2.set_xlabel("Context Length (tokens)")
    ax2.set_ylabel("Peak VRAM (GB)")
    ax2.set_title("Peak VRAM vs Context Length\n(Mistral-7B, batch=1)")
    ax2.legend(framealpha=0.9, fontsize=9)
    ax2.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.axhline(y=192, color="red", linestyle=":", linewidth=1.5, label="MI300X limit (192 GB)")

    plt.tight_layout()
    out = FIGURES_DIR / "throughput_vs_context.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 2: Memory savings by compression level
# ──────────────────────────────────────────────────────────────────────────────

def plot_memory_savings():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    configs = ["fp16", "fp8", "int4", "tq4", "tq3"]
    ratios  = [1.0,    2.0,   4.0,   3.76,  4.92]
    labels  = [LABELS.get(c, c) for c in configs]
    colors  = [COLORS.get(c, "gray") for c in configs]

    # Left: compression ratio bar
    ax = axes[0]
    bars = ax.bar(labels, ratios, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("Compression Ratio vs FP16")
    ax.set_title("KV Cache Compression Ratios")
    ax.set_ylim(0, 6)
    for bar, ratio in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{ratio:.2f}×", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.tick_params(axis="x", rotation=25)

    # Right: effective KV size at 128K context (Mistral-7B: 32 layers × 8 KV heads × 128 dim)
    ax = axes[1]
    n_kv_elems = 128 * 1024 * 32 * 8 * 128 * 2  # 128K ctx × layers × heads × dim × K+V
    fp16_bytes = n_kv_elems * 2  # 2B per FP16 element
    kv_gbs = [fp16_bytes / r / 1e9 for r in ratios]

    bars2 = ax.bar(labels, kv_gbs, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_ylabel("KV Cache Size (GB)")
    ax.set_title("KV Cache Size at 128K Context\n(Mistral-7B: 32L × 8H × 128D)")
    for bar, size in zip(bars2, kv_gbs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{size:.1f} GB", ha="center", va="bottom", fontsize=9)
    ax.tick_params(axis="x", rotation=25)

    plt.tight_layout()
    out = FIGURES_DIR / "memory_vs_compression.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 3: Quality vs compression
# ──────────────────────────────────────────────────────────────────────────────

def plot_quality(results_dir: Path):
    quality_path = results_dir / "bench_quality_mistralai_Mistral-7B-v0.1.json"
    data = load_json(quality_path)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: perplexity bar
    ax = axes[0]
    if data and "perplexity" in data:
        ppl_data = data["perplexity"]
        configs = [c for c in ["fp16", "fp8", "int4", "tq4", "tq3"] if c in ppl_data and "perplexity" in ppl_data[c]]
        ppls    = [ppl_data[c]["perplexity"] for c in configs]
        colors  = [COLORS.get(c, "gray") for c in configs]
        labels  = [LABELS.get(c, c) for c in configs]

        bars = ax.bar(labels, ppls, color=colors, edgecolor="black", linewidth=0.8)
        baseline = ppl_data.get("fp16", {}).get("perplexity", ppls[0] if ppls else 1)
        for bar, ppl, cfg in zip(bars, ppls, configs):
            delta = ppl - baseline
            label_str = f"{ppl:.1f}" + (f" (+{delta:.1f})" if delta > 0.1 else "")
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                    label_str, ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Perplexity (lower is better)")
        ax.set_title("Perplexity on WikiText-103\n(Mistral-7B)")
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.text(0.5, 0.5, "No perplexity data\n(run bench_quality.py first)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Perplexity (data pending)")

    # Right: KV reconstruction cosine similarity
    ax = axes[1]
    if data and "kv_reconstruction" in data:
        kv_data = data["kv_reconstruction"]
        configs = [c for c in ["fp8", "int4", "tq4", "tq3"] if c in kv_data and "mean_cosine_sim" in kv_data[c]]
        cos_sims = [kv_data[c]["mean_cosine_sim"] for c in configs]
        colors   = [COLORS.get(c, "gray") for c in configs]
        labels   = [LABELS.get(c, c) for c in configs]

        bars = ax.bar(labels, cos_sims, color=colors, edgecolor="black", linewidth=0.8)
        ax.set_ylim(0.8, 1.01)
        for bar, cos in zip(bars, cos_sims):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                    f"{cos:.4f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Mean KV Cosine Similarity")
        ax.set_title("KV Reconstruction Quality\n(per-layer, averaged)")
        ax.axhline(y=1.0, color="red", linestyle=":", linewidth=1)
        ax.tick_params(axis="x", rotation=25)
    else:
        ax.text(0.5, 0.5, "No KV quality data\n(run bench_quality.py first)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("KV Quality (data pending)")

    plt.tight_layout()
    out = FIGURES_DIR / "quality_vs_compression.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 4: Kernel throughput comparison
# ──────────────────────────────────────────────────────────────────────────────

def plot_kernel_throughput(results_dir: Path):
    data = load_json(results_dir / "bench_kernels.json")

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: Python wrapper throughput
    ax = axes[0]
    if data and "python_wrapper" in data:
        pw = data["python_wrapper"]
        ops   = list(pw.keys())
        gbs   = [pw[o].get("throughput_gbs", 0) for o in ops]
        labels = [pw[o].get("op", o).replace("tq3_", "").replace("tq4_", "") for o in ops]
        bar_colors = [COLORS["tq3"]] * len(ops)
        bar_colors[-1] = "#999999"  # FP16 reference in gray

        bars = ax.bar(labels, gbs, color=bar_colors, edgecolor="black", linewidth=0.8)
        for bar, g in zip(bars, gbs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                    f"{g:.0f}", ha="center", va="bottom", fontsize=9)
        ax.set_ylabel("Throughput (GB/s)")
        ax.set_title("Python Wrapper Throughput\n(torch.matmul → MFMA, 65K vectors)")
        # HBM3 peak bandwidth reference
        ax.axhline(y=5300, color="red", linestyle=":", linewidth=1.5, label="HBM3 peak (5.3 TB/s)")
        ax.legend(fontsize=9)
        ax.tick_params(axis="x", rotation=20)
    else:
        ax.text(0.5, 0.5, "No kernel data\n(run bench_kernels.py first)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Kernel Throughput (data pending)")

    # Right: Speedup at various context lengths (from attention bench)
    ax = axes[1]
    tq_data = load_json(results_dir / "bench_tq3_attention.json")
    if tq_data:
        results = tq_data.get("results", [])
        xs = [r["n_kv"] for r in results if r.get("tq3_speedup")]
        ys = [r["tq3_speedup"] for r in results if r.get("tq3_speedup")]
        if xs:
            ax.plot(xs, ys, marker="D", color=COLORS["tq3"], linewidth=2,
                    markersize=8, label="TQ3 speedup")
            ax.axhline(y=1.0, color="red", linestyle="--", linewidth=1.5, label="FP16 baseline")
            ax.fill_between(xs, 1.0, ys, alpha=0.15, color=COLORS["tq3"],
                            where=[y > 1 for y in ys])
            ax.set_xscale("log", base=2)
            ax.set_xlabel("Context Length (KV tokens)")
            ax.set_ylabel("Attention Speedup vs FP16")
            ax.set_title("TQ3 Attention Speedup\nvs FP16 (synthetic, 8 KV heads)")
            ax.legend(fontsize=9)
            ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    else:
        ax.text(0.5, 0.5, "No attention bench data\n(run bench_tq_attention.py first)",
                ha="center", va="center", transform=ax.transAxes, color="gray")
        ax.set_title("Attention Speedup (data pending)")

    plt.tight_layout()
    out = FIGURES_DIR / "kernel_throughput.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Plot 5: KV reconstruction quality heat map per layer
# ──────────────────────────────────────────────────────────────────────────────

def plot_kv_reconstruction(results_dir: Path):
    """Placeholder — requires per-layer data from extended quality run."""
    data = load_json(results_dir / "bench_quality_mistralai_Mistral-7B-v0.1.json")

    fig, ax = plt.subplots(figsize=(9, 5))

    schemes = ["fp8", "int4", "tq4", "tq3"]
    if data and "kv_reconstruction" in data:
        kv = data["kv_reconstruction"]
        cos_vals = [kv.get(s, {}).get("mean_cosine_sim", 0) for s in schemes]
        mse_vals = [kv.get(s, {}).get("mean_mse", 0) for s in schemes]
        labels   = [LABELS.get(s, s) for s in schemes]
        colors   = [COLORS.get(s, "gray") for s in schemes]

        x = np.arange(len(schemes))
        width = 0.35
        bars1 = ax.bar(x - width/2, cos_vals, width, label="Cosine Similarity",
                       color=colors, edgecolor="black", linewidth=0.8, alpha=0.85)
        ax2 = ax.twinx()
        bars2 = ax2.bar(x + width/2, mse_vals, width, label="MSE",
                        color=colors, edgecolor="black", linewidth=0.8, alpha=0.45, hatch="//")

        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=20)
        ax.set_ylabel("KV Cosine Similarity (higher=better)")
        ax2.set_ylabel("Per-element MSE (lower=better)")
        ax.set_title("KV Cache Reconstruction Quality by Scheme\n(Mistral-7B, averaged over layers+heads)")
        ax.set_ylim(0.8, 1.01)
        ax.legend(loc="lower left", fontsize=9)
        ax2.legend(loc="lower right", fontsize=9)
    else:
        ax.text(0.5, 0.5, "No KV reconstruction data\n(run bench_quality.py first)",
                ha="center", va="center", transform=ax.transAxes, color="gray", fontsize=13)
        ax.set_title("KV Reconstruction (data pending)")

    plt.tight_layout()
    out = FIGURES_DIR / "kv_reconstruction.png"
    plt.savefig(out, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out}")


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Generate analysis plots from benchmark results")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    args = parser.parse_args()

    print(f"=== Generating Analysis Plots ===")
    print(f"Results dir: {args.results_dir}")
    print(f"Figures dir: {FIGURES_DIR}")
    print()

    print("Plot 1: Throughput vs context length ...")
    plot_throughput(args.results_dir)

    print("Plot 2: Memory savings by compression level ...")
    plot_memory_savings()

    print("Plot 3: Quality vs compression ...")
    plot_quality(args.results_dir)

    print("Plot 4: Kernel throughput comparison ...")
    plot_kernel_throughput(args.results_dir)

    print("Plot 5: KV reconstruction quality ...")
    plot_kv_reconstruction(args.results_dir)

    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print("Run the benchmarks to populate with real data:")
    print("  python3 benchmarks/bench_kernels.py")
    print("  python3 benchmarks/bench_tq_attention.py")
    print("  python3 baselines/fp16_baseline.py --model mistralai/Mistral-7B-v0.1")
    print("  python3 baselines/fp8_baseline.py  --model mistralai/Mistral-7B-v0.1")
    print("  python3 baselines/int4_baseline.py --model mistralai/Mistral-7B-v0.1")
    print("  python3 benchmarks/bench_quality.py --model mistralai/Mistral-7B-v0.1")


if __name__ == "__main__":
    main()
