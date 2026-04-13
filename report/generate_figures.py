"""
generate_figures.py — Generate all analysis figures for the TurboQuant MI300X report.

Reads result JSONs from ../results/ and writes PNGs to ./figures/.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as ticker
import numpy as np

RESULTS = Path(__file__).parent.parent / "results"
FIGURES = Path(__file__).parent / "figures"
FIGURES.mkdir(exist_ok=True)

# ── AMD brand palette ────────────────────────────────────────────────────────
AMD_RED   = "#ED1C24"
AMD_DARK  = "#1A1A1A"
AMD_GREY  = "#4A4A4A"
BG_DARK   = "#0F0F0F"
BG_CARD   = "#1C1C1E"
GRID_COL  = "#2C2C2E"

SCHEME_COLORS = {
    "fp16": "#60A5FA",   # blue
    "fp8":  "#34D399",   # green
    "int4": "#FBBF24",   # amber
    "tq4":  "#F87171",   # red-light
    "tq3":  AMD_RED,     # AMD red
}
SCHEME_LABELS = {
    "fp16": "FP16 (baseline)",
    "fp8":  "FP8 E4M3 (2×)",
    "int4": "INT4 (4×)",
    "tq4":  "TQ4 (3.76×)",
    "tq3":  "TQ3 (4.92×)",
}

def style():
    plt.rcParams.update({
        "figure.facecolor":  BG_DARK,
        "axes.facecolor":    BG_CARD,
        "axes.edgecolor":    GRID_COL,
        "axes.labelcolor":   "#E5E5EA",
        "axes.titlecolor":   "#FFFFFF",
        "xtick.color":       "#8E8E93",
        "ytick.color":       "#8E8E93",
        "grid.color":        GRID_COL,
        "grid.linewidth":    0.6,
        "legend.facecolor":  BG_CARD,
        "legend.edgecolor":  GRID_COL,
        "legend.labelcolor": "#E5E5EA",
        "text.color":        "#E5E5EA",
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.titlesize":    13,
        "axes.labelsize":    11,
        "figure.dpi":        150,
    })

style()


# ── Load data ────────────────────────────────────────────────────────────────
def load(name):
    p = RESULTS / name
    return json.loads(p.read_text())

fp16_data  = load("fp16_baseline_mistralai_Mistral-7B-v0.1.json")
fp8_data   = load("fp8_baseline_mistralai_Mistral-7B-v0.1.json")
int4_data  = load("int4_baseline_mistralai_Mistral-7B-v0.1.json")
tq_data    = load("bench_tq3_decode_mistralai_Mistral-7B-v0.1.json")
qual_data  = load("bench_quality_mistralai_Mistral-7B-v0.1.json")
attn_data        = load("bench_tq3_attention.json")   # Python wrapper, 8 KV heads (historical)
triton_attn_data = load("bench_triton_attention.json") # actual Triton kernel, 32 KV heads
kern_data  = load("bench_kernels.json")

# ── Flatten helpers ──────────────────────────────────────────────────────────
def fp16_by_seq():
    return {b["seq_len"]: b for b in fp16_data["benchmarks"]}

def fp8_by_seq():
    return {b["seq_len"]: b for b in fp8_data["benchmarks"]}

def int4_by_seq():
    return {b["seq_len"]: b for b in int4_data["benchmarks"]}

def tq_by_seq_mode():
    d = {}
    for r in tq_data["results"]:
        d[(r["seq_len"], r["mode"])] = r
    return d


# ════════════════════════════════════════════════════════════════════════════
# Figure 1 — Decode Throughput vs Context Length (all schemes)
# ════════════════════════════════════════════════════════════════════════════
def fig_throughput():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_DARK)

    # FP16 — all 6 seq_lens
    fp16 = fp16_by_seq()
    seqs_fp16 = sorted(fp16)
    ax.plot(seqs_fp16, [fp16[s]["tokens_per_sec"] for s in seqs_fp16],
            "o-", color=SCHEME_COLORS["fp16"], lw=2.2, ms=7, label=SCHEME_LABELS["fp16"])

    # FP8 (3 pts)
    fp8 = fp8_by_seq()
    seqs_fp8 = sorted(fp8)
    ax.plot(seqs_fp8, [fp8[s]["tokens_per_sec"] for s in seqs_fp8],
            "s-", color=SCHEME_COLORS["fp8"], lw=2.2, ms=7, label=SCHEME_LABELS["fp8"])

    # INT4 (3 pts)
    int4 = int4_by_seq()
    seqs_int4 = sorted(int4)
    ax.plot(seqs_int4, [int4[s]["tokens_per_sec"] for s in seqs_int4],
            "^-", color=SCHEME_COLORS["int4"], lw=2.2, ms=7, label=SCHEME_LABELS["int4"])

    # TQ3 & TQ4 (3 pts)
    tq = tq_by_seq_mode()
    seqs_tq = sorted({k[0] for k in tq})
    for mode in ("tq4", "tq3"):
        vals = [tq.get((s, mode), {}).get("tokens_per_sec") for s in seqs_tq]
        marker = "D" if mode == "tq3" else "v"
        ax.plot(seqs_tq, vals, f"{marker}-",
                color=SCHEME_COLORS[mode], lw=2.2, ms=7, label=SCHEME_LABELS[mode])

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Decode Throughput (tok/s)")
    ax.set_title("Decode Throughput vs Context Length\nMistral-7B-v0.1 · AMD MI300X VF · batch=1")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="lower left", fontsize=9.5)

    # Annotation: "Python-level quant overhead"
    ax.annotate("Python-level\nquant/dequant\noverhead", xy=(2048, 9.12),
                xytext=(512, 4), fontsize=8.5, color="#8E8E93",
                arrowprops=dict(arrowstyle="->", color="#8E8E93", lw=0.8),
                ha="center")

    fig.tight_layout()
    out = FIGURES / "fig1_throughput_vs_context.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 2 — Decode Latency vs Context (ms per token)
# ════════════════════════════════════════════════════════════════════════════
def fig_latency():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_DARK)

    fp16 = fp16_by_seq(); seqs_fp16 = sorted(fp16)
    fp8  = fp8_by_seq();  seqs_fp8  = sorted(fp8)
    int4 = int4_by_seq(); seqs_int4 = sorted(int4)
    tq   = tq_by_seq_mode(); seqs_tq = sorted({k[0] for k in tq})

    ax.plot(seqs_fp16, [fp16[s]["latency_ms"] for s in seqs_fp16],
            "o-", color=SCHEME_COLORS["fp16"], lw=2.2, ms=7, label=SCHEME_LABELS["fp16"])
    ax.plot(seqs_fp8, [fp8[s]["latency_ms"] for s in seqs_fp8],
            "s-", color=SCHEME_COLORS["fp8"], lw=2.2, ms=7, label=SCHEME_LABELS["fp8"])
    ax.plot(seqs_int4, [int4[s]["latency_ms"] for s in seqs_int4],
            "^-", color=SCHEME_COLORS["int4"], lw=2.2, ms=7, label=SCHEME_LABELS["int4"])
    for mode in ("tq4", "tq3"):
        vals = [tq.get((s, mode), {}).get("latency_ms") for s in seqs_tq]
        marker = "D" if mode == "tq3" else "v"
        ax.plot(seqs_tq, vals, f"{marker}-",
                color=SCHEME_COLORS[mode], lw=2.2, ms=7, label=SCHEME_LABELS[mode])

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Decode Latency (ms/token)")
    ax.set_title("Decode Latency vs Context Length\nMistral-7B-v0.1 · AMD MI300X VF · batch=1")
    ax.grid(True, alpha=0.35)
    ax.legend(loc="upper left", fontsize=9.5)

    fig.tight_layout()
    out = FIGURES / "fig2_latency_vs_context.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 3 — Compression Ratio vs KV Cache Size (memory analysis)
# ════════════════════════════════════════════════════════════════════════════
def fig_memory():
    fig, (ax_bar, ax_line) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG_DARK)

    # Left: compression ratio bars
    schemes = ["fp16", "fp8", "int4", "tq4", "tq3"]
    ratios  = [1.0, 2.0, 4.0, 3.76, 4.923]
    colors  = [SCHEME_COLORS[s] for s in schemes]
    labels  = ["FP16", "FP8\nE4M3", "INT4", "TQ4", "TQ3"]
    bars = ax_bar.bar(labels, ratios, color=colors, width=0.55, edgecolor=BG_DARK, linewidth=1.5)
    for bar, ratio in zip(bars, ratios):
        ax_bar.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.06,
                    f"{ratio:.2f}×", ha="center", va="bottom", fontsize=10.5, color="#FFFFFF")
    ax_bar.set_ylabel("Compression Ratio vs FP16")
    ax_bar.set_title("KV Cache Compression Ratios")
    ax_bar.set_ylim(0, 6.2)
    ax_bar.axhline(1, color=SCHEME_COLORS["fp16"], lw=0.8, ls="--", alpha=0.4)
    ax_bar.grid(True, axis="y", alpha=0.3)

    # Right: KV cache size at different context lengths
    contexts = [512, 2048, 8192, 32768, 65536, 131072]
    fp16_sizes = fp16_by_seq()
    # kv_bytes_per_seq from FP16 data: for Mistral-7B, 32 layers × 2 × 8 heads × seq × 128 × 2 bytes
    # We can compute: 32 * 2 * 8 * seq * 128 * 2 bytes = 131072 * seq bytes
    def kv_gb(seq, ratio):
        n_bytes = 32 * 2 * 8 * seq * 128 * 2  # FP16 bytes
        return n_bytes / ratio / 1e9

    for scheme, ratio in [("fp16", 1.0), ("fp8", 2.0), ("int4", 4.0), ("tq4", 3.76), ("tq3", 4.923)]:
        sizes = [kv_gb(s, ratio) for s in contexts]
        ax_line.plot(contexts, sizes, "o-",
                     color=SCHEME_COLORS[scheme], lw=2, ms=6, label=SCHEME_LABELS[scheme])

    # Mark MI300X VRAM (192 GB total; ~14 GB for model weights → ~178 GB for KV)
    ax_line.axhline(178, color="#FCD34D", lw=1.2, ls="--", alpha=0.7)
    ax_line.text(contexts[-1], 178 + 3, "MI300X VRAM limit\n(after model weights ~178 GB)",
                 ha="right", va="bottom", fontsize=8, color="#FCD34D")

    ax_line.set_xscale("log", base=2)
    ax_line.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_line.set_xlabel("Context Length (tokens)")
    ax_line.set_ylabel("KV Cache Size (GB)")
    ax_line.set_title("KV Cache Memory vs Context Length\nMistral-7B (32L, 8 KV heads, d=128)")
    ax_line.grid(True, alpha=0.3)
    ax_line.legend(loc="upper left", fontsize=8.5)

    fig.tight_layout(pad=2.0)
    out = FIGURES / "fig3_memory_analysis.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 4 — KV Reconstruction Quality
# ════════════════════════════════════════════════════════════════════════════
def fig_quality():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor(BG_DARK)

    kv_rec = qual_data["kv_reconstruction"]

    # Schemes with data
    schemes_q  = []
    cos_sims   = []
    mses       = []
    cols       = []
    known = {"fp8": "FP8 E4M3", "tq3": "TQ3", "tq4": "TQ4"}
    for k, label in known.items():
        if k in kv_rec and not (kv_rec[k].get("mean_cosine_sim") != kv_rec[k].get("mean_cosine_sim")):  # NaN check
            cs = kv_rec[k]["mean_cosine_sim"]
            ms = kv_rec[k]["mean_mse"]
            if cs == cs:  # not NaN
                schemes_q.append(label)
                cos_sims.append(cs)
                mses.append(ms)
                cols.append(SCHEME_COLORS[k])

    # Cosine similarity (higher is better)
    bars1 = ax1.bar(schemes_q, cos_sims, color=cols, width=0.4,
                    edgecolor=BG_DARK, linewidth=1.5)
    for bar, val in zip(bars1, cos_sims):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.006,
                 f"{val:.4f}", ha="center", va="top", fontsize=10.5, color="#FFFFFF", fontweight="bold")
    ax1.set_ylim(0.97, 1.002)
    ax1.axhline(1.0, color="#FFFFFF", lw=0.8, ls="--", alpha=0.3)
    ax1.set_ylabel("Mean Cosine Similarity (↑ better)")
    ax1.set_title("KV Reconstruction Quality\n(Cosine Similarity per head vector)")
    ax1.grid(True, axis="y", alpha=0.3)

    # MSE (lower is better)
    bars2 = ax2.bar(schemes_q, mses, color=cols, width=0.4,
                    edgecolor=BG_DARK, linewidth=1.5)
    for bar, val in zip(bars2, mses):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.0003,
                 f"{val:.4f}", ha="center", va="bottom", fontsize=10.5, color="#FFFFFF", fontweight="bold")
    ax2.set_ylabel("Mean MSE (↓ better)")
    ax2.set_title("KV Reconstruction Quality\n(Mean Squared Error per head vector)")
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("KV Cache Reconstruction Quality · Mistral-7B · MI300X",
                 fontsize=13, color="#FFFFFF", y=1.01)
    fig.tight_layout()
    out = FIGURES / "fig4_kv_quality.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 5 — Kernel Throughput (GB/s) — standalone binary vs Python wrapper
# ════════════════════════════════════════════════════════════════════════════
def fig_kernel_throughput():
    fig, (ax_tq, ax_ref) = plt.subplots(1, 2, figsize=(13, 5.5),
                                          gridspec_kw={"width_ratios": [3, 1]})
    fig.patch.set_facecolor(BG_DARK)

    # Left: TQ3 operations
    tq_ops = [
        ("TQ3\nCompress",   10.6, 11.8,  "#60A5FA", "#3B82F6"),
        ("TQ3\nDecompress", 198.1, 58.4, "#34D399", "#10B981"),
        ("TQ3\nFused Dot",  93.1, 33.1,  AMD_RED, "#DC2626"),
    ]
    x = np.arange(len(tq_ops))
    w = 0.35

    bars_hw = ax_tq.bar(x - w/2, [o[1] for o in tq_ops], w,
                        color=[o[3] for o in tq_ops], edgecolor=BG_DARK, linewidth=1.2,
                        label="Standalone HIP binary (ROCm 7.2)")
    bars_py = ax_tq.bar(x + w/2, [o[2] for o in tq_ops], w,
                        color=[o[4] for o in tq_ops], alpha=0.75, edgecolor=BG_DARK, linewidth=1.2,
                        label="Python wrapper (torch.matmul → rocBLAS)")

    for bar in list(bars_hw) + list(bars_py):
        h = bar.get_height()
        ax_tq.text(bar.get_x() + bar.get_width()/2, h + 2,
                   f"{h:.0f}", ha="center", va="bottom", fontsize=9, color="#E5E5EA")

    ax_tq.set_xticks(x)
    ax_tq.set_xticklabels([o[0] for o in tq_ops], fontsize=10)
    ax_tq.set_ylabel("Throughput (GB/s)")
    ax_tq.set_title("TQ3 Kernel Operations")
    ax_tq.set_ylim(0, 240)
    ax_tq.grid(True, axis="y", alpha=0.3)
    ax_tq.legend(loc="upper left", fontsize=8.5)

    # Right: FP16 matmul reference (separate scale)
    ax_ref.bar([0], [767.7], color="#6B7280", width=0.5, edgecolor=BG_DARK, linewidth=1.2,
               label="HIP / PyTorch")
    ax_ref.text(0, 767.7 + 10, "768", ha="center", va="bottom", fontsize=10, color="#E5E5EA")
    # HBM3 peak line
    ax_ref.axhline(5300, color="#FCD34D", lw=0.8, ls="--", alpha=0.6)
    ax_ref.text(0.45, 5350, "HBM3\npeak\n5,300", ha="right", fontsize=7.5, color="#FCD34D")
    ax_ref.set_xticks([0])
    ax_ref.set_xticklabels(["FP16\nMatmul\n(reference)"], fontsize=10)
    ax_ref.set_ylim(0, 5600)
    ax_ref.set_ylabel("Throughput (GB/s)")
    ax_ref.set_title("FP16 Reference")
    ax_ref.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Kernel Throughput · AMD MI300X VF · n=65,536 vectors · head_dim=128",
                 fontsize=12, color="#FFFFFF")
    fig.tight_layout()
    out = FIGURES / "fig5_kernel_throughput.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 6 — VRAM Usage vs Context Length
# ════════════════════════════════════════════════════════════════════════════
def fig_vram():
    fig, ax = plt.subplots(figsize=(10, 5.5))
    fig.patch.set_facecolor(BG_DARK)

    fp16 = fp16_by_seq()
    seqs = sorted(fp16)
    vrams = [fp16[s]["vram_peak_gb"] for s in seqs]

    ax.fill_between(seqs, vrams, alpha=0.18, color=SCHEME_COLORS["fp16"])
    ax.plot(seqs, vrams, "o-", color=SCHEME_COLORS["fp16"], lw=2.5, ms=8,
            label="FP16 measured VRAM")

    # Annotate each point
    for s, v in zip(seqs, vrams):
        ax.annotate(f"{v:.1f} GB", (s, v), xytext=(0, 10),
                    textcoords="offset points", ha="center", fontsize=8.5,
                    color="#E5E5EA")

    # MI300X total VRAM
    ax.axhline(192, color="#FCD34D", lw=1.2, ls="--", alpha=0.7)
    ax.text(seqs[0], 194, "MI300X capacity: 192 GB", fontsize=8.5, color="#FCD34D")

    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax.set_xlabel("Context Length (tokens)")
    ax.set_ylabel("Peak VRAM (GB)")
    ax.set_title("Peak VRAM vs Context Length (FP16 KV Cache)\nMistral-7B-v0.1 · AMD MI300X VF")
    ax.set_ylim(0, 210)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9.5)

    # Extrapolate TQ3 VRAM (14 GB model + KV in TQ3)
    model_gb = 14.0
    def tq3_vram(seq):
        kv_fp16 = 32 * 2 * 8 * seq * 128 * 2 / 1e9
        return model_gb + kv_fp16 / 4.923

    tq3_vrams = [tq3_vram(s) for s in seqs]
    ax.plot(seqs, tq3_vrams, "D--", color=SCHEME_COLORS["tq3"], lw=2, ms=6, alpha=0.85,
            label="TQ3 projected VRAM")

    # Annotate TQ3 at 131K
    ax.annotate(f"{tq3_vrams[-1]:.1f} GB\n(TQ3 projected)",
                xy=(seqs[-1], tq3_vrams[-1]), xytext=(-80, -25),
                textcoords="offset points", fontsize=8, color=SCHEME_COLORS["tq3"],
                arrowprops=dict(arrowstyle="->", color=SCHEME_COLORS["tq3"], lw=0.8))
    ax.legend(fontsize=9.5)

    fig.tight_layout()
    out = FIGURES / "fig6_vram_vs_context.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 7 — TQ3 Attention Speedup vs Context Length (measured, 32 KV heads)
# ════════════════════════════════════════════════════════════════════════════
def fig_attn_speedup():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
    fig.patch.set_facecolor(BG_DARK)

    # Use the real Triton benchmark data (bench_triton_attention.json, 32 KV heads)
    results    = triton_attn_data["results"]
    seq_ks     = [r["seq_k"]     for r in results]
    fp16_ms    = [r["fp16_ms"]   for r in results]
    pywrap_ms  = [r["pywrap_ms"] for r in results]
    triton_ms  = [r["triton_ms"] for r in results]
    spd_fp16   = [r["speedup_vs_fp16"]   for r in results]   # Triton vs FP16
    spd_pywrap = [r["speedup_vs_pywrap"] for r in results]   # Triton vs Python TQ3

    # Left: attention latency
    ax1.plot(seq_ks, fp16_ms,   "o-", color=SCHEME_COLORS["fp16"], lw=2.2, ms=7, label="FP16 SDPA (baseline)")
    ax1.plot(seq_ks, pywrap_ms, "s-", color="#FBBF24",              lw=2.2, ms=7, label="TQ3 Python wrapper (decompress+SDPA)")
    ax1.plot(seq_ks, triton_ms, "^-", color=SCHEME_COLORS["tq3"],  lw=2.5, ms=7, label="TQ3 Triton fused kernel (measured)")

    ax1.set_xscale("log", base=2)
    ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax1.set_xlabel("KV Cache Length (tokens)")
    ax1.set_ylabel("Attention Latency (ms, log scale)")
    ax1.set_title("Attention Latency vs KV Length\n32 KV heads, head_dim=128, S_q=1")
    ax1.set_yscale("log")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(fontsize=9)

    # Right: speedup of Triton vs FP16 and vs Python TQ3
    ax2.axhline(1.0, color="#FFFFFF", lw=0.8, ls="--", alpha=0.4, label="break-even (1×)")
    ax2.plot(seq_ks, spd_pywrap, "^-", color=SCHEME_COLORS["tq3"], lw=2.2, ms=7,
             label="Triton TQ3 vs Python TQ3 wrapper")
    ax2.plot(seq_ks, spd_fp16,  "o-", color="#60A5FA", lw=2.2, ms=7,
             label="Triton TQ3 vs FP16")

    ax2.set_xscale("log", base=2)
    ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax2.set_xlabel("KV Cache Length (tokens)")
    ax2.set_ylabel("Speedup (×)")
    ax2.set_title("TQ3 Triton Kernel Speedup\n(measured, not projected)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle("Attention Throughput: FP16 vs TQ3 (Triton, measured) · AMD MI300X VF · 32 heads",
                 fontsize=12, color="#FFFFFF")
    fig.tight_layout()
    out = FIGURES / "fig7_attention_speedup.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 8 — Summary Dashboard (2×2 grid)
# ════════════════════════════════════════════════════════════════════════════
def fig_dashboard():
    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor(BG_DARK)
    gs = fig.add_gridspec(2, 2, hspace=0.38, wspace=0.3)

    ax_thr = fig.add_subplot(gs[0, 0])
    ax_mem = fig.add_subplot(gs[0, 1])
    ax_qua = fig.add_subplot(gs[1, 0])
    ax_rat = fig.add_subplot(gs[1, 1])

    # ── Throughput (top-left) ────────────────────────────────────────────────
    fp16 = fp16_by_seq(); seqs_fp16 = sorted(fp16)
    fp8  = fp8_by_seq();  seqs_fp8  = sorted(fp8)
    int4 = int4_by_seq(); seqs_int4 = sorted(int4)
    tq   = tq_by_seq_mode(); seqs_tq = sorted({k[0] for k in tq})

    ax_thr.plot(seqs_fp16, [fp16[s]["tokens_per_sec"] for s in seqs_fp16],
                "o-", color=SCHEME_COLORS["fp16"], lw=2, ms=5, label="FP16")
    ax_thr.plot(seqs_fp8, [fp8[s]["tokens_per_sec"] for s in seqs_fp8],
                "s-", color=SCHEME_COLORS["fp8"], lw=2, ms=5, label="FP8 (2×)")
    ax_thr.plot(seqs_int4, [int4[s]["tokens_per_sec"] for s in seqs_int4],
                "^-", color=SCHEME_COLORS["int4"], lw=2, ms=5, label="INT4 (4×)")
    for mode in ("tq4", "tq3"):
        vals = [tq.get((s, mode), {}).get("tokens_per_sec") for s in seqs_tq]
        ax_thr.plot(seqs_tq, vals, "D-" if mode == "tq3" else "v-",
                    color=SCHEME_COLORS[mode], lw=2, ms=5,
                    label=f"TQ3 (4.92×)" if mode == "tq3" else "TQ4 (3.76×)")
    ax_thr.set_xscale("log", base=2)
    ax_thr.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_thr.set_xlabel("Context Length", fontsize=9)
    ax_thr.set_ylabel("tok/s", fontsize=9)
    ax_thr.set_title("Decode Throughput", fontsize=10)
    ax_thr.legend(fontsize=7.5, loc="lower left")
    ax_thr.grid(True, alpha=0.3)

    # ── Memory (top-right) ───────────────────────────────────────────────────
    contexts = [512, 2048, 8192, 32768, 65536, 131072]
    def kv_gb(seq, ratio):
        return 32 * 2 * 8 * seq * 128 * 2 / ratio / 1e9
    for scheme, ratio in [("fp16", 1.0), ("fp8", 2.0), ("int4", 4.0), ("tq4", 3.76), ("tq3", 4.923)]:
        sizes = [kv_gb(s, ratio) for s in contexts]
        lbl = {"fp16": "FP16", "fp8": "FP8 (2×)", "int4": "INT4 (4×)", "tq3": "TQ3 (4.92×)", "tq4": "TQ4 (3.76×)"}[scheme]
        ax_mem.plot(contexts, sizes, "o-", color=SCHEME_COLORS[scheme], lw=1.8, ms=4, label=lbl)
    ax_mem.axhline(178, color="#FCD34D", lw=0.8, ls="--", alpha=0.7)
    ax_mem.set_xscale("log", base=2)
    ax_mem.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{int(x):,}"))
    ax_mem.set_xlabel("Context Length", fontsize=9)
    ax_mem.set_ylabel("KV Cache Size (GB)", fontsize=9)
    ax_mem.set_title("KV Cache Memory", fontsize=10)
    ax_mem.legend(fontsize=7.5, loc="upper left")
    ax_mem.grid(True, alpha=0.3)

    # ── Quality (bottom-left) ────────────────────────────────────────────────
    kv_rec = qual_data["kv_reconstruction"]
    q_schemes = []
    q_cos = []
    q_cols = []
    for k, lbl in [("fp8", "FP8"), ("tq3", "TQ3"), ("tq4", "TQ4")]:
        if k in kv_rec:
            cs = kv_rec[k].get("mean_cosine_sim", float("nan"))
            if cs == cs:
                q_schemes.append(lbl)
                q_cos.append(cs)
                q_cols.append(SCHEME_COLORS[k])
    if q_schemes:
        bars = ax_qua.bar(q_schemes, q_cos, color=q_cols, width=0.4, edgecolor=BG_DARK)
        for bar, val in zip(bars, q_cos):
            ax_qua.text(bar.get_x() + bar.get_width()/2, bar.get_height() - 0.005,
                        f"{val:.4f}", ha="center", va="top", fontsize=9, color="white", fontweight="bold")
        ax_qua.set_ylim(0.97, 1.002)
        ax_qua.axhline(1.0, color="#FFFFFF", lw=0.6, ls="--", alpha=0.3)
    ax_qua.set_ylabel("Cosine Similarity (↑ better)", fontsize=9)
    ax_qua.set_title("KV Reconstruction Quality", fontsize=10)
    ax_qua.grid(True, axis="y", alpha=0.3)

    # ── Compression ratio bars (bottom-right) ────────────────────────────────
    schemes_r = ["FP16", "FP8\nE4M3", "INT4", "TQ4", "TQ3"]
    ratios_r  = [1.0, 2.0, 4.0, 3.76, 4.923]
    colors_r  = [SCHEME_COLORS[s] for s in ["fp16", "fp8", "int4", "tq4", "tq3"]]
    bars_r = ax_rat.bar(schemes_r, ratios_r, color=colors_r, width=0.5, edgecolor=BG_DARK, linewidth=1.2)
    for bar, ratio in zip(bars_r, ratios_r):
        ax_rat.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                    f"{ratio:.2f}×", ha="center", va="bottom", fontsize=9, color="white")
    ax_rat.set_ylabel("Compression vs FP16", fontsize=9)
    ax_rat.set_title("KV Compression Ratios", fontsize=10)
    ax_rat.set_ylim(0, 6.0)
    ax_rat.grid(True, axis="y", alpha=0.3)

    fig.suptitle("TurboQuant KV Cache Compression · AMD MI300X VF · Mistral-7B-v0.1",
                 fontsize=14, color="#FFFFFF", y=1.01, fontweight="bold")

    out = FIGURES / "fig8_dashboard.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 9 — Effective Max Context (how much context fits in 192 GB)
# ════════════════════════════════════════════════════════════════════════════
def fig_max_context():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor(BG_DARK)

    model_gb = 14.72  # measured from FP16 baseline at 512 tokens (mostly weights)
    avail_gb = 192 - model_gb

    schemes = ["FP16", "FP8\n(2×)", "INT4\n(4×)", "TQ4\n(3.76×)", "TQ3\n(4.92×)"]
    ratios  = [1.0, 2.0, 4.0, 3.76, 4.923]
    colors  = [SCHEME_COLORS[s] for s in ["fp16", "fp8", "int4", "tq4", "tq3"]]

    # bytes per token for Mistral-7B: 32 layers × 2 KV × 8 heads × 128 dim × 2 bytes
    bytes_per_tok_fp16 = 32 * 2 * 8 * 128 * 2  # = 131072 bytes
    def max_ctx_k(ratio):
        kv_gb_per_tok = bytes_per_tok_fp16 / ratio / 1e9
        return avail_gb / kv_gb_per_tok / 1000  # in K tokens

    max_ctxs = [max_ctx_k(r) for r in ratios]

    bars = ax.bar(schemes, max_ctxs, color=colors, width=0.55, edgecolor=BG_DARK, linewidth=1.5)
    for bar, val in zip(bars, max_ctxs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f"{val:.0f}K", ha="center", va="bottom", fontsize=11, color="white", fontweight="bold")

    ax.set_ylabel("Max Context Length (K tokens)")
    ax.set_title(f"Maximum Context Length on MI300X (192 GB)\nModel weights: {model_gb:.1f} GB → {avail_gb:.1f} GB available for KV")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, max(max_ctxs) * 1.18)

    # Highlight TQ3 bar
    bars[-1].set_edgecolor(AMD_RED)
    bars[-1].set_linewidth(2.5)

    fig.tight_layout()
    out = FIGURES / "fig9_max_context.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Figure 10 — Triton fused TQ3 vs FP16 and Python TQ3 wrapper
# ════════════════════════════════════════════════════════════════════════════
def fig_triton_speedup():
    path = RESULTS / "bench_triton_attention.json"
    if not path.exists():
        print(f"  ⚠ {path.name} not found — skipping fig10")
        return

    data = json.loads(path.read_text())["results"]
    seq_ks      = [r["seq_k"] for r in data]
    ms_fp16     = [r["fp16_ms"] for r in data]
    ms_pywrap   = [r["pywrap_ms"] for r in data]
    ms_triton   = [r["triton_ms"] for r in data]
    spd_pywrap  = [r["speedup_vs_pywrap"] for r in data]
    bw_triton   = [r["triton_eff_bw_gbs"] for r in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG_DARK)

    # ── Left: latency comparison ───────────────────────────────────────────
    ax = axes[0]
    ax.set_facecolor(BG_CARD)
    x_labels = [f"{s//1024}K" for s in seq_ks]

    ax.plot(x_labels, ms_fp16,   "o-", color="#60A5FA", lw=2, ms=6, label="FP16 SDPA (baseline)")
    ax.plot(x_labels, ms_pywrap, "s-", color="#FBBF24", lw=2, ms=6, label="Python TQ3 (decompress+SDPA)")
    ax.plot(x_labels, ms_triton, "^-", color=AMD_RED,   lw=2.5, ms=7, label="Triton fused TQ3 (this work)")

    ax.set_xlabel("KV Context Length")
    ax.set_ylabel("Latency (ms, log scale)")
    ax.set_title("Attention Latency: FP16 vs TQ3 Implementations")
    ax.set_yscale("log")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)

    # Annotate Triton points with speedup vs PyWrapper
    for i, (xl, spd) in enumerate(zip(x_labels, spd_pywrap)):
        ax.annotate(f"{spd:.1f}×\nfaster\nvs PyTQ3",
                    xy=(i, ms_triton[i]),
                    xytext=(i, ms_triton[i] * 1.8),
                    fontsize=7, color=AMD_RED, ha="center",
                    arrowprops=dict(arrowstyle="-", color=AMD_RED, lw=0.8))

    # ── Right: speedup bars ────────────────────────────────────────────────
    ax2 = axes[1]
    ax2.set_facecolor(BG_CARD)
    x = np.arange(len(seq_ks))
    bars = ax2.bar(x, spd_pywrap, color=AMD_RED, width=0.55,
                   edgecolor=BG_DARK, linewidth=1.2)
    for bar, val in zip(bars, spd_pywrap):
        ax2.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 0.05,
                 f"{val:.2f}×", ha="center", va="bottom",
                 fontsize=10, color="white", fontweight="bold")

    ax2.axhline(1.0, color="#60A5FA", lw=1.5, ls="--", label="Python TQ3 baseline (1×)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(x_labels)
    ax2.set_xlabel("KV Context Length")
    ax2.set_ylabel("Speedup vs Python TQ3 Wrapper")
    ax2.set_title("Triton Kernel Speedup over Python TQ3 Wrapper\n(Cosine similarity = 1.0000 — numerically exact)")
    ax2.set_ylim(0, max(spd_pywrap) * 1.25)
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(fontsize=9)

    fig.suptitle("Triton Fused TQ3 Attention · AMD MI300X VF · 32 heads · decode step (S_q=1)",
                 fontsize=13, color="#FFFFFF", fontweight="bold")
    fig.tight_layout()

    out = FIGURES / "fig10_triton_speedup.png"
    fig.savefig(out, bbox_inches="tight", facecolor=BG_DARK)
    plt.close(fig)
    print(f"  ✓ {out.name}")


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Generating figures...")
    fig_throughput()
    fig_latency()
    fig_memory()
    fig_quality()
    fig_kernel_throughput()
    fig_vram()
    fig_attn_speedup()
    fig_dashboard()
    fig_max_context()
    fig_triton_speedup()
    print(f"\nAll figures saved to {FIGURES}/")
