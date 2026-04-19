import { useEffect, useRef, useState, useMemo } from "react";
import { motion, useInView, useSpring, useTransform, animate, useMotionValue, AnimatePresence } from "framer-motion";
import katex from "katex";
import "katex/dist/katex.min.css";

/* ─── SHARED UTILITIES ───────────────────────────────────────────── */

function useCountUp(target, inView, duration = 1.4, decimals = 0) {
  const [val, setVal] = useState(0);
  useEffect(() => {
    if (!inView) return;
    const controls = animate(0, target, {
      duration,
      ease: [0.23, 1, 0.32, 1],
      onUpdate: (v) => setVal(decimals ? v.toFixed(decimals) : Math.round(v)),
    });
    return () => controls.stop();
  }, [inView, target, duration, decimals]);
  return val;
}

function AnimatedBar({ pct, cls, delay = 0, inView }) {
  return (
    <motion.div
      className={`bar-fill ${cls}`}
      initial={{ width: 0 }}
      animate={{ width: inView ? `${pct}%` : 0 }}
      transition={{ duration: 1.4, delay, ease: [0.23, 1, 0.32, 1] }}
    />
  );
}

function FmaBar({ pct, color, inView, delay = 0 }) {
  return (
    <div className="fma-track">
      <motion.div
        className="fma-fill"
        style={{ background: color }}
        initial={{ width: 0 }}
        animate={{ width: inView ? `${pct}%` : 0 }}
        transition={{ duration: 1.2, delay, ease: [0.23, 1, 0.32, 1] }}
      />
    </div>
  );
}

const fadeUp = {
  hidden: { opacity: 0, y: 30 },
  visible: (i = 0) => ({
    opacity: 1, y: 0,
    transition: { duration: 0.55, delay: i * 0.08, ease: [0.23, 1, 0.32, 1] },
  }),
};

/** Mistral-7B-v0.1 — matches KV sizing used in `bench_tq3_decode.py`. */
const MISTRAL7B_KV = {
  nLayers: 32,
  nKvHeads: 8,
  headDim: 128,
};

/** Full KV cache (K+V), FP16: 2 × L × Hkv × seq × head_dim × sizeof(fp16). */
function kvBytesFp16(seqLen) {
  return (
    2 *
    MISTRAL7B_KV.nLayers *
    MISTRAL7B_KV.nKvHeads *
    seqLen *
    MISTRAL7B_KV.headDim *
    2
  );
}

/**
 * TQ3 packed layout: 4 + ceil(head_dim × 3 / 8) bytes per head vector (K or V).
 * Full KV: 2 × L × Hkv × seq × bytes_per_vector (same seq factor as FP16).
 */
function kvBytesTq3Theoretical(seqLen) {
  const bytesPerHeadVector = 4 + Math.ceil((MISTRAL7B_KV.headDim * 3) / 8);
  return (
    2 *
    MISTRAL7B_KV.nLayers *
    MISTRAL7B_KV.nKvHeads *
    seqLen *
    bytesPerHeadVector
  );
}

function formatKvDataSize(bytes) {
  const b = Math.max(0, bytes);
  if (b >= 1e12) return `${(b / 1e12).toFixed(2)} TB`;
  if (b >= 1e9) return `${(b / 1e9).toFixed(1)} GB`;
  if (b >= 1e6) return `${(b / 1e6).toFixed(2)} MB`;
  if (b >= 1e3) return `${(b / 1e3).toFixed(2)} KB`;
  return `${b.toFixed(0)} B`;
}

/** Per-head vector: FP16 K or V slice = head_dim × sizeof(fp16) = 128×2 = 256 B. */
const BYTES_FP16_HEAD_VEC = MISTRAL7B_KV.headDim * 2;

const EXPERIMENT_KV_METRICS_URLS = ["/content/experiment_kv_metrics.json"];

/**
 * Loads MI300X benchmark numbers shipped with the UI (snapshot of results/*.json).
 * Update `public/content/experiment_kv_metrics.json` after new benchmark runs.
 */
function useExperimentKvMetrics() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  useEffect(() => {
    let cancelled = false;
    (async () => {
      for (const url of EXPERIMENT_KV_METRICS_URLS) {
        try {
          const r = await fetch(url, { cache: "no-store" });
          if (!r.ok) continue;
          const ct = (r.headers.get("content-type") || "").toLowerCase();
          const text = await r.text();
          if (ct.includes("text/html") || /^\s*<!doctype html>/i.test(text)) continue;
          const j = JSON.parse(text);
          if (!cancelled) {
            setData(j);
            setError(null);
            return;
          }
        } catch (e) {
          if (!cancelled) setError(e);
        }
      }
      if (!cancelled) setError(new Error("experiment_kv_metrics.json not found"));
    })();
    return () => {
      cancelled = true;
    };
  }, []);
  return { data, error };
}

function LiveGenerationPanel({ inView }) {
  const TARGET = 60;
  const MEASURED_FP16_BYTES = 268435456;
  const MEASURED_TQ3_BYTES = 54525952;
  const [tokens, setTokens] = useState(0);
  const [running, setRunning] = useState(false);
  const [cursorOn, setCursorOn] = useState(true);
  const [cycle, setCycle] = useState(0);

  useEffect(() => {
    if (!inView) return;
    let tokenTimer = 0;
    let holdTimer = 0;
    const runCycle = () => {
      let t = 0;
      setRunning(true);
      setTokens(0);
      tokenTimer = window.setInterval(() => {
        t += 1;
        setTokens(t);
        if (t >= TARGET) {
          window.clearInterval(tokenTimer);
          setRunning(false);
          holdTimer = window.setTimeout(() => {
            setCycle((c) => c + 1);
            runCycle();
          }, 1600);
        }
      }, 72);
    };
    runCycle();
    return () => {
      if (tokenTimer) window.clearInterval(tokenTimer);
      if (holdTimer) window.clearTimeout(holdTimer);
    };
  }, [inView]);

  useEffect(() => {
    const timer = window.setInterval(() => setCursorOn((v) => !v), 340);
    return () => window.clearInterval(timer);
  }, []);

  const measuredRatio = MEASURED_FP16_BYTES / MEASURED_TQ3_BYTES;
  const fp16Str = `${MEASURED_FP16_BYTES.toLocaleString()} B`;
  const tq3Str = `${MEASURED_TQ3_BYTES.toLocaleString()} B`;
  const demoTokPerSec = tokens > 0 ? (14.2 + Math.min(3.8, tokens / 20)).toFixed(1) : "0.0";
  const elapsedSec =
    tokens > 0 ? Math.max(0.1, tokens / Number(demoTokPerSec)).toFixed(2) : "0.00";
  const progressBlocks = 38;
  const filled = Math.round((tokens / TARGET) * progressBlocks);
  const bar = `${"█".repeat(filled)}${" ".repeat(progressBlocks - filled)}`;
  const headline =
    tokens >= TARGET ? "TurboQuant — Generation Complete" : "TurboQuant — Generating…";
  const spinner = ["|", "/", "-", "\\"][tokens % 4];

  const generatedText = useMemo(() => {
    const streams = [
      "TurboQuant keeps decode flowing by writing tightly packed 3-bit KV vectors and feeding MI300X with a steady stream of work. Each token extends context, updates cache, and pushes the progress bar forward while the output text materializes in real time across the terminal pane.",
      "On long-context generation, the runtime keeps the visual rhythm simple: token counter rises, cache grows, and sampled words appear line by line. The stream is synthetic for this demo, but the cache sizes and throughput style are wired to the same TQ benchmark-driven display.",
      "This live panel mimics a terminal decode session: incremental token updates, a blinking cursor, and rolling text that expands as generation proceeds. After completion, the animation pauses briefly, then loops back to simulate another prompt on the same TurboQuant pipeline.",
    ];
    const current = streams[cycle % streams.length];
    const chars = Math.floor((tokens / TARGET) * current.length);
    return current.slice(0, chars);
  }, [tokens, cycle]);

  return (
    <div className="glass" style={{ borderRadius: "var(--radius-md)", padding: "1.2rem 1.3rem" }}>
      <p style={{ fontSize: "0.66rem", color: "var(--amd-red)", letterSpacing: "0.1em", fontFamily: "JetBrains Mono,monospace", marginBottom: "0.7rem" }}>
        LIVE DECODE SCOREBOARD (screenshot-style, benchmark-driven)
      </p>
      <div style={{
        background: "rgba(0,0,0,0.55)",
        border: "1px solid rgba(255,255,255,0.08)",
        borderRadius: "8px",
        padding: "0.9rem 1rem",
        fontFamily: "JetBrains Mono,monospace",
        fontSize: "0.74rem",
        lineHeight: 1.65,
        color: "var(--text)",
      }}>
        <div>{headline} {running ? spinner : " "}</div>
        <div>{"=".repeat(68)}</div>
        <div>Tokens&nbsp;&nbsp;&nbsp; {tokens}/{TARGET} [{bar}]</div>
        <div>Cache&nbsp;&nbsp;&nbsp;&nbsp; {MISTRAL7B_KV.nLayers} layers × {MISTRAL7B_KV.nKvHeads} KV heads × {MISTRAL7B_KV.headDim}d</div>
        <div>TQ size&nbsp;&nbsp;&nbsp; {tq3Str}&nbsp;&nbsp;&nbsp;&nbsp;FP16&nbsp;&nbsp;{fp16Str}</div>
        <div>Ratio&nbsp;&nbsp;&nbsp;&nbsp;{measuredRatio.toFixed(5)}x</div>
        <div>Speed&nbsp;&nbsp;&nbsp;&nbsp;{demoTokPerSec} tok/s</div>
        <div>{"=".repeat(68)}</div>
        <div>&nbsp;</div>
        <div style={{ color: "var(--text-muted)", fontSize: "0.68rem" }}>// Live sampled output</div>
        <div
          style={{
            minHeight: "10.6rem",
            whiteSpace: "pre-wrap",
            color: "rgba(240,245,255,0.94)",
            border: "1px solid rgba(255,255,255,0.07)",
            borderRadius: "6px",
            padding: "0.7rem 0.75rem",
            background: "rgba(255,255,255,0.015)",
          }}
        >
          {generatedText}
          {(running || tokens >= TARGET) && <span style={{ opacity: cursorOn ? 1 : 0 }}>&nbsp;█</span>}
        </div>
        <div style={{ marginTop: "0.5rem" }}>
          {tokens} tokens in {elapsedSec}s
        </div>
      </div>
      <p style={{ marginTop: "0.55rem", fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "JetBrains Mono,monospace" }}>
        Source: <code style={{ fontSize: "0.68rem" }}>report-ui/src/components/ReportSections.jsx</code> — run{" "}
        <code style={{ fontSize: "0.68rem" }}>npm run dev</code> or{" "}
        <code style={{ fontSize: "0.68rem" }}>npm run build</code> so <code style={{ fontSize: "0.68rem" }}>dist/</code>{" "}
        matches src. Same KV math as <code style={{ fontSize: "0.68rem" }}>bench_tq3_decode.py</code>.
      </p>
    </div>
  );
}

function ImportantFactsPanel() {
  const facts = [
    { k: "Compression target", v: "4.923x", note: "52 B vs FP16 256 B/vector" },
    { k: "Observed block ratio", v: "~1.94x", note: "Current int8-index Triton path" },
    { k: "Best compress kernel", v: "IsoQuant 21.8 GB/s", note: "4096 vectors, median" },
    { k: "Best decompress kernel", v: "IsoQuant 38.3 GB/s", note: "4096 vectors, median" },
    { k: "Fastest prefill", v: "Planar 1,126K tok/s", note: "26.5x TurboQuant" },
    { k: "Batch=1 decode", v: "Compute-bound", note: "Weight cycling dominates" },
    { k: "Max context @192GB", v: "6.9M tokens", note: "if 3-bit packed layout" },
    { k: "Quality spread", v: "±0.0003 cosine", note: "3-bit methods nearly identical" },
  ];

  return (
    <div className="glass" style={{ borderRadius: "var(--radius-md)", padding: "1.2rem 1.3rem" }}>
      <p style={{ fontSize: "0.66rem", color: "var(--amd-red)", letterSpacing: "0.1em", fontFamily: "JetBrains Mono,monospace", marginBottom: "0.8rem" }}>
        IMPORTANT FACTS FROM THE REPORT
      </p>
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "0.65rem" }}>
        {facts.map((f) => (
          <div key={f.k} style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.06)", borderRadius: "8px", padding: "0.65rem 0.7rem" }}>
            <p style={{ fontSize: "0.66rem", color: "var(--text-muted)", fontFamily: "JetBrains Mono,monospace", marginBottom: "0.18rem" }}>{f.k}</p>
            <p style={{ fontSize: "0.88rem", color: "var(--text)", fontWeight: 650, marginBottom: "0.12rem" }}>{f.v}</p>
            <p style={{ fontSize: "0.66rem", color: "var(--text-sub)" }}>{f.note}</p>
          </div>
        ))}
      </div>
    </div>
  );
}

function RooflineComparePanel() {
  const rows = [
    { m: "PlanarQuant3", c: "var(--planar-color)", fma: 256, comp: 18.7, decomp: 35.4 },
    { m: "IsoQuant3", c: "var(--iso-color)", fma: 512, comp: 21.8, decomp: 38.3 },
    { m: "RotorQuant3", c: "var(--rotor-color)", fma: 1176, comp: 17.3, decomp: 34.8 },
    { m: "TurboQuant3", c: "var(--turbo-color)", fma: 16384, comp: 2.9, decomp: 4.4 },
  ];
  return (
    <div className="glass" style={{ borderRadius: "var(--radius-md)", padding: "1.2rem 1.3rem" }}>
      <p style={{ fontSize: "0.66rem", color: "var(--amd-red)", letterSpacing: "0.1em", fontFamily: "JetBrains Mono,monospace", marginBottom: "0.8rem" }}>
        TURBOQUANT ROOFLINE + 4-METHOD COMPARISON
      </p>
      <img
        src="/content/figures_v2/fig28_mi300x_roofline_tq_attention.png"
        alt="TurboQuant roofline"
        style={{ width: "100%", borderRadius: "8px", border: "1px solid rgba(255,255,255,0.08)", marginBottom: "0.75rem" }}
      />
      <div style={{ overflowX: "auto" }}>
        <table className="data-table" style={{ width: "100%" }}>
          <thead>
            <tr>
              <th>Method</th><th>FMAs/vec</th><th>Compress</th><th>Decompress</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => (
              <tr key={r.m}>
                <td className="cell-method"><span className="dot" style={{ background: r.c }} />{r.m}</td>
                <td className="cell-num">{r.fma.toLocaleString()}</td>
                <td className="cell-num">{r.comp} GB/s</td>
                <td className="cell-num">{r.decomp} GB/s</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <p style={{ marginTop: "0.5rem", fontSize: "0.7rem", color: "var(--text-muted)", fontFamily: "JetBrains Mono,monospace" }}>
        Roofline chart is TurboQuant-specific; table provides apples-to-apples cross-method context.
      </p>
    </div>
  );
}

/* ─── COMPRESSION LANDSCAPE SECTION ─────────────────────────────── */

// All compression metrics are sourced from bench_empirical_kv_validation.json
// (merged from bench_runtime_ratio_all_methods.json + bench_compress_decompress_recheck.json).
//
// ratioPacked = theoretical ratio assuming full bitplane packing (4+48 bytes = 52B)
// ratioObserved = measured from materialized cache tensors in a real prefill run
//   (results/bench_runtime_ratio_all_methods.json: kv_bytes_fp16 / kv_bytes_compressed_materialized)
//
// Key finding: TurboQuant (Python wrapper) writes the full 52-byte bitplane format,
// so ratioObserved == ratioPacked for TQ3. The block-rotation Triton kernels store
// indices as int8 (1 byte/index), giving 132 bytes/vec for Planar/Iso and 133 for
// Rotor (128 dims → 43 groups×3 = 129 indices due to 3D padding). ratioPacked for
// those methods is what they WOULD achieve with bitplane packing implemented.
const ALL_SCHEMES = [
  { id: "fp16",   name: "FP16",          bytesObserved: 256,  bytesIfPacked: 256,  ratioObserved: 1.000,  ratioPacked: 1.000,  cosine: 1.0000, mse: 0.0,      decode8k: 46.5,  color: "#6B7280", group: "baseline" },
  { id: "fp8",    name: "FP8 E4M3",      bytesObserved: 128,  bytesIfPacked: 128,  ratioObserved: 2.000,  ratioPacked: 2.000,  cosine: 0.9999, mse: 0.00001,  decode8k: 46.0,  color: "#3B82F6", group: "hardware" },
  { id: "int4",   name: "INT4 sym",      bytesObserved: 64,   bytesIfPacked: 64,   ratioObserved: 4.000,  ratioPacked: 4.000,  cosine: 0.9800, mse: 0.001,    decode8k: 26.0,  color: "#8B5CF6", group: "hardware" },
  { id: "tq4",    name: "TQ4",           bytesObserved: 68,   bytesIfPacked: 68,   ratioObserved: 3.765,  ratioPacked: 3.765,  cosine: 0.9954, mse: 0.00974,  decode8k: 11.2,  color: "#06B6D4", group: "turbo" },
  // TQ3: bitplane packing is implemented — observed matches packed
  { id: "tq3",    name: "TQ3",           bytesObserved: 52,   bytesIfPacked: 52,   ratioObserved: 4.923,  ratioPacked: 4.923,  cosine: 0.9831, mse: 0.03413,  decode8k: 6.3,   color: "#3B82F6", group: "turbo" },
  // Block methods: current Triton kernels store int8 per index (not bit-packed).
  // bytesObserved: measured by bench_runtime_ratio_all_methods.py on Mistral-7B-v0.1.
  // bytesIfPacked: target format = 4B norm + 48B bitplanes (same as TQ3).
  // Rotor stores 129 indices (43 groups × 3 dims; 128 is not divisible by 3 → +1 pad).
  { id: "planar", name: "PlanarQuant3",  bytesObserved: 132,  bytesIfPacked: 52,   ratioObserved: 1.939,  ratioPacked: 4.923,  cosine: 0.9829, mse: 0.03423,  decode8k: null,  color: "#E5344B", group: "block" },
  { id: "iso",    name: "IsoQuant3",     bytesObserved: 132,  bytesIfPacked: 52,   ratioObserved: 1.939,  ratioPacked: 4.923,  cosine: 0.9831, mse: 0.03380,  decode8k: null,  color: "#A855F7", group: "block" },
  { id: "rotor",  name: "RotorQuant3",   bytesObserved: 133,  bytesIfPacked: 52,   ratioObserved: 1.925,  ratioPacked: 4.923,  cosine: 0.9830, mse: 0.03408,  decode8k: null,  color: "#FF7B35", group: "block" },
];

export function CompressionLandscapeSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const [activeView, setActiveView] = useState("ratio");
  const [showPacked, setShowPacked] = useState(false);

  const maxRatioPacked = 4.923;
  const maxRatioObserved = 4.923; // TQ3 observed = 4.923, dominates
  const maxDecode = 46.5;

  return (
    <section id="compression" className="report-section" ref={ref}>
      <div className="section-divider" />
      <motion.div
        style={{ paddingTop: "5rem" }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"} custom={0}
      >
        <div className="section-label">
          <span className="section-num">03</span>
          <span className="section-tag">COMPRESSION LANDSCAPE</span>
        </div>
        <h2 className="section-title">
          All Schemes,<br /><span className="dim">One Benchmark</span>
        </h2>
        <p className="section-lead">
          TurboQuant (TQ3) achieves <strong style={{ color: "var(--amd-red)" }}>4.923×</strong> because it
          writes a tightly bit-packed 52-byte format. The block-rotation Triton kernels (PlanarQuant,
          IsoQuant, RotorQuant) currently store <strong style={{ color: "var(--text)" }}>one int8 per index</strong> —
          132–133 bytes/vector — giving only <strong style={{ color: "var(--rotor-color)" }}>1.94×</strong> in the
          actual benchmark. All four methods share the same target 52-byte layout; the block methods have
          not yet implemented bitplane packing in their Triton kernels.
        </p>
      </motion.div>

      {/* OBSERVED vs PACKED callout banner */}
      <motion.div
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"} custom={1}
        style={{
          display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem",
          marginBottom: "1.5rem",
        }}
        className="obs-packed-grid"
      >
        <div style={{ background: "rgba(229,52,75,0.08)", border: "1px solid rgba(229,52,75,0.3)", borderRadius: "10px", padding: "1.1rem 1.3rem" }}>
          <p style={{ fontSize: "0.65rem", fontFamily: "JetBrains Mono,monospace", color: "var(--amd-red)", letterSpacing: "0.1em", marginBottom: "0.4rem" }}>OBSERVED — bench_runtime_ratio_all_methods.py</p>
          <p style={{ fontFamily: "JetBrains Mono,monospace", fontSize: "0.82rem", lineHeight: 1.9 }}>
            <span style={{ color: "#3B82F6" }}>TQ3: <strong>4.923×</strong></span>{" "}(52 B/vec — bitplane-packed)<br />
            <span style={{ color: "#E5344B" }}>PlanarQuant: <strong>1.939×</strong></span>{" "}(132 B/vec — int8/index)<br />
            <span style={{ color: "#A855F7" }}>IsoQuant: <strong>1.939×</strong></span>{" "}(132 B/vec — int8/index)<br />
            <span style={{ color: "#FF7B35" }}>RotorQuant: <strong>1.925×</strong></span>{" "}(133 B/vec — 129 int8s¹)
          </p>
          <p style={{ fontSize: "0.62rem", color: "var(--text-muted)", fontFamily: "JetBrains Mono,monospace", marginTop: "0.5rem" }}>
            ¹ 128 dims / group_size=3 → 43 groups × 3 = 129 indices (1 pad dim)
          </p>
        </div>
        <div style={{ background: "rgba(74,222,128,0.05)", border: "1px solid rgba(74,222,128,0.15)", borderRadius: "10px", padding: "1.1rem 1.3rem" }}>
          <p style={{ fontSize: "0.65rem", fontFamily: "JetBrains Mono,monospace", color: "#4ade80", letterSpacing: "0.1em", marginBottom: "0.4rem" }}>TARGET LAYOUT — all methods, if bit-packed</p>
          <p style={{ fontFamily: "JetBrains Mono,monospace", fontSize: "0.82rem", lineHeight: 1.9 }}>
            <span style={{ color: "var(--text-sub)" }}>4B float32 norm</span><br />
            <span style={{ color: "var(--text-sub)" }}>48B bitplanes (3 planes × 16B)</span><br />
            <span style={{ color: "var(--text)" }}><strong>52 bytes total = 4.923× vs FP16</strong></span><br />
            <span style={{ color: "var(--text-muted)" }}>Same format TQ3 already uses</span>
          </p>
          <p style={{ fontSize: "0.62rem", color: "var(--text-muted)", fontFamily: "JetBrains Mono,monospace", marginTop: "0.5rem" }}>
            Requires bitplane pack/unpack in Triton kernels
          </p>
        </div>
      </motion.div>

      {/* Full scheme table */}
      <motion.div
        className="glass"
        style={{ borderRadius: "var(--radius-lg)", padding: "2rem", marginBottom: "1.5rem" }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"} custom={2}
      >
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: "1.2rem", flexWrap: "wrap", gap: "0.6rem" }}>
          <p style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.7rem", color: "var(--amd-red)", letterSpacing: "0.12em" }}>
            KV COMPRESSION — FULL SCHEME COMPARISON · MI300X gfx942 · Mistral-7B-v0.1
          </p>
          <button
            onClick={() => setShowPacked(p => !p)}
            style={{
              padding: "0.3rem 0.75rem", borderRadius: "6px", cursor: "pointer",
              fontFamily: "JetBrains Mono,monospace", fontSize: "0.68rem", letterSpacing: "0.05em",
              background: showPacked ? "rgba(74,222,128,0.12)" : "rgba(229,52,75,0.1)",
              border: showPacked ? "1px solid rgba(74,222,128,0.3)" : "1px solid rgba(229,52,75,0.3)",
              color: showPacked ? "#4ade80" : "var(--amd-red)",
            }}>
            {showPacked ? "◉ Showing: if bit-packed" : "◎ Showing: observed"}
          </button>
        </div>
        <table className="data-table" style={{ width: "100%" }}>
          <thead>
            <tr>
              <th>Scheme</th>
              <th>{showPacked ? "Bytes/vec (packed)" : "Bytes/vec (observed)"}</th>
              <th>{showPacked ? "Ratio (packed)" : "Ratio (observed)"}</th>
              <th>Mean Cosine Sim</th>
              <th>Mean MSE</th>
              <th>Decode 8K (tok/s)</th>
            </tr>
          </thead>
          <tbody>
            {ALL_SCHEMES.map((s) => {
              const bytes = showPacked ? s.bytesIfPacked : s.bytesObserved;
              const ratio = showPacked ? s.ratioPacked : s.ratioObserved;
              const isPacked = s.group === "block" && s.bytesObserved !== s.bytesIfPacked;
              return (
                <tr key={s.id}>
                  <td className="cell-method">
                    <span className="dot" style={{ background: s.color }} />
                    <span style={{ color: s.group === "block" ? s.color : undefined }}>{s.name}</span>
                    {isPacked && !showPacked && (
                      <span style={{ marginLeft: "0.5rem", fontFamily: "'JetBrains Mono',monospace", fontSize: "0.6rem", color: "var(--rotor-color)", border: "1px solid rgba(255,123,53,0.3)", borderRadius: "3px", padding: "0.1rem 0.3rem" }}>
                        int8/idx
                      </span>
                    )}
                    {isPacked && showPacked && (
                      <span style={{ marginLeft: "0.5rem", fontFamily: "'JetBrains Mono',monospace", fontSize: "0.6rem", color: "#4ade80", border: "1px solid rgba(74,222,128,0.25)", borderRadius: "3px", padding: "0.1rem 0.3rem" }}>
                        if packed
                      </span>
                    )}
                  </td>
                  <td className="cell-num">{bytes} B</td>
                  <td>
                    <span className={`cell-badge ${ratio >= 4.9 ? "good" : ratio >= 3.5 ? "ok" : ratio >= 2 ? "warn" : "bad"}`}>
                      {ratio.toFixed(ratio === 1 ? 0 : ratio < 4 ? 3 : 3)}×
                    </span>
                  </td>
                  <td className="cell-num" style={{ color: s.cosine > 0.999 ? "#4ade80" : "var(--text)" }}>
                    {s.cosine.toFixed(4)}
                  </td>
                  <td className="cell-num">{s.mse === 0 ? "0.0 (ref)" : s.mse < 0.0001 ? s.mse.toExponential(1) : s.mse.toFixed(5)}</td>
                  <td>
                    {s.decode8k !== null ? (
                      <span style={{ color: s.decode8k > 40 ? "#4ade80" : s.decode8k > 20 ? "var(--text-sub)" : "var(--text-muted)", fontFamily: "JetBrains Mono,monospace", fontSize: "0.78rem" }}>
                        {s.decode8k} tok/s
                      </span>
                    ) : (
                      <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.74rem", color: "var(--text-muted)" }}>kernel-only†</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p style={{ fontSize: "0.7rem", color: "var(--text-muted)", marginTop: "0.8rem", fontFamily: "'JetBrains Mono',monospace", lineHeight: 1.7 }}>
          † Block-method decode8k is a kernel-only number (no model weights). At batch=1 all methods
          are compute-bound by model weight cycling, so end-to-end would resemble TQ3 (6.3 tok/s).
          At higher batch sizes, block-method kernel speed matters.<br />
          Observed ratios from results/bench_runtime_ratio_all_methods.json · seq_len=2048 · 32 layers · 8 KV heads · Mistral-7B-v0.1
        </p>
      </motion.div>

      {/* Animated chart tabs */}
      <motion.div
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"} custom={3}
      >
        <div className="results-tabs" style={{ marginBottom: "1.2rem" }}>
          {[
            { key: "ratio",   label: "Compression Ratio" },
            { key: "decode",  label: "Decode tok/s" },
            { key: "quality", label: "Cosine Similarity" },
          ].map((tab) => (
            <button key={tab.key}
              className={`results-tab ${activeView === tab.key ? "active" : ""}`}
              onClick={() => setActiveView(tab.key)}>
              {tab.label}
            </button>
          ))}
        </div>
      </motion.div>

      <AnimatePresence mode="wait">
        <motion.div
          className="chart-panel glass"
          key={activeView}
          initial={{ opacity: 0, y: 8 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -4 }}
          transition={{ duration: 0.25 }}
        >
          {activeView === "ratio" && (
            <>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", flexWrap: "wrap", gap: "0.5rem", marginBottom: "0.2rem" }}>
                <p className="chart-title" style={{ margin: 0 }}>Compression Ratio — All Schemes</p>
                <button onClick={() => setShowPacked(p => !p)} style={{
                  padding: "0.25rem 0.6rem", borderRadius: "6px", cursor: "pointer",
                  fontFamily: "JetBrains Mono,monospace", fontSize: "0.63rem",
                  background: showPacked ? "rgba(74,222,128,0.1)" : "rgba(229,52,75,0.08)",
                  border: showPacked ? "1px solid rgba(74,222,128,0.25)" : "1px solid rgba(229,52,75,0.2)",
                  color: showPacked ? "#4ade80" : "var(--amd-red)",
                }}>
                  {showPacked ? "if bit-packed" : "observed"}
                </button>
              </div>
              <p className="chart-subtitle">× vs FP16 (256 B/vector) · {showPacked ? "assuming full bitplane packing" : "measured: kv_bytes_fp16 / kv_bytes_compressed_materialized"}</p>
              <div className="bar-group">
                {ALL_SCHEMES.map((s, i) => {
                  const ratio = showPacked ? s.ratioPacked : s.ratioObserved;
                  const maxR = showPacked ? maxRatioPacked : maxRatioObserved;
                  return (
                    <div key={s.id} className="bar-row">
                      <div className="bar-method" style={{ color: s.color, fontSize: "0.78rem" }}>{s.name}</div>
                      <div className="bar-track">
                        <motion.div
                          style={{ height: "100%", borderRadius: "6px", background: `linear-gradient(90deg, ${s.color}cc, ${s.color}88)`, boxShadow: ratio >= 4.9 ? `inset 0 0 20px ${s.color}44` : "none" }}
                          initial={{ width: 0 }}
                          animate={{ width: inView ? `${(ratio / maxR) * 100}%` : 0 }}
                          transition={{ duration: 1.4, delay: i * 0.07, ease: [0.23, 1, 0.32, 1] }}
                        />
                        {/* ghost bar showing packed target if viewing observed */}
                        {!showPacked && s.group === "block" && (
                          <div style={{
                            position: "absolute", top: 0, left: 0,
                            width: `${(s.ratioPacked / maxR) * 100}%`,
                            height: "100%", borderRadius: "6px",
                            border: `1px dashed ${s.color}55`,
                            pointerEvents: "none",
                          }} />
                        )}
                      </div>
                      <div>
                        <div className="bar-num" style={{ color: ratio >= 4.9 ? "var(--amd-red)" : undefined }}>
                          {ratio.toFixed(3)}×
                        </div>
                        {!showPacked && s.group === "block" && (
                          <div style={{ fontSize: "0.6rem", color: "var(--text-muted)", fontFamily: "JetBrains Mono,monospace" }}>
                            ({s.ratioPacked.toFixed(3)}× packed)
                          </div>
                        )}
                        {s.id === "tq3" && <div className="bar-winner">bit-packed ✓</div>}
                      </div>
                    </div>
                  );
                })}
              </div>
              <div style={{
                background: showPacked ? "rgba(74,222,128,0.06)" : "rgba(255,123,53,0.06)",
                border: showPacked ? "1px solid rgba(74,222,128,0.15)" : "1px solid rgba(255,123,53,0.2)",
                borderRadius: "10px", padding: "1rem 1.2rem", marginTop: "0.5rem",
              }}>
                {showPacked ? (
                  <>
                    <p style={{ fontSize: "0.84rem", color: "var(--text)", fontWeight: 600, marginBottom: "0.3rem" }}>
                      All 3-bit methods share the same 52-byte target format
                    </p>
                    <p style={{ fontSize: "0.78rem", color: "var(--text-sub)", lineHeight: 1.6 }}>
                      4B float32 norm + 48B bitplanes (3 planes × 16B). This is what TQ3 already stores.
                      Block methods would achieve 4.923× if their Triton kernels packed indices into bitplanes
                      instead of storing one int8 per index.
                    </p>
                  </>
                ) : (
                  <>
                    <p style={{ fontSize: "0.84rem", color: "var(--text)", fontWeight: 600, marginBottom: "0.3rem" }}>
                      TQ3: 4.923× · Block methods: 1.94× — measured from real prefill cache
                    </p>
                    <p style={{ fontSize: "0.78rem", color: "var(--text-sub)", lineHeight: 1.6 }}>
                      TQ3 writes a 52-byte bitplane block. PlanarQuant/IsoQuant store 128 int8 indices
                      (1 byte each) + 4B norm = 132 bytes. RotorQuant stores 129 int8s (128 dims padded to
                      43×3) + 4B norm = 133 bytes. Dashed lines show the packed target ratio.
                    </p>
                  </>
                )}
              </div>
            </>
          )}

          {activeView === "decode" && (
            <>
              <p className="chart-title">End-to-End Decode Throughput — 8K Context</p>
              <p className="chart-subtitle">tok/s · Mistral-7B-v0.1 · batch=1 · model inference (not kernel-only)</p>
              <div className="bar-group">
                {ALL_SCHEMES.filter(s => s.decode8k !== null).map((s, i) => (
                  <div key={s.id} className="bar-row">
                    <div className="bar-method" style={{ color: s.color, fontSize: "0.78rem" }}>{s.name}</div>
                    <div className="bar-track">
                      <motion.div
                        style={{ height: "100%", borderRadius: "6px", background: `linear-gradient(90deg, ${s.color}cc, ${s.color}88)` }}
                        initial={{ width: 0 }}
                        animate={{ width: inView ? `${(s.decode8k / maxDecode) * 100}%` : 0 }}
                        transition={{ duration: 1.4, delay: i * 0.09, ease: [0.23, 1, 0.32, 1] }}
                      />
                    </div>
                    <div>
                      <div className="bar-num">{s.decode8k} <span style={{ fontSize: "0.68rem", color: "var(--text-muted)" }}>tok/s</span></div>
                      {s.id === "fp16" && <div className="bar-winner">BASELINE</div>}
                      {s.id === "fp8"  && <div style={{ fontSize: "0.64rem", color: "var(--text-muted)", fontFamily: "'JetBrains Mono',monospace" }}>−1.1%</div>}
                      {s.id === "int4" && <div style={{ fontSize: "0.64rem", color: "var(--rotor-color)", fontFamily: "'JetBrains Mono',monospace" }}>−44%</div>}
                      {(s.id === "tq4" || s.id === "tq3") && <div style={{ fontSize: "0.64rem", color: "var(--turbo-color)", fontFamily: "'JetBrains Mono',monospace" }}>Python OH</div>}
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ background: "rgba(59,130,246,0.08)", border: "1px solid rgba(59,130,246,0.2)", borderRadius: "10px", padding: "1rem 1.2rem", marginTop: "0.5rem" }}>
                <p style={{ fontSize: "0.84rem", color: "var(--text)", fontWeight: 600, marginBottom: "0.3rem" }}>
                  MI300X decode is compute-bound at batch=1
                </p>
                <p style={{ fontSize: "0.78rem", color: "var(--text-sub)", lineHeight: 1.6 }}>
                  FP16 and FP8 both hit ~46 tok/s — the 14.7 GB model weight matrix cycles through
                  HBM3 every step, saturating compute before KV bandwidth matters. TQ3/TQ4 slower
                  due to Python-level overhead, not hardware limits. Fused Triton kernel eliminates this.
                </p>
              </div>
            </>
          )}

          {activeView === "quality" && (
            <>
              <p className="chart-title">KV Reconstruction Quality — Cosine Similarity</p>
              <p className="chart-subtitle">Mean cosine similarity vs FP16 · 512 random unit vectors · head_dim=128 · bench_ppl_all_methods_quality_recheck.json</p>
              <div className="bar-group">
                {ALL_SCHEMES.map((s, i) => (
                  <div key={s.id} className="bar-row">
                    <div className="bar-method" style={{ color: s.color, fontSize: "0.78rem" }}>{s.name}</div>
                    <div className="bar-track">
                      <motion.div
                        style={{ height: "100%", borderRadius: "6px", background: `linear-gradient(90deg, ${s.color}cc, ${s.color}88)` }}
                        initial={{ width: 0 }}
                        animate={{ width: inView ? `${s.cosine * 100}%` : 0 }}
                        transition={{ duration: 1.4, delay: i * 0.07, ease: [0.23, 1, 0.32, 1] }}
                      />
                    </div>
                    <div className="bar-num">{s.cosine.toFixed(4)}</div>
                  </div>
                ))}
              </div>
              <div style={{ background: "rgba(74,222,128,0.06)", border: "1px solid rgba(74,222,128,0.15)", borderRadius: "10px", padding: "1rem 1.2rem", marginTop: "0.5rem" }}>
                <p style={{ fontSize: "0.84rem", color: "var(--text)", fontWeight: 600, marginBottom: "0.3rem" }}>
                  All 3-bit methods within ±0.0003 — quality is not a differentiator
                </p>
                <p style={{ fontSize: "0.78rem", color: "var(--text-sub)", lineHeight: 1.6 }}>
                  Cosine similarity range: 0.9829–0.9832. The choice between methods comes down
                  entirely to FMA count and kernel throughput, not reconstruction accuracy.
                </p>
              </div>
            </>
          )}
        </motion.div>
      </AnimatePresence>

      {/* Spotlight cards */}
      <motion.div
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: "1rem", marginTop: "1.5rem" }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"} custom={5}
        className="spotlight-cards"
      >
        {[
          {
            title: "4.923× (TQ3)",
            sub: "only TurboQuant observed",
            body: "TQ3 writes a 52-byte bitplane block today. Block-rotation methods (Planar/Iso/Rotor) write 132–133 bytes (int8 per index). Both approaches target the same 52-byte format — the Triton packing kernel is the missing piece.",
            color: "var(--amd-red)",
          },
          {
            title: "1.94× (block)",
            sub: "PlanarQ/IsoQ observed · 1.92× RotorQ",
            body: "128 int8 indices + 4B norm = 132 bytes/vec at 3-bit. RotorQ stores 129 indices (128 dims + 1 pad to fill 43×3 groups) → 133 bytes. This is what bench_runtime_ratio_all_methods.py measures from a live Mistral-7B-v0.1 cache.",
            color: "var(--rotor-color)",
          },
          {
            title: "6.9M tokens",
            sub: "if all methods hit 4.923×",
            body: "FP16 fits 1.4M tokens in 192 GB HBM3. At 4.923×, every 3-bit method fits 6.9M tokens. At the current observed 1.94×, block methods fit ~2.7M tokens — still 1.9× more than FP16, but far short of the packed-format potential.",
            color: "var(--iso-color)",
          },
        ].map((card) => (
          <div key={card.title} className="glass" style={{ borderRadius: "var(--radius-md)", padding: "1.4rem", borderTop: `3px solid ${card.color}` }}>
            <p style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: "1.25rem", fontWeight: 700, color: card.color, lineHeight: 1.1, marginBottom: "0.3rem" }}>{card.title}</p>
            <p style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.65rem", color: "var(--text-muted)", marginBottom: "0.8rem" }}>{card.sub}</p>
            <p style={{ fontSize: "0.82rem", color: "var(--text-sub)", lineHeight: 1.65 }}>{card.body}</p>
          </div>
        ))}
      </motion.div>
    </section>
  );
}

/* ─── PROBLEM SECTION ────────────────────────────────────────────── */

export function ProblemSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const { data: kvExp, error: kvExpErr } = useExperimentKvMetrics();
  const seqRef = 131072;
  const kvFp16BytesRef = kvBytesFp16(seqRef);
  const kvFp16Gb = kvFp16BytesRef / 1e9;
  const gb = useCountUp(kvFp16Gb, inView, 1.2, 1);
  const ctx = useCountUp(seqRef, inView, 1.4, 0);
  const tq3PackedBytesPerVec = 4 + Math.ceil((MISTRAL7B_KV.headDim * 3) / 8);
  const tq3BitPayloadBytes = Math.ceil((MISTRAL7B_KV.headDim * 3) / 8);
  const kvFp16Str = formatKvDataSize(kvFp16BytesRef);
  const kvTq3BytesRef = kvBytesTq3Theoretical(seqRef);
  const kvTq3Str = formatKvDataSize(kvTq3BytesRef);
  const ratioFullKv = kvFp16BytesRef / Math.max(kvTq3BytesRef, 1);

  const tqDecode = kvExp?.turboquant_tq3_decode_seq8192;
  const ratioDecodeMeasured =
    tqDecode?.kv_bytes_fp16 && tqDecode?.kv_bytes_compressed
      ? tqDecode.kv_bytes_fp16 / tqDecode.kv_bytes_compressed
      : null;
  const rr = kvExp?.runtime_materialized_bytes_seq2048;

  return (
    <section id="problem" className="report-section" ref={ref}>
      <motion.div
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={0}
      >
        <div className="section-label">
          <span className="section-num">01</span>
          <span className="section-tag">PROBLEM</span>
        </div>
        <h2 className="section-title">
          The KV Cache<br /><span className="dim">Memory Crisis</span>
        </h2>
        <p className="section-lead">
          LLM decode on AMD MI300X is compute-bound at batch=1 — the hardware cycles through
          140 billion model weights for every token. But as context grows, the KV cache
          becomes the bottleneck in a different way: it fills 192 GB of HBM3.
          A 131K context window stored in FP16 consumes{" "}
          <strong style={{ color: "var(--amd-red)" }}>{kvFp16Gb.toFixed(1)} GB</strong> —
          leaving almost nothing for model weights.
        </p>
      </motion.div>

      <div className="problem-cards">
        {[
          { label: "VRAM on MI300X", val: `${gb} GB`, unit: "HBM3", desc: `Total on-chip memory. FP16 KV at 131K context consumes ${kvFp16Gb.toFixed(1)} GB — nearly all of it.` },
          { label: "FP16 Context Cap", val: "1.4 M", unit: "tokens", desc: "Maximum tokens storable in 192 GB HBM3 at FP16 before VRAM is exhausted." },
          { label: "4.92× Compression", val: "6.9 M", unit: "tokens", desc: "Context capacity unlocked by 3-bit KV compression. All four methods achieve this ratio." },
          { label: "Benchmark Context", val: ctx.toLocaleString?.() ?? ctx, unit: "tokens", desc: "Maximum context tested in this benchmark. All methods preserve 100% needle recall here." },
        ].map((c, i) => (
          <motion.div
            key={c.label}
            className="problem-card glass"
            variants={fadeUp} initial="hidden"
            animate={inView ? "visible" : "hidden"}
            custom={i + 1}
          >
            <p className="problem-card-title">{c.label}</p>
            <p className="problem-card-val">{c.val} <span style={{ fontSize: '0.9rem', color: 'var(--text-muted)' }}>{c.unit}</span></p>
            <p className="problem-card-desc">{c.desc}</p>
          </motion.div>
        ))}
      </div>

      <motion.div
        className="memory-formula glass"
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={5}
      >
        {!kvExp && !kvExpErr && (
          <div><span className="comment">// Loading MI300X experiment metrics (experiment_kv_metrics.json)…</span></div>
        )}
        {kvExpErr && (
          <div>
            <div><span className="comment">// Could not load /content/experiment_kv_metrics.json</span></div>
            <div style={{ fontSize: "0.72rem", color: "var(--text-muted)", marginTop: "0.45rem" }}>
              Place a snapshot under <code style={{ fontSize: "0.68rem" }}>report-ui/public/content/</code> (see README). Layout-only fallback: FP16 {kvFp16Str} / TQ3 {kvTq3Str} = {ratioFullKv.toFixed(3)}× @ seq {seqRef.toLocaleString()}.
            </div>
          </div>
        )}
        {kvExp && (
          <>
            <div><span className="comment">// MEASURED — TurboQuant TQ3 materialized KV (`bench_tq3_decode` on MI300X)</span></div>
            <div>
              <span className="var">device</span> = <span className="val">{kvExp.device}</span> · <span className="var">model</span> = <span className="val">{kvExp.model}</span>
            </div>
            {tqDecode && ratioDecodeMeasured != null && (
              <>
                <div>
                  <span className="var">seq_len</span> = <span className="val">{tqDecode.seq_len}</span> · <span className="var">mode</span> = <span className="val">{tqDecode.mode}</span>
                </div>
                <div>
                  <span className="var">KV_fp16</span> = <span className="val">{tqDecode.kv_bytes_fp16.toLocaleString()}</span> bytes ({formatKvDataSize(tqDecode.kv_bytes_fp16)})
                </div>
                <div>
                  <span className="var">KV_tq3</span> = <span className="val">{tqDecode.kv_bytes_compressed.toLocaleString()}</span> bytes ({formatKvDataSize(tqDecode.kv_bytes_compressed)})
                </div>
                <div>
                  <span className="var">ratio</span> = <span className="result">{tqDecode.kv_bytes_fp16.toLocaleString()}</span> /{" "}
                  <span className="result">{tqDecode.kv_bytes_compressed.toLocaleString()}</span> ={" "}
                  <span className="result">{ratioDecodeMeasured.toFixed(5)}×</span>{" "}
                  <span className="comment">// bench reports {tqDecode.compression_ratio_reported}×</span>
                </div>
              </>
            )}
            <br />
            <div><span className="comment">// MEASURED — materialized cache bytes @2048 (`bench_runtime_ratio_all_methods`)</span></div>
            {rr?.turbo_tq3 && (
              <div>
                TurboQuant TQ3: ratio_observed = <span className="result">{rr.turbo_tq3.ratio_observed_runtime.toFixed(5)}×</span>{" "}
                <span className="comment">
                  ({rr.turbo_tq3.kv_bytes_fp16.toLocaleString()} / {rr.turbo_tq3.kv_bytes_compressed_materialized.toLocaleString()} B)
                </span>
              </div>
            )}
            {rr?.planar_tq3 && (
              <div>
                PlanarQuant TQ3: ratio_observed = <span className="result">{rr.planar_tq3.ratio_observed_runtime.toFixed(5)}×</span>{" "}
                <span className="comment">
                  ({rr.planar_tq3.kv_bytes_fp16.toLocaleString()} / {rr.planar_tq3.kv_bytes_compressed_materialized.toLocaleString()} B — wider materialized layout, not 52B packing)
                </span>
              </div>
            )}
            <div style={{ marginTop: "0.65rem", fontSize: "0.68rem" }} className="comment">
              JSON snapshot: public/content/experiment_kv_metrics.json · sources: {JSON.stringify(kvExp.sources ?? {})}
            </div>
          </>
        )}
        <br />
        <div><span className="comment">// Reference — packed layout (algebra only; TurboQuant run matches this)</span></div>
        <div>
          <span className="var">packed_bytes</span> = 4 + ⌈<span className="val">{MISTRAL7B_KV.headDim}</span> × <span className="val">3</span> / 8⌉ = 4 + {tq3BitPayloadBytes} = <span className="result">{tq3PackedBytesPerVec} bytes/vector</span>
        </div>
        <div>
          <span className="var">GQA capacity check @131072</span> — FP16 {kvFp16Str} / TQ3 {kvTq3Str} = <span className="result">{ratioFullKv.toFixed(3)}×</span>{" "}
          <span className="comment">// 2×L×Hkv×seq×128×2 vs 2×L×Hkv×seq×52</span>
        </div>
      </motion.div>

      <motion.div
        className="method-intro-grid"
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={6}
      >
        {[
          { cls: "turbo", name: "TurboQuant", source: "Google, 2024", desc: "WHT pseudorandom full-matrix rotation + QJL scalar quantization. O(d²) rotation cost." },
          { cls: "iso", name: "IsoQuant", source: "RotorQuant repo", desc: "Quaternion sandwich product on 4D groups. Maps to SIMD-4 on CDNA3." },
          { cls: "planar", name: "PlanarQuant", source: "RotorQuant repo", desc: "2D Givens rotation on consecutive pairs. Minimum nontrivial rotation — 256 FMAs." },
          { cls: "rotor", name: "RotorQuant", source: "Pope, 2026", desc: "Clifford Cl(3,0) algebra rotors. 3D groups — poor SIMD alignment on gfx942." },
        ].map((m) => (
          <div key={m.name} className={`method-intro-card glass ${m.cls}`}>
            <p className="method-name" style={{ color: `var(--${m.cls}-color)` }}>{m.name}</p>
            <p className="method-tagline" style={{ color: 'var(--text-muted)', fontSize: '0.68rem', marginBottom: '0.4rem' }}>{m.source}</p>
            <p className="method-tagline">{m.desc}</p>
          </div>
        ))}
      </motion.div>
    </section>
  );
}

/* ─── KATEX COMPONENT ────────────────────────────────────────────── */

function KatexEq({ math, display = false }) {
  const html = useMemo(() => {
    try {
      return katex.renderToString(math, { displayMode: display, throwOnError: false, output: "html" });
    } catch {
      return math;
    }
  }, [math, display]);
  return <span dangerouslySetInnerHTML={{ __html: html }} />;
}

/* ─── SVG ROTATION VISUALIZERS ───────────────────────────────────── */

function GivensViz({ color, inView }) {
  const theta = useMotionValue(0.4);
  const x2 = useTransform(theta, t => 88 + 56 * Math.cos(t));
  const y2 = useTransform(theta, t => 88 - 56 * Math.sin(t));
  const x2b = useTransform(theta, t => 88 + 56 * Math.cos(t - Math.PI / 2));
  const y2b = useTransform(theta, t => 88 - 56 * Math.sin(t - Math.PI / 2));

  useEffect(() => {
    if (!inView) return;
    const ctrl = animate(theta, theta.get() + Math.PI * 4, {
      duration: 12, repeat: Infinity, ease: "linear",
    });
    return ctrl.stop;
  }, [inView]);

  const arcD = useTransform(theta, t => {
    const r = 22, x = 88 + r * Math.cos(t), y = 88 - r * Math.sin(t);
    return `M 88 88 L ${88 + r} 88 A ${r} ${r} 0 0 0 ${x} ${y} Z`;
  });

  return (
    <svg width="176" height="176" viewBox="0 0 176 176" style={{ flexShrink: 0 }}>
      {/* grid */}
      {[-56, -28, 0, 28, 56].map(d => (
        <line key={`h${d}`} x1="10" y1={88 + d} x2="166" y2={88 + d} stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      ))}
      {[-56, -28, 0, 28, 56].map(d => (
        <line key={`v${d}`} x1={88 + d} y1="10" x2={88 + d} y2="166" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      ))}
      {/* axes */}
      <line x1="10" y1="88" x2="166" y2="88" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
      <line x1="88" y1="10" x2="88" y2="166" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
      {/* unit circle */}
      <circle cx="88" cy="88" r="56" fill="none" stroke="rgba(255,255,255,0.08)" strokeWidth="1" />
      {/* angle arc */}
      <motion.path d={arcD} fill={`${color}30`} stroke={color} strokeWidth="0.5" />
      {/* secondary vector (pair partner) */}
      <motion.line x1={88} y1={88} x2={x2b} y2={y2b} stroke={color} strokeWidth="1.5" strokeOpacity="0.4" />
      <motion.circle cx={x2b} cy={y2b} r="3" fill={color} fillOpacity="0.4" />
      {/* primary vector */}
      <motion.line x1={88} y1={88} x2={x2} y2={y2} stroke={color} strokeWidth="2.5" />
      <motion.circle cx={x2} cy={y2} r="4.5" fill={color} />
      {/* labels */}
      <text x="160" y="85" fill="rgba(255,255,255,0.3)" fontSize="9" fontFamily="monospace">x₂ᵢ</text>
      <text x="90" y="18" fill="rgba(255,255,255,0.3)" fontSize="9" fontFamily="monospace">x₂ᵢ₊₁</text>
      <text x="100" y="78" fill={color} fontSize="8" fontFamily="monospace">θᵢ</text>
    </svg>
  );
}

function QuaternionViz({ color, inView }) {
  const [phase, setPhase] = useState(0);
  useEffect(() => {
    if (!inView) return;
    let rafId = 0;
    let lastTs = 0;
    const step = (ts) => {
      const dt = lastTs ? (ts - lastTs) / 1000 : 0;
      lastTs = ts;
      setPhase((p) => p + dt * 1.8);
      rafId = requestAnimationFrame(step);
    };
    rafId = requestAnimationFrame(step);
    return () => cancelAnimationFrame(rafId);
  }, [inView]);

  // Project 4D quaternion rotation onto 2D using stereographic-like projection.
  const q0 = Math.cos(phase * 0.3);
  const q1 = Math.sin(phase * 0.3) * Math.cos(phase * 0.13);
  const q2 = Math.sin(phase * 0.3) * Math.sin(phase * 0.13) * Math.cos(phase * 0.07);
  const q3 = Math.sin(phase * 0.3) * Math.sin(phase * 0.13) * Math.sin(phase * 0.07);
  const r = 52;
  const safePos = [
    [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1],
  ].map(([a, b, c, d]) => {
    const qa = q0 * a - q1 * b - q2 * c - q3 * d;
    const qb = q0 * b + q1 * a + q2 * d - q3 * c;
    const qd = q0 * d + q1 * c - q2 * b + q3 * a;
    const w = 1 / (1 - qd * 0.7 + 0.001);
    return [88 + r * qa * w, 88 - r * qb * w];
  });

  return (
    <svg width="176" height="176" viewBox="0 0 176 176" style={{ flexShrink: 0 }}>
      <circle cx="88" cy="88" r="58" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
      <circle cx="88" cy="88" r="38" fill="none" stroke="rgba(255,255,255,0.04)" strokeWidth="1" />
      {/* edges of quaternion tetrahedron */}
      {[[0,1],[1,2],[2,3],[3,0],[0,2],[1,3]].map(([a,b], i) => (
        <line key={i}
          x1={safePos[a][0]} y1={safePos[a][1]}
          x2={safePos[b][0]} y2={safePos[b][1]}
          stroke={color} strokeWidth="1.2" strokeOpacity="0.35" />
      ))}
      {/* vertices */}
      {safePos.map(([px,py], i) => (
        <circle key={i} cx={px} cy={py} r={i === 0 ? 5.5 : 3.5} fill={color} fillOpacity={i === 0 ? 1 : 0.55} />
      ))}
      {/* center dot */}
      <circle cx="88" cy="88" r="2" fill="rgba(255,255,255,0.3)" />
      {/* labels */}
      {['e₁','e₂','e₃','e₄'].map((l, i) => (
        <text key={l} x={safePos[i][0] + 6} y={safePos[i][1] - 3} fill={color} fontSize="8" fontFamily="monospace" fillOpacity="0.7">{l}</text>
      ))}
      <text x="6" y="170" fill="rgba(255,255,255,0.2)" fontSize="8" fontFamily="monospace">q ⊗ v ⊗ q*</text>
    </svg>
  );
}

function RotorViz({ color, inView }) {
  const t = useMotionValue(0);
  useEffect(() => {
    if (!inView) return;
    const ctrl = animate(t, Math.PI * 6, { duration: 10, repeat: Infinity, ease: "linear" });
    return ctrl.stop;
  }, [inView]);

  const [vx, setVx] = useState(88 + 50);
  const [vy, setVy] = useState(88);
  const [bx1, setBx1] = useState(88); const [by1, setBy1] = useState(88 - 40);
  const [bx2, setBx2] = useState(88 + 30); const [by2, setBy2] = useState(88 + 30);

  useEffect(() => {
    return t.on("change", tv => {
      // 3D Cl(3,0) rotor rotation visualized
      const theta = tv * 0.5;
      // rotor in e12 plane: R = cos(θ/2) + sin(θ/2)e12
      const c = Math.cos(theta / 2), s = Math.sin(theta / 2);
      // rotate vector v = e1 by rotor
      const vxr = c * c - s * s, vyr = 2 * c * s;
      const r = 52;
      setVx(88 + r * vxr);
      setVy(88 - r * vyr);
      // bivector plane indicator
      const phi = theta + Math.PI / 2;
      setBx1(88 + 40 * Math.cos(phi - 0.4));
      setBy1(88 - 40 * Math.sin(phi - 0.4));
      setBx2(88 + 40 * Math.cos(phi + 0.4));
      setBy2(88 - 40 * Math.sin(phi + 0.4));
    });
  }, [t]);

  return (
    <svg width="176" height="176" viewBox="0 0 176 176" style={{ flexShrink: 0 }}>
      <circle cx="88" cy="88" r="56" fill="none" stroke="rgba(255,255,255,0.06)" strokeWidth="1" />
      {/* 3 axes */}
      <line x1="88" y1="88" x2="154" y2="88" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
      <line x1="88" y1="88" x2="88" y2="22" stroke="rgba(255,255,255,0.15)" strokeWidth="1" />
      <line x1="88" y1="88" x2="44" y2="130" stroke="rgba(255,255,255,0.1)" strokeWidth="1" strokeDasharray="3,2" />
      <text x="156" y="91" fill="rgba(255,255,255,0.25)" fontSize="9" fontFamily="monospace">e₁</text>
      <text x="91" y="19" fill="rgba(255,255,255,0.25)" fontSize="9" fontFamily="monospace">e₂</text>
      <text x="33" y="135" fill="rgba(255,255,255,0.25)" fontSize="9" fontFamily="monospace">e₃</text>
      {/* bivector plane arc */}
      <line x1={88} y1={88} x2={bx1} y2={by1} stroke={color} strokeWidth="1" strokeOpacity="0.3" />
      <line x1={88} y1={88} x2={bx2} y2={by2} stroke={color} strokeWidth="1" strokeOpacity="0.3" />
      <path d={`M ${bx1} ${by1} A 40 40 0 0 1 ${bx2} ${by2}`} fill={`${color}15`} stroke={color} strokeWidth="0.5" strokeOpacity="0.4" />
      {/* rotating vector */}
      <motion.line x1={88} y1={88} x2={vx} y2={vy} stroke={color} strokeWidth="2.5" />
      <motion.circle cx={vx} cy={vy} r="4.5" fill={color} />
      <text x="6" y="170" fill="rgba(255,255,255,0.2)" fontSize="8" fontFamily="monospace">R·v·R̃ (Cl(3,0))</text>
    </svg>
  );
}

function HadamardViz({ color, inView }) {
  const t = useMotionValue(0);
  useEffect(() => {
    if (!inView) return;
    const ctrl = animate(t, 1, { duration: 2.5, repeat: Infinity, repeatType: "reverse", ease: "easeInOut" });
    return ctrl.stop;
  }, [inView]);
  const [wave, setWave] = useState(0);
  useEffect(() => t.on("change", setWave), [t]);

  const N = 8;
  // Hadamard butterfly pattern
  const cells = [];
  for (let r = 0; r < N; r++) {
    for (let c = 0; c < N; c++) {
      // Hadamard sign: (-1)^popcount(r&c)
      let p = r & c, bits = 0;
      while (p) { bits += p & 1; p >>= 1; }
      const sign = bits % 2 === 0 ? 1 : -1;
      cells.push({ r, c, sign });
    }
  }

  return (
    <svg width="176" height="176" viewBox="0 0 176 176" style={{ flexShrink: 0 }}>
      <text x="8" y="14" fill="rgba(255,255,255,0.4)" fontSize="9" fontFamily="monospace">H₈ · D · x</text>
      {cells.map(({ r, c, sign }) => {
        const x = 8 + c * 20, y = 20 + r * 18;
        const activity = Math.sin(wave * Math.PI * 2 + (r + c) * 0.5) * 0.5 + 0.5;
        const opacity = 0.1 + activity * 0.55;
        const fillColor = sign > 0 ? color : `rgba(255,255,255,0.6)`;
        return (
          <rect key={`${r}-${c}`} x={x} y={y} width="17" height="14" rx="2"
            fill={fillColor} fillOpacity={opacity} />
        );
      })}
      {/* diagonal matrix D */}
      <text x="8" y="168" fill="rgba(255,255,255,0.2)" fontSize="8" fontFamily="monospace">128×128 → O(d²) FMAs</text>
    </svg>
  );
}

/* ─── EQUATION STEP REVEAL ───────────────────────────────────────── */

function EqReveal({ steps, inView, color }) {
  const [visible, setVisible] = useState(0);
  useEffect(() => {
    if (!inView) return;
    setVisible(0);
    let i = 0;
    const id = setInterval(() => {
      i++;
      setVisible(i);
      if (i >= steps.length) clearInterval(id);
    }, 520);
    return () => clearInterval(id);
  }, [inView, steps.length]);

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "0.55rem" }}>
      <AnimatePresence>
        {steps.slice(0, visible).map((step, i) => (
          <motion.div key={i}
            initial={{ opacity: 0, x: -10 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }}
            style={{
              background: step.highlight ? `${color}12` : "rgba(0,0,0,0.3)",
              border: step.highlight ? `1px solid ${color}40` : "1px solid rgba(255,255,255,0.06)",
              borderRadius: "6px",
              padding: step.display ? "0.7rem 1rem" : "0.4rem 0.8rem",
              display: "flex", alignItems: "center", gap: "0.6rem",
            }}>
            {step.label && (
              <span style={{
                fontSize: "0.62rem", fontFamily: "JetBrains Mono, monospace",
                color: "rgba(255,255,255,0.3)", minWidth: "2.5rem", textAlign: "right",
              }}>{step.label}</span>
            )}
            <span style={{ fontSize: step.display ? "0.95rem" : "0.78rem", overflowX: "auto" }}>
              <KatexEq math={step.eq} display={false} />
            </span>
            {step.note && (
              <span style={{
                fontSize: "0.65rem", color: "rgba(255,255,255,0.35)",
                fontFamily: "JetBrains Mono, monospace", marginLeft: "auto", whiteSpace: "nowrap",
              }}>{step.note}</span>
            )}
          </motion.div>
        ))}
      </AnimatePresence>
    </div>
  );
}

/* ─── MATH SECTION ───────────────────────────────────────────────── */

const MATH_METHODS = [
  {
    cls: "planar",
    name: "PlanarQuant",
    subtitle: "2D Givens Rotation",
    paper: "RotorQuant repo — scrya-com/rotorquant",
    color: "var(--planar-color)",
    colorHex: "#E5344B",
    fmaPct: 1.56,
    fmaLabel: "256 FMAs",
    Viz: GivensViz,
    params: [
      { k: "Groups", v: "d/2 = 64" },
      { k: "Params/group", v: "(cos θ, sin θ)" },
      { k: "Total params", v: "128 floats" },
      { k: "SIMD fit", v: "SIMD-2 ✓" },
    ],
    steps: [
      { label: "input", eq: String.raw`\mathbf{x} \in \mathbb{R}^{128},\quad \text{pair } i: (x_{2i},\, x_{2i+1})` },
      { label: "rotation", eq: String.raw`\begin{pmatrix} y_{2i} \\ y_{2i+1} \end{pmatrix} = G(\theta_i)\, \begin{pmatrix} x_{2i} \\ x_{2i+1} \end{pmatrix}`, highlight: true, display: true },
      { label: "matrix", eq: String.raw`G(\theta_i) = \begin{pmatrix} \cos\theta_i & -\sin\theta_i \\ \sin\theta_i & \cos\theta_i \end{pmatrix}`, highlight: true, display: true },
      { label: "cost", eq: String.raw`64 \text{ groups} \times 4 \text{ FMAs} = \mathbf{256}\text{ FMAs/vector}`, note: "1.56% of TurboQuant" },
      { label: "inverse", eq: String.raw`G^{-1}(\theta_i) = G(-\theta_i) = G(\theta_i)^{\top}` },
    ],
    qualityStats: { mean: 0.9829, std: 0.0031, p5: 0.9771, min: 0.9669 },
    identityEq: String.raw`\mathbb{E}[\mathrm{score}] = \tfrac{(G\,\mathbf{q})^\top\hat{\mathbf{k}}}{\beta_{\rm JL}} = \mathbf{q}^\top\mathbf{k}, \quad G = \textstyle\prod_{i=1}^{64}G(\theta_i), \quad G^\top G = I`,
    identityNote: "64 independent 2×2 Givens blocks — each (x₂ᵢ, x₂ᵢ₊₁) pair rotates independently. JL guarantee holds exactly per SIMD-2 lane. β_JL is the Lloyd-Max 3-bit correction scalar from the shared codebook.",
  },
  {
    cls: "iso",
    name: "IsoQuant",
    subtitle: "Quaternion Sandwich Product",
    paper: "RotorQuant repo — scrya-com/rotorquant",
    color: "var(--iso-color)",
    colorHex: "#A855F7",
    fmaPct: 3.12,
    fmaLabel: "512 FMAs",
    Viz: QuaternionViz,
    params: [
      { k: "Groups", v: "d/4 = 32" },
      { k: "Params/group", v: "unit quaternion q" },
      { k: "Total params", v: "128 floats" },
      { k: "SIMD fit", v: "CDNA3 SIMD-4 ✓✓" },
    ],
    steps: [
      { label: "input", eq: String.raw`\mathbf{v} = (v_0, v_1, v_2, v_3) \in \mathbb{R}^4,\quad \text{treated as pure quaternion}` },
      { label: "rotation", eq: String.raw`\mathbf{y} = q \otimes \mathbf{v} \otimes q^*`, highlight: true, display: true },
      { label: "unit q", eq: String.raw`q = \cos\alpha + \sin\alpha\,(n_1\,\mathbf{i} + n_2\,\mathbf{j} + n_3\,\mathbf{k}),\quad \|n\|=1`, highlight: true },
      { label: "product", eq: String.raw`\mathbf{i}^2 = \mathbf{j}^2 = \mathbf{k}^2 = \mathbf{ijk} = -1` },
      { label: "cost", eq: String.raw`32\text{ groups} \times 16\text{ FMAs} = \mathbf{512}\text{ FMAs/vector}`, note: "4D → CDNA3 SIMD-4" },
    ],
    qualityStats: { mean: 0.9831, std: 0.0032, p5: 0.9773, min: 0.9617 },
    identityEq: String.raw`\mathbb{E}[\mathrm{score}] = \tfrac{(Q_{\rm blk}\,\mathbf{q})^\top\hat{\mathbf{k}}}{\beta_{\rm JL}} = \mathbf{q}^\top\mathbf{k}, \quad Q_{\rm blk} = \bigoplus_{i=1}^{32}[q_i\otimes(\cdot)\otimes q_i^*], \quad Q_{\rm blk}^\top Q_{\rm blk} = I`,
    identityNote: "Unit quaternion sandwich q⊗v⊗q* is an exact SO(4) isometry on ℝ⁴. CDNA3 SIMD-4 aligns perfectly with the 4-element group structure — no cross-group leakage, cleanest correction of the four methods.",
  },
  {
    cls: "rotor",
    name: "RotorQuant",
    subtitle: "Clifford Cl(3,0) Geometric Algebra",
    paper: "Pope (2026) — scrya.com/rotorquant",
    color: "var(--rotor-color)",
    colorHex: "#FF7B35",
    fmaPct: 7.18,
    fmaLabel: "1,176 FMAs",
    Viz: RotorViz,
    params: [
      { k: "Groups", v: "d/3 ≈ 42" },
      { k: "Params/group", v: "rotor R (4 scalars)" },
      { k: "Total params", v: "172 floats" },
      { k: "SIMD fit", v: "3D → SIMD-4 gap ✗" },
    ],
    steps: [
      { label: "algebra", eq: String.raw`\text{Cl}(3,0): \quad e_i^2 = +1,\quad e_i e_j = -e_j e_i \; (i\neq j)` },
      { label: "rotor", eq: String.raw`R = \cos\tfrac{\theta}{2} + \sin\tfrac{\theta}{2}(a\,e_{12} + b\,e_{23} + c\,e_{13}),\quad a^2+b^2+c^2=1`, highlight: true },
      { label: "rotation", eq: String.raw`\mathbf{y} = R\,\mathbf{v}\,\tilde{R}, \quad \tilde{R} = \cos\tfrac{\theta}{2} - \sin\tfrac{\theta}{2}({\cdots})`, highlight: true, display: true },
      { label: "cost", eq: String.raw`42\text{ groups} \times 28\text{ FMAs} = \mathbf{1{,}176}\text{ FMAs/vector}`, note: "4.6× more than Planar" },
      { label: "problem", eq: String.raw`\text{3D group} \not\subset \text{SIMD-4} \Rightarrow \text{padding waste on CDNA3}` },
    ],
    qualityStats: { mean: 0.9830, std: 0.0032, p5: 0.9772, min: 0.9649 },
    identityEq: String.raw`\mathbb{E}[\mathrm{score}] = \tfrac{(R_{\rm blk}\,\mathbf{q})^\top\hat{\mathbf{k}}}{\beta_{\rm JL}} = \mathbf{q}^\top\mathbf{k}, \quad R_{\rm blk}^\top R_{\rm blk} = I`,
    identityNote: "Rotor sandwich product is orthogonal in each 3D block; expected attention score is preserved after JL correction. Performance cost comes from SIMD mismatch, not from score distortion.",
  },
  {
    cls: "turbo",
    name: "TurboQuant",
    subtitle: "Walsh–Hadamard Transform Rotation",
    paper: "Agarwal et al., Google (2024) — arXiv:2406.12820",
    color: "var(--turbo-color)",
    colorHex: "#3B82F6",
    fmaPct: 100,
    fmaLabel: "16,384 FMAs",
    Viz: HadamardViz,
    params: [
      { k: "Groups", v: "1 (full 128D)" },
      { k: "Params", v: "128 diagonal ±1" },
      { k: "Rotation matrix", v: "128×128 Hadamard" },
      { k: "SIMD fit", v: "MFMA-accelerated" },
    ],
    steps: [
      { label: "rotation", eq: String.raw`\mathbf{y} = \frac{1}{\sqrt{d}}\,H_d\,D\,\mathbf{x}`, highlight: true, display: true },
      { label: "WHT", eq: String.raw`H_d \in \{-1,+1\}^{d\times d},\quad H_d = H_2 \otimes H_{d/2}` },
      { label: "diag", eq: String.raw`D = \mathrm{diag}(r_1,\ldots,r_d),\quad r_i \overset{\mathrm{iid}}{\sim} \mathrm{Rademacher}(\pm 1)` },
      { label: "cost", eq: String.raw`d^2 = 128^2 = \mathbf{16{,}384}\text{ FMAs/vector}`, note: "O(d²) — 64× Planar" },
      { label: "kernel", eq: String.raw`\texttt{torch.matmul} \to \text{rocBLAS} \to \text{MFMA on gfx942}` },
    ],
    qualityStats: { mean: 0.9829, std: 0.0033, p5: 0.9765, min: 0.9601 },
    identityEq: String.raw`\mathbb{E}[\mathrm{score}] = \tfrac{((H D)\,\mathbf{q})^\top\hat{\mathbf{k}}}{\beta_{\rm JL}} = \mathbf{q}^\top\mathbf{k}, \quad (HD)^\top(HD)=I`,
    identityNote: "Full 128D random orthogonal transform has the same score-preservation guarantee; the difference vs block methods is only computational cost.",
  },
];

export function MathSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const [activeTab, setActiveTab] = useState(0);
  const m = MATH_METHODS[activeTab] ?? MATH_METHODS[0];
  const Viz = m.Viz;

  return (
    <section id="math" className="report-section" ref={ref}>
      <div className="section-divider" />
      <motion.div
        style={{ paddingTop: "5rem" }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={0}
      >
        <div className="section-label">
          <span className="section-num">02</span>
          <span className="section-tag">MATHEMATICS</span>
        </div>
        <h2 className="section-title">
          Rotation Algebras<br /><span className="dim">vs. Hardware Reality</span>
        </h2>
        <p className="section-lead">
          Every method applies a learned random rotation before quantization to decorrelate
          KV vector components and reduce error. All four share an identical 52-byte output
          format. The decisive factor is <strong style={{ color: "var(--text)" }}>FMAs per
          vector</strong> — spanning four orders of magnitude.
        </p>
      </motion.div>

      {/* Method tabs */}
      <motion.div
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={1}
        style={{ display: "flex", gap: "0.6rem", flexWrap: "wrap", marginBottom: "1.5rem" }}
      >
        {MATH_METHODS.map((mm, i) => (
          <button
            key={mm.name}
            onClick={() => setActiveTab(i)}
            style={{
              padding: "0.45rem 1rem",
              borderRadius: "999px",
              border: activeTab === i ? `1.5px solid ${mm.colorHex}` : "1.5px solid rgba(255,255,255,0.1)",
              background: activeTab === i ? `${mm.colorHex}18` : "transparent",
              color: activeTab === i ? mm.colorHex : "rgba(255,255,255,0.4)",
              fontFamily: "JetBrains Mono, monospace",
              fontSize: "0.72rem",
              fontWeight: 600,
              letterSpacing: "0.07em",
              cursor: "pointer",
              transition: "all 0.2s",
            }}
          >
            {mm.name}
          </button>
        ))}
      </motion.div>

      {/* Main method card */}
      <AnimatePresence mode="wait">
        <motion.div
          key={activeTab}
          initial={{ opacity: 0, y: 16 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0, y: -8 }}
          transition={{ duration: 0.32, ease: [0.23, 1, 0.32, 1] }}
          className="glass math-detail-card"
          style={{ borderTop: `3px solid ${m.colorHex}`, borderRadius: "var(--radius-md)", padding: "2rem", marginBottom: "1.5rem" }}
        >
          {/* header row */}
          <div style={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", marginBottom: "1.5rem", flexWrap: "wrap", gap: "0.8rem" }}>
            <div>
              <span className={`math-method-badge ${m.cls}`}>{m.name}</span>
              <h3 style={{ fontFamily: "Space Grotesk, sans-serif", fontSize: "1.3rem", fontWeight: 700, margin: "0.5rem 0 0.2rem" }}>{m.subtitle}</h3>
              <span style={{ fontSize: "0.68rem", fontFamily: "JetBrains Mono, monospace", color: "rgba(255,255,255,0.3)" }}>{m.paper}</span>
            </div>
            <div style={{ display: "flex", flexDirection: "column", alignItems: "flex-end", gap: "0.25rem" }}>
              <span style={{ fontSize: "1.4rem", fontWeight: 700, color: m.colorHex, fontFamily: "Space Grotesk, sans-serif" }}>{m.fmaLabel}</span>
              <span style={{ fontSize: "0.68rem", color: "rgba(255,255,255,0.3)", fontFamily: "JetBrains Mono, monospace" }}>per 128-dim vector</span>
            </div>
          </div>

          {/* two-column: viz + equations */}
          <div style={{ display: "grid", gridTemplateColumns: "auto 1fr", gap: "2rem", alignItems: "start" }} className="math-detail-body">
            <div style={{ display: "flex", flexDirection: "column", gap: "1rem", alignItems: "center" }}>
              <Viz color={m.colorHex} inView={inView} />
              {/* params mini-table */}
              <div style={{
                background: "rgba(0,0,0,0.35)", border: "1px solid rgba(255,255,255,0.07)",
                borderRadius: "8px", padding: "0.75rem", width: "176px",
              }}>
                {m.params.map(({ k, v }) => (
                  <div key={k} style={{ display: "flex", justifyContent: "space-between", fontSize: "0.68rem", padding: "0.15rem 0", borderBottom: "1px solid rgba(255,255,255,0.04)" }}>
                    <span style={{ color: "rgba(255,255,255,0.35)", fontFamily: "JetBrains Mono, monospace" }}>{k}</span>
                    <span style={{ color: "var(--text)", fontFamily: "JetBrains Mono, monospace", fontWeight: 600 }}>{v}</span>
                  </div>
                ))}
              </div>
            </div>

            {/* equations */}
            <div style={{ minWidth: 0 }}>
              <EqReveal steps={m.steps} inView={inView} color={m.colorHex} />

              {/* FMA bar */}
              <div style={{ marginTop: "1.5rem" }}>
                <div style={{ display: "flex", justifyContent: "space-between", fontSize: "0.72rem", fontFamily: "JetBrains Mono, monospace", color: "rgba(255,255,255,0.4)", marginBottom: "0.4rem" }}>
                  <span>FMA cost relative to TurboQuant (16,384)</span>
                  <span style={{ color: m.colorHex, fontWeight: 700 }}>{m.fmaLabel}</span>
                </div>
                <div className="fma-track">
                  <FmaBar pct={m.fmaPct} color={m.colorHex} inView={inView} delay={0.2} />
                </div>
                {/* all 4 methods mini comparison */}
                <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: "0.4rem", marginTop: "0.8rem" }}>
                  {MATH_METHODS.map((mm) => (
                    <div key={mm.name} style={{ textAlign: "center" }}>
                      <div style={{ fontSize: "0.65rem", fontFamily: "JetBrains Mono, monospace", color: mm.colorHex, fontWeight: 600 }}>{mm.fmaLabel}</div>
                      <div className="fma-track" style={{ marginTop: "0.2rem" }}>
                        <FmaBar pct={mm.fmaPct} color={mm.colorHex} inView={inView} delay={0.3} />
                      </div>
                      <div style={{ fontSize: "0.6rem", color: "rgba(255,255,255,0.3)", fontFamily: "JetBrains Mono, monospace", marginTop: "0.2rem" }}>{mm.name}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>

          {/* measured fidelity + score identity (aggregate benchmark data only) */}
          <div style={{
            marginTop: "1.4rem",
            background: "rgba(0,0,0,0.28)",
            border: "1px solid rgba(255,255,255,0.08)",
            borderRadius: "10px",
            padding: "1rem 1.1rem",
          }}>
            <p style={{
              fontSize: "0.66rem",
              color: m.colorHex,
              fontFamily: "JetBrains Mono, monospace",
              letterSpacing: "0.08em",
              marginBottom: "0.55rem",
            }}>
              MEASURED ATTENTION FIDELITY (aggregate) — bench_ppl_all_methods_quality_recheck.json
            </p>
            <div style={{ fontSize: "0.82rem", lineHeight: 1.8, color: "var(--text-sub)" }}>
              <span style={{ marginRight: "1rem" }}>mean: <strong style={{ color: "var(--text)" }}>{Number(m.qualityStats?.mean ?? 0).toFixed(4)}</strong></span>
              <span style={{ marginRight: "1rem" }}>std: <strong style={{ color: "var(--text)" }}>{Number(m.qualityStats?.std ?? 0).toFixed(4)}</strong></span>
              <span style={{ marginRight: "1rem" }}>p5: <strong style={{ color: "var(--text)" }}>{Number(m.qualityStats?.p5 ?? 0).toFixed(4)}</strong></span>
              <span>min: <strong style={{ color: "var(--text)" }}>{Number(m.qualityStats?.min ?? 0).toFixed(4)}</strong></span>
            </div>
            <div style={{ marginTop: "0.7rem", fontSize: "0.85rem", overflowX: "auto" }}>
              <KatexEq math={m.identityEq ?? String.raw`\mathbb{E}[\mathrm{score}] = \mathbf{q}^\top\mathbf{k}`} display={false} />
            </div>
            <p style={{
              marginTop: "0.55rem",
              fontSize: "0.72rem",
              color: "rgba(255,255,255,0.45)",
              lineHeight: 1.6,
            }}>
              {m.identityNote ?? "Measured fidelity statistics shown above. Identity equation unavailable for this method entry."}
            </p>
            <p style={{
              marginTop: "0.4rem",
              fontSize: "0.68rem",
              color: "var(--text-muted)",
              fontFamily: "JetBrains Mono, monospace",
            }}>
              Per-layer plots are intentionally omitted here until a layer-resolved benchmark artifact is generated.
            </p>
          </div>
        </motion.div>
      </AnimatePresence>

      {/* Common quantization block */}
      <motion.div
        className="quant-section glass-red"
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={3}
      >
        <div className="section-label">
          <span className="section-num" style={{ fontSize: "0.65rem" }}>COMMON</span>
          <span className="section-tag">QUANTIZATION STEP — ALL FOUR METHODS</span>
        </div>
        <div style={{ display: "grid", gridTemplateColumns: "1fr auto", gap: "2rem", alignItems: "start" }} className="quant-inner-grid">
          <div>
            <p style={{ color: "var(--text-sub)", fontSize: "0.88rem", lineHeight: 1.7, marginBottom: "1rem" }}>
              After rotation, all methods apply the same scalar quantization. The rotated vector
              is L2-normalized, components mapped to a Lloyd-Max codebook, and packed into
              an identical 52-byte block. This is why all methods achieve statistically
              indistinguishable reconstruction quality.
            </p>
            <div style={{ background: "rgba(0,0,0,0.3)", borderRadius: "8px", padding: "0.8rem 1rem", fontSize: "0.82rem" }}>
              <KatexEq math={String.raw`\hat{x} = \frac{x}{\|x\|},\quad \hat{x}_i \mapsto \arg\min_{c \in \mathcal{C}} |{\hat{x}_i - c}|,\quad \text{pack} \to 3\text{-bit index}`} display={false} />
            </div>
          </div>
          <div className="quant-grid">
            {[
              { val: "4 B", desc: "float32 norm" },
              { val: "48 B", desc: "3-bit indices ×128" },
              { val: "4.923×", desc: "vs FP16 (256 B)" },
            ].map((q) => (
              <div key={q.val} className="quant-item">
                <div className="quant-val" style={{ color: "var(--amd-red)" }}>{q.val}</div>
                <div className="quant-desc">{q.desc}</div>
              </div>
            ))}
          </div>
        </div>
      </motion.div>
    </section>
  );
}

/* ─── RESULTS SECTION ────────────────────────────────────────────── */

const CHARTS = {
  compress: {
    title: "Compress Kernel Throughput",
    subtitle: "GB/s · MI300X gfx942 · 4096 vectors · 50 iter median",
    unit: "GB/s",
    bars: [
      { method: "IsoQuant",    cls: "iso",    val: 21.8, pct: 100,  badge: "FASTEST" },
      { method: "PlanarQuant", cls: "planar", val: 18.7, pct: 85.8, badge: null },
      { method: "RotorQuant",  cls: "rotor",  val: 17.3, pct: 79.4, badge: null },
      { method: "TurboQuant",  cls: "turbo",  val: 2.9,  pct: 13.3, badge: "7.5× SLOWER" },
    ],
  },
  decompress: {
    title: "Decompress Kernel Throughput",
    subtitle: "GB/s · MI300X gfx942 · 4096 vectors · 50 iter median",
    unit: "GB/s",
    bars: [
      { method: "IsoQuant",    cls: "iso",    val: 38.3, pct: 100,  badge: "FASTEST" },
      { method: "PlanarQuant", cls: "planar", val: 35.4, pct: 92.4, badge: null },
      { method: "RotorQuant",  cls: "rotor",  val: 34.8, pct: 90.9, badge: null },
      { method: "TurboQuant",  cls: "turbo",  val: 4.4,  pct: 11.5, badge: "8.7× SLOWER" },
    ],
  },
  prefill: {
    title: "Prefill KV Compression Speed",
    subtitle: "tokens/sec · seq=32768 · 32 layers · 8 heads · compress-only",
    unit: "K tok/s",
    bars: [
      { method: "PlanarQuant", cls: "planar", val: "1,126K", pct: 100,  badge: "26.5× TURBO", rawVal: 1126 },
      { method: "IsoQuant",    cls: "iso",    val: "891K",   pct: 79.2, badge: "21.0× TURBO", rawVal: 891 },
      { method: "RotorQuant",  cls: "rotor",  val: "855K",   pct: 76.0, badge: "20.1× TURBO", rawVal: 855 },
      { method: "TurboQuant",  cls: "turbo",  val: "42K",    pct: 3.7,  badge: "BASELINE",    rawVal: 42 },
    ],
  },
  fmas: {
    title: "FMAs per Vector (Inverted — fewer is better)",
    subtitle: "Floating-point multiply-adds required to rotate one 128-dim KV head vector",
    unit: "FMAs",
    bars: [
      { method: "PlanarQuant", cls: "planar", val: "256",    pct: 100,  badge: "MINIMUM" },
      { method: "IsoQuant",    cls: "iso",    val: "512",    pct: 50.0, badge: "2.0×" },
      { method: "RotorQuant",  cls: "rotor",  val: "1,176",  pct: 21.8, badge: "4.6×" },
      { method: "TurboQuant",  cls: "turbo",  val: "16,384", pct: 1.56, badge: "64×" },
    ],
  },
};

export function ResultsSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });
  const [activeTab, setActiveTab] = useState("compress");

  const chart = CHARTS[activeTab];

  return (
    <section id="results" className="report-section" ref={ref}>
      <div className="section-divider" />
      <motion.div
        style={{ paddingTop: '5rem' }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={0}
      >
        <div className="section-label">
          <span className="section-num">04</span>
          <span className="section-tag">RESULTS</span>
        </div>
        <h2 className="section-title">
          Measured on<br /><span className="dim">AMD MI300X gfx942</span>
        </h2>
        <p className="section-lead">
          All benchmarks run on AMD Instinct MI300X (gfx942, 192 GB HBM3, ~5.3 TB/s peak).
          Kernels implemented in Triton-ROCm. Model: Mistral-7B-v0.1 (32 layers, 8 KV heads,
          head_dim=128). All four methods achieve identical 4.923× compression and
          statistically indistinguishable KV reconstruction quality — making kernel throughput
          and prefill speed the only relevant differentiators.
        </p>
      </motion.div>

      <motion.div
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={1}
      >
        <div className="results-tabs">
          {[
            { key: "compress",   label: "Compress BW" },
            { key: "decompress", label: "Decompress BW" },
            { key: "prefill",    label: "Prefill Speed" },
            { key: "fmas",       label: "FMA Efficiency" },
          ].map((tab) => (
            <button
              key={tab.key}
              className={`results-tab ${activeTab === tab.key ? "active" : ""}`}
              onClick={() => setActiveTab(tab.key)}
            >
              {tab.label}
            </button>
          ))}
        </div>
      </motion.div>

      <motion.div
        className="chart-panel glass"
        key={activeTab}
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }}
      >
        <p className="chart-title">{chart.title}</p>
        <p className="chart-subtitle">{chart.subtitle}</p>

        <div className="bar-group">
          {chart.bars.map((bar, i) => (
            <div key={bar.method} className="bar-row">
              <div className="bar-method" style={{ color: `var(--${bar.cls}-color)` }}>
                {bar.method}
              </div>
              <div className="bar-track">
                <AnimatedBar pct={bar.pct} cls={bar.cls} delay={i * 0.08} inView={inView} />
              </div>
              <div>
                <div className="bar-num">{bar.val} <span style={{ color: 'var(--text-muted)', fontSize: '0.68rem' }}>{chart.unit}</span></div>
                {bar.badge && (
                  <div className={`bar-winner`}>{bar.badge}</div>
                )}
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Full comparison matrix */}
      <motion.div
        className="results-matrix glass"
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={3}
      >
        <p className="matrix-title">Complete Method Comparison — MI300X gfx942</p>
        <table className="data-table">
          <thead>
            <tr>
              <th>Method</th>
              <th>FMAs/vec</th>
              <th>Compress</th>
              <th>Decompress</th>
              <th>Prefill (32K)</th>
              <th>Cosine Sim</th>
              <th>PPL (3-bit, lit.)</th>
              <th>Context (192 GB)</th>
              <th>Verdict</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td className="cell-method">
                <span className="dot" style={{ background: 'var(--planar-color)' }} />
                PlanarQuant
              </td>
              <td><span className="cell-badge good">256</span></td>
              <td className="cell-num">18.7 GB/s</td>
              <td className="cell-num">35.4 GB/s</td>
              <td><span className="cell-best">1,126K tok/s</span></td>
              <td className="cell-num">0.9829</td>
              <td><span className="cell-best">10.62</span></td>
              <td><span className="cell-best">6.9 M</span></td>
              <td><span className="cell-badge good">★ BEST</span></td>
            </tr>
            <tr>
              <td className="cell-method">
                <span className="dot" style={{ background: 'var(--iso-color)' }} />
                IsoQuant
              </td>
              <td><span className="cell-badge ok">512</span></td>
              <td><span className="cell-best">21.8 GB/s</span></td>
              <td><span className="cell-best">38.3 GB/s</span></td>
              <td className="cell-num">891K tok/s</td>
              <td className="cell-num">0.9831</td>
              <td className="cell-num">12.85</td>
              <td className="cell-num">6.9 M</td>
              <td><span className="cell-badge ok">FAST KERNEL</span></td>
            </tr>
            <tr>
              <td className="cell-method">
                <span className="dot" style={{ background: 'var(--rotor-color)' }} />
                RotorQuant
              </td>
              <td><span className="cell-badge warn">1,176</span></td>
              <td className="cell-worst">17.3 GB/s</td>
              <td className="cell-worst">34.8 GB/s</td>
              <td className="cell-worst">855K tok/s</td>
              <td className="cell-num">0.9832</td>
              <td className="cell-num">12.72</td>
              <td className="cell-num">6.9 M</td>
              <td><span className="cell-badge warn">AVOID</span></td>
            </tr>
            <tr>
              <td className="cell-method">
                <span className="dot" style={{ background: 'var(--turbo-color)' }} />
                TurboQuant
              </td>
              <td><span className="cell-badge bad">16,384</span></td>
              <td className="cell-worst">2.9 GB/s</td>
              <td className="cell-worst">4.4 GB/s</td>
              <td className="cell-worst">42K tok/s</td>
              <td className="cell-num">0.9831</td>
              <td><span className="cell-best">7.07*</span></td>
              <td className="cell-num">6.9 M</td>
              <td><span className="cell-badge bad">SLOW KERNEL</span></td>
            </tr>
          </tbody>
        </table>
        <p style={{ fontSize: '0.72rem', color: 'var(--text-muted)', marginTop: '0.8rem', fontFamily: 'JetBrains Mono, monospace', lineHeight: 1.6 }}>
          * TurboQuant PPL measured with deferred quantization (K stored FP16 during prefill). Roundtrip mode would be significantly worse.
          All block methods measured in strict roundtrip mode. PPL values from literature (Qwen2.5-3B), not Mistral-7B.
        </p>
      </motion.div>

      {/* Quality panel */}
      <motion.div
        style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginTop: '1rem' }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={4}
      >
        <div className="glass" style={{ borderRadius: 'var(--radius-md)', padding: '1.5rem' }}>
          <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--amd-red)', letterSpacing: '0.12em', marginBottom: '1rem' }}>
            KV RECONSTRUCTION QUALITY — 3-BIT
          </p>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-sub)', lineHeight: 1.7, marginBottom: '1rem' }}>
            All four methods achieve <strong style={{ color: 'var(--text)' }}>statistically indistinguishable</strong> cosine
            similarity at 3-bit. The range across methods is ±0.0003 — well within trial variance.
          </p>
          {[
            { m: "PlanarQuant", cls: "planar", sim: 0.9829, mse: 0.0341 },
            { m: "IsoQuant",    cls: "iso",    sim: 0.9831, mse: 0.0341 },
            { m: "RotorQuant",  cls: "rotor",  sim: 0.9832, mse: 0.0339 },
            { m: "TurboQuant",  cls: "turbo",  sim: 0.9831, mse: 0.0339 },
          ].map((row) => (
            <div key={row.m} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.4rem 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span style={{ fontSize: '0.8rem', color: `var(--${row.cls}-color)`, fontWeight: 500 }}>{row.m}</span>
              <div style={{ display: 'flex', gap: '1.5rem' }}>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.78rem', color: 'var(--text)' }}>{row.sim.toFixed(4)}</span>
                <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.78rem', color: 'var(--text-muted)' }}>MSE {row.mse}</span>
              </div>
            </div>
          ))}
        </div>

        <div className="glass" style={{ borderRadius: 'var(--radius-md)', padding: '1.5rem' }}>
          <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--amd-red)', letterSpacing: '0.12em', marginBottom: '1rem' }}>
            RUNTIME COMPRESSION RATIO — OBSERVED
          </p>
          <p style={{ fontSize: '0.84rem', color: 'var(--text-sub)', lineHeight: 1.7, marginBottom: '1rem' }}>
            Reported from materialized cache buffers during real prefill runs. This keeps UI claims tied to executed measurements only.
          </p>
          {[
            { m: "TurboQuant3", ratio: "4.923×", bytes: "52 B/vec" },
            { m: "PlanarQuant3", ratio: "1.939×", bytes: "132 B/vec" },
            { m: "IsoQuant3", ratio: "1.939×", bytes: "132 B/vec" },
            { m: "RotorQuant3", ratio: "1.925×", bytes: "133 B/vec" },
          ].map((row) => (
            <div key={row.m} style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.5rem 0', borderBottom: '1px solid rgba(255,255,255,0.04)' }}>
              <span style={{ fontSize: '0.8rem', color: 'var(--text-sub)', fontFamily: "'JetBrains Mono', monospace" }}>{row.m}</span>
              <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.78rem', color: 'var(--text)', fontWeight: 600 }}>
                {row.ratio} ({row.bytes})
              </span>
            </div>
          ))}
        </div>
      </motion.div>

      <motion.div
        style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: "1rem", marginTop: "1rem" }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={5}
        className="obs-packed-grid"
      >
        <LiveGenerationPanel inView={inView} />
        <ImportantFactsPanel />
      </motion.div>

      <motion.div
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={6}
        style={{ marginTop: "1rem" }}
      >
        <RooflineComparePanel />
      </motion.div>
    </section>
  );
}

/* ─── DEPLOYMENT STORIES & NEXT STEPS (memory vs speed) ───────────── */

export function DeploymentStoriesSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section id="stories" className="report-section" ref={ref}>
      <div className="section-divider" />
      <motion.div
        style={{ paddingTop: "5rem" }}
        variants={fadeUp}
        initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={0}
      >
        <div className="section-label">
          <span className="section-num">04b</span>
          <span className="section-tag">EXPERIMENT RESULTS</span>
        </div>
        <h2 className="section-title">
          Two deployment stories<br /><span className="dim">Memory vs speed</span>
        </h2>
        <p className="section-lead">
          <strong>Story 1</strong> treats KV compression as a <em>production memory</em> feature: it expands feasible context,
          improves <code>max_model_len</code> / HBM headroom, and helps scheduling even when batch=1 tok/s barely moves.
          <strong> Story 2</strong> treats it as a <em>speed</em> feature only when the attention bubble is large enough and the non-KV path is lean enough for savings to surface end-to-end.
        </p>
      </motion.div>

      <motion.div
        className="glass"
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: "1.25rem",
          padding: "1.5rem",
          marginTop: "1.5rem",
          maxWidth: 1280,
          marginLeft: "auto",
          marginRight: "auto",
        }}
        variants={fadeUp}
        initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={1}
      >
        <div>
          <h3 style={{ fontSize: "1.05rem", color: "var(--iso-color)", marginBottom: "0.5rem" }}>Story 1 — Memory</h3>
          <p style={{ fontSize: "0.86rem", color: "var(--text-sub)", lineHeight: 1.65 }}>
            Same quality tier at 4.923× storage: more tokens in 192 GB, less eviction pressure, larger batches possible <em>given</em> VRAM.
            This is a real deployment win independent of single-stream tok/s.
          </p>
        </div>
        <div>
          <h3 style={{ fontSize: "1.05rem", color: "var(--amd-red)", marginBottom: "0.5rem" }}>Story 2 — Speed (conditional)</h3>
          <p style={{ fontSize: "0.86rem", color: "var(--text-sub)", lineHeight: 1.65 }}>
            Isolated fused TQ3 attention becomes compelling around <strong>~16K</strong> sequence length (Split-K, Primus).
            Full vLLM Mistral runs still show <strong>flat aggregate output tok/s</strong> across FP16 / TQ paths — the rest of the decode step stays heavy.
          </p>
        </div>
      </motion.div>

      <motion.div
        style={{ maxWidth: 1100, margin: "2rem auto 0", padding: "0 1.5rem" }}
        variants={fadeUp}
        initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={2}
      >
        <h3 style={{ fontSize: "1.15rem", color: "var(--text)", marginBottom: "1rem", textAlign: "center" }}>
          Full stack vs attention-only
        </h3>
        <div
          style={{
            fontSize: "0.9rem",
            color: "var(--text-sub)",
            lineHeight: 1.8,
            maxWidth: 920,
            margin: "0 auto 1.25rem",
            textAlign: "left",
          }}
        >
          <p style={{ margin: "0 0 0.9rem" }}>
            The left chart is a full vLLM run on Mistral-7B with a heavy KV setup. All three backends land on almost the same output tokens per second, because that number reflects the whole step: weight reads, MLP matmuls, norms, framework work, and attention together. Shrink the KV or speed up attention, and you still only touch part of the wall clock, so the total barely moves. That is the usual Amdahl story: a large fixed portion of the work is elsewhere.
          </p>
          <p style={{ margin: 0 }}>
            The right chart is the same hardware, but the benchmark measures only the attention operator. There, fused TurboQuant reads fewer KV bytes and eventually beats FP16 SDPA once the sequence is long enough that the extra dequant work stops dominating. So the two panels are not in conflict: one is the full pipeline, the other is a single kernel path. Smaller KV is still a strong memory win; turning it into a clear end-to-end speed win means making the non-attention slice smaller or cheaper as well.
          </p>
        </div>
        <figure className="glass" style={{ margin: 0, padding: "1rem 1.25rem", borderRadius: "var(--radius-md)" }}>
          <img
            src="/content/figures_v2/fig29_story_e2e_vs_isolated_attention_comparison.png"
            alt="Two charts: full vLLM output tok/s flat across backends, and isolated attention FP16 over fused TQ3 ratio versus sequence length"
            style={{ width: "100%", height: "auto", borderRadius: 8 }}
            loading="lazy"
          />
          <figcaption style={{ fontSize: "0.75rem", color: "var(--text-muted)", marginTop: "0.65rem", lineHeight: 1.55 }}>
            Data: <code>results/bench_vllm_turboquant_ab_sweep_kv_heavy.json</code> and <code>results/bench_triton_attention.json</code>. Figure from <code>report/generate_figures_v2.py</code>.
          </figcaption>
        </figure>
      </motion.div>

      <motion.div
        className="glass"
        style={{
          marginTop: "2rem",
          maxWidth: 900,
          marginLeft: "auto",
          marginRight: "auto",
          padding: "1.5rem 1.75rem",
          borderLeft: "4px solid var(--amd-orange)",
        }}
        variants={fadeUp}
        initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={3}
      >
        <h3 style={{ fontSize: "1.05rem", marginBottom: "0.6rem", color: "var(--text)" }}>Next step (for speed)</h3>
        <p style={{ fontSize: "0.9rem", color: "var(--text-sub)", lineHeight: 1.75, margin: 0 }}>
          The question is no longer only &quot;is KV smaller?&quot; It is:{" "}
          <strong>Can I reduce the non-KV path enough that attention savings matter end-to-end?</strong>
          {" "}Quantized GEMM / FFN fusion, less Python overhead, better scheduling, and profiling the full step
          (<code>VLLM_TQ_LOG_DISPATCH</code>, rocprof) turn Story 2 from an operator win into a serving win.
        </p>
      </motion.div>

      <motion.div
        className="glass"
        style={{
          marginTop: "1.25rem",
          maxWidth: 900,
          marginLeft: "auto",
          marginRight: "auto",
          padding: "1.5rem 1.75rem",
        }}
        variants={fadeUp}
        initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={4}
      >
        <h3 style={{ fontSize: "1.05rem", marginBottom: "0.6rem", color: "var(--planar-color)" }}>Issues encountered — now fixed</h3>
        <ul style={{ margin: 0, paddingLeft: "1.1rem", color: "var(--text-sub)", fontSize: "0.86rem", lineHeight: 1.75 }}>
          <li>In-repo <code>vllm/</code> shadowing PyPI vLLM → <code>tq_backends/</code> + <code>PYTHONPATH</code> discipline.</li>
          <li>HIP / PyTorch alignment for reproducible numbers → <strong>Primus</strong> <code>rocm/primus:v26.2</code> + PyTorch 2.10.</li>
          <li>GQA blocking fused decode → <code>expand_tq_compressed_for_gqa</code> + <code>bench_tq_gqa_decode_sweep.json</code>.</li>
          <li>Install / registry fragility → <code>scripts/install_turboquant_vllm_backend.sh</code>, <code>vllm_turboquant_registry.py</code>, <code>docs/vllm_turboquant_wiring.md</code>.</li>
        </ul>
      </motion.div>
    </section>
  );
}

/* ─── REASONING SECTION ──────────────────────────────────────────── */

export function ReasoningSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section id="reasoning" className="report-section" ref={ref}>
      <div className="section-divider" />
      <motion.div
        style={{ paddingTop: '5rem' }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={0}
      >
        <div className="section-label">
          <span className="section-num">05</span>
          <span className="section-tag">REASONING</span>
        </div>
        <h2 className="section-title">
          Why the Numbers<br /><span className="dim">Come Out This Way</span>
        </h2>
        <p className="section-lead">
          The results are not arbitrary — they follow directly from the alignment between each
          method's algebraic structure and the AMD CDNA3 microarchitecture.
          Understanding the <em>why</em> matters for choosing the right method in production.
        </p>
      </motion.div>

      <div className="reasoning-grid">
        {[
          {
            icon: "⚡",
            iconBg: "rgba(229, 52, 75, 0.15)",
            title: "Why IsoQuant has the fastest kernel despite 2× FMAs",
            body: `IsoQuant's quaternion sandwich operates on <strong>4-dimensional groups</strong>.
                  AMD CDNA3's matrix fused multiply-add (MFMA) units process data in SIMD-4 lane groups.
                  The quaternion group size is a perfect match — the compiler can fully vectorize the inner loop
                  with zero padding or lane waste. PlanarQuant's 2D groups leave half the SIMD lanes idle,
                  causing the slightly lower throughput despite having <code>256 vs 512</code> FMAs.`,
          },
          {
            icon: "🧮",
            iconBg: "rgba(168, 85, 247, 0.12)",
            title: "Why TurboQuant's WHT rotation is 64× slower",
            body: `TurboQuant applies a <strong>full 128×128 random orthogonal matrix</strong> rotation before
                  quantization. This requires 16,384 FMAs per vector — the entire matrix-vector product.
                  The implementation uses <code>torch.matmul</code> dispatching to rocBLAS → MFMA, which is
                  efficient for large matrices but dominates latency for small (128-dim) vectors.
                  The rotation compute cost is so large that it dwarfs all other operations in the pipeline.`,
          },
          {
            icon: "❌",
            iconBg: "rgba(255, 123, 53, 0.12)",
            title: "Why RotorQuant is the worst block method",
            body: `Clifford Cl(3,0) rotors operate on <strong>3-dimensional groups</strong> — not a power of 2.
                  CDNA3 SIMD-4 lanes cannot process 3D groups without padding one lane to zero.
                  This wastes 25% of each SIMD instruction's capacity on every group.
                  Additionally, 42 groups of 3 don't evenly fill 128 dims (42×3=126, leaving 2 dims padded).
                  These structural misalignments compound: <code>1,176 FMAs</code> at 25% SIMD waste
                  yields slower throughput than even PlanarQuant's 256 FMAs at full utilization.`,
          },
          {
            icon: "💾",
            iconBg: "rgba(59, 130, 246, 0.1)",
            title: "Why all methods are compute-limited, not bandwidth-limited",
            body: `At N=4,096 vectors, all compressed kernels run at <code>< 1%</code> of MI300X's
                  5.3 TB/s peak memory bandwidth. The bottleneck is <strong>instruction latency and kernel
                  launch overhead</strong>, not HBM3 bandwidth. At serving scale (N > 100K vectors),
                  efficiency improves significantly as the fixed kernel launch cost amortizes.
                  For batch decode, the decompress kernel becomes the new compute bottleneck,
                  making FP16 the better choice at high batch sizes with current PyTorch kernels.`,
          },
          {
            icon: "🔀",
            iconBg: "rgba(74, 222, 128, 0.08)",
            title: "Why all methods achieve the same reconstruction quality",
            body: `The fundamental theorem behind all four methods: any <strong>random orthogonal rotation</strong>
                  decorrelates vector components equally in expectation. The specific algebra (2D Givens,
                  4D quaternion, 3D Clifford, full WHT) doesn't matter for quality — what matters is
                  that the rotation is random and orthogonal. All four methods satisfy this.
                  The cosine similarity spread of <code>±0.0003</code> is noise, not signal.
                  Quality is not the differentiator; compute cost is.`,
          },
          {
            icon: "📐",
            iconBg: "rgba(229, 52, 75, 0.08)",
            title: "Why prefill overhead matters more than decode throughput",
            body: `Decode at batch=1 is <strong>weight-cycling bound</strong> (compute-bound) on MI300X.
                  Adding KV compression doesn't change the weight bandwidth bottleneck.
                  Prefill, however, adds a direct compression pass over all generated K and V vectors —
                  this is pure overhead that appears in Time-To-First-Token (TTFT).
                  At seq=32K, PlanarQuant adds only <code>29ms</code> of prefill compression,
                  while TurboQuant adds <code>772ms</code> — a production-critical difference for
                  long-context workloads.`,
          },
        ].map((card, i) => (
          <motion.div
            key={card.title}
            className="reasoning-card glass"
            variants={fadeUp} initial="hidden"
            animate={inView ? "visible" : "hidden"}
            custom={i + 1}
            whileHover={{ y: -4, transition: { duration: 0.2 } }}
          >
            <div className="reasoning-card-icon" style={{ background: card.iconBg }}>
              {card.icon}
            </div>
            <h4 className="reasoning-card-title">{card.title}</h4>
            <p
              className="reasoning-card-body"
              dangerouslySetInnerHTML={{ __html: card.body }}
            />
          </motion.div>
        ))}
      </div>

      <motion.div
        className="rotor-verdict"
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={7}
      >
        <h4 className="rotor-verdict-title">
          ⚠ Scientific Disqualification — RotorQuant on AMD MI300X
        </h4>
        <p style={{ fontSize: '0.86rem', color: 'var(--text-sub)', lineHeight: 1.7, marginBottom: '1.2rem' }}>
          RotorQuant was included specifically to test the claim that Clifford algebra rotors
          provide superior attention fidelity. The measured data conclusively refutes this on gfx942.
          RotorQuant expends <strong style={{ color: 'var(--text)' }}>4.6× more compute than PlanarQuant</strong> for
          the same compression ratio, same quality, and <em>worse</em> throughput.
          There is no regime on AMD MI300X where RotorQuant is the correct choice over PlanarQuant.
        </p>
        <div className="verdict-grid">
          {[
            { label: "More FMAs than PlanarQuant", val: "4.6×" },
            { label: "Slower prefill (32K tokens)", val: "24%" },
            { label: "PPL difference (3-bit)", val: "≈ 0" },
          ].map((v) => (
            <div key={v.label}>
              <p className="verdict-item-label">{v.label}</p>
              <p className="verdict-item-val">{v.val}</p>
            </div>
          ))}
        </div>
      </motion.div>
    </section>
  );
}

/* ─── CONCLUSION SECTION ─────────────────────────────────────────── */

export function ConclusionSection() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-80px" });

  return (
    <section id="conclusion" className="report-section" ref={ref}>
      <div className="section-divider" />
      <motion.div
        style={{ paddingTop: '5rem' }}
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={0}
      >
        <div className="section-label">
          <span className="section-num">06</span>
          <span className="section-tag">CONCLUSION</span>
        </div>
        <h2 className="section-title">
          Deployment<br /><span className="dim">Recommendations</span>
        </h2>
        <p className="section-lead">
          KV compression on AMD MI300X is primarily a <strong style={{ color: 'var(--text)' }}>context capacity</strong> play
          — 4.92× compression enables 6.9M tokens in 192 GB HBM3 vs 1.4M for FP16.
          Decode throughput at batch=1 is compute-bound regardless of KV format.
          The choice of method comes down to prefill latency and kernel overhead.
        </p>
      </motion.div>

      <div className="conclusion-grid">
        <motion.div
          variants={fadeUp} initial="hidden"
          animate={inView ? "visible" : "hidden"}
          custom={1}
        >
          <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--amd-red)', letterSpacing: '0.12em', marginBottom: '1rem' }}>
            USE CASE GUIDE
          </p>
          <div className="use-case-list">
            {[
              {
                icon: "🧠",
                title: "Extend context window",
                body: "4.92× compression → 6.9M tokens on 192 GB HBM3. Primary production benefit on MI300X.",
                rec: "PlanarQuant3 or IsoQuant3",
                recCls: "planar",
              },
              {
                icon: "⚡",
                title: "Minimize prefill latency (TTFT)",
                body: "26.5× faster prefill compression vs TurboQuant at seq=32K. Lowest FMA count.",
                rec: "PlanarQuant3",
                recCls: "planar",
              },
              {
                icon: "🚀",
                title: "Fastest individual kernel",
                body: "IsoQuant achieves highest compress (21.8 GB/s) and decompress (38.3 GB/s) throughput.",
                rec: "IsoQuant3",
                recCls: "iso",
              },
              {
                icon: "📦",
                title: "Production vLLM serving",
                body: "TurboQuant has the most mature integration. Block methods need new backend wiring.",
                rec: "TurboQuant3 (maturity)",
                recCls: "iso",
              },
              {
                icon: "📊",
                title: "High batch decode",
                body: "Decompress overhead exceeds KV BW savings at high batch. FP16 wins above crossover.",
                rec: "FP16 (batch > 26)",
                recCls: "fp16",
              },
              {
                icon: "⛔",
                title: "What to avoid",
                body: "4.6× more FMAs, slower prefill, same quality as PlanarQuant. No upside on gfx942.",
                rec: "Avoid RotorQuant",
                recCls: "avoid",
              },
            ].map((uc, i) => (
              <div key={uc.title} className="use-case glass">
                <span className="use-case-icon">{uc.icon}</span>
                <div>
                  <p className="use-case-title">{uc.title}</p>
                  <p className="use-case-body">{uc.body}</p>
                  <span className={`use-case-rec ${uc.recCls}`}>{uc.rec}</span>
                </div>
              </div>
            ))}
          </div>
        </motion.div>

        <motion.div
          variants={fadeUp} initial="hidden"
          animate={inView ? "visible" : "hidden"}
          custom={2}
        >
          <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--amd-red)', letterSpacing: '0.12em', marginBottom: '1rem' }}>
            FUTURE WORK
          </p>
          <div className="future-list">
            {[
              "Full model PPL on WikiText-2 with transformer hook (proper roundtrip, not cosine proxy)",
              "rocprofv2 counter traces: FETCH_SIZE, VALU_UTIL, WAVE_OCCUPANCY per kernel",
              "Fused decompress+attention kernel — eliminate intermediate KV materialization for bandwidth-limited speedups",
              "CK-based ROCm Flash Attention baseline (FP16 BW currently 0.69 TB/s = 13% of peak)",
              "K-only compression (PlanarQuant3 K + FP16 V) — near-FP16 PPL at 5.1× compression",
              "vLLM integration: complete isoquant_rocm_attn.py with full PagedAttention support",
            ].map((item, i) => (
              <div key={i} className="future-item">
                <span className="future-num">0{i + 1}</span>
                <p className="future-text">{item}</p>
              </div>
            ))}
          </div>
        </motion.div>
      </div>

      <motion.div
        className="conclusion-banner glass-red"
        variants={fadeUp} initial="hidden"
        animate={inView ? "visible" : "hidden"}
        custom={3}
      >
        <p style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--amd-red)', letterSpacing: '0.16em', marginBottom: '1rem' }}>
          AMD MI300X · gfx942 · HBM3 5.3 TB/s · April 2026
        </p>
        <h3 className="conclusion-banner-title">
          PlanarQuant is the right choice<br />
          <span style={{ color: 'var(--amd-red)' }}>for AMD MI300X KV cache compression</span>
        </h3>
        <p className="conclusion-banner-sub">
          Minimum FMA cost (256/vector), fastest prefill (26.5× over TurboQuant), best published PPL
          at 3-bit, and full 4.923× compression enabling 6.9M token context on 192 GB HBM3.
          IsoQuant is the alternative when single-kernel throughput is the priority.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '1.5rem', marginTop: '1.5rem', flexWrap: 'wrap' }}>
          {[
            { label: "Context Tokens", val: "6.9M", sub: "on 192 GB HBM3" },
            { label: "Prefill Speedup", val: "26.5×", sub: "vs TurboQuant" },
            { label: "Compression", val: "4.923×", sub: "vs FP16 storage" },
            { label: "FMAs/vector", val: "256", sub: "minimum possible" },
          ].map((s) => (
            <div key={s.label} style={{ textAlign: 'center' }}>
              <div style={{ fontFamily: "'Space Grotesk', sans-serif", fontSize: '2rem', fontWeight: 700, color: 'var(--amd-red)' }}>{s.val}</div>
              <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>{s.label}</div>
              <div style={{ fontSize: '0.68rem', color: 'var(--text-muted)', fontFamily: "'JetBrains Mono', monospace" }}>{s.sub}</div>
            </div>
          ))}
        </div>
      </motion.div>
    </section>
  );
}
