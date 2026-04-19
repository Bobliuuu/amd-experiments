import { useRef } from "react";
import { motion, useScroll, useTransform, useInView } from "framer-motion";
import NavBar from "./components/NavBar";
import {
  ProblemSection,
  MathSection,
  CompressionLandscapeSection,
  ResultsSection,
  DeploymentStoriesSection,
  ReasoningSection,
  ConclusionSection,
} from "./components/ReportSections";
import FigureGallery from "./components/FigureGallery";
import { getFigureUrlsAll } from "./dataSources";

function Hero() {
  const { scrollY } = useScroll();
  const sceneY = useTransform(scrollY, [0, 500], [0, -100]);
  const heroOpacity = useTransform(scrollY, [0, 400], [1, 0]);

  const stats = [
    { val: "4.923×", label: "KV Compression",    color: "var(--amd-red)" },
    { val: "6.9M",   label: "Max Tokens (192 GB)", color: "var(--iso-color)" },
    { val: "26.5×",  label: "Prefill Speedup",    color: "var(--amd-orange)" },
    { val: "4",      label: "Methods Benchmarked", color: "var(--text-sub)" },
  ];

  return (
    <section id="hero" className="hero-shell">
      <div className="hero-orb hero-orb-a" />
      <div className="hero-orb hero-orb-b" />
      <div className="hero-orb hero-orb-c" />

      <motion.div style={{ opacity: heroOpacity, textAlign: "center", display: "flex", flexDirection: "column", alignItems: "center" }}>
        <motion.p className="hero-eyebrow"
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}>
          AMD Instinct MI300X · gfx942 · April 2026
        </motion.p>

        <motion.h1 className="hero-title"
          initial={{ opacity: 0, y: 30 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.7, delay: 0.2 }}>
          KV Cache Compression<br />
          <span className="highlight">on AMD MI300X</span>
        </motion.h1>

        <motion.p className="hero-sub"
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.35 }}>
          A full four-way benchmark of TurboQuant, IsoQuant, PlanarQuant, and RotorQuant —
          Triton-ROCm kernels on Mistral-7B-v0.1. Compress/decompress bandwidth,
          prefill overhead, batch decode, and KV reconstruction quality.
        </motion.p>

        <motion.div className="hero-stat-row"
          initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}>
          {stats.map((s, i) => (
            <motion.div key={s.label} className="hero-stat glass"
              whileHover={{ y: -4, transition: { duration: 0.18 } }}
              initial={{ opacity: 0, y: 20 }} animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5, delay: 0.5 + i * 0.08 }}>
              <div className="hero-stat-val" style={{ color: s.color }}>{s.val}</div>
              <div className="hero-stat-label">{s.label}</div>
            </motion.div>
          ))}
        </motion.div>

        <motion.div className="hero-cta-row"
          initial={{ opacity: 0, y: 16 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5, delay: 0.75 }}>
          <a className="cta-primary" href="#problem">Read the Report</a>
          <a className="cta-ghost" href="#results">Jump to Results</a>
          <a className="cta-ghost" href="#stories">Memory vs speed</a>
          <a className="cta-ghost" href="#figures">View Figures</a>
        </motion.div>

        <motion.div className="scene-dock" style={{ y: sceneY }}
          initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.9, delay: 0.6, ease: [0.23, 1, 0.32, 1] }}>
          <motion.div className="scene-card scene-main"
            animate={{ y: [-5, 8, -5], rotateY: [-2, 3, -2] }}
            transition={{ repeat: Infinity, duration: 7, ease: "easeInOut" }}>
            <img src="/content/figures_v2/fig15_compress_decompress_bw.png"
              alt="Compress/Decompress Bandwidth"
              onError={(e) => { e.target.style.display = "none"; }} />
          </motion.div>
          <motion.div className="scene-card scene-left"
            animate={{ y: [10, -8, 10], rotateY: [10, -6, 10] }}
            transition={{ repeat: Infinity, duration: 8.2, ease: "easeInOut" }}>
            <img src="/content/figures_v2/fig14_prefill_comparison.png"
              alt="Prefill comparison"
              onError={(e) => { e.target.style.display = "none"; }} />
          </motion.div>
          <motion.div className="scene-card scene-right"
            animate={{ y: [0, 12, 0], rotateY: [-10, 6, -10] }}
            transition={{ repeat: Infinity, duration: 7.6, ease: "easeInOut" }}>
            <img src="/content/figures_v2/fig12_decode_all_methods.png"
              alt="Decode all methods"
              onError={(e) => { e.target.style.display = "none"; }} />
          </motion.div>
        </motion.div>

        <motion.a className="scroll-cue" href="#problem"
          animate={{ opacity: [0.5, 1, 0.5], y: [0, 6, 0] }}
          transition={{ repeat: Infinity, duration: 2.2, delay: 1.4 }}>
          <span className="scroll-cue-arrow">↓</span>
          SCROLL TO REPORT
        </motion.a>
      </motion.div>
    </section>
  );
}

function MethodLegend() {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-40px" });

  const methods = [
    { name: "PlanarQuant", color: "var(--planar-color)", fma: "256 FMAs", verdict: "★ RECOMMENDED" },
    { name: "IsoQuant",    color: "var(--iso-color)",    fma: "512 FMAs", verdict: "Fastest Kernel" },
    { name: "RotorQuant",  color: "var(--rotor-color)",  fma: "1,176 FMAs", verdict: "Avoid on gfx942" },
    { name: "TurboQuant",  color: "var(--turbo-color)",  fma: "16,384 FMAs", verdict: "Slow Kernel" },
  ];

  return (
    <div ref={ref} style={{
      display: "grid", gridTemplateColumns: "repeat(4,1fr)", gap: "1rem",
      maxWidth: 1280, margin: "0 auto", padding: "0 1.5rem",
    }}>
      {methods.map((m, i) => (
        <motion.div key={m.name} className="glass"
          style={{ borderRadius: "var(--radius-md)", padding: "1rem 1.2rem", borderTop: `3px solid ${m.color}` }}
          initial={{ opacity: 0, y: 20 }}
          animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
          transition={{ duration: 0.5, delay: i * 0.07, ease: [0.23, 1, 0.32, 1] }}
          whileHover={{ y: -4, transition: { duration: 0.2 } }}>
          <p style={{ fontSize: "0.88rem", fontWeight: 600, color: m.color, marginBottom: "0.25rem" }}>{m.name}</p>
          <p style={{ fontSize: "0.72rem", color: "var(--text-muted)", fontFamily: "'JetBrains Mono',monospace", marginBottom: "0.35rem" }}>{m.fma}</p>
          <p style={{ fontSize: "0.72rem", color: "var(--text-sub)" }}>{m.verdict}</p>
        </motion.div>
      ))}
    </div>
  );
}

export default function App() {
  const figures = getFigureUrlsAll();

  return (
    <div className="app-root">
      <div className="bg-layer bg-gradient" />
      <div className="bg-layer bg-grid" />
      <div className="bg-layer bg-scanlines" />

      <div className="content-wrap">
        <NavBar />
        <Hero />

        <div style={{ paddingBottom: "1.5rem" }}>
          <MethodLegend />
        </div>

        <div className="page-wrap">
          <ProblemSection />
          <MathSection />
          <CompressionLandscapeSection />
          <ResultsSection />
          <DeploymentStoriesSection />
          <ReasoningSection />
          <ConclusionSection />
        </div>

        <div id="figures" style={{ maxWidth: 1280, margin: "0 auto", padding: "0 1.5rem" }}>
          <div className="section-divider" style={{ marginBottom: "4rem" }} />
          <FigureGallery figures={figures} />
        </div>

        <footer style={{
          textAlign: "center", padding: "3rem 1.5rem",
          borderTop: "1px solid rgba(229,52,75,0.08)",
          color: "var(--text-muted)", fontSize: "0.76rem",
          fontFamily: "'JetBrains Mono',monospace", letterSpacing: "0.06em",
        }}>
          <p>AMD MI300X · gfx942 · 192 GB HBM3 · Primus ROCm 7.2 + PyTorch 2.10 · Mistral-7B-v0.1</p>
          <p style={{ marginTop: "0.4rem" }}>
            TurboQuant · IsoQuant · PlanarQuant · RotorQuant — KV cache compression — April 2026 · vLLM TQ wiring + GQA fused decode; e2e tok/s still step-limited (see reports)
          </p>
        </footer>
      </div>
    </div>
  );
}
