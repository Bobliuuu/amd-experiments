import { motion, useScroll, useTransform } from "framer-motion";
import NavBar from "./components/NavBar";
import ReportView from "./components/ReportView";

function LandingSection() {
  const { scrollY } = useScroll();
  const rotateX = useTransform(scrollY, [0, 500], [0, 12]);
  const rotateY = useTransform(scrollY, [0, 500], [0, -10]);
  const floatY = useTransform(scrollY, [0, 400], [0, 60]);
  const sceneDepth = useTransform(scrollY, [0, 700], [0, -120]);
  const sceneSpin = useTransform(scrollY, [0, 700], [0, 18]);

  return (
    <section id="landing" className="landing-shell">
      <div className="orb orb-a" />
      <div className="orb orb-b" />
      <motion.div
        className="landing-panel glass"
        style={{ rotateX, rotateY, y: floatY }}
        initial={{ opacity: 0, y: 40 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.7, ease: "easeOut" }}
      >
        <p className="eyebrow">Quantum Observatory</p>
        <h1>TurboQuant Decode Observatory</h1>
        <p className="landing-sub">
          Explore a cinematic data story: an animated landing sequence, then both reports in one
          continuous scroll narrative.
        </p>
        <motion.div className="landing-3d-scene" style={{ y: sceneDepth, rotateZ: sceneSpin }}>
          <motion.article
            className="scene-card scene-card-main"
            animate={{ y: [-4, 8, -4], rotateY: [-3, 4, -3] }}
            transition={{ repeat: Infinity, duration: 6.5, ease: "easeInOut" }}
          >
            <img src="/content/figures/fig8_dashboard.png" alt="TurboQuant dashboard preview" />
          </motion.article>
          <motion.article
            className="scene-card scene-card-left"
            animate={{ y: [12, -8, 12], rotateY: [8, -6, 8] }}
            transition={{ repeat: Infinity, duration: 7.4, ease: "easeInOut" }}
          >
            <img src="/content/figures/fig1_throughput_vs_context.png" alt="Throughput chart" />
          </motion.article>
          <motion.article
            className="scene-card scene-card-right"
            animate={{ y: [0, 14, 0], rotateY: [-8, 6, -8] }}
            transition={{ repeat: Infinity, duration: 7.9, ease: "easeInOut" }}
          >
            <img src="/content/figures_v2/fig12_decode_all_methods.png" alt="Methods comparison chart" />
          </motion.article>
        </motion.div>
        <div className="landing-actions">
          <a className="landing-cta" href="#report-v1">
            Enter Report Sequence
          </a>
          <a className="landing-ghost" href="#report-v2">
            Jump To V2
          </a>
        </div>
        <motion.a
          className="scroll-down-cue"
          href="#report-v1"
          animate={{ y: [0, 8, 0], opacity: [0.65, 1, 0.65] }}
          transition={{ repeat: Infinity, duration: 1.4, ease: "easeInOut" }}
        >
          Scroll Down
          <span>↓</span>
        </motion.a>
      </motion.div>
    </section>
  );
}

export default function App() {
  return (
    <div className="app-bg">
      <div className="bg-noise" />
      <div className="bg-grid" />
      <NavBar />
      <LandingSection />
      <section id="report-v1" className="report-anchor">
        <div className="report-title-band glass">
          <p className="eyebrow">Phase 1</p>
          <h2>Primary Report</h2>
        </div>
        <ReportView kind="v1" />
      </section>
      <section id="report-v2" className="report-anchor">
        <div className="report-title-band glass">
          <p className="eyebrow">Phase 2</p>
          <h2>Comparative Report V2</h2>
        </div>
        <ReportView kind="v2" />
      </section>
    </div>
  );
}
