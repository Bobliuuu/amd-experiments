/**
 * FigureGallery.jsx — Interactive figure carousel for benchmark results
 *
 * Renders two independent carousels (V2 = 15 figures, V1 = 11 figures) with:
 *   - Directional slide animations via AnimatePresence
 *   - Scrollable thumbnail strip (auto-scrolls active into view)
 *   - Fullscreen lightbox with keyboard navigation (←/→/Esc)
 *   - Figures served from /public/content/ (Vite static assets)
 */
import { useEffect, useRef, useState, useCallback } from "react";
import { motion, AnimatePresence, useInView } from "framer-motion";

/* ─── CONSTANTS ──────────────────────────────────────────────────── */

const METHOD_ACCENT = {
  "figures_v2": "#E5344B",
  "figures":    "#3B82F6",
};

function getAccent(figure) {
  return figure.srcCandidates[0].includes("figures_v2")
    ? "#E5344B"
    : "#3B82F6";
}

/* ─── LIGHTBOX ───────────────────────────────────────────────────── */

function Lightbox({ figure, onClose, onPrev, onNext, total, index }) {
  useEffect(() => {
    const handler = (e) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowRight") onNext();
      if (e.key === "ArrowLeft") onPrev();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onClose, onNext, onPrev]);

  const accent = getAccent(figure);

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.2 }}
      onClick={onClose}
      style={{
        position: "fixed", inset: 0, zIndex: 2000,
        background: "rgba(0,0,0,0.92)",
        backdropFilter: "blur(16px)",
        display: "flex", alignItems: "center", justifyContent: "center",
        padding: "2rem",
      }}
    >
      {/* Prev */}
      <button
        onClick={(e) => { e.stopPropagation(); onPrev(); }}
        style={navBtnStyle("left")}
        aria-label="Previous"
      >‹</button>

      {/* Content */}
      <motion.div
        key={figure.name}
        initial={{ opacity: 0, scale: 0.94 }}
        animate={{ opacity: 1, scale: 1 }}
        exit={{ opacity: 0, scale: 0.94 }}
        transition={{ duration: 0.22 }}
        onClick={(e) => e.stopPropagation()}
        style={{
          display: "grid",
          gridTemplateColumns: "1fr 380px",
          gap: 0,
          maxWidth: "min(1200px, 92vw)",
          maxHeight: "88vh",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          border: `1px solid ${accent}44`,
          boxShadow: `0 40px 100px rgba(0,0,0,0.8), 0 0 60px ${accent}18`,
          background: "var(--surface)",
        }}
      >
        {/* Image pane */}
        <div style={{ background: "#000", display: "flex", alignItems: "center", justifyContent: "center", minHeight: "300px", maxHeight: "88vh" }}>
          <img
            src={figure.src}
            alt={figure.label}
            style={{ maxWidth: "100%", maxHeight: "88vh", objectFit: "contain", display: "block" }}
          />
        </div>

        {/* Description pane */}
        <div style={{ padding: "2rem", overflowY: "auto", display: "flex", flexDirection: "column", gap: "1rem", borderLeft: `1px solid ${accent}22` }}>
          <div>
            <p style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.68rem", color: accent, letterSpacing: "0.12em", marginBottom: "0.5rem" }}>
              {figure.name.replace(".png", "")} · {index + 1} / {total}
            </p>
            <h3 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: "1.05rem", fontWeight: 600, color: "var(--text)", lineHeight: 1.3, margin: 0 }}>
              {figure.label}
            </h3>
          </div>

          {figure.desc && (
            <p style={{ fontSize: "0.82rem", color: "var(--text-sub)", lineHeight: 1.75, margin: 0 }}>
              {figure.desc}
            </p>
          )}

          <button
            onClick={onClose}
            style={{
              marginTop: "auto", alignSelf: "flex-start",
              background: "rgba(229,52,75,0.12)", border: "1px solid rgba(229,52,75,0.3)",
              color: "var(--amd-red)", borderRadius: "6px",
              padding: "0.4rem 0.9rem", fontSize: "0.75rem",
              fontFamily: "'JetBrains Mono',monospace", cursor: "pointer",
              letterSpacing: "0.06em",
            }}
          >
            CLOSE  ESC
          </button>
        </div>
      </motion.div>

      {/* Next */}
      <button
        onClick={(e) => { e.stopPropagation(); onNext(); }}
        style={navBtnStyle("right")}
        aria-label="Next"
      >›</button>
    </motion.div>
  );
}

function navBtnStyle(side) {
  return {
    position: "absolute", [side]: "1.5rem",
    top: "50%", transform: "translateY(-50%)",
    background: "rgba(229,52,75,0.15)", border: "1px solid rgba(229,52,75,0.35)",
    color: "var(--text)", borderRadius: "50%",
    width: "3rem", height: "3rem",
    fontSize: "1.5rem", lineHeight: 1,
    cursor: "pointer", display: "flex", alignItems: "center", justifyContent: "center",
    transition: "background 0.18s, box-shadow 0.18s",
    zIndex: 10,
  };
}

/* ─── THUMBNAIL STRIP ─────────────────────────────────────────────── */

function ThumbStrip({ figures, current, onSelect }) {
  const stripRef = useRef(null);

  // Scroll active thumb into view
  useEffect(() => {
    const el = stripRef.current?.querySelector(`[data-idx="${current}"]`);
    if (el) el.scrollIntoView({ block: "nearest", inline: "center", behavior: "smooth" });
  }, [current]);

  return (
    <div
      ref={stripRef}
      style={{
        display: "flex",
        gap: "0.5rem",
        overflowX: "auto",
        padding: "0.75rem 0.5rem",
        scrollbarWidth: "thin",
        scrollbarColor: "rgba(229,52,75,0.3) transparent",
        borderTop: "1px solid rgba(255,255,255,0.05)",
        background: "rgba(0,0,0,0.3)",
      }}
    >
      {figures.map((fig, i) => {
        const accent = getAccent(fig);
        const active = i === current;
        return (
          <motion.button
            key={fig.name}
            data-idx={i}
            onClick={() => onSelect(i)}
            animate={{ opacity: active ? 1 : 0.45, scale: active ? 1 : 0.94 }}
            transition={{ duration: 0.2 }}
            style={{
              flexShrink: 0,
              width: "72px", height: "52px",
              borderRadius: "6px",
              overflow: "hidden",
              border: active ? `2px solid ${accent}` : "2px solid transparent",
              background: "#000",
              cursor: "pointer",
              padding: 0,
              boxShadow: active ? `0 0 12px ${accent}55` : "none",
              transition: "border-color 0.18s, box-shadow 0.18s",
            }}
          >
            <img
              src={fig.src}
              alt={fig.label}
              style={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }}
              onError={(e) => { e.target.style.display = "none"; }}
            />
          </motion.button>
        );
      })}
    </div>
  );
}

/* ─── MAIN CAROUSEL ──────────────────────────────────────────────── */

function Carousel({ figures, groupLabel, groupColor }) {
  const [current, setCurrent] = useState(0);
  const [direction, setDirection] = useState(1);
  const [lightbox, setLightbox] = useState(false);
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  const go = useCallback((next) => {
    const n = (next + figures.length) % figures.length;
    setDirection(next > current ? 1 : -1);
    setCurrent(n);
  }, [current, figures.length]);

  const prev = useCallback(() => go(current - 1), [go, current]);
  const next = useCallback(() => go(current + 1), [go, current]);

  // Keyboard on carousel (when not in lightbox)
  useEffect(() => {
    if (lightbox) return;
    const handler = (e) => {
      if (e.key === "ArrowRight") next();
      if (e.key === "ArrowLeft") prev();
    };
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [lightbox, next, prev]);

  const fig = figures[current];
  const accent = getAccent(fig);

  const variants = {
    enter: (d) => ({ x: d > 0 ? "40%" : "-40%", opacity: 0 }),
    center: { x: 0, opacity: 1 },
    exit: (d) => ({ x: d > 0 ? "-40%" : "40%", opacity: 0 }),
  };

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 24 }}
      animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 24 }}
      transition={{ duration: 0.5 }}
      style={{
        borderRadius: "var(--radius-lg)",
        overflow: "hidden",
        border: `1px solid ${groupColor}22`,
        background: "linear-gradient(160deg, rgba(20,25,38,0.85), rgba(10,12,20,0.75))",
        backdropFilter: "blur(24px)",
        boxShadow: "0 24px 64px rgba(0,0,0,0.55)",
      }}
    >
      {/* Main slide */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", minHeight: "360px", position: "relative" }}>

        {/* Image pane */}
        <div style={{ background: "#000", position: "relative", overflow: "hidden", minHeight: "320px" }}>
          <AnimatePresence initial={false} custom={direction} mode="popLayout">
            <motion.div
              key={fig.name}
              custom={direction}
              variants={variants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.35, ease: [0.23, 1, 0.32, 1] }}
              style={{ position: "absolute", inset: 0, display: "flex", alignItems: "center", justifyContent: "center" }}
            >
              <img
                src={fig.src}
                alt={fig.label}
                style={{ width: "100%", height: "100%", objectFit: "contain", display: "block", cursor: "zoom-in" }}
                onClick={() => setLightbox(true)}
                onError={(e) => {
                  e.target.style.display = "none";
                  e.target.parentElement.style.background = "rgba(229,52,75,0.05)";
                }}
              />
            </motion.div>
          </AnimatePresence>

          {/* Prev / Next overlays */}
          <button onClick={prev} style={slideNavStyle("left")}>‹</button>
          <button onClick={next} style={slideNavStyle("right")}>›</button>

          {/* Counter badge */}
          <div style={{
            position: "absolute", top: "0.75rem", left: "0.75rem",
            fontFamily: "'JetBrains Mono',monospace", fontSize: "0.68rem",
            color: accent, background: "rgba(0,0,0,0.65)", backdropFilter: "blur(8px)",
            border: `1px solid ${accent}44`, borderRadius: "5px",
            padding: "0.2rem 0.5rem", letterSpacing: "0.08em",
          }}>
            {current + 1} / {figures.length}
          </div>

          {/* Expand button */}
          <button
            onClick={() => setLightbox(true)}
            style={{
              position: "absolute", bottom: "0.75rem", right: "0.75rem",
              background: "rgba(0,0,0,0.65)", backdropFilter: "blur(8px)",
              border: `1px solid ${accent}44`, color: accent,
              borderRadius: "5px", padding: "0.25rem 0.55rem",
              fontSize: "0.68rem", fontFamily: "'JetBrains Mono',monospace",
              cursor: "pointer", letterSpacing: "0.06em",
            }}
          >
            EXPAND ↗
          </button>
        </div>

        {/* Description pane */}
        <div style={{ padding: "2rem 2rem 1.5rem", display: "flex", flexDirection: "column", gap: "0.75rem", borderLeft: `1px solid ${accent}18` }}>
          <AnimatePresence mode="wait">
            <motion.div
              key={fig.name + "_desc"}
              initial={{ opacity: 0, y: 8 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -8 }}
              transition={{ duration: 0.25 }}
              style={{ display: "flex", flexDirection: "column", gap: "0.65rem", height: "100%" }}
            >
              <p style={{
                fontFamily: "'JetBrains Mono',monospace", fontSize: "0.66rem",
                color: accent, letterSpacing: "0.12em", margin: 0,
              }}>
                {fig.name.replace(".png", "")}
              </p>

              <h4 style={{
                fontFamily: "'Space Grotesk',sans-serif", fontSize: "1rem",
                fontWeight: 600, color: "var(--text)", lineHeight: 1.3, margin: 0,
              }}>
                {fig.label}
              </h4>

              {fig.desc ? (
                <p style={{ fontSize: "0.82rem", color: "var(--text-sub)", lineHeight: 1.75, margin: 0, flex: 1, overflowY: "auto" }}>
                  {fig.desc}
                </p>
              ) : (
                <p style={{ fontSize: "0.78rem", color: "var(--text-muted)", fontFamily: "'JetBrains Mono',monospace" }}>No description.</p>
              )}

              {/* Dot pager */}
              <div style={{ display: "flex", gap: "0.35rem", marginTop: "auto", flexWrap: "wrap" }}>
                {figures.map((_, i) => (
                  <motion.button
                    key={i}
                    onClick={() => go(i)}
                    animate={{ opacity: i === current ? 1 : 0.3, scale: i === current ? 1.2 : 1 }}
                    transition={{ duration: 0.15 }}
                    style={{
                      width: "6px", height: "6px", borderRadius: "50%",
                      background: i === current ? accent : "var(--text-muted)",
                      border: "none", padding: 0, cursor: "pointer",
                    }}
                    aria-label={`Go to figure ${i + 1}`}
                  />
                ))}
              </div>
            </motion.div>
          </AnimatePresence>
        </div>
      </div>

      {/* Thumbnail strip */}
      <ThumbStrip figures={figures} current={current} onSelect={(i) => go(i)} />

      {/* Lightbox */}
      <AnimatePresence>
        {lightbox && (
          <Lightbox
            figure={fig}
            index={current}
            total={figures.length}
            onClose={() => setLightbox(false)}
            onPrev={() => { prev(); }}
            onNext={() => { next(); }}
          />
        )}
      </AnimatePresence>
    </motion.div>
  );
}

function slideNavStyle(side) {
  return {
    position: "absolute", [side]: 0, top: 0, bottom: 0,
    width: "3rem",
    background: `linear-gradient(${side === "left" ? "90deg" : "270deg"}, rgba(0,0,0,0.5), transparent)`,
    border: "none", color: "rgba(255,255,255,0.7)",
    fontSize: "1.6rem", cursor: "pointer",
    display: "flex", alignItems: "center", justifyContent: "center",
    transition: "background 0.18s, color 0.18s",
    zIndex: 5,
  };
}

/* ─── GALLERY WRAPPER ─────────────────────────────────────────────── */

export default function FigureGallery({ figures }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  const v2 = figures.filter(f => f.srcCandidates[0].includes("figures_v2"));
  const v1 = figures.filter(f => !f.srcCandidates[0].includes("figures_v2"));

  return (
    <section className="gallery-wrap" ref={ref}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
        transition={{ duration: 0.5 }}
        style={{ marginBottom: "2.5rem" }}
      >
        <div style={{ display: "flex", alignItems: "center", gap: "0.8rem", marginBottom: "0.5rem" }}>
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.7rem", color: "var(--amd-red)", letterSpacing: "0.18em" }}>07</span>
          <span style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.68rem", color: "var(--text-muted)", letterSpacing: "0.12em", background: "rgba(229,52,75,0.08)", border: "1px solid rgba(229,52,75,0.2)", padding: "0.2rem 0.6rem", borderRadius: "4px" }}>FIGURE GALLERY</span>
        </div>
        <h3 style={{ fontFamily: "'Space Grotesk',sans-serif", fontSize: "1.3rem", fontWeight: 600, marginBottom: "0.4rem" }}>
          All Benchmark Figures
        </h3>
        <p style={{ color: "var(--text-muted)", fontFamily: "'JetBrains Mono',monospace", fontSize: "0.78rem", marginBottom: 0 }}>
          ← → arrow keys to navigate · click image to expand · {figures.length} total figures
        </p>
      </motion.div>

      {/* V2 carousel */}
      <div style={{ marginBottom: "2.5rem" }}>
        <p style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.7rem", color: "var(--amd-red)", letterSpacing: "0.14em", marginBottom: "1rem", paddingBottom: "0.5rem", borderBottom: "1px solid rgba(229,52,75,0.12)" }}>
          ▸ REPORT V2 — TurboQuant / IsoQuant / PlanarQuant / RotorQuant · {v2.length} figures
        </p>
        {v2.length > 0 && <Carousel figures={v2} groupLabel="V2" groupColor="#E5344B" />}
      </div>

      {/* V1 carousel */}
      <div>
        <p style={{ fontFamily: "'JetBrains Mono',monospace", fontSize: "0.7rem", color: "var(--turbo-color)", letterSpacing: "0.14em", marginBottom: "1rem", paddingBottom: "0.5rem", borderBottom: "1px solid rgba(59,130,246,0.12)" }}>
          ▸ REPORT V1 — TurboQuant Baseline (FP16 / FP8 / INT4 / TQ3 / TQ4) · {v1.length} figures
        </p>
        {v1.length > 0 && <Carousel figures={v1} groupLabel="V1" groupColor="#3B82F6" />}
      </div>
    </section>
  );
}
