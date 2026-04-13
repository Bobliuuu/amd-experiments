import { useEffect, useState, useRef } from "react";
import { motion, useInView } from "framer-motion";

function FigureCard({ figure, idx }) {
  const [src, setSrc] = useState("");

  useEffect(() => {
    let mounted = true;
    const test = async () => {
      for (const candidate of figure.srcCandidates) {
        try {
          const res = await fetch(candidate, { method: "HEAD" });
          if (res.ok) {
            if (mounted) setSrc(candidate);
            return;
          }
        } catch {
          // Try next
        }
      }
    };
    test();
    return () => { mounted = false; };
  }, [figure.srcCandidates]);

  return (
    <motion.article
      className="figure-card glass"
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: idx * 0.035, duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
      whileHover={{ scale: 1.02, y: -4, transition: { duration: 0.2 } }}
    >
      <h4>{figure.label}</h4>
      {src ? (
        <img src={src} alt={figure.label} loading="lazy" />
      ) : (
        <div className="missing-figure">unavailable at runtime</div>
      )}
    </motion.article>
  );
}

export default function FigureGallery({ figures }) {
  const ref = useRef(null);
  const inView = useInView(ref, { once: true, margin: "-60px" });

  return (
    <section className="gallery-wrap" ref={ref}>
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={inView ? { opacity: 1, y: 0 } : { opacity: 0, y: 20 }}
        transition={{ duration: 0.5 }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.8rem', marginBottom: '0.5rem' }}>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.7rem', color: 'var(--amd-red)', letterSpacing: '0.18em' }}>06</span>
          <span style={{ fontFamily: "'JetBrains Mono', monospace", fontSize: '0.68rem', color: 'var(--text-muted)', letterSpacing: '0.12em', background: 'rgba(229,52,75,0.08)', border: '1px solid rgba(229,52,75,0.2)', padding: '0.2rem 0.6rem', borderRadius: '4px' }}>FIGURE GALLERY</span>
        </div>
        <h3 className="gallery-title">All Benchmark Figures</h3>
        <p className="gallery-sub">figures_v2/ + figures/ · MI300X gfx942 · {figures.length} total</p>
      </motion.div>

      <div className="figure-grid">
        {figures.map((figure, idx) => (
          <FigureCard figure={figure} idx={idx} key={figure.name} />
        ))}
      </div>
    </section>
  );
}
