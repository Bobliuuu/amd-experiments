import { useEffect, useState } from "react";
import { motion } from "framer-motion";

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
          // Try next candidate.
        }
      }
    };
    test();
    return () => {
      mounted = false;
    };
  }, [figure.srcCandidates]);

  return (
    <motion.article
      className="figure-card glass"
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: idx * 0.04, duration: 0.3 }}
      whileHover={{ scale: 1.02, y: -2 }}
    >
      <h4>{figure.label}</h4>
      {src ? (
        <img src={src} alt={figure.label} loading="lazy" />
      ) : (
        <div className="missing-figure">Figure unavailable at runtime</div>
      )}
    </motion.article>
  );
}

export default function FigureGallery({ title, figures }) {
  return (
    <section className="section-stack">
      <div className="section-heading">
        <h3>{title}</h3>
      </div>
      <div className="figure-grid">
        {figures.map((figure, idx) => (
          <FigureCard figure={figure} idx={idx} key={figure.name} />
        ))}
      </div>
    </section>
  );
}
