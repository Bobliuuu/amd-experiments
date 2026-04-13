import { useEffect, useMemo, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { motion } from "framer-motion";
import { getFigureUrls, loadReportMarkdown, resolveImageSrc } from "../dataSources";
import FigureGallery from "./FigureGallery";

function Hero({ title, subtitle, statA, statB }) {
  return (
    <motion.section
      className="hero glass"
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      <div>
        <p className="eyebrow">Quantum Observatory</p>
        <h1>{title}</h1>
        <p className="hero-sub">{subtitle}</p>
      </div>
      <div className="hero-metrics">
        <div className="metric-card">
          <span>{statA.label}</span>
          <strong>{statA.value}</strong>
        </div>
        <div className="metric-card metric-card-secondary">
          <span>{statB.label}</span>
          <strong>{statB.value}</strong>
        </div>
      </div>
    </motion.section>
  );
}

export default function ReportView({ kind }) {
  const [markdown, setMarkdown] = useState("");
  const [sourceUrl, setSourceUrl] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    let mounted = true;
    setError("");
    setMarkdown("");
    loadReportMarkdown(kind)
      .then(({ text, sourceUrl: resolvedUrl }) => {
        if (!mounted) return;
        setMarkdown(text);
        setSourceUrl(resolvedUrl);
      })
      .catch((err) => {
        if (!mounted) return;
        setError(String(err.message || err));
      });

    return () => {
      mounted = false;
    };
  }, [kind]);

  const heroProps = useMemo(() => {
    if (kind === "v2") {
      return {
        title: "Benchmark Comparison",
        subtitle:
          "TurboQuant, IsoQuant, PlanarQuant, and RotorQuant on MI300X — data-first, visually immersive.",
        statA: { label: "Peak Prefill Gain", value: "26.5x" },
        statB: { label: "KV Compression", value: "4.92x" },
      };
    }
    return {
      title: "TurboQuant MI300X Report",
      subtitle:
        "Hardware-aware KV compression research presented through an interactive liquid glass interface.",
      statA: { label: "KV Compression", value: "4.923x" },
      statB: { label: "Max Context", value: "~6.7M" },
    };
  }, [kind]);

  const figures = getFigureUrls(kind);

  return (
    <main className="report-shell">
      <Hero {...heroProps} />
      {error ? (
        <section className="glass error-block">
          <h3>Unable to load report markdown</h3>
          <p>{error}</p>
          <p>
            Expected from local repo paths. You can also place files in
            `/public/content`.
          </p>
        </section>
      ) : null}

      <motion.section
        className="markdown-wrap glass"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.05 }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            img: ({ src, alt }) => (
              <img
                src={resolveImageSrc(sourceUrl, src || "")}
                alt={alt || "report figure"}
                loading="lazy"
              />
            ),
          }}
        >
          {markdown || "Loading report content..."}
        </ReactMarkdown>
      </motion.section>

      <FigureGallery
        title={kind === "v2" ? "Figure Deck (V2)" : "Figure Deck (Primary)"}
        figures={figures}
      />
    </main>
  );
}
