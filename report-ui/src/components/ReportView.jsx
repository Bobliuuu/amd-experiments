/**
 * ReportView.jsx — Legacy markdown-based report viewer (no longer used)
 *
 * Replaced by the structured ReportSections.jsx components which render
 * benchmark data directly without parsing markdown. Kept for reference.
 */
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

function extractSnapshot(markdown) {
  const lines = markdown.split("\n");
  const title = lines.find((line) => line.startsWith("# "))?.replace(/^#\s+/, "") || "";
  const lead =
    lines.find(
      (line) =>
        line.trim() &&
        !line.startsWith("#") &&
        !line.startsWith("|") &&
        !line.startsWith("```")
    ) || "";
  const sections = lines
    .filter((line) => line.startsWith("## "))
    .map((line) => line.replace(/^##\s+/, "").trim())
    .slice(0, 6);

  const numericMatches =
    markdown.match(/\b\d+(?:\.\d+)?(?:x|%|M|K|MB|GB|us|ms|tok\/s)\b/gi) || [];
  const highlights = [...new Set(numericMatches)].slice(0, 4);
  return { title, lead, sections, highlights };
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
  const snapshot = useMemo(() => extractSnapshot(markdown), [markdown]);

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
        className="snapshot-grid"
        initial={{ opacity: 0, y: 12 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.04 }}
      >
        <article className="snapshot-main glass">
          <p className="eyebrow">{kind === "v2" ? "Report V2 Snapshot" : "Report Snapshot"}</p>
          <h3>{snapshot.title || heroProps.title}</h3>
          <p>{snapshot.lead || heroProps.subtitle}</p>
        </article>
        <article className="snapshot-rail glass">
          <h4>Key Signals</h4>
          <div className="snapshot-pills">
            {(snapshot.highlights.length ? snapshot.highlights : [heroProps.statA.value]).map(
              (item) => (
                <span key={item}>{item}</span>
              )
            )}
          </div>
          <h4>Top Sections</h4>
          <ul>
            {snapshot.sections.length ? (
              snapshot.sections.map((section) => <li key={section}>{section}</li>)
            ) : (
              <li>Loading sections...</li>
            )}
          </ul>
        </article>
      </motion.section>

      <motion.section
        className="markdown-wrap glass"
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.35, delay: 0.05 }}
      >
        <ReactMarkdown
          remarkPlugins={[remarkGfm]}
          components={{
            h2: ({ children }) => <h2 className="md-section">{children}</h2>,
            h3: ({ children }) => <h3 className="md-subsection">{children}</h3>,
            p: ({ children }) => <p className="md-paragraph">{children}</p>,
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
