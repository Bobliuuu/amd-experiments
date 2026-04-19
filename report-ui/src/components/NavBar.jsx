/**
 * NavBar.jsx — Fixed top navigation bar
 *
 * AMD-branded header with section pill links (smooth-scroll anchors)
 * and a live status indicator.
 */
import { motion } from "framer-motion";

export default function NavBar() {
  return (
    <header className="topbar glass">
      <a className="brand" href="#hero" aria-label="AMD Research Hub">
        <div className="brand-amd">AMD</div>
        <div>
          <span className="brand-text">MI300X RESEARCH</span>
          <span className="brand-sub">KV CACHE COMPRESSION</span>
        </div>
      </a>

      <nav className="nav-pills">
        <a href="#problem" className="nav-pill">Problem</a>
        <a href="#math" className="nav-pill">Mathematics</a>
        <a href="#compression" className="nav-pill">Compression</a>
        <a href="#results" className="nav-pill">Results</a>
        <a href="#stories" className="nav-pill">Stories</a>
        <a href="#reasoning" className="nav-pill">Reasoning</a>
        <a href="#conclusion" className="nav-pill">Conclusion</a>
        <a href="#figures" className="nav-pill">Figures</a>
      </nav>

      <div className="nav-status">
        <motion.div
          className="status-dot pulse-dot"
          animate={{ opacity: [0.5, 1, 0.5] }}
          transition={{ repeat: Infinity, duration: 2.4, ease: "easeInOut" }}
        />
        <span>gfx942 · Apr 2026</span>
      </div>
    </header>
  );
}
