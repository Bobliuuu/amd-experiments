import { motion } from "framer-motion";

export default function NavBar() {
  return (
    <header className="topbar glass">
      <a className="brand" href="#landing">
        <span className="brand-mark" />
        <span>QUANTUM_OBSERVATORY</span>
      </a>
      <nav className="nav-tabs">
        <a href="#landing" className="tab-link">
          Landing
        </a>
        <a href="#report-v1" className="tab-link">
          Report
        </a>
        <a href="#report-v2" className="tab-link">
          Report V2
        </a>
      </nav>
      <motion.div
        className="status-chip"
        initial={{ opacity: 0.5 }}
        animate={{ opacity: [0.5, 1, 0.5] }}
        transition={{ repeat: Infinity, duration: 2.6, ease: "easeInOut" }}
      >
        LIVE
      </motion.div>
    </header>
  );
}
