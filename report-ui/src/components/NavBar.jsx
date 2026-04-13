import { Link, NavLink } from "react-router-dom";
import { motion } from "framer-motion";

export default function NavBar() {
  return (
    <header className="topbar glass">
      <Link className="brand" to="/">
        <span className="brand-mark" />
        <span>QUANTUM_OBSERVATORY</span>
      </Link>
      <nav className="nav-tabs">
        <NavLink to="/report" className="tab-link">
          Report
        </NavLink>
        <NavLink to="/report-v2" className="tab-link">
          Report V2
        </NavLink>
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
