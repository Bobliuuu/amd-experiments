import { Navigate, Route, Routes } from "react-router-dom";
import NavBar from "./components/NavBar";
import ReportView from "./components/ReportView";

export default function App() {
  return (
    <div className="app-bg">
      <div className="bg-noise" />
      <div className="bg-grid" />
      <NavBar />
      <Routes>
        <Route path="/" element={<Navigate to="/report" replace />} />
        <Route path="/report" element={<ReportView kind="v1" />} />
        <Route path="/report-v2" element={<ReportView kind="v2" />} />
      </Routes>
    </div>
  );
}
