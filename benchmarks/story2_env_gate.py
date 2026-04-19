#!/usr/bin/env python3
"""
story2_env_gate.py — Artifact-backed check: vLLM site-packages + optional TQ smoke + dispatch logs.

Writes results/story2_env_gate.json (Story 2 plan Phase 0).

  python3 benchmarks/story2_env_gate.py
  python3 benchmarks/story2_env_gate.py --skip-vllm-smoke   # import probe only
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
INSTALL_SCRIPT = ROOT / "scripts" / "install_turboquant_vllm_backend.sh"


def _default_python() -> Path:
    v = ROOT / ".venv" / "bin" / "python3"
    if v.is_file():
        return v
    return Path(sys.executable)


def _vllm_probe(py: Path) -> dict:
    code = r"""
import json, pathlib
import vllm
p = pathlib.Path(vllm.__file__)
s = str(p)
if "site-packages" in s or "dist-packages" in s:
    kind = "pypi_or_system"
elif "amd-experiments" in s or "workspace" in s:
    kind = "repo_stub"
else:
    kind = "unknown"
print(json.dumps({"vllm_file": s, "kind": kind}))
"""
    r = subprocess.run(
        [str(py), "-c", code],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(ROOT),
    )
    if r.returncode != 0:
        return {
            "vllm_file": "",
            "kind": "import_failed",
            "stderr": (r.stderr or "")[-4000:],
            "stdout": (r.stdout or "")[-2000:],
        }
    line = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    try:
        return json.loads(line)
    except json.JSONDecodeError:
        return {"vllm_file": "", "kind": "parse_error", "raw": line[:2000]}


def _run_smoke(py: Path, model: str, max_model_len: int) -> dict:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT / 'kernels'}:{ROOT}"
    env["VLLM_ATTENTION_BACKEND"] = "TURBOQUANT_ROCM"
    env["VLLM_TQ_USE_FUSED_KERNEL"] = "1"
    env["VLLM_TQ_LOG_DISPATCH"] = "1"
    # Prefer V0 if stack is flaky on V1 custom backend
    env.setdefault("VLLM_USE_V1", "0")

    cmd = [
        str(py),
        str(ROOT / "benchmarks" / "bench_vllm_turboquant_ab.py"),
        "--only-backend",
        "turboquant_fused",
        "--model",
        model,
        "--input-len",
        "32",
        "--output-len",
        "8",
        "--num-prompts",
        "2",
        "--max-model-len",
        str(max_model_len),
        "--enforce-eager",
        "--output",
        str(RESULTS / "_story2_gate_smoke.json"),
    ]
    t0 = time.perf_counter()
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=600,
        env=env,
        cwd=str(ROOT),
    )
    elapsed = round(time.perf_counter() - t0, 2)
    merged = (proc.stdout or "") + "\n" + (proc.stderr or "")
    dispatch_lines = [ln for ln in merged.splitlines() if "[TQ_DISPATCH]" in ln]
    fused_hits = sum(1 for ln in dispatch_lines if "FUSED_TRITON" in ln)
    return {
        "exit_code": proc.returncode,
        "elapsed_s": elapsed,
        "dispatch_line_count": len(dispatch_lines),
        "dispatch_lines_sample": dispatch_lines[:12],
        "fused_dispatch_hints": fused_hits,
        "tail_stderr": (proc.stderr or "")[-2500:],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Story 2 Phase 0: vLLM + TurboQuant env gate")
    p.add_argument("--python", type=Path, default=_default_python())
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument(
        "--skip-vllm-smoke",
        action="store_true",
        help="Only record vLLM import path; do not load LLM (no GPU required).",
    )
    args = p.parse_args()

    # Do not Path.resolve() the venv interpreter: it often symlinks to /usr/bin/python3
    # and the child would lose venv site-packages (vllm would appear missing).
    py = Path(args.python)
    out: dict = {
        "story2_phase": "env_gate",
        "time_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "interpreter": str(py),
        "install_turboquant_backend_script": str(INSTALL_SCRIPT),
        "install_hint": "After vLLM upgrades: bash scripts/install_turboquant_vllm_backend.sh",
        "vllm_probe": _vllm_probe(Path(py)),
        "vllm_smoke": None,
        "gate_passed": False,
    }

    probe = out["vllm_probe"]
    ok_import = probe.get("kind") == "pypi_or_system"

    if args.skip_vllm_smoke:
        out["gate_passed"] = ok_import
        out["vllm_smoke"] = {"skipped": True}
    elif not ok_import:
        out["vllm_smoke"] = {
            "skipped": True,
            "reason": "vLLM not importable from site-packages; fix interpreter / install vLLM",
        }
        out["gate_passed"] = False
    else:
        try:
            out["vllm_smoke"] = _run_smoke(Path(py), args.model, args.max_model_len)
        except subprocess.TimeoutExpired as e:
            out["vllm_smoke"] = {"error": "timeout", "detail": str(e)}
            out["gate_passed"] = False
        except Exception as e:
            out["vllm_smoke"] = {"error": type(e).__name__, "detail": str(e)}
            out["gate_passed"] = False
        else:
            sm = out["vllm_smoke"]
            err = sm.get("tail_stderr") or ""
            if sm.get("exit_code") not in (None, 0) and "custom_graph_pass" in err:
                sm["hint"] = (
                    "torch and vLLM versions are mismatched (torch._inductor.custom_graph_pass). "
                    "Align ROCm torch + vLLM wheels, then re-run without --skip-vllm-smoke."
                )
            passed = (
                sm.get("exit_code") == 0
                and sm.get("dispatch_line_count", 0) >= 1
                and sm.get("fused_dispatch_hints", 0) >= 1
            )
            out["gate_passed"] = bool(passed)
            if sm.get("exit_code") == 0 and sm.get("dispatch_line_count", 0) >= 1 and not passed:
                out["dispatch_path_note"] = (
                    "Expected path=FUSED_TRITON in [TQ_DISPATCH] lines for turboquant_fused; got other paths."
                )

    RESULTS.mkdir(parents=True, exist_ok=True)
    outp = RESULTS / "story2_env_gate.json"
    outp.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(outp.read_text())
    sys.exit(0 if out["gate_passed"] else 1)


if __name__ == "__main__":
    main()
