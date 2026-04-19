#!/usr/bin/env python3
"""
Benchmark qualification: ROCm torch stack + optional vLLM decode smoke.
Exit 0 = benchmark_ready (or explicit waiver), 2 = install OK but not benchmark_ready, 1 = hard fail.

For the written boundary between repo-delivered mitigations and deployment-stack
throughput work, see docs/repo_decode_bottleneck_closure.md.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone


def main() -> int:
    p = argparse.ArgumentParser(description="Qualify ROCm72 uv venv for MI300X benchmarking")
    p.add_argument("--python", default=sys.executable, help="Interpreter under test")
    p.add_argument("--out-json", required=True, help="Write qualification_report.json here")
    p.add_argument("--torch-only", action="store_true", help="Do not import vLLM or run decode smoke")
    p.add_argument("--require-mi300", action="store_true", help="Fail if GPU name lacks MI300")
    p.add_argument(
        "--decode-model",
        default=os.environ.get("ROCM72_QUALIFY_MODEL", "facebook/opt-125m"),
        help="HF model for vLLM decode smoke (ignored with --torch-only)",
    )
    p.add_argument("--skip-decode", action="store_true", help="Skip decode (benchmark_ready false unless --waive-decode)")
    p.add_argument("--waive-decode", action="store_true", help="Mark benchmark_ready despite skipped/failed decode policy")
    args = p.parse_args()
    py = args.python

    rep: dict = {
        "schema": "rocm72_vllm_qualification_v1",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "python_executable": py,
        "torch_only_mode": bool(args.torch_only),
        "checks": {},
        "benchmark_ready": False,
        "notes": [],
    }

    def subp(code: str, timeout: int = 600) -> dict:
        r = subprocess.run([py, "-c", code], capture_output=True, text=True, timeout=timeout)
        if r.returncode != 0:
            rep["checks"]["subprocess_error"] = {"stderr": (r.stderr or "")[-12000:], "stdout": (r.stdout or "")[-6000:]}
            raise RuntimeError("subprocess_failed")
        lines = [ln for ln in (r.stdout or "").strip().splitlines() if ln.strip()]
        return json.loads(lines[-1] if lines else "{}")

    rep["checks"]["which_python"] = shutil.which(py) or py
    try:
        uv_r = subprocess.run(["uv", "--version"], capture_output=True, text=True, timeout=15)
        rep["checks"]["uv_version"] = (uv_r.stdout or uv_r.stderr).strip() if uv_r.returncode == 0 else None
    except FileNotFoundError:
        rep["checks"]["uv_version"] = None

    rv_path = "/opt/rocm/.info/version"
    if os.path.isfile(rv_path):
        with open(rv_path, encoding="utf-8") as f:
            rep["rocm_container_version"] = f.readline().strip()
    else:
        rep["rocm_container_version"] = None

    for name in ("rocprof", "rocprofv2"):
        rep["checks"][f"has_{name}"] = shutil.which(name) is not None

    gfx = None
    if shutil.which("rocminfo"):
        try:
            ri = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60)
            for line in (ri.stdout or "").splitlines():
                if "gfx942" in line or ("Name:" in line and "gfx" in line):
                    gfx = (gfx or "") + line.strip() + "\n"
            if gfx:
                rep["checks"]["rocminfo_gfx_snippet"] = gfx[:4000]
        except (subprocess.TimeoutExpired, FileNotFoundError):
            rep["checks"]["rocminfo_gfx_snippet"] = None

    core = subp(
        r"""
import json, torch, torchvision, torchaudio
ver = torch.__version__
hip = getattr(torch.version, "hip", None) or ""
bad = ("+cu" in ver) or ("cu12" in ver) or (not hip)
print(json.dumps({
  "torch": ver, "torchvision": torchvision.__version__, "torchaudio": torchaudio.__version__,
  "hip": hip, "cuda_available": bool(torch.cuda.is_available()),
  "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
  "rocm_torch_ok": not bad,
}))
"""
    )
    rep["checks"]["torch_stack"] = core
    if not core.get("rocm_torch_ok"):
        _write(args.out_json, rep)
        return 1
    if not core.get("cuda_available"):
        rep["notes"].append("torch.cuda.is_available() is false")
        _write(args.out_json, rep)
        return 1
    if args.require_mi300:
        gn = (core.get("gpu_name") or "").upper()
        if "MI300" not in gn:
            rep["notes"].append(f"--require-mi300: gpu_name={core.get('gpu_name')!r}")
            _write(args.out_json, rep)
            return 1

    if args.torch_only:
        rep["checks"]["vllm_import"] = {"skipped": True}
        rep["checks"]["decode_smoke"] = {"skipped": True, "reason": "torch_only"}
        rep["benchmark_ready"] = True
        rep["notes"].append("torch_only: no vLLM decode required")
        _write(args.out_json, rep)
        return 0

    try:
        vm = subp(
            r"""
import json
try:
    import vllm
    print(json.dumps({"ok": True, "vllm_version": getattr(vllm, "__version__", "?")}))
except Exception as e:
    print(json.dumps({"ok": False, "error": str(e)}))
"""
        )
    except RuntimeError:
        _write(args.out_json, rep)
        return 1
    rep["checks"]["vllm_import"] = vm
    if not vm.get("ok"):
        _write(args.out_json, rep)
        return 1

    if args.skip_decode or not (args.decode_model or "").strip():
        rep["checks"]["decode_smoke"] = {"skipped": True, "model": args.decode_model}
        rep["benchmark_ready"] = bool(args.waive_decode)
        if not args.waive_decode:
            rep["notes"].append("decode skipped without waiver — not benchmark_ready")
        _write(args.out_json, rep)
        return 0 if args.waive_decode else 2

    model = args.decode_model.strip()
    code = f"""
import json, gc, os
os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")
from vllm import LLM, SamplingParams
llm = LLM(
    model={model!r},
    enforce_eager=True,
    max_model_len=512,
    gpu_memory_utilization=float(os.environ.get("ROCM72_QUALIFY_GPU_MEM", "0.25")),
    trust_remote_code=True,
)
out = llm.generate(["bench qualify"], SamplingParams(max_tokens=4, temperature=0.0))
n = len(out[0].outputs[0].token_ids)
del llm
gc.collect()
print(json.dumps({{"ok": True, "output_tokens": n}}))
"""
    t0 = time.perf_counter()
    try:
        dr = subprocess.run(
            [py, "-c", code],
            capture_output=True,
            text=True,
            timeout=int(os.environ.get("ROCM72_QUALIFY_DECODE_TIMEOUT_SEC", "900")),
        )
    except subprocess.TimeoutExpired:
        rep["checks"]["decode_smoke"] = {"ok": False, "error": "timeout", "model": model}
        _write(args.out_json, rep)
        return 1
    dt = time.perf_counter() - t0
    if dr.returncode != 0:
        rep["checks"]["decode_smoke"] = {
            "ok": False,
            "model": model,
            "seconds": round(dt, 3),
            "stderr": (dr.stderr or "")[-16000:],
            "stdout": (dr.stdout or "")[-8000:],
        }
        _write(args.out_json, rep)
        return 1
    try:
        payload = json.loads(dr.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        rep["checks"]["decode_smoke"] = {"ok": False, "parse_error": True, "stdout_tail": dr.stdout[-2000:]}
        _write(args.out_json, rep)
        return 1
    rep["checks"]["decode_smoke"] = {"ok": True, "model": model, "seconds": round(dt, 3), "detail": payload}
    rep["benchmark_ready"] = True
    rep["notes"].append("decode_smoke passed")
    _write(args.out_json, rep)
    return 0


def _write(path: str, obj: dict) -> None:
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError:
        sys.exit(1)
