#!/usr/bin/env bash
# =============================================================================
# Benchmark-grade installer: uv venv + AMD ROCm 7.2 nightly torch triple + optional
# vLLM with explicit stack identity, drift detection, and runtime qualification.
#
# Run ONLY inside an existing ROCm 7.2.x GPU container (see docs/rocm72_uv_torch_vllm_venv.md).
#
# Primary selector: ROCM72_UV_STACK
#   torch_only           — AMD nightly torch+vision+audio only (stack_id=amd_rocm72_nightly_torch_triple)
#   strict_uv_vllm_721   — ROCm 7.2.1 only; vLLM via uv; torch* must NOT drift vs freeze_torch_only
#   pip_owned_vllm_721   — vLLM via pip from rocm721 nightlies; requires ACK env vars; torch* may repin
#
# Rerun safety: existing $WORKDIR/.venv requires ROCM72_UV_FORCE=1 (or --fresh) unless
# --resume-qualify-only or ROCM72_UV_RESUME_QUALIFY_ONLY=1 (re-freeze + qualify only).
#
# --gates-only: stack-vs-ROCm policy + /opt/rocm + 7.2.* + uv (no downloads; no qualifier run).
#
# Artifacts under ROCM72_UV_WORKDIR:
#   freeze_torch_only.txt, freeze_post_install.txt, freeze_diff.patch
#   stack_manifest.json, qualification_report.json, install_rocm72_uv_report.txt
#   pytorch_rocm72_manifest.txt (copy of freeze_torch_only for backward compat)
#   install_state.json — written on installer failure
#
# Default ROCM72_UV_WORKDIR: <repo>/.benchmark_mi300_vllm_frozen (see docs/benchmark_mi300_locked_env.md).
# References: AMD PyTorch-on-ROCm install doc; vLLM GPU install doc (rocm721 row = ROCm 7.2.1).
# =============================================================================
set -euo pipefail

GATES_ONLY=0
RESUME_QUALIFY=0
for arg in "$@"; do
  case "$arg" in
    --gates-only|-n) GATES_ONLY=1 ;;
    --fresh) export ROCM72_UV_FORCE=1 ;;
    --resume-qualify-only) RESUME_QUALIFY=1 ;;
    -h|--help)
      sed -n '1,55p' "$0"
      exit 0
      ;;
  esac
done
[[ "${ROCM72_UV_RESUME_QUALIFY_ONLY:-0}" == "1" ]] && RESUME_QUALIFY=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
WORKDIR="${ROCM72_UV_WORKDIR:-$REPO_ROOT/.benchmark_mi300_vllm_frozen}"
FORCE="${ROCM72_UV_FORCE:-0}"
QUALIFY_PY="$REPO_ROOT/scripts/qualify_rocm72_vllm_stack.py"

die() {
  echo "[install_rocm72_uv_torch_vllm] ERROR: $*" >&2
  if [[ -n "${WORKDIR:-}" ]] && [[ -d "$WORKDIR" ]]; then
    printf '%s\n' "{\"schema\":\"install_state_v1\",\"failed\":true,\"reason\":$(printf '%s' "$*" | python3 -c 'import json,sys; print(json.dumps(sys.stdin.read()))')}" >"$WORKDIR/install_state.json" 2>/dev/null || true
  fi
  exit 1
}
info() { echo "[install_rocm72_uv_torch_vllm] $*"; }
warn() { echo "[install_rocm72_uv_torch_vllm] WARN: $*" >&2; }

# --- Resolve ROCM72_UV_STACK (strict; deprecate SKIP_VLLM / ROCM72_UV_VLLM_MODE) ---
STACK="${ROCM72_UV_STACK:-}"
if [[ "${SKIP_VLLM:-0}" == "1" ]]; then
  [[ -z "$STACK" ]] || die "Conflicting: SKIP_VLLM=1 and ROCM72_UV_STACK=$STACK"
  STACK="torch_only"
  warn "SKIP_VLLM=1 is deprecated; use ROCM72_UV_STACK=torch_only"
fi
if [[ -n "${ROCM72_UV_VLLM_MODE:-}" ]]; then
  warn "ROCM72_UV_VLLM_MODE is deprecated; migrate to ROCM72_UV_STACK=..."
  case "${ROCM72_UV_VLLM_MODE}" in
    off) [[ -z "$STACK" ]] && STACK="torch_only" ;;
    uv) [[ -z "$STACK" ]] && STACK="strict_uv_vllm_721" ;;
    pip) [[ -z "$STACK" ]] && STACK="pip_owned_vllm_721" ;;
    auto)
      if [[ -z "$STACK" ]]; then
        die "ROCM72_UV_VLLM_MODE=auto is removed. Set ROCM72_UV_STACK to torch_only | strict_uv_vllm_721 | pip_owned_vllm_721 (see docs)."
      fi
      ;;
    *) die "Unknown ROCM72_UV_VLLM_MODE=${ROCM72_UV_VLLM_MODE}" ;;
  esac
fi
if [[ -z "$STACK" ]]; then
  die "Set ROCM72_UV_STACK (torch_only | strict_uv_vllm_721 | pip_owned_vllm_721). Deprecated auto defaults removed."
fi
case "$STACK" in
  torch_only|strict_uv_vllm_721|pip_owned_vllm_721) ;;
  *) die "Invalid ROCM72_UV_STACK=$STACK" ;;
esac

info "=== ROCm72 benchmark installer ==="
info "REPO_ROOT=$REPO_ROOT WORKDIR=$WORKDIR STACK=$STACK FORCE=$FORCE RESUME_QUALIFY=$RESUME_QUALIFY"

if [[ ! -d /opt/rocm ]]; then
  die "/opt/rocm missing — not a ROCm container layout."
fi
ROCM_VER_FILE="/opt/rocm/.info/version"
[[ -f "$ROCM_VER_FILE" ]] || die "Missing $ROCM_VER_FILE"
ROCM_VER="$(head -1 "$ROCM_VER_FILE" | tr -d ' \t\r')"
info "ROCm container version: $ROCM_VER"
[[ "$ROCM_VER" =~ ^7\.2\. ]] || die "ROCm must be 7.2.x (got $ROCM_VER)"

# True ROCm 7.2.1 patchline only (exclude 7.2.10, etc.)
IS_721=0
[[ "$ROCM_VER" =~ ^7\.2\.1([^0-9]|$) ]] && IS_721=1

command -v uv >/dev/null 2>&1 || die "uv not on PATH"
UV_VER="$(uv --version)"

# --- Stack vs ROCm policy (no silent unsupported paths; included in --gates-only) ---
if [[ "$STACK" == "strict_uv_vllm_721" ]]; then
  [[ "$IS_721" == 1 ]] || die "strict_uv_vllm_721 requires ROCm 7.2.1 (got $ROCM_VER). Use torch_only or pip_owned_vllm_721 with ACKs, or a 7.2.1 image."
fi
if [[ "$STACK" == "pip_owned_vllm_721" ]]; then
  [[ "${ROCM72_UV_ACK_VLLM_PIP_REPINS_TORCH:-}" == "YES" ]] || die "pip_owned_vllm_721 requires ROCM72_UV_ACK_VLLM_PIP_REPINS_TORCH=YES"
  if [[ "$IS_721" != 1 ]]; then
    [[ "${ROCM72_UV_ACK_ROCM720_USES_ROCM721_WHEEL:-}" == "YES" ]] || die "ROCm $ROCM_VER: pip rocm721 wheel is documented for 7.2.1. Set ROCM72_UV_ACK_ROCM720_USES_ROCM721_WHEEL=YES to proceed, or use torch_only / 7.2.1 image."
  fi
fi

if [[ "$GATES_ONLY" == "1" ]]; then
  info "--gates-only OK (stack policy + uv + ROCm gates)."
  exit 0
fi

[[ -f "$QUALIFY_PY" ]] || die "Missing qualifier: $QUALIFY_PY"

# --- Refuse accidental venv mutation ---
if [[ "$RESUME_QUALIFY" == "1" ]]; then
  [[ -x "$WORKDIR/.venv/bin/python" ]] || die "--resume-qualify-only requires existing $WORKDIR/.venv"
else
  if [[ -e "$WORKDIR/.venv" ]] && [[ "$FORCE" != "1" ]]; then
    die "Refusing to reuse $WORKDIR/.venv without ROCM72_UV_FORCE=1 or --fresh. rm -rf workdir or set FORCE."
  fi
fi

mkdir -p "$WORKDIR"
cd "$WORKDIR"
PY="$WORKDIR/.venv/bin/python"

write_freezes_and_diff() {
  uv pip freeze --python "$PY" | sort >"$WORKDIR/freeze_torch_only.txt"
  cp -f "$WORKDIR/freeze_torch_only.txt" "$WORKDIR/pytorch_rocm72_manifest.txt"
  uv pip show torch torchvision torchaudio --python "$PY" 2>/dev/null | awk -F': ' '/^Name:|^Version:/{print}' >"$WORKDIR/torch_triple_show_torch_only.txt" || true
}

torch_triple_lines_changed() {
  # return 0 if torch/vision/audio pin lines differ between two sorted freezes
  local a="$1" b="$2"
  grep -E '^(torch|torchvision|torchaudio)(==|@)' "$a" 2>/dev/null | sort >"$WORKDIR/.cmp_a" || true
  grep -E '^(torch|torchvision|torchaudio)(==|@)' "$b" 2>/dev/null | sort >"$WORKDIR/.cmp_b" || true
  ! cmp -s "$WORKDIR/.cmp_a" "$WORKDIR/.cmp_b"
}

write_stack_manifest_partial() {
  local stack_id="$1" wheel_class="$2"
  export STACK_ID_JSON="$stack_id" WHEEL_CLASS_JSON="$wheel_class" ROCM_JSON="$ROCM_VER" WORKDIR_JSON="$WORKDIR"
  export UV_JSON="$UV_VER" REPO_JSON="$REPO_ROOT"
  export PYTORCH_INDEX_JSON="${PYTORCH_INDEX:-}"
  export VLLM_INSTALL_METHOD_JSON="${VLLM_METHOD:-none}"
  export VLLM_WHEEL_VARIANT_JSON="${VLLM_WHEEL_VARIANT:-}"
  export TOOLCHAIN_JSON="${TOOLCHAIN_JSON:-[]}"
  "$PY" <<'PY'
import json, os, shutil, subprocess

wd = os.environ["WORKDIR_JSON"]
py = os.path.join(wd, ".venv", "bin", "python")


def sh(code: str) -> str:
    return subprocess.check_output([py, "-c", code], text=True).strip()


def ver(pkg: str):
    try:
        return subprocess.check_output(
            [py, "-c", f"import {pkg} as p; print(p.__version__)"], text=True
        ).strip()
    except subprocess.CalledProcessError:
        return None


def rocminfo_gfx942():
    if not shutil.which("rocminfo"):
        return "rocminfo_absent"
    try:
        r = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=60)
    except subprocess.TimeoutExpired:
        return "rocminfo_timeout"
    if r.returncode != 0:
        return "rocminfo_failed"
    for line in (r.stdout or "").splitlines():
        if "gfx942" in line:
            return line.strip()[:2000]
    return "gfx942_line_not_found"


idx = os.environ.get("PYTORCH_INDEX_JSON") or ""
vmeth = os.environ.get("VLLM_INSTALL_METHOD_JSON") or "none"
variant = os.environ.get("VLLM_WHEEL_VARIANT_JSON") or ""
steps = [
    {
        "step": "uv_pip_amd_nightly_torch_triple",
        "index_url": idx,
        "packages": ["torch", "torchvision", "torchaudio"],
    }
]
sid = os.environ["STACK_ID_JSON"]
if sid == "amd_rocm72_nightly_torch_triple":
    pass
elif sid == "vllm_uv_rocm721_preserved_amd_torch":
    if vmeth == "uv":
        steps.append(
            {
                "step": "uv_pip_vllm",
                "extra_index_url": f"https://wheels.vllm.ai/rocm/nightly/{variant}" if variant else "https://wheels.vllm.ai/rocm/nightly/<variant>",
                "index_strategy": "unsafe-best-match",
            }
        )
    elif vmeth.startswith("pip_fallback"):
        steps.append(
            {
                "step": "uv_pip_vllm_failed",
                "note": "uv install returned non-zero; operator allowed pip fallback",
            }
        )
        steps.append(
            {
                "step": "pip_install_vllm_rocm721",
                "extra_index_url": "https://wheels.vllm.ai/rocm/nightly/rocm721",
            }
        )
elif sid == "vllm_pip_rocm721_owned_torch":
    steps.append(
        {
            "step": "pip_install_vllm_rocm721",
            "extra_index_url": "https://wheels.vllm.ai/rocm/nightly/rocm721",
        }
    )

try:
    ext = json.loads(os.environ.get("TOOLCHAIN_JSON") or "[]")
    if isinstance(ext, list) and ext:
        steps = ext
except json.JSONDecodeError:
    pass

meta = {
    "schema": "stack_manifest_v1",
    "stack_id": sid,
    "wheel_class": os.environ["WHEEL_CLASS_JSON"],
    "stability_tier": "nightly_wheel",
    "rocm_container_version": os.environ["ROCM_JSON"],
    "repo": os.environ["REPO_JSON"],
    "workdir": wd,
    "uv_version": os.environ["UV_JSON"],
    "which_python_under_test": subprocess.run(["which", py], capture_output=True, text=True).stdout.strip() or py,
    "python": sh("import sys; print(sys.version.split()[0])"),
    "python_executable": py,
    "torch": ver("torch"),
    "torchvision": ver("torchvision"),
    "torchaudio": ver("torchaudio"),
    "toolchain_steps": steps,
    "vllm_install_method": vmeth,
    "vllm_wheel_variant": (variant or None),
}
try:
    import vllm  # noqa: E402

    meta["vllm"] = getattr(vllm, "__version__", None)
except Exception:
    meta["vllm"] = None
try:
    import torch

    meta["torch_version_full"] = torch.__version__
    meta["hip"] = getattr(torch.version, "hip", None)
    meta["gpu_name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    meta["cuda_available"] = torch.cuda.is_available()
except Exception as e:
    meta["torch_probe_error"] = str(e)
rg = rocminfo_gfx942()
meta["rocminfo_gfx942_line"] = rg
path = os.path.join(wd, "stack_manifest.json")
with open(path, "w", encoding="utf-8") as f:
    json.dump(meta, f, indent=2, sort_keys=True)
print(path)
PY
}

merge_qualification_into_manifest() {
  local qjson="$WORKDIR/qualification_report.json"
  local mjson="$WORKDIR/stack_manifest.json"
  [[ -f "$qjson" && -f "$mjson" ]] || return 0
  QUAL_JSON="$qjson" MAN_JSON="$mjson" python3 <<'PY'
import json, os
with open(os.environ["MAN_JSON"], encoding="utf-8") as f:
    m = json.load(f)
with open(os.environ["QUAL_JSON"], encoding="utf-8") as f:
    q = json.load(f)
ch = q.get("checks") or {}
m["qualification_summary"] = {
    "benchmark_ready": q.get("benchmark_ready"),
    "torch_only_mode": q.get("torch_only_mode"),
    "checks_keys": sorted(ch.keys()),
    "has_rocprof": bool(ch.get("has_rocprof")),
    "has_rocprofv2": bool(ch.get("has_rocprofv2")),
}
m["qualification_report_path"] = os.environ["QUAL_JSON"]
with open(os.environ["MAN_JSON"], "w", encoding="utf-8") as f:
    json.dump(m, f, indent=2, sort_keys=True)
PY
}

if [[ "$RESUME_QUALIFY" == "1" ]]; then
  info "Resume: re-freeze + manifest + qualification only"
  [[ -f "$WORKDIR/freeze_torch_only.txt" ]] || die "--resume-qualify-only requires existing $WORKDIR/freeze_torch_only.txt from a prior install"
  VLLM_METHOD="none"
  VLLM_WHEEL_VARIANT=""
  if [[ -f "$WORKDIR/stack_manifest.json" ]]; then
    VLLM_METHOD="$(WORKDIR="$WORKDIR" python3 -c 'import json,os;print(json.load(open(os.path.join(os.environ["WORKDIR"],"stack_manifest.json"))).get("vllm_install_method") or "none")' 2>/dev/null || echo none)"
    VLLM_WHEEL_VARIANT="$(WORKDIR="$WORKDIR" python3 -c 'import json,os;print(json.load(open(os.path.join(os.environ["WORKDIR"],"stack_manifest.json"))).get("vllm_wheel_variant") or "")' 2>/dev/null || true)"
  fi
  uv pip freeze --python "$PY" | sort >"$WORKDIR/freeze_post_install.txt"
  diff -u "$WORKDIR/freeze_torch_only.txt" "$WORKDIR/freeze_post_install.txt" >"$WORKDIR/freeze_diff.patch" 2>/dev/null || true
  case "$STACK" in
    torch_only) write_stack_manifest_partial "amd_rocm72_nightly_torch_triple" "pytorch_nightly_rocm7_2" ;;
    strict_uv_vllm_721) write_stack_manifest_partial "vllm_uv_rocm721_preserved_amd_torch" "pytorch_nightly_rocm7_2_plus_vllm_uv" ;;
    pip_owned_vllm_721) write_stack_manifest_partial "vllm_pip_rocm721_owned_torch" "vllm_nightly_rocm721_owned_torch" ;;
  esac
else
  if [[ ! -x "$PY" ]]; then
    info "Creating venv …"
    uv venv --python 3.12 --seed
    PY="$WORKDIR/.venv/bin/python"
  fi
  "$PY" -c 'import sys; assert sys.version_info[:2]==(3,12), "require Python 3.12.x"' || die "venv Python must be 3.12.x"

  PYTORCH_INDEX="https://download.pytorch.org/whl/nightly/rocm7.2"
  info "Installing AMD nightly torch triple from $PYTORCH_INDEX"
  uv pip install --python "$PY" --pre --upgrade --index-url "$PYTORCH_INDEX" torch torchvision torchaudio

  info "Validating torch (ROCm / HIP / GPU visibility) …"
  "$PY" - <<'PY'
import torch
ver = torch.__version__
hip = getattr(torch.version, "hip", None)
if "+cu" in ver or "cu12" in ver:
    raise SystemExit(f"CUDA torch rejected: {ver}")
if not hip:
    raise SystemExit("empty HIP — not ROCm torch")
if not torch.cuda.is_available():
    raise SystemExit("torch.cuda.is_available() is False — no GPU for MI300X benchmark path")
print("torch", ver, "hip", hip, "gpu", torch.cuda.get_device_name(0))
PY

  if [[ "${ROCM72_UV_ALLOW_NON_MI300_GPU:-0}" != "1" ]]; then
    "$PY" -c "import torch; assert 'MI300' in torch.cuda.get_device_name(0).upper(), 'GPU must be MI300* (or set ROCM72_UV_ALLOW_NON_MI300_GPU=1)'" || die "MI300 GPU check failed"
  else
    warn "ROCM72_UV_ALLOW_NON_MI300_GPU=1 — allowing non-MI300 GPU"
  fi

  write_freezes_and_diff
  info "Wrote freeze_torch_only.txt (+ pytorch_rocm72_manifest.txt compat)"

  STACK_ID=""
  WHEEL_CLASS=""
  VLLM_METHOD="none"
  VLLM_WHEEL_VARIANT=""

  case "$STACK" in
    torch_only)
      STACK_ID="amd_rocm72_nightly_torch_triple"
      WHEEL_CLASS="pytorch_nightly_rocm7_2"
      cp -f "$WORKDIR/freeze_torch_only.txt" "$WORKDIR/freeze_post_install.txt"
      printf '' >"$WORKDIR/freeze_diff.patch"
      ;;
    strict_uv_vllm_721)
      STACK_ID="vllm_uv_rocm721_preserved_amd_torch"
      WHEEL_CLASS="pytorch_nightly_rocm7_2_plus_vllm_uv"
      VAR="$(curl -fsSL "https://wheels.vllm.ai/rocm/nightly/" | grep -oE 'rocm[0-9]+' | head -1 || true)"
      [[ -n "$VAR" ]] || die "Could not discover rocm variant from wheels.vllm.ai"
      VLLM_WHEEL_VARIANT="$VAR"
      info "Installing vLLM via uv (variant=$VAR) …"
      set +e
      uv pip install --python "$PY" --pre vllm \
        --extra-index-url "https://wheels.vllm.ai/rocm/nightly/${VAR}" \
        --index-strategy unsafe-best-match
      UV_RC=$?
      set -e
      if [[ "$UV_RC" != "0" ]]; then
        if [[ "${ROCM72_UV_ALLOW_PIP_FALLBACK:-}" == "YES" ]] && [[ "${ROCM72_UV_ACK_VLLM_PIP_REPINS_TORCH:-}" == "YES" ]]; then
          warn "uv vLLM failed; ROCM72_UV_ALLOW_PIP_FALLBACK=YES — switching to pip (torch will repin); stack becomes pip-owned."
          STACK_ID="vllm_pip_rocm721_owned_torch"
          WHEEL_CLASS="vllm_nightly_rocm721_owned_torch"
          VLLM_METHOD="pip_fallback_after_uv_fail"
          "$PY" -m pip install -q --upgrade "pip<82" setuptools wheel
          "$PY" -m pip install --pre vllm --extra-index-url "https://wheels.vllm.ai/rocm/nightly/rocm721"
        else
          die "uv pip install vllm failed (exit $UV_RC). Fix URL-dep issue or set ROCM72_UV_ALLOW_PIP_FALLBACK=YES with ROCM72_UV_ACK_VLLM_PIP_REPINS_TORCH=YES, or use ROCM72_UV_STACK=pip_owned_vllm_721 with ACKs."
        fi
      else
        VLLM_METHOD="uv"
      fi
      uv pip freeze --python "$PY" | sort >"$WORKDIR/freeze_post_install.txt"
      diff -u "$WORKDIR/freeze_torch_only.txt" "$WORKDIR/freeze_post_install.txt" >"$WORKDIR/freeze_diff.patch" || true
      if [[ "$VLLM_METHOD" == "uv" ]]; then
        if torch_triple_lines_changed "$WORKDIR/freeze_torch_only.txt" "$WORKDIR/freeze_post_install.txt"; then
          die "torch/vision/audio pins changed after uv vLLM — preserved stack violated. See freeze_diff.patch"
        fi
      fi
      ;;
    pip_owned_vllm_721)
      STACK_ID="vllm_pip_rocm721_owned_torch"
      WHEEL_CLASS="vllm_nightly_rocm721_owned_torch"
      VLLM_WHEEL_VARIANT="rocm721"
      info "Installing vLLM via pip (torch* expected to repin to vLLM-owned builds) …"
      "$PY" -m pip install -q --upgrade "pip<82" setuptools wheel
      "$PY" -m pip install --pre vllm --extra-index-url "https://wheels.vllm.ai/rocm/nightly/rocm721"
      VLLM_METHOD="pip"
      uv pip freeze --python "$PY" | sort >"$WORKDIR/freeze_post_install.txt"
      diff -u "$WORKDIR/freeze_torch_only.txt" "$WORKDIR/freeze_post_install.txt" >"$WORKDIR/freeze_diff.patch" || true
      ;;
  esac

  write_stack_manifest_partial "$STACK_ID" "$WHEEL_CLASS"
  info "Wrote stack_manifest.json (partial)"
fi

# --- Qualification (runtime, not import-only) ---
QUAL_ARGS=(--python "$PY" --out-json "$WORKDIR/qualification_report.json" --require-mi300)
if [[ "$STACK" == "torch_only" ]]; then
  QUAL_ARGS+=(--torch-only)
elif [[ "${ROCM72_UV_QUALIFY_WAIVE_DECODE:-0}" == "1" ]]; then
  QUAL_ARGS+=(--skip-decode --waive-decode)
fi
info "Running qualification: python3 $QUALIFY_PY ${QUAL_ARGS[*]}"
set +e
python3 "$QUALIFY_PY" "${QUAL_ARGS[@]}"
QRC=$?
set -e
merge_qualification_into_manifest

if [[ "$STACK" != "torch_only" ]]; then
  if [[ "$QRC" != "0" ]]; then
    die "Qualification failed (exit $QRC). See qualification_report.json. For decode skip with waiver: ROCM72_UV_QUALIFY_WAIVE_DECODE=1 (not benchmark_ready)."
  fi
else
  if [[ "$QRC" != "0" ]]; then
    die "Qualification failed for torch_only (exit $QRC)"
  fi
fi

# --- Human report ---
{
  echo "=== install_rocm72_uv_report ==="
  date -u "+%Y-%m-%dT%H:%M:%SZ"
  echo "stack=$STACK rocm=$ROCM_VER workdir=$WORKDIR"
  echo "which_python=$PY"
  echo "uv=$UV_VER"
  echo "dockerenv=$([[ -f /.dockerenv ]] && echo yes || echo no)"
  echo "qualification_exit=$QRC"
  cat "$WORKDIR/stack_manifest.json" 2>/dev/null || true
} | tee "$WORKDIR/install_rocm72_uv_report.txt"

info "SUCCESS: install + qualification complete."
info "Artifacts: stack_manifest.json qualification_report.json freeze_*.txt freeze_diff.patch"
info "Activate: source $WORKDIR/.venv/bin/activate"
exit 0
