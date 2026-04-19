# MI300X — locked benchmark Python environment (canonical)

This document is the **single source of truth** for the **project-standard** ROCm 7.2 + PyTorch + vLLM stack used for **all future GPU / vLLM benchmarks** in this repo. Operators and automation should treat paths and env vars here as **normative**, not suggestions.

**Related:** general installer mechanics — [`rocm72_uv_torch_vllm_venv.md`](rocm72_uv_torch_vllm_venv.md); scripts — [`scripts/install_rocm72_uv_torch_vllm.sh`](../scripts/install_rocm72_uv_torch_vllm.sh), [`scripts/qualify_rocm72_vllm_stack.py`](../scripts/qualify_rocm72_vllm_stack.py).

---

## 1. What was changed (exact record, 2026-04-19 lab)

These are the **exact** steps and settings that produced the first locked tree. Repeating them on a **fresh** machine with the same ROCm line + network access reproduces the **same installer behavior**; pinned **nightly** wheels may still drift over calendar time (see §7).

| Step | Exact action |
|------|----------------|
| **Removed** | Previous failed partial tree **`$REPO/.rocm72-uv-bench`** (only contained an `install_state.json` from `strict_uv_vllm_721` on ROCm 7.2.0). |
| **Did not remove** | Any Docker image in use by a running container (e.g. image `rocm` / `rocm:latest`). |
| **Workdir** | **`ROCM72_UV_WORKDIR=$REPO/.benchmark_mi300_vllm_frozen`** (fixed path under repo root). |
| **Stack** | **`ROCM72_UV_STACK=pip_owned_vllm_721`** — required on **ROCm 7.2.0** Primus-style images because **`strict_uv_vllm_721`** is **refused** unless `/opt/rocm/.info/version` is the **7.2.1** patchline. |
| **ACKs** | **`ROCM72_UV_ACK_VLLM_PIP_REPINS_TORCH=YES`** and **`ROCM72_UV_ACK_ROCM720_USES_ROCM721_WHEEL=YES`** (mandatory for pip + `rocm721` on 7.2.0). |
| **Force** | **`ROCM72_UV_FORCE=1`** on first provision (empty workdir; FORCE documents operator intent if `.venv` already exists). |
| **Installer** | `bash scripts/install_rocm72_uv_torch_vllm.sh` (no `--gates-only` for full install). |
| **PATH** | `export PATH="$HOME/.local/bin:$PATH"` so **`uv`** resolves (same as other docs). |
| **Qualification** | Produced automatically by installer: **`qualification_report.json`** with **`benchmark_ready: true`**, **`decode_smoke.ok: true`** (`facebook/opt-125m`), **`vllm_import.ok: true`**, **`has_rocprof` / `has_rocprofv2`: true**. |
| **Extra decode smoke** | Manual check after install (same venv): `facebook/opt-125m`, `enforce_eager=True`, `max_model_len=512`, `gpu_memory_utilization=0.25`, `max_tokens=16` — succeeded. |
| **Representative benchmark** | `benchmarks/bench_flash_attn_check.py --skip-dispatch-check --seq-lens 512 1024 --causal-seq-lens 4096 --matrix-seq-lens 1024 4096` using **`$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/python`** → wrote **`results/bench_flash_attn_check.json`**. |
| **Archive (“freeze”)** | Copied freezes + manifests + reports + representative bench JSON into **`results/benchmark_env_mi300_20260419T063618Z/`** and wrote **`env_lock.json`** pointing at the live workdir. |

**Hardware / OS context for that run:** AMD Instinct **MI300X VF**, **`/opt/rocm/.info/version` = `7.2.0`**, host **not** inside Docker (`/.dockerenv` absent). Other labs may use **`docker_run_amd_mi300x.sh`**; set **`ROCM72_UV_WORKDIR`** to a path **inside the container mount** (e.g. `/workspace/amd-experiments/.benchmark_mi300_vllm_frozen`).

---

## 2. Canonical paths (copy-paste)

Let **`$REPO`** be the absolute path to the **`amd-experiments`** git root (e.g. `/workspace/amd-experiments` in Docker, or your host checkout).

| Role | Path |
|------|------|
| **Live venv (activate this for benchmarks)** | **`$REPO/.benchmark_mi300_vllm_frozen/.venv`** |
| **Installer + freeze artifacts** | **`$REPO/.benchmark_mi300_vllm_frozen/`** (`freeze_*.txt`, `stack_manifest.json`, `qualification_report.json`, …) |
| **First audited snapshot folder** | **`$REPO/results/benchmark_env_mi300_20260419T063618Z/`** (`env_lock.json` + copies of the above + `bench_flash_attn_check.representative.json`) |

**Activate:**

```bash
source "$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/activate"
```

**Always invoke benchmarks with this interpreter** (or an activated shell from it):

```bash
"$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/python" benchmarks/bench_flash_attn_check.py
```

---

## 3. Versions recorded at lock time (`stack_manifest.json`)

These values are taken from the locked **`stack_manifest.json`** snapshot. They are **not** a guarantee for future `pip`/`uv` resolves on a different day.

| Component | Version (at lock) |
|-----------|---------------------|
| **Python** | 3.12.3 (venv) |
| **torch** | `2.10.0+git8514f05` |
| **torchvision** | `0.24.1+d801a34` |
| **torchaudio** | `2.9.0+eaa9e4e` |
| **HIP** | `7.2.53211` |
| **vLLM** | `0.19.2rc1.dev9+g4353c9cb4` |
| **stack_id** | `vllm_pip_rocm721_owned_torch` |
| **uv** | `uv 0.11.7 (x86_64-unknown-linux-gnu)` |

**Note:** `rocminfo` was **not** on `PATH` on that host; manifest records `rocminfo_gfx942_line: "rocminfo_absent"`. Install **`rocminfo`** in the image if you want a real gfx942 line in the manifest for audits.

---

## 4. Exact reinstall commands (from zero)

```bash
cd "$REPO"
export PATH="$HOME/.local/bin:$PATH"
export ROCM72_UV_WORKDIR="$REPO/.benchmark_mi300_vllm_frozen"
export ROCM72_UV_STACK=pip_owned_vllm_721
export ROCM72_UV_ACK_VLLM_PIP_REPINS_TORCH=YES
export ROCM72_UV_ACK_ROCM720_USES_ROCM721_WHEEL=YES
export ROCM72_UV_FORCE=1   # required if .venv already exists
bash scripts/install_rocm72_uv_torch_vllm.sh
```

**Green qualification:** read **`$ROCM72_UV_WORKDIR/qualification_report.json`** and assert **`benchmark_ready`** is **`true`** and **`checks.decode_smoke.ok`** is **`true`** (unless you intentionally used a waiver env — not used for this lock).

---

## 5. Freezing a new snapshot after any reinstall

When you intentionally rebuild the venv, archive artifacts again (UTC stamp). Set **`REPO`** to your checkout root, then:

```bash
export REPO=/absolute/path/to/amd-experiments
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
export LOCKDIR="$REPO/results/benchmark_env_mi300_${STAMP}"
mkdir -p "$LOCKDIR"
W="$REPO/.benchmark_mi300_vllm_frozen"
cp -a "$W/freeze_torch_only.txt" "$W/freeze_post_install.txt" "$W/freeze_diff.patch" \
      "$W/stack_manifest.json" "$W/qualification_report.json" "$W/install_rocm72_uv_report.txt" \
      "$LOCKDIR/"
# optional: copy a key results/ JSON you just ran for traceability
cp -a "$REPO/results/bench_flash_attn_check.json" "$LOCKDIR/bench_flash_attn_check.representative.json" 2>/dev/null || true
REPO="$REPO" LOCKDIR="$LOCKDIR" python3 -c "
import json, datetime, os
repo = os.environ['REPO']
lock = os.environ['LOCKDIR']
w = f'{repo}/.benchmark_mi300_vllm_frozen'
meta = {
    'locked_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
    'rocm72_uv_workdir': w,
    'activate': f'source {w}/.venv/bin/activate',
    'rocm72_uv_stack': 'pip_owned_vllm_721',
    'note': 'Re-locked after reinstall; see docs/benchmark_mi300_locked_env.md',
}
open(f'{lock}/env_lock.json', 'w', encoding='utf-8').write(json.dumps(meta, indent=2))
"
```

---

## 6. Integration with other entrypoints

| Entrypoint | How to align with this lock |
|------------|------------------------------|
| **`install_and_run.sh`** | Set **`VENV_DIR=$REPO/.benchmark_mi300_vllm_frozen/.venv`** (parent of `bin/python`) **or** set **`AMDEXP_PYTHON=$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/python`** when using system-python mode is inappropriate. |
| **`docker_run_amd_mi300x.sh`** | Mount repo at a stable path; use **`ROCM72_UV_WORKDIR`** under that mount; do **not** set **`AMDEXP_PIP_REQUIREMENTS=1`** in the same session unless you intend to fight the image torch. |
| **`scripts/verify_mi300x_stack.sh`** | **`AMDEXP_PYTHON=$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/python`** so verification matches benchmark interpreter. |

---

## 7. Reproducibility limits (read once)

1. **Nightlies move:** AMD PyTorch nightly index and vLLM `rocm721` nightlies can publish new builds any day. **`freeze_post_install.txt`** is the **ground truth** for what was installed; **`stack_manifest.json`** summarizes it.
2. **HF models** used in decode smoke are not “pinned” by this repo; for strict paper repro, record **`ROCM72_QUALIFY_MODEL`** revision or cache model dirs.
3. **ROCm patch:** This lock is on **7.2.0** + **`pip_owned_vllm_721`**. A **7.2.1** image can use **`strict_uv_vllm_721`** instead; that is a **different** `stack_id` — document a new lock file if you switch.

---

## 8. “Green” checklist before publishing benchmark numbers

- [ ] `source "$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/activate"`
- [ ] `python -c "import torch; assert torch.version.hip; assert torch.cuda.is_available(); assert 'MI300' in torch.cuda.get_device_name(0).upper()"`
- [ ] `python -c "import vllm; print(vllm.__version__)"`
- [ ] `test -f "$REPO/.benchmark_mi300_vllm_frozen/qualification_report.json" && jq -e '.benchmark_ready == true' "$REPO/.benchmark_mi300_vllm_frozen/qualification_report.json"`

If any step fails, do **not** treat results as comparable to the locked snapshot until fixed and re-qualified.
