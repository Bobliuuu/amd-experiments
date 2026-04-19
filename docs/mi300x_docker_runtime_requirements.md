# MI300X + Docker + ROCm — what must be true (and how to debug)

This repo’s **canonical** path is **AMD ROCm 7.2** on **MI300X**, usually via **`rocm/primus`** and **`docker_run_amd_mi300x.sh`**. Benchmarks **require a GPU visible to PyTorch** inside the environment where you run them.

## Requirements (check in order)

### 1. Hardware and host driver

- You are on a machine with **AMD Instinct MI300X** (or equivalent gfx942-class GPU you intend to use).
- **ROCm user-space + kernel driver** are installed on the **host** (`rocm-smi` works on the host outside Docker).

### 2. Device nodes inside the container

PyTorch on ROCm needs the usual device passthrough. This repo’s Docker wrapper already passes:

- `/dev/kfd`
- `/dev/dri`
- `--group-add video`, `--ipc host`, `--privileged`, etc. (aligned with [AMD Primus / PyTorch training Docker](https://rocm.docs.amd.com/en/docs-7.2.1/how-to/rocm-for-ai/training/benchmark-docker/pytorch-training.html))

**Sanity check (run inside the container):**

```bash
ls -l /dev/kfd /dev/dri
rocm-smi --showproductname || rocm-smi
```

If `/dev/kfd` is missing or `rocm-smi` shows no GPU, Docker is not seeing the host GPU (wrong host, wrong runtime, or policy blocking devices).

### 3. PyTorch must be a **ROCm** build (not a CUDA `+cu*` wheel)

Inside the same interpreter you use for benchmarks:

```bash
python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("hip:  ", getattr(torch.version, "hip", None))
print("gpu:  ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("name: ", torch.cuda.get_device_name(0))
PY
```

Interpretation:

| `torch.version.hip` | `torch.__version__` often looks like | Meaning |
|---------------------|--------------------------------------|---------|
| `7.2.x…` (non-empty) | `2.x.x+rocm7.2` (or similar) | **ROCm PyTorch** — good for this repo |
| `None` | `…+cu12…` / CUDA stack | **CUDA PyTorch** — **wrong** for MI300X ROCm workflows |

If you see **`hip=None`** and **`+cu`** in the version string, **`pip install -r requirements.txt`** (or another pip step) may have **replaced** the image’s ROCm torch. Mitigation used here:

- **`docker_run_amd_mi300x.sh`** sets **`AMDEXP_PIP_REQUIREMENTS=0`** by default so `install_and_run.sh` **skips** `pip install -r requirements.txt` in system-Python mode and **preserves** the vendor ROCm torch.
- Set **`AMDEXP_PIP_REQUIREMENTS=1`** only when you intentionally want to refresh Python deps and you know your index will **not** swap in a CUDA torch.

### 4. PyTorch must **see** the GPU (HIP path still uses `torch.cuda.*`)

On ROCm, PyTorch still reports devices through **`torch.cuda.is_available()`** and **`torch.device("cuda:0")`** — that is the **supported API**, not evidence of an NVIDIA stack.

You need:

- **`torch.cuda.is_available()` → `True`**
- A sensible **`torch.cuda.get_device_name(0)`** (e.g. MI300X)

If **`hip`** is set (ROCm build) but **`is_available()`** is **`False`**, the process still has **no usable GPU** (devices, groups, `HIP_VISIBLE_DEVICES`, or running on a host that did not attach the GPU to this container).

### 5. Interpreter selection in Primus

`install_and_run.sh` with **`AMDEXP_USE_SYSTEM_PYTHON=1`** defaults **`AMDEXP_PYTHON`** to **`/opt/venv/bin/python3`** when that path exists, so you match the image’s intended venv.

Override only if you know what you are doing:

```bash
export AMDEXP_PYTHON=/path/to/rocm/python
```

## One-shot diagnostics

From the repo root (host **or** inside the container):

```bash
bash scripts/verify_mi300x_stack.sh
```

It prints device nodes, `rocm-smi` (if present), identity/groups, and the PyTorch lines above.

**Interpreter:** `verify_mi300x_stack.sh` does not assume system `python3` has PyTorch. If **`AMDEXP_PYTHON`** is set, it must be runnable and able to `import torch` (otherwise the script exits with a clear error). Otherwise it tries **`/opt/venv/bin/python3`** (Primus / many ROCm images), then **`<repo>/.benchmark_mi300_vllm_frozen/.venv/bin/python3`**, then **`<repo>/.venv/bin/python3`**, then **`python3` on `PATH`**. The log line `resolved_python:` shows which interpreter was used. Example overrides:

```bash
AMDEXP_PYTHON=~/workspace/amd-experiments/.benchmark_mi300_vllm_frozen/.venv/bin/python3 bash scripts/verify_mi300x_stack.sh
# legacy local venv (not the locked stack):
AMDEXP_PYTHON=~/workspace/amd-experiments/.venv/bin/python3 bash scripts/verify_mi300x_stack.sh
```

### 6. Locked benchmark `uv` venv (vLLM + ROCm torch)

For **vLLM** and benchmark scripts that must match the project’s pinned installer stack, point tools at the **`install_rocm72_uv_torch_vllm.sh`** workdir (default **`<repo>/.benchmark_mi300_vllm_frozen`**):

```bash
export AMDEXP_PYTHON="$REPO/.benchmark_mi300_vllm_frozen/.venv/bin/python"
bash scripts/verify_mi300x_stack.sh
```

Authoritative copy-paste, freeze archives, and “green” **`qualification_report.json`** checks: **`docs/benchmark_mi300_locked_env.md`**.

## About `AMDEXP_ALLOW_NO_GPU=1`

**Not for MI300X production use.** It only exists so broken CI / sandboxes can skip checks. **Benchmarks need a real GPU**; do not use this on a machine where you expect meaningful results.

## Quick references

| Topic | Location |
|--------|-----------|
| Docker wrapper | `docker_run_amd_mi300x.sh` |
| Install + benchmark driver | `install_and_run.sh` |
| Locked benchmark `uv` venv (torch + vLLM) | **`docs/benchmark_mi300_locked_env.md`**, `docs/rocm72_uv_torch_vllm_venv.md`, `scripts/install_rocm72_uv_torch_vllm.sh` |
| vLLM + TurboQuant wiring | `docs/vllm_turboquant_wiring.md` |
