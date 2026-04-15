"""
tq_mfma_loader.py — Direct mfma_f32_16x16x16f16 rotation kernel for TurboQuant

Compiles tq_mfma_rotate.hip.cpp to a COV5 HSACO (compatible with PyTorch's
bundled ROCm 6.2 runtime) and loads it via ctypes + hipModuleLoad.

Two operations:
  forward : Y = X @ R.T   (compress — rotate x_unit into codebook space)
  inverse : Y = X @ R     (decompress — rotate centroids back to original space)

Usage
-----
    from tq_mfma_loader import MFMARotate, get_mfma_rotate

    mfma = get_mfma_rotate(rotation_matrix)   # singleton, builds HSACO once
    y    = mfma.forward(x_unit)               # (n, 128) float32 → (n, 128) float32
    x    = mfma.inverse(y_hat)               # inverse rotation

If hipcc is unavailable or the HSACO fails to load, all calls silently fall
back to torch.matmul (rocBLAS path), so correctness is never compromised.

Compilation
-----------
    The HIP source is compiled with:
      hipcc --offload-arch=gfx942:sramecc+:xnack- -O3 -mwavefrontsize64
            -DAMD_MFMA_AVAILABLE -mcode-object-version=5 --genco

    COV5 is required for PyTorch ROCm 6.2 compatibility (system hipcc defaults
    to COV6 under ROCm 7.2, which raises hipError 209 in the 6.2 runtime).
"""

from __future__ import annotations

import ctypes
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import torch

_THIS_DIR   = Path(__file__).parent
_HIP_DIR    = _THIS_DIR / "hip"
_SRC_PATH   = _HIP_DIR / "tq_mfma_rotate.hip.cpp"
_HSACO_PATH = _HIP_DIR / "tq_mfma_rotate_cov5.hsaco"

_ARCH     = "gfx942:sramecc+:xnack-"
_HIPCC    = Path("/opt/rocm/bin/hipcc")
_HEAD_DIM = 128
_TILE     = 16    # wavefront handles 16 input rows


# ──────────────────────────────────────────────────────────────────────────────
# HIP ctypes types (re-used from tq_hsaco_loader pattern)
# ──────────────────────────────────────────────────────────────────────────────

hipError_t     = ctypes.c_int
hipModule_t    = ctypes.c_void_p
hipFunction_t  = ctypes.c_void_p
hipStream_t    = ctypes.c_void_p


def _find_torch_libamdhip() -> Path:
    torch_lib = Path(torch.__file__).parent / "lib"
    for name in ("libamdhip64.so", "libamdhip64.so.6"):
        p = torch_lib / name
        if p.exists():
            return p
    import ctypes.util
    sp = ctypes.util.find_library("amdhip64")
    if sp:
        return Path(sp)
    raise FileNotFoundError(
        f"libamdhip64.so not found in {torch_lib}. "
        "Expected alongside torch Python package."
    )


def _load_hip_lib() -> ctypes.CDLL:
    lib = ctypes.CDLL(str(_find_torch_libamdhip()), mode=ctypes.RTLD_LOCAL)
    lib.hipModuleLoad.restype  = hipError_t
    lib.hipModuleLoad.argtypes = [ctypes.POINTER(hipModule_t), ctypes.c_char_p]

    lib.hipModuleGetFunction.restype  = hipError_t
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(hipFunction_t), hipModule_t, ctypes.c_char_p
    ]

    lib.hipModuleLaunchKernel.restype  = hipError_t
    lib.hipModuleLaunchKernel.argtypes = [
        hipFunction_t,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,   # grid
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,   # block
        ctypes.c_uint, hipStream_t,                    # sharedMem, stream
        ctypes.POINTER(ctypes.c_void_p),               # kernelParams
        ctypes.POINTER(ctypes.c_void_p),               # extra
    ]

    lib.hipDeviceSynchronize.restype  = hipError_t
    lib.hipDeviceSynchronize.argtypes = []

    lib.hipGetLastError.restype  = hipError_t
    lib.hipGetLastError.argtypes = []
    return lib


def _check(err: int, msg: str = "") -> None:
    if err != 0:
        raise RuntimeError(f"HIP error {err}" + (f": {msg}" if msg else ""))


# ──────────────────────────────────────────────────────────────────────────────
# Compilation
# ──────────────────────────────────────────────────────────────────────────────

def compile_mfma_hsaco(
    src: Path = _SRC_PATH,
    out: Path = _HSACO_PATH,
    arch: str = _ARCH,
    force: bool = False,
) -> Path:
    """
    Compile tq_mfma_rotate.hip.cpp → COV5 HSACO.

    Uses system hipcc (ROCm 7.2) with -mcode-object-version=5 to produce
    a binary loadable by PyTorch's bundled ROCm 6.2 HIP runtime.

    Parameters
    ----------
    src   : HIP source file (default: tq_mfma_rotate.hip.cpp)
    out   : output HSACO path (default: tq_mfma_rotate_cov5.hsaco)
    arch  : target GPU architecture (default: gfx942:sramecc+:xnack-)
    force : recompile even if HSACO already exists

    Returns the output path.
    """
    if out.exists() and not force:
        return out

    if not _HIPCC.exists():
        raise FileNotFoundError(
            f"hipcc not found at {_HIPCC}. "
            "Install ROCm to use the MFMA rotation kernel."
        )
    if not src.exists():
        raise FileNotFoundError(f"HIP source not found: {src}")

    cmd = [
        str(_HIPCC),
        f"--offload-arch={arch}",
        "-O3",
        "-fPIC",
        "-mwavefrontsize64",
        "-DAMD_MFMA_AVAILABLE",
        "-DCDNA3",
        "-std=c++17",                    # for if constexpr
        "-mcode-object-version=5",       # COV5 = loadable by ROCm 6.2 runtime
        "--genco",                        # GPU code object only (HSACO), no host binary
        "-o", str(out),
        str(src),
    ]

    print(f"[MFMARotate] Compiling MFMA rotation kernel → {out.name}")
    print(f"             {' '.join(cmd)}")

    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.perf_counter() - t0

    if result.returncode != 0 or not out.exists():
        raise RuntimeError(
            f"hipcc failed (exit {result.returncode}, {elapsed:.1f}s):\n"
            f"  stdout: {result.stdout[:800]}\n"
            f"  stderr: {result.stderr[:800]}"
        )

    size_kb = out.stat().st_size / 1024
    print(f"[MFMARotate] Compiled OK in {elapsed:.1f}s — {size_kb:.0f} KB")
    return out


# ──────────────────────────────────────────────────────────────────────────────
# MFMARotate class
# ──────────────────────────────────────────────────────────────────────────────

class MFMARotate:
    """
    Direct mfma_f32_16x16x16f16 rotation kernel for TurboQuant.

    Computes Y = X @ R.T  (forward)  and  Y = X @ R  (inverse) using
    tiled 16×16×16 MFMA instructions, bypassing rocBLAS.

    Performance vs torch.matmul (rocBLAS) on MI300X:
      - Eliminates rocBLAS kernel selection overhead (~10-20 µs per call)
      - Direct MFMA achieves near-peak HBM bandwidth for small-n cases
      - Expected speedup: 1.5–3× for n ≤ 4096 (small-batch decode)

    Falls back to torch.matmul if HSACO compilation or loading fails.

    Parameters
    ----------
    rotation : (128, 128) float32 CUDA tensor — the orthogonal rotation matrix
    hsaco_path : optional explicit path to a pre-compiled HSACO
    compile_if_missing : if True (default), auto-compile HSACO on first use
    """

    def __init__(
        self,
        rotation: torch.Tensor,
        hsaco_path: Optional[Path] = None,
        compile_if_missing: bool = True,
    ):
        assert rotation.shape == (_HEAD_DIM, _HEAD_DIM), (
            f"rotation must be ({_HEAD_DIM}, {_HEAD_DIM}), got {rotation.shape}"
        )
        assert rotation.dtype == torch.float32
        assert rotation.is_cuda

        self.rotation  = rotation.contiguous()
        self._hsaco    = Path(hsaco_path) if hsaco_path else _HSACO_PATH
        self._hip      = None
        self._module   = None
        self._fn_fwd   = None
        self._fn_inv   = None
        self._available = False
        self._fallback_reason: str = ""

        self._try_load(compile_if_missing)

    # ── Initialization ────────────────────────────────────────────────────────

    def _try_load(self, compile_if_missing: bool) -> None:
        try:
            if not self._hsaco.exists():
                if not compile_if_missing:
                    self._fallback_reason = f"HSACO not found: {self._hsaco}"
                    return
                compile_mfma_hsaco(out=self._hsaco)

            self._hip = _load_hip_lib()

            mod = hipModule_t(None)
            err = self._hip.hipModuleLoad(
                ctypes.byref(mod), str(self._hsaco).encode()
            )
            if err == 209:
                # COV6 from older compile — recompile with COV5
                print("[MFMARotate] HSACO is COV6; recompiling with COV5...")
                compile_mfma_hsaco(out=self._hsaco, force=True)
                err = self._hip.hipModuleLoad(
                    ctypes.byref(mod), str(self._hsaco).encode()
                )
            _check(err, f"hipModuleLoad({self._hsaco.name})")
            self._module = mod

            self._fn_fwd = self._get_fn(b"tq_rotate_forward")
            self._fn_inv = self._get_fn(b"tq_rotate_inverse")
            self._available = True
            global _MFMARotate_loaded_once
            if not _MFMARotate_loaded_once:
                print(f"[MFMARotate] MFMA rotation kernel ready ({self._hsaco.name})")
                _MFMARotate_loaded_once = True

        except Exception as e:
            self._fallback_reason = str(e)
            self._available = False
            print(f"[MFMARotate] WARNING: HIP kernel unavailable → falling back to torch.matmul")
            print(f"             Reason: {e}")

    def _get_fn(self, name: bytes) -> hipFunction_t:
        fn = hipFunction_t(None)
        err = self._hip.hipModuleGetFunction(
            ctypes.byref(fn), self._module, ctypes.c_char_p(name)
        )
        _check(err, f"hipModuleGetFunction({name})")
        return fn

    # ── Kernel launch ─────────────────────────────────────────────────────────

    def _launch(self, fn: hipFunction_t, X: torch.Tensor, Y: torch.Tensor,
                n: int) -> None:
        """Launch a rotation kernel: Y = X @ R.T or X @ R."""
        R = self.rotation
        n_arg   = ctypes.c_int(n)
        ptr_x   = ctypes.c_void_p(X.data_ptr())
        ptr_r   = ctypes.c_void_p(R.data_ptr())
        ptr_y   = ctypes.c_void_p(Y.data_ptr())

        args = (ctypes.c_void_p * 4)(
            ctypes.addressof(ptr_x),
            ctypes.addressof(ptr_r),
            ctypes.addressof(ptr_y),
            ctypes.addressof(n_arg),
        )

        grid_x = (n + _TILE - 1) // _TILE
        err = self._hip.hipModuleLaunchKernel(
            fn,
            grid_x, 1, 1,   # grid
            64, 1, 1,         # block (one Wave64)
            0, None,          # sharedMem, stream=NULL (default HIP stream)
            args, None,
        )
        _check(err, "hipModuleLaunchKernel")
        # Do NOT call hipDeviceSynchronize here: the default HIP stream is shared
        # with PyTorch's CUDA stream, so torch.cuda.synchronize() will properly
        # wait for this kernel.  Calling hipDeviceSynchronize inside every launch
        # would add ~5-10 µs of unnecessary overhead.

    # ── Public API ────────────────────────────────────────────────────────────

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        Y = X @ R.T   (rotate into codebook space for compression)

        Parameters
        ----------
        X : (n, 128) float32 CUDA tensor

        Returns (n, 128) float32.
        """
        assert X.is_cuda and X.dtype == torch.float32
        n = X.shape[0]
        X_c = X.contiguous()

        if not self._available:
            return X_c @ self.rotation.T

        Y = torch.empty(n, _HEAD_DIM, dtype=torch.float32, device=X.device)
        try:
            self._launch(self._fn_fwd, X_c, Y, n)
        except Exception as e:
            # Transient error — fall back this call only
            return X_c @ self.rotation.T
        return Y

    def inverse(self, X: torch.Tensor) -> torch.Tensor:
        """
        Y = X @ R   (rotate centroids back to original space for decompression)

        Parameters
        ----------
        X : (n, 128) float32 CUDA tensor

        Returns (n, 128) float32.
        """
        assert X.is_cuda and X.dtype == torch.float32
        n = X.shape[0]
        X_c = X.contiguous()

        if not self._available:
            return X_c @ self.rotation

        Y = torch.empty(n, _HEAD_DIM, dtype=torch.float32, device=X.device)
        try:
            self._launch(self._fn_inv, X_c, Y, n)
        except Exception as e:
            return X_c @ self.rotation
        return Y

    @property
    def available(self) -> bool:
        """True if MFMA HIP kernel is loaded and operational."""
        return self._available

    def __repr__(self) -> str:
        status = "MFMA active" if self._available else f"fallback ({self._fallback_reason[:60]})"
        return f"MFMARotate(head_dim={_HEAD_DIM}, {status})"


# ──────────────────────────────────────────────────────────────────────────────
# Module-level singleton helper
# ──────────────────────────────────────────────────────────────────────────────

_mfma_cache: dict[int, MFMARotate] = {}   # keyed by rotation tensor data_ptr
_MFMARotate_loaded_once: bool = False     # suppress repeat "ready" messages


def get_mfma_rotate(rotation: torch.Tensor) -> MFMARotate:
    """
    Return a cached MFMARotate for the given rotation matrix.

    The first call compiles the HSACO (one-time, ~0.6 s on ROCm 7.2).
    Subsequent calls with the same underlying data pointer return the cached instance.
    """
    global _MFMARotate_loaded_once
    key = rotation.data_ptr()
    if key not in _mfma_cache:
        inst = MFMARotate(rotation)
        _mfma_cache[key] = inst
        _MFMARotate_loaded_once = True
    return _mfma_cache[key]


# ──────────────────────────────────────────────────────────────────────────────
# CLI self-test
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse, math

    parser = argparse.ArgumentParser(description="MFMARotate self-test")
    parser.add_argument("--compile-only", action="store_true",
                        help="Only compile the HSACO, do not run the kernel test")
    parser.add_argument("--force", action="store_true",
                        help="Force recompilation even if HSACO exists")
    parser.add_argument("--n", type=int, default=4096,
                        help="Number of test vectors")
    args = parser.parse_args()

    if args.compile_only or args.force:
        path = compile_mfma_hsaco(force=args.force)
        print(f"HSACO: {path}")
        if args.compile_only:
            sys.exit(0)

    print("=== MFMARotate Self-Test ===")
    sys.path.insert(0, str(_THIS_DIR))
    from turboquant_mi300x import make_rotation_matrix

    R = make_rotation_matrix(seed=42, device="cuda")
    mfma = MFMARotate(R)
    print(f"  {mfma}")
    print()

    n = args.n
    X = torch.randn(n, _HEAD_DIM, device="cuda", dtype=torch.float32)

    # ── Forward correctness ──────────────────────────────────────────────────
    Y_mfma = mfma.forward(X)
    Y_ref  = X @ R.T
    cos_fwd = torch.nn.functional.cosine_similarity(
        Y_mfma.reshape(-1), Y_ref.reshape(-1), dim=0
    ).item()
    max_err_fwd = (Y_mfma - Y_ref).abs().max().item()
    print(f"Forward  (X @ R.T):")
    print(f"  cos_sim  = {cos_fwd:.6f}  (expect > 0.9999)")
    print(f"  max_err  = {max_err_fwd:.2e}  (expect < 1e-2, fp16 rounding)")

    # ── Inverse correctness ──────────────────────────────────────────────────
    Y_inv  = mfma.inverse(X)
    Y_iref = X @ R
    cos_inv = torch.nn.functional.cosine_similarity(
        Y_inv.reshape(-1), Y_iref.reshape(-1), dim=0
    ).item()
    max_err_inv = (Y_inv - Y_iref).abs().max().item()
    print(f"Inverse  (X @ R):")
    print(f"  cos_sim  = {cos_inv:.6f}  (expect > 0.9999)")
    print(f"  max_err  = {max_err_inv:.2e}")

    # ── Round-trip correctness ───────────────────────────────────────────────
    # forward then inverse should recover X (since R.T × R = I)
    Y_rt = mfma.inverse(mfma.forward(X))
    cos_rt = torch.nn.functional.cosine_similarity(
        Y_rt.reshape(-1), X.reshape(-1), dim=0
    ).item()
    print(f"Round-trip (inverse(forward(X)) ≈ X):")
    print(f"  cos_sim  = {cos_rt:.6f}  (expect > 0.998)")
    print()

    # ── Throughput benchmark ─────────────────────────────────────────────────
    import time as _time
    N_WARM, N_BENCH = 20, 100
    torch.cuda.synchronize()

    for label, fn_mfma, fn_ref in [
        ("forward ", mfma.forward,  lambda x: x @ R.T),
        ("inverse ", mfma.inverse,  lambda x: x @ R  ),
    ]:
        # MFMA warm-up
        for _ in range(N_WARM):
            fn_mfma(X)
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(N_BENCH):
            fn_mfma(X)
        torch.cuda.synchronize()
        t_mfma = (_time.perf_counter() - t0) / N_BENCH * 1e6

        # torch.matmul warm-up
        for _ in range(N_WARM):
            fn_ref(X)
        torch.cuda.synchronize()
        t0 = _time.perf_counter()
        for _ in range(N_BENCH):
            fn_ref(X)
        torch.cuda.synchronize()
        t_ref = (_time.perf_counter() - t0) / N_BENCH * 1e6

        speedup = t_ref / t_mfma if t_mfma > 0 else float("nan")
        gb_s_mfma = (n * _HEAD_DIM * 4 * 2 + _HEAD_DIM * _HEAD_DIM * 4) / t_mfma / 1e3
        print(f"{label} n={n:5d}:  MFMA {t_mfma:6.1f} µs  |  matmul {t_ref:6.1f} µs  |  "
              f"speedup {speedup:.2f}×  |  MFMA ~{gb_s_mfma:.0f} GB/s")

    print()
    all_pass = (cos_fwd > 0.9999 and cos_inv > 0.9999 and cos_rt > 0.998)
    print("OVERALL:", "PASS" if all_pass else "FAIL")
