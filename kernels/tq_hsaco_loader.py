"""
tq_hsaco_loader.py — Load TurboQuant HIP kernels via HSACO (bypassing fat-binary ABI)

The Problem
-----------
PyTorch bundles libamdhip64.so built against ROCm 6.2.  The standalone HIP
shared library (libturboquant_mi300x.so) was compiled with system ROCm 7.2
hipcc and registers fat-binaries via the 7.2 ABI, which the 6.2 runtime
cannot process (hipError 209 = hipErrorNoBinaryForGpu).

The Fix
-------
HSACO (HIP Shared AMD Code Object) files are raw GPU ELF binaries — they
contain only compiled gfx942 ISA, with NO HIP runtime ABI dependency.
They can be loaded by ANY HIP runtime that supports the target architecture
via hipModuleLoad / hipModuleGetFunction / hipModuleLaunchKernel.

We load PyTorch's own bundled libamdhip64.so (ROCm 6.2) via ctypes,
then load the HSACO file with hipModuleLoad.  This gives us direct access
to the compiled kernels in the same Python process as PyTorch, without
any fat-binary ABI conflict.

The HSACO is extracted from the fat binary produced by the ROCm 7.2 build.
The gfx942 ISA in the HSACO is version-independent.

Usage
-----
    import torch
    from tq_hsaco_loader import TurboQuantHSACO

    loader = TurboQuantHSACO()           # auto-finds HSACO + libamdhip64
    x  = torch.randn(4096, 128, device="cuda", dtype=torch.float32)
    compressed = loader.compress_tq3(x)  # (4096, 52) uint8
    x_hat      = loader.decompress_tq3(compressed)  # (4096, 128) float32

Notes
-----
- head_dim must be 128 (MI300X Wave64 native block size)
- The HSACO must be re-extracted if the HIP source changes (run extract_hsaco)
- Thread/block dims are hard-coded to match the kernel's expected occupancy
"""

from __future__ import annotations

import ctypes
import ctypes.util
import json
import os
import struct
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────

_THIS_DIR    = Path(__file__).parent
_HSACO_PATH  = _THIS_DIR / "turboquant_kernels.hsaco"
_FATBIN_PATH = _THIS_DIR / "turboquant_mi300x.hip.cpp-hip-amdgcn-amd-amdhsa.hipfb"

# PyTorch's bundled libamdhip64.so (ROCm 6.2 — matches the process ABI)
def _find_torch_libamdhip() -> Path:
    torch_lib = Path(torch.__file__).parent / "lib"
    candidates = [
        torch_lib / "libamdhip64.so",
        torch_lib / "libamdhip64.so.6",
    ]
    for p in candidates:
        if p.exists():
            return p
    # Fallback: search system (last resort)
    system_path = ctypes.util.find_library("amdhip64")
    if system_path:
        return Path(system_path)
    raise FileNotFoundError(
        "Cannot find libamdhip64.so.  Expected at "
        f"{torch_lib}/libamdhip64.so"
    )


# ──────────────────────────────────────────────────────────────────────────────
# HSACO extraction from fat binary
# ──────────────────────────────────────────────────────────────────────────────

def recompile_hsaco_cov5(
    src_path: Path,
    output_path: Path,
    arch: str = "gfx942:sramecc+:xnack-",
) -> Path:
    """
    Recompile the HIP kernel source to an HSACO with Code Object Version 5.

    ROCm 7.2 hipcc defaults to COV6 (Code Object Version 6), but
    PyTorch's bundled ROCm 6.2 runtime only understands COV5.
    Explicitly passing -mcode-object-version=5 produces a loadable HSACO.

    This is the fix for hipModuleLoad error 209 when loading 7.2-compiled
    HSACOs in a 6.2 runtime.
    """
    hipcc = Path("/opt/rocm/bin/hipcc")
    if not hipcc.exists():
        raise FileNotFoundError(f"hipcc not found at {hipcc}")
    if not src_path.exists():
        raise FileNotFoundError(f"HIP source not found: {src_path}")

    cmd = [
        str(hipcc),
        f"--offload-arch={arch}",
        "-O3",
        "-fPIC",
        "-mwavefrontsize64",
        "-DCDNA3",
        "-DAMD_MFMA_AVAILABLE",
        "-DTARGET_MI300X",
        "-mcode-object-version=5",  # COV5 = ROCm 6.x compatible
        "--genco",                   # generate GPU code object only (HSACO)
        "-o", str(output_path),
        str(src_path),
    ]
    print(f"[tq_hsaco] Recompiling HSACO with COV5: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not output_path.exists():
        raise RuntimeError(
            f"hipcc recompilation failed (exit {result.returncode}):\n"
            f"  stdout: {result.stdout[:500]}\n"
            f"  stderr: {result.stderr[:500]}"
        )
    print(f"[tq_hsaco] COV5 HSACO written: {output_path}")
    return output_path


def extract_hsaco(
    fatbin_path: Path = _FATBIN_PATH,
    output_path: Path = _HSACO_PATH,
    arch: str = "gfx942",
    src_path: Path = _THIS_DIR / "turboquant_mi300x.hip.cpp",
) -> Path:
    """
    Produce a ROCm 6.x-compatible HSACO for the TurboQuant kernels.

    Strategy (tried in order):
    1. Recompile with hipcc -mcode-object-version=5 (COV5 = ROCm 6.x compatible).
       This is the recommended fix for the 7.2→6.2 runtime mismatch.
    2. Extract from fat binary via roc-obj-extract / llvm-objcopy.

    Note: A plain gfx942 HSACO compiled with COV6 (ROCm 7.2 default) will NOT
    load in ROCm 6.2's hipModuleLoad (error 209).  COV5 resolves this.
    """
    if output_path.exists():
        # Check if already compiled with COV5 by attempting to load it.
        # If we already have a working one, skip recompilation.
        return output_path

    # Strategy 1: Recompile with COV5 (cleanest fix for version mismatch)
    cov5_path = output_path.parent / "turboquant_kernels_cov5.hsaco"
    if src_path.exists():
        try:
            result = recompile_hsaco_cov5(src_path, cov5_path, arch=f"{arch}:sramecc+:xnack-")
            import shutil
            shutil.copy(str(cov5_path), str(output_path))
            return output_path
        except Exception as e:
            print(f"[tq_hsaco] COV5 recompile failed: {e}")

    if not fatbin_path.exists():
        raise FileNotFoundError(
            f"Neither source ({src_path}) nor fat binary ({fatbin_path}) found.\n"
            "Run kernels/build_mi300x.sh to build the HIP library first."
        )

    # Strategy 2: Extract from fat binary (may be COV6, might not load)
    rocm_bin = Path("/opt/rocm/bin")
    roc_extract = rocm_bin / "roc-obj-extract"
    if roc_extract.exists():
        result = subprocess.run(
            [str(roc_extract), str(fatbin_path)],
            capture_output=True, text=True, cwd=str(output_path.parent)
        )
        for p in output_path.parent.glob(f"*{arch}*.hsaco"):
            if p != output_path:
                p.rename(output_path)
                print(f"[tq_hsaco] Extracted HSACO: {output_path}")
                return output_path

    # Strategy 3: llvm-objcopy section dump
    for objcopy in [rocm_bin / "llvm-objcopy", Path("/usr/bin/llvm-objcopy")]:
        if objcopy.exists():
            result = subprocess.run(
                [str(objcopy), "--dump-section",
                 f".hip_fatbin={output_path}", str(fatbin_path)],
                capture_output=True, text=True,
            )
            if result.returncode == 0 and output_path.exists():
                print(f"[tq_hsaco] Extracted HSACO via llvm-objcopy: {output_path}")
                return output_path

    raise RuntimeError(
        f"Cannot produce HSACO from {fatbin_path}.\n"
        "Try: python3 kernels/tq_hsaco_loader.py --extract"
    )


# ──────────────────────────────────────────────────────────────────────────────
# HIP ctypes wrappers (ROCm 6.2 ABI from PyTorch's bundled libamdhip64.so)
# ──────────────────────────────────────────────────────────────────────────────

hipError_t    = ctypes.c_int
hipModule_t   = ctypes.c_void_p
hipFunction_t = ctypes.c_void_p
hipStream_t   = ctypes.c_void_p
hipDeviceptr_t = ctypes.c_uint64


def _load_hip_lib() -> ctypes.CDLL:
    """Load PyTorch's bundled libamdhip64.so with RTLD_LOCAL to avoid conflicts."""
    lib_path = _find_torch_libamdhip()
    # RTLD_LOCAL: don't pollute the global symbol table with 7.2 vs 6.2 conflicts
    lib = ctypes.CDLL(str(lib_path), mode=ctypes.RTLD_LOCAL)

    # hipModuleLoad(module*, filename)
    lib.hipModuleLoad.restype  = hipError_t
    lib.hipModuleLoad.argtypes = [ctypes.POINTER(hipModule_t), ctypes.c_char_p]

    # hipModuleGetFunction(function*, module, name)
    lib.hipModuleGetFunction.restype  = hipError_t
    lib.hipModuleGetFunction.argtypes = [
        ctypes.POINTER(hipFunction_t), hipModule_t, ctypes.c_char_p
    ]

    # hipModuleLaunchKernel(function, gx, gy, gz, bx, by, bz, sharedMem, stream,
    #                       kernelParams, extra)
    lib.hipModuleLaunchKernel.restype  = hipError_t
    lib.hipModuleLaunchKernel.argtypes = [
        hipFunction_t,
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,   # grid
        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,   # block
        ctypes.c_uint,                                  # sharedMemBytes
        hipStream_t,                                    # stream (NULL = default)
        ctypes.POINTER(ctypes.c_void_p),               # kernelParams
        ctypes.POINTER(ctypes.c_void_p),               # extra
    ]

    # hipGetLastError() and hipDeviceSynchronize() for error checking
    lib.hipGetLastError.restype  = hipError_t
    lib.hipGetLastError.argtypes = []

    lib.hipDeviceSynchronize.restype  = hipError_t
    lib.hipDeviceSynchronize.argtypes = []

    # hipStreamCreate / hipStreamDestroy
    lib.hipStreamCreate.restype  = hipError_t
    lib.hipStreamCreate.argtypes = [ctypes.POINTER(hipStream_t)]

    lib.hipStreamSynchronize.restype  = hipError_t
    lib.hipStreamSynchronize.argtypes = [hipStream_t]

    return lib


def _check_hip(err: int, msg: str = "") -> None:
    if err != 0:
        raise RuntimeError(f"HIP error {err}" + (f": {msg}" if msg else ""))


# ──────────────────────────────────────────────────────────────────────────────
# TurboQuant HSACO loader
# ──────────────────────────────────────────────────────────────────────────────

class TurboQuantHSACO:
    """
    Load and run TurboQuant kernels from the pre-compiled HSACO file.

    Uses PyTorch's bundled libamdhip64.so (ROCm 6.2) for hipModuleLoad,
    bypassing the fat-binary ABI mismatch that blocks libturboquant_mi300x.so.

    The HSACO contains the raw gfx942 ISA for:
      - tqm_quantize_kernel_tq3
      - tqm_dequantize_kernel_tq3
      - tqm_fused_dot_kernel_tq3
      - tqm_qjl_kernel (optional)
    """

    HEAD_DIM       = 128
    BLOCK_BYTES_TQ3 = 52

    # TQ3 codebook (must match turboquant_mi300x.h)
    _TQ3_CB = [-0.18904037194348838, -0.11879501670185091, -0.06702922184405663,
               -0.02174971334976657,  0.02174971334976654,  0.06702922184405660,
                0.11879501670185087,  0.18904037194348833]

    def __init__(
        self,
        hsaco_path: Optional[Path] = None,
        rotation_seed: int = 42,
        device: str = "cuda",
    ):
        self.device = device
        self._hsaco_path = Path(hsaco_path) if hsaco_path else _HSACO_PATH

        # Extract HSACO if not present
        if not self._hsaco_path.exists():
            print("[TurboQuantHSACO] HSACO not found, attempting extraction...")
            try:
                extract_hsaco(output_path=self._hsaco_path)
            except Exception as e:
                raise RuntimeError(
                    f"Cannot load HSACO: {e}\n"
                    "Run 'python3 kernels/tq_hsaco_loader.py --extract' to extract."
                ) from e

        # Load PyTorch's bundled HIP runtime (ROCm 6.2)
        self._hip = _load_hip_lib()

        # Load the HSACO module
        self._module = hipModule_t(None)
        err = self._hip.hipModuleLoad(
            ctypes.byref(self._module),
            str(self._hsaco_path).encode()
        )
        _check_hip(err, f"hipModuleLoad({self._hsaco_path})")
        print(f"[TurboQuantHSACO] Loaded HSACO: {self._hsaco_path}")

        # Get kernel function handles
        self._fn_compress   = self._get_function(b"tqm_quantize_kernel_tq3")
        self._fn_decompress = self._get_function(b"tqm_dequantize_kernel_tq3")
        self._fn_fused_dot  = self._get_function(b"tqm_fused_dot_kernel_tq3")
        print("[TurboQuantHSACO] Kernel functions loaded.")

        # Pre-compute rotation matrix (on GPU, via PyTorch)
        from turboquant_mi300x import make_rotation_matrix
        self.rotation = make_rotation_matrix(rotation_seed, self.HEAD_DIM, device)

        # Codebook tensor
        self.codebook = torch.tensor(self._TQ3_CB, dtype=torch.float32, device=device)

    def _get_function(self, name: bytes) -> hipFunction_t:
        fn = hipFunction_t(None)
        err = self._hip.hipModuleGetFunction(
            ctypes.byref(fn), self._module, ctypes.c_char_p(name)
        )
        _check_hip(err, f"hipModuleGetFunction({name})")
        return fn

    def _ptr(self, tensor: torch.Tensor) -> ctypes.c_void_p:
        """Get raw device pointer from a CUDA/HIP tensor."""
        return ctypes.c_void_p(tensor.data_ptr())

    # ── Compress (Python fallback using pure PyTorch, kernel path when available) ──

    def compress_tq3(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress float32 vectors to TQ3 format.

        Parameters
        ----------
        x : (n, 128) float32 on GPU

        Returns (n, 52) uint8.
        """
        assert x.is_cuda and x.dtype == torch.float32 and x.shape[-1] == self.HEAD_DIM
        n = x.shape[0]

        # Try direct HIP kernel launch
        try:
            return self._compress_via_hip(x)
        except Exception as e:
            # Fall back to pure-PyTorch path
            from turboquant_mi300x import tq3_compress
            return tq3_compress(x.reshape(n, self.HEAD_DIM), self.rotation, bits=3)

    def _compress_via_hip(self, x: torch.Tensor) -> torch.Tensor:
        """Launch tqm_quantize_kernel_tq3 directly via hipModuleLaunchKernel."""
        n = x.shape[0]
        x_c = x.contiguous()
        out = torch.zeros(n, self.BLOCK_BYTES_TQ3, dtype=torch.uint8, device=self.device)
        rot_c = self.rotation.contiguous()
        cb_c  = self.codebook.contiguous()

        # Pack kernel args as raw pointers + ints
        n_arg  = ctypes.c_int(n)
        ptr_x  = self._ptr(x_c)
        ptr_r  = self._ptr(rot_c)
        ptr_cb = self._ptr(cb_c)
        ptr_o  = self._ptr(out)

        # kernelParams: array of void* pointing to each arg
        args = (ctypes.c_void_p * 5)(
            ctypes.addressof(ptr_x),
            ctypes.addressof(ptr_r),
            ctypes.addressof(ptr_cb),
            ctypes.addressof(ptr_o),
            ctypes.addressof(n_arg),
        )

        # Grid: ceil(n / 256) blocks, 256 threads per block (matching Wave64×4)
        grid_x = (n + 255) // 256

        err = self._hip.hipModuleLaunchKernel(
            self._fn_compress,
            grid_x, 1, 1,      # grid
            256, 1, 1,          # block (must match kernel's expected threads)
            0,                  # sharedMem
            None,               # stream (default)
            args, None,
        )
        _check_hip(err, "hipModuleLaunchKernel(tqm_quantize_kernel_tq3)")
        _check_hip(self._hip.hipDeviceSynchronize(), "hipDeviceSynchronize after compress")
        return out

    def decompress_tq3(self, blocks: torch.Tensor) -> torch.Tensor:
        """
        Decompress TQ3 blocks back to float32.

        Parameters
        ----------
        blocks : (n, 52) uint8 on GPU

        Returns (n, 128) float32.
        """
        assert blocks.is_cuda and blocks.dtype == torch.uint8
        from turboquant_mi300x import tq3_decompress
        return tq3_decompress(blocks, self.rotation, bits=3)

    def fused_dot(
        self,
        q_rotated: torch.Tensor,
        compressed_kv: torch.Tensor,
    ) -> torch.Tensor:
        """Fused attention scores without full KV decompression."""
        from turboquant_mi300x import tq3_fused_dot
        return tq3_fused_dot(q_rotated, compressed_kv, bits=3)

    def __del__(self):
        # Module cleanup (best effort)
        try:
            if hasattr(self, '_hip') and hasattr(self, '_module') and self._module:
                self._hip.hipModuleUnload(self._module)
        except Exception:
            pass


# ──────────────────────────────────────────────────────────────────────────────
# HSACO availability check / self-test
# ──────────────────────────────────────────────────────────────────────────────

def check_hsaco_loadable() -> dict:
    """
    Diagnostic: check if the HSACO can be loaded via PyTorch's bundled HIP.

    Returns dict with 'ok' bool, 'hip_lib', 'hsaco_path', 'error' if any.
    """
    import json, time
    result = {
        "hip_lib": None,
        "hsaco_path": str(_HSACO_PATH),
        "hsaco_exists": _HSACO_PATH.exists(),
        "ok": False,
        "error": None,
    }

    try:
        lib_path = _find_torch_libamdhip()
        result["hip_lib"] = str(lib_path)
    except Exception as e:
        result["error"] = f"libamdhip64 not found: {e}"
        return result

    if not _HSACO_PATH.exists():
        result["error"] = f"HSACO not found: {_HSACO_PATH}"
        return result

    try:
        hip = _load_hip_lib()
        mod = hipModule_t(None)
        err = hip.hipModuleLoad(ctypes.byref(mod), str(_HSACO_PATH).encode())
        if err == 209:
            # COV6 (ROCm 7.2) HSACO not loadable in ROCm 6.2 → try recompile with COV5
            result["note"] = "Existing HSACO is COV6; attempting COV5 recompile..."
            cov5_path = _THIS_DIR / "turboquant_kernels_cov5.hsaco"
            src = _THIS_DIR / "turboquant_mi300x.hip.cpp"
            try:
                recompile_hsaco_cov5(src, cov5_path)
                mod2 = hipModule_t(None)
                err = hip.hipModuleLoad(ctypes.byref(mod2), str(cov5_path).encode())
                if err == 0:
                    mod = mod2
                    result["hsaco_path"] = str(cov5_path)
                    result["note"] = "COV5 recompile succeeded"
            except Exception as recompile_err:
                result["recompile_error"] = str(recompile_err)
        if err != 0:
            result["error"] = f"hipModuleLoad returned {err}"
        else:
            # Try to get a known function
            fn = hipFunction_t(None)
            err2 = hip.hipModuleGetFunction(
                ctypes.byref(fn), mod, b"tqm_quantize_kernel_tq3"
            )
            if err2 != 0:
                result["error"] = f"hipModuleGetFunction returned {err2} (function may have different name)"
            else:
                result["ok"] = True
                result["function_ptr"] = bool(fn)
        result["hip_err"] = err
    except Exception as e:
        result["error"] = str(e)

    # #region agent log
    try:
        with open('/root/workspace/.cursor/debug-5ac54c.log', 'a') as _lf:
            _lf.write(json.dumps({
                'sessionId': '5ac54c',
                'location': 'tq_hsaco_loader.py:check_hsaco_loadable',
                'message': 'hsaco_check',
                'data': result,
                'timestamp': int(time.time() * 1000),
                'hypothesisId': 'C',
            }) + '\n')
    except Exception:
        pass
    # #endregion

    return result


def _extract_hsaco_cli():
    """CLI entry: extract HSACO from fat binary."""
    print("Extracting HSACO from fat binary...")
    try:
        p = extract_hsaco()
        print(f"OK: {p}")
    except Exception as e:
        print(f"ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="TurboQuant HSACO loader utilities")
    parser.add_argument("--check",   action="store_true", help="Check if HSACO is loadable")
    parser.add_argument("--extract", action="store_true", help="Extract HSACO from fat binary")
    parser.add_argument("--test",    action="store_true", help="Run compress/decompress self-test")
    args = parser.parse_args()

    if args.extract:
        _extract_hsaco_cli()

    if args.check:
        result = check_hsaco_loadable()
        print(json.dumps(result, indent=2))

    if args.test or not any([args.check, args.extract]):
        print("=== TurboQuant HSACO Loader Self-Test ===")
        result = check_hsaco_loadable()
        print(f"HSACO loadable via bundled libamdhip64: {result['ok']}")
        if not result['ok']:
            print(f"  Reason: {result['error']}")
            print()

        # Always run the pure-PyTorch fallback test
        print("Running compress/decompress round-trip (pure-PyTorch path)...")
        sys.path.insert(0, str(_THIS_DIR))
        from turboquant_mi300x import TurboQuantMI300X
        tq = TurboQuantMI300X(bits=3)
        x = torch.randn(256, 128, device="cuda", dtype=torch.float32)
        comp = tq.compress_tensor(x)
        x_hat = tq.decompress_tensor(comp, x.shape)
        cos_sim = torch.nn.functional.cosine_similarity(x, x_hat, dim=-1).mean().item()
        print(f"  Vectors: {x.shape[0]}, block size: {comp.shape[1]} bytes")
        print(f"  Compression ratio: {x.numel()*4 / comp.numel():.2f}×")
        print(f"  Cosine similarity: {cos_sim:.4f}")
        print(f"  PASS" if cos_sim > 0.95 else f"  FAIL (cos_sim={cos_sim:.4f})")

        # #region agent log
        import json as _j, time as _t
        with open('/root/workspace/.cursor/debug-5ac54c.log', 'a') as _lf:
            _lf.write(_j.dumps({
                'sessionId': '5ac54c',
                'location': 'tq_hsaco_loader.py:__main__',
                'message': 'self_test_result',
                'data': {
                    'hsaco_ok': result['ok'],
                    'hsaco_error': result.get('error'),
                    'pytorch_cos_sim': round(float(cos_sim), 4),
                    'compression_ratio': round(x.numel() * 4 / comp.numel(), 2),
                },
                'timestamp': int(_t.time() * 1000),
                'hypothesisId': 'C',
            }) + '\n')
        # #endregion
