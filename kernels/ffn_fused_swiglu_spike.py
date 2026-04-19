"""
ffn_fused_swiglu_spike.py — Microbench: separate SiLU×mul vs fused (Triton) SwiGLU gate path.

This is a **Phase B spike** for custom FFN fusion (not wired into vLLM). It measures
whether fusing SiLU(gate)*up for decode-shaped `[M, N]` batches is worth pursuing
after weight-only quantization lands.

  PYTHONPATH=amd-experiments/kernels python3 amd-experiments/kernels/ffn_fused_swiglu_spike.py
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional, Tuple

import torch
import torch.nn.functional as F

# Module-level Triton kernel: nested @triton.jit loses ``tl`` in the compiler on some versions.
try:
    import triton
    import triton.language as tl

    @triton.jit
    def _silu_mul_fused_kernel(
        gate_ptr,
        up_ptr,
        out_ptr,
        n_elements,
        BLOCK: tl.constexpr,
    ):
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(gate_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        u = tl.load(up_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        s = g / (1.0 + tl.exp(-g))
        tl.store(out_ptr + offs, (s * u).to(tl.float16), mask=mask)

    _TRITON_SILU_ERR: Optional[str] = None
except Exception as _triton_exc:  # pragma: no cover
    _silu_mul_fused_kernel = None  # type: ignore[misc, assignment]
    _TRITON_SILU_ERR = repr(_triton_exc)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def torch_separate(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    return F.silu(gate) * up


def _make_triton_kernel() -> Tuple[Optional[Callable[..., None]], str]:
    if _silu_mul_fused_kernel is not None:
        return _silu_mul_fused_kernel, "ok"
    return None, _TRITON_SILU_ERR or "triton unavailable"


def triton_fused(
    gate: torch.Tensor, up: torch.Tensor, silu_mul_fn: Any
) -> torch.Tensor:
    n = gate.numel()
    out = torch.empty_like(gate)
    BLOCK = 1024
    grid = (triton.cdiv(n, BLOCK),)
    silu_mul_fn[grid](gate, up, out, n, BLOCK=BLOCK)
    return out


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    M, H = 64, 14336  # decode-ish batch × Mistral-class FFN intermediate
    gate = torch.randn(M, H, device=device, dtype=torch.float16)
    up = torch.randn(M, H, device=device, dtype=torch.float16)

    silu_mul_fn, msg = _make_triton_kernel()
    ok = silu_mul_fn is not None
    report: dict = {
        "device": device,
        "shape_m_h": [M, H],
        "triton_kernel": ok,
        "triton_import": msg,
    }

    for _ in range(3):
        torch_separate(gate, up)
    _sync()
    t0 = time.perf_counter()
    for _ in range(50):
        y1 = torch_separate(gate, up)
    _sync()
    t_sep = (time.perf_counter() - t0) / 50

    if ok and device == "cuda":
        y2 = triton_fused(gate, up, silu_mul_fn)
        max_err = (y1.float() - y2.float()).abs().max().item()
        report["max_abs_err_vs_torch"] = max_err
        for _ in range(3):
            triton_fused(gate, up, silu_mul_fn)
        _sync()
        t0 = time.perf_counter()
        for _ in range(50):
            triton_fused(gate, up, silu_mul_fn)
        _sync()
        t_f = (time.perf_counter() - t0) / 50
        report["ms_per_iter_torch_separate"] = round(t_sep * 1000, 4)
        report["ms_per_iter_triton_fused"] = round(t_f * 1000, 4)
        report["speedup_triton_vs_torch"] = round(t_sep / t_f, 3) if t_f > 0 else None
    else:
        report["ms_per_iter_torch_separate"] = round(t_sep * 1000, 4)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
