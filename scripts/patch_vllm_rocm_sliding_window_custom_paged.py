#!/usr/bin/env python3
"""Relax vLLM ROCm ``use_rocm_custom_paged_attention`` sliding-window gate.

Upstream disables the custom paged-attention path whenever ``sliding_window`` is
non-zero, even when ``sliding_window == max_seq_len - 1`` (no effective sliding
window for common Mistral-style configs). That forces the Triton fallback and
can leave decode throughput on the table on MI300X.

This patch is **idempotent** and targets the installed ``vllm`` tree (same
interpreter you use to run vLLM). See ``docs/bottleneck_improvement_mi300.md``.

Run after ``pip install vllm`` (ROCm build). Safe to re-run.
"""
from __future__ import annotations

import pathlib
import sys


MARKER = "_amd_exps_sliding_window_allows_rocm_custom_paged"

HELPER = f'''
def {MARKER}(
    sliding_window: int | tuple[int, int],
    max_seq_len: int,
) -> bool:
    """AMD_EXP_ROCM_SLIDING_PATCH_V1: allow custom paged attn when slide is off or ineffective."""
    if sliding_window in (0, (-1, -1)):
        return True
    if isinstance(sliding_window, int):
        if max_seq_len <= 0:
            return False
        return sliding_window >= max_seq_len - 1
    return False


'''


def main() -> int:
    import vllm

    p = pathlib.Path(vllm.__path__[0]) / "platforms" / "rocm.py"
    text = p.read_text()
    if MARKER in text:
        print(f"[patch_vllm_rocm_sliding_window_custom_paged] already patched: {p}")
        return 0

    anchor = (
        "def on_gfx950() -> bool:\n"
        "    return _ON_GFX950\n\n\n"
        "@cache\n"
        "def use_rocm_custom_paged_attention("
    )
    if anchor not in text:
        print(
            f"[patch_vllm_rocm_sliding_window_custom_paged] ERROR: anchor not found in {p}",
            file=sys.stderr,
        )
        return 2

    text = text.replace(anchor, anchor.replace("\n\n\n@cache", f"\n{HELPER}@cache"), 1)

    old = "(sliding_window == 0 or sliding_window == (-1, -1))"
    count = text.count(old)
    if count != 2:
        print(
            f"[patch_vllm_rocm_sliding_window_custom_paged] ERROR: expected 2 "
            f"occurrences of sliding-window check, found {count} in {p}",
            file=sys.stderr,
        )
        return 2

    text = text.replace(
        old,
        f"{MARKER}(sliding_window, max_seq_len)",
    )
    p.write_text(text)
    print(f"[patch_vllm_rocm_sliding_window_custom_paged] patched: {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
