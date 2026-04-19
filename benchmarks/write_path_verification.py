#!/usr/bin/env python3
"""Write results/path_verification.json — where fused TQ can actually run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    vllm_file = ""
    vllm_kind = "unavailable"
    try:
        import vllm as _v

        vllm_file = str(getattr(_v, "__file__", ""))
        if "site-packages" in vllm_file or "dist-packages" in vllm_file:
            vllm_kind = "pypi_or_system"
        elif "amd-experiments" in vllm_file or "workspace" in vllm_file:
            vllm_kind = "repo_stub_minimal_package"
        else:
            vllm_kind = "unknown"
    except Exception as e:
        vllm_file = repr(e)
        vllm_kind = "import_failed"

    data = {
        "hf_transformers_mistral_decode": {
            "uses_turboquant_rocm_backend": False,
            "attention": "PyTorch SDPA on FP16 DynamicCache (or BF16); no TQ3 fused kernel.",
        },
        "vllm_import_probe": {
            "kind": vllm_kind,
            "vllm_module_file": vllm_file,
            "note": "Real fused-TQ serving needs PyPI vLLM + install script + registry (see docs/vllm_turboquant_wiring.md). Backend source: tq_backends/.",
        },
        "turboquant_fused_gqa_decode": {
            "implementation": "tq_backends/attention/backends/rocm_flash_attn.py::_forward_decode_fused",
            "standalone_validation": "benchmarks/validate_tq_gqa_fused_decode.py",
            "dispatch_logging": "VLLM_TQ_LOG_DISPATCH=1",
        },
    }
    outp = ROOT / "results" / "path_verification.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(data, indent=2))
    print(outp.read_text())


if __name__ == "__main__":
    main()
