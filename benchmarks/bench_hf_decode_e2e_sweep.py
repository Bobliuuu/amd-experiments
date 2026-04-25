#!/usr/bin/env python3
"""
End-to-end HuggingFace decode sweep: FP16 vs TQ3 *simulated* KV (compress+decompress
each step after cache update) — same protocol as bench_batch_decode.py.

This is NOT vLLM fused Triton TQ (HF never calls that backend). Labels in JSON
are explicit: mode "fp16" vs "tq_sim_roundtrip".

Usage (kernels on PYTHONPATH; do NOT put repo root before site-packages if a
real vLLM install must be imported):

  PYTHONPATH=./kernels python3 benchmarks/bench_hf_decode_e2e_sweep.py
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
KERNELS = ROOT / "kernels"
sys.path.insert(0, str(KERNELS))

from cache_utils import (
    add_swa_args,
    print_swa_status,
    resolve_swa_window,
)


def _load_bench_batch_decode():
    path = ROOT / "benchmarks" / "bench_batch_decode.py"
    spec = importlib.util.spec_from_file_location("_bbd", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--seq-lens", type=int, nargs="+", default=[4096, 8192, 16384, 32768])
    p.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--n-measure", type=int, default=12)
    p.add_argument("--skip-tq", action="store_true")
    p.add_argument("--skip-fp16", action="store_true")
    p.add_argument(
        "--output",
        type=str,
        default=str(ROOT / "results" / "bench_hf_e2e_decode_sweep.json"),
    )
    add_swa_args(p)
    args = p.parse_args()

    if str(KERNELS) not in sys.path:
        sys.path.insert(0, str(KERNELS))

    from dotenv import load_dotenv

    load_dotenv(ROOT / ".env")
    if tok := os.environ.get("HF_TOKEN"):
        from huggingface_hub import login

        login(token=tok, add_to_git_credential=False)

    bbd = _load_bench_batch_decode()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from turboquant_mi300x import TurboQuantMI300X

    print(f"Device: {torch.cuda.get_device_name(0)}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    tq = TurboQuantMI300X(bits=3, rotation_seed=42)

    effective_window = resolve_swa_window(args.swa, model, args.window)
    print_swa_status(args.swa, effective_window)

    rows = []
    for seq_len in args.seq_lens:
        for bs in args.batch_sizes:
            if not args.skip_fp16:
                try:
                    r = bbd.bench_one(
                        model, tok, tq, seq_len, bs, "fp16", args.n_warmup, args.n_measure,
                        swa_window=effective_window,
                    )
                    r["mode_label"] = "hf_fp16_sdpa"
                    rows.append(r)
                    print(
                        f"fp16  seq={seq_len} batch={bs}  tok/s={r['tokens_per_sec']:.2f}  "
                        f"ms/step={r['latency_ms']:.2f}",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"fp16  seq={seq_len} batch={bs}  OOM", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

            if not args.skip_tq:
                try:
                    r = bbd.bench_one(
                        model, tok, tq, seq_len, bs, "tq3", args.n_warmup, args.n_measure,
                        swa_window=effective_window,
                    )
                    r["mode_label"] = "hf_tq_sim_roundtrip_per_step"
                    rows.append(r)
                    print(
                        f"tqsim seq={seq_len} batch={bs}  tok/s={r['tokens_per_sec']:.2f}  "
                        f"ms/step={r['latency_ms']:.2f}",
                        flush=True,
                    )
                except torch.cuda.OutOfMemoryError:
                    print(f"tqsim seq={seq_len} batch={bs}  OOM", flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    break
                finally:
                    gc.collect()
                    torch.cuda.empty_cache()

    out = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "hip": getattr(torch.version, "hip", None),
        "swa": args.swa,
        "swa_window": effective_window,
        "disclaimer": (
            "HF path uses Transformers SDPA on FP16 KV. 'tq_sim' runs TurboQuant "
            "compress+decompress on the DynamicCache after each decode step — not "
            "fused Triton and not vLLM."
        ),
        "rows": rows,
    }
    outp = Path(args.output)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"Wrote {outp}", flush=True)


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    main()
