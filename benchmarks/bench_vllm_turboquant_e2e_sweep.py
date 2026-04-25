#!/usr/bin/env python3
"""
Sweep vLLM end-to-end decode: max_model_len × input_len × concurrent prompts × backends.

Supports optional ``--quant-model`` / ``quant_*`` modes, and
``--max-num-batched-tokens`` / ``--max-num-seqs`` for KV-sensitive / batching sweeps.

Requires installed vLLM + ``scripts/install_turboquant_vllm_backend.sh``.
See docs/vllm_turboquant_wiring.md.

  python3 benchmarks/bench_vllm_turboquant_e2e_sweep.py \\
    --model mistralai/Mistral-7B-v0.1 \\
    --max-model-lens 4096,8192 \\
    --input-lens 512,4096 \\
    --num-prompts-list 1,4,16 \\
    --max-num-batched-tokens 8192 \\
    --output-len 64 --enforce-eager
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"
sys.path.insert(0, str(ROOT / "benchmarks"))
sys.path.insert(0, str(ROOT / "kernels"))

from bench_vllm_turboquant_ab import _make_prompts, run_turboquant_vllm_cell  # noqa: E402
from cache_utils import add_swa_args, print_swa_status, vllm_swa_warn  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--input-len", type=int, default=512)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument(
        "--max-model-lens",
        type=str,
        default="4096,8192",
        help="Comma-separated max_model_len values (must cover input_len + output_len)",
    )
    p.add_argument(
        "--num-prompts-list",
        type=str,
        default="1,2,4,8",
        help="Comma-separated concurrent prompt counts",
    )
    p.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument(
        "--input-lens",
        type=str,
        default="",
        help="Comma-separated input_len values; if empty, use --input-len once.",
    )
    p.add_argument(
        "--max-num-batched-tokens",
        type=int,
        default=0,
        help="If >0, pass through to LLM() when supported (larger effective batch).",
    )
    p.add_argument(
        "--max-num-seqs",
        type=int,
        default=0,
        help="If >0, pass through to LLM() when supported.",
    )
    p.add_argument(
        "--modes",
        type=str,
        default="fp16,turboquant_decompress,turboquant_fused",
        help="Comma-separated: fp16, turboquant_decompress, turboquant_fused, quant_*",
    )
    p.add_argument(
        "--quant-model",
        default="",
        help="HF id for quantized weights (required for quant_* modes).",
    )
    p.add_argument("--quantization", default="awq")
    p.add_argument("--output", type=str, default="")
    add_swa_args(p)
    args = p.parse_args()
    print_swa_status(args.swa, args.window if args.swa == "on" else None)
    sweep_max = max([int(x.strip()) for x in args.max_model_lens.split(",") if x.strip()] or [4096])
    vllm_swa_warn(args.swa, sweep_max)

    max_lens = [int(x.strip()) for x in args.max_model_lens.split(",") if x.strip()]
    nplist = [int(x.strip()) for x in args.num_prompts_list.split(",") if x.strip()]
    modelens = [m.strip() for m in args.modes.split(",") if m.strip()]
    input_lens = (
        [int(x.strip()) for x in args.input_lens.split(",") if x.strip()]
        if args.input_lens.strip()
        else [args.input_len]
    )
    max_bt = args.max_num_batched_tokens if args.max_num_batched_tokens > 0 else None
    max_seqs = args.max_num_seqs if args.max_num_seqs > 0 else None

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    rows = []

    for mlen in max_lens:
        for ilen in input_lens:
            for np in nplist:
                prompts = _make_prompts(tok, np, ilen)
                for mode in modelens:
                    for env_key in ("VLLM_TQ_USE_FUSED_KERNEL",):
                        os.environ.pop(env_key, None)
                    use_tq = mode not in ("fp16", "quant_fp16_kv")
                    fused = mode in ("turboquant_fused", "quant_turboquant_fused")
                    quant = None
                    eff_model = args.model
                    if mode.startswith("quant_"):
                        if not args.quant_model:
                            rows.append(
                                {
                                    "backend": mode,
                                    "error": "quant_* modes need --quant-model",
                                    "sweep_max_model_len": mlen,
                                    "sweep_num_prompts": np,
                                    "sweep_input_len": ilen,
                                }
                            )
                            print(
                                f"SKIP mode={mode} max_len={mlen} prompts={np} in_len={ilen} (no --quant-model)",
                                flush=True,
                            )
                            continue
                        quant = args.quantization
                        eff_model = args.quant_model
                    if use_tq:
                        os.environ["VLLM_TQ_USE_FUSED_KERNEL"] = "1" if fused else "0"
                    try:
                        r = run_turboquant_vllm_cell(
                            mode,
                            eff_model,
                            prompts,
                            args.output_len,
                            args.gpu_memory_utilization,
                            enforce_eager=args.enforce_eager,
                            max_model_len=mlen,
                            use_turboquant=use_tq,
                            fused=fused,
                            quantization=quant,
                            max_num_batched_tokens=max_bt,
                            max_num_seqs=max_seqs,
                        )
                        r["sweep_max_model_len"] = mlen
                        r["sweep_num_prompts"] = np
                        r["sweep_input_len"] = ilen
                        r["sweep_max_num_batched_tokens"] = max_bt
                        r["sweep_max_num_seqs"] = max_seqs
                        rows.append(r)
                        print(
                            f"OK mode={mode} max_len={mlen} in_len={ilen} prompts={np} tok/s={r['throughput_output_tps']}",
                            flush=True,
                        )
                    except Exception as e:
                        rows.append(
                            {
                                "backend": mode,
                                "error": str(e),
                                "sweep_max_model_len": mlen,
                                "sweep_num_prompts": np,
                                "sweep_input_len": ilen,
                            }
                        )
                        print(
                            f"FAIL mode={mode} max_len={mlen} in_len={ilen} prompts={np}: {e}",
                            flush=True,
                        )

    out = {
        "model": args.model,
        "quant_model": args.quant_model or None,
        "quantization": args.quantization,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
        "swa": args.swa,
        "swa_window": args.window if args.swa == "on" else None,
        "rows": rows,
    }
    outp = Path(args.output) if args.output else RESULTS_DIR / "bench_vllm_turboquant_e2e_sweep.json"
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(out, indent=2))
    print(f"Saved → {outp}")


if __name__ == "__main__":
    main()
