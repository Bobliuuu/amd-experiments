#!/usr/bin/env python3
"""
profile_full_model_decode.py — Full-model decode: attention vs rest (GPU time).

Uses HuggingFace Transformers + torch.profiler (CUDA) on a real Mistral-class
stack. This answers "attention-only microbench vs full step" without vLLM.

Output: JSON with self_cuda_time aggregates by category and top ops.

Optional: wrap the same command with rocprofv2 for kernel names:

  rocprofv2 -d results/rocprof_full_decode_fp16 --kernel-trace \\
    python3 benchmarks/profile_full_model_decode.py --model mistralai/Mistral-7B-v0.1 \\
      --seq-len 8192 --n-decode 8 --batch-size 1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.profiler import ProfilerActivity, profile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _categorize(key: str) -> str:
    k = key.lower()
    if any(
        s in k
        for s in (
            "scaled_dot_product_attention",
            "scaled_dot_product",
            "_scaled_dot_product",
            "flash_attn",
            "flash_attention",
            "efficient_attention",
            "mem_efficient_attention",
            "_fused_sdp_choice",
            "sdp_utils",
            "attn_fwd",
            "attention_forward",
            "fused_attention",
        )
    ):
        return "sdpa_attention"
    if any(s in k for s in ("layer_norm", "native_layer_norm", "rmsnorm", "rms_norm")):
        return "norm"
    if any(s in k for s in ("linear", "addmm", "mm", "bmm", "matmul")):
        return "gemm_matmul"
    # ROCm / hipBLASLt kernel symbols on MI300X
    if "cijk_" in k and "alik" in k and "bljk" in k:
        return "gemm_matmul"
    if "embedding" in k:
        return "embedding"
    if any(s in k for s in ("silu", "gelu", "swiglu", "activation")):
        return "activation"
    return "other"


def _make_prompt_ids(tokenizer, seq_len: int, batch_size: int) -> torch.Tensor:
    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device="cuda")
    ids[:, 0] = tokenizer.bos_token_id or 1
    return ids


def run_profile(
    model_name: str,
    seq_len: int,
    n_decode: int,
    n_warmup: int,
    batch_size: int,
    max_memory_gb: float | None,
) -> Dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    torch.cuda.empty_cache()
    kwargs = dict(
        torch_dtype=torch.float16,
        attn_implementation="sdpa",
    )
    if max_memory_gb is not None and max_memory_gb > 0:
        kwargs["device_map"] = "auto"
        kwargs["max_memory"] = {0: f"{int(max_memory_gb)}GiB", "cpu": "256GiB"}
    else:
        kwargs["device_map"] = "cuda"
    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()

    prompt_ids = _make_prompt_ids(tok, seq_len, batch_size)
    with torch.no_grad():
        torch.cuda.synchronize()
        out = model(prompt_ids, use_cache=True)
    cache = out.past_key_values
    del out
    torch.cuda.synchronize()

    next_token = torch.full(
        (batch_size, 1),
        tok.eos_token_id or 1,
        dtype=torch.long,
        device="cuda",
    )

    for _ in range(n_warmup):
        with torch.no_grad():
            o = model(next_token, past_key_values=cache, use_cache=True)
        cache = o.past_key_values
        next_token = o.logits[:, -1:, :].argmax(dim=-1)
        del o
    torch.cuda.synchronize()

    # Profile only the decode window (fixed n_decode steps from fresh branch optional — reuse cache)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    t_wall0 = time.perf_counter()
    with profile(activities=activities, record_shapes=False, profile_memory=False) as prof:
        for _ in range(n_decode):
            with torch.no_grad():
                o = model(next_token, past_key_values=cache, use_cache=True)
            cache = o.past_key_values
            next_token = o.logits[:, -1:, :].argmax(dim=-1)
            del o
    torch.cuda.synchronize()
    wall_s = time.perf_counter() - t_wall0

    stats = prof.key_averages()
    rows: List[Tuple[str, str, float]] = []
    bucket_cuda_ms: Dict[str, float] = {}
    total_cuda_self = 0.0

    for e in stats:
        key = e.key
        cat = _categorize(key)
        # self_device_time_total is microseconds
        self_cuda_ms = e.self_device_time_total / 1000.0
        if self_cuda_ms <= 0:
            continue
        rows.append((cat, key, self_cuda_ms))
        bucket_cuda_ms[cat] = bucket_cuda_ms.get(cat, 0.0) + self_cuda_ms
        total_cuda_self += self_cuda_ms

    rows.sort(key=lambda x: -x[2])
    top = [{"category": c, "key": k, "self_cuda_ms": round(ms, 4)} for c, k, ms in rows[:40]]

    sdpa_ms = bucket_cuda_ms.get("sdpa_attention", 0.0)
    share = (100.0 * sdpa_ms / total_cuda_self) if total_cuda_self > 0 else 0.0
    if share >= 25.0:
        interp = (
            "Attention (SDPA/flash) is a large fraction of measured CUDA self-time — "
            "KV bandwidth / attention implementation can move end-to-end decode."
        )
    elif share >= 10.0:
        interp = (
            "Attention is a meaningful but not dominant fraction of CUDA self-time — "
            "KV compression may help some; matmul/norm still matter."
        )
    else:
        interp = (
            "Attention is a small fraction of CUDA self-time in this run — "
            "decode is dominated by GEMM/other ops; KV compression alone may not move tok/s much at batch=1."
        )

    return {
        "model": model_name,
        "seq_len": seq_len,
        "n_decode": n_decode,
        "n_warmup": n_warmup,
        "batch_size": batch_size,
        "wall_time_decode_s": round(wall_s, 4),
        "tokens_per_sec": round(n_decode * batch_size / wall_s, 2),
        "profiler_self_cuda_ms_total": round(total_cuda_self, 3),
        "bucket_self_cuda_ms": {k: round(v, 3) for k, v in sorted(bucket_cuda_ms.items(), key=lambda x: -x[1])},
        "sdpa_attention_pct_of_cuda_self": round(share, 2),
        "interpretation": interp,
        "top_ops": top,
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Profile full-model decode: attention vs rest")
    p.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    p.add_argument("--seq-len", type=int, default=8192)
    p.add_argument("--n-decode", type=int, default=8)
    p.add_argument("--n-warmup", type=int, default=2)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--max-memory-gb",
        type=float,
        default=0.0,
        help="If >0, use device_map=auto with this GPU cap (GiB) to avoid OOM on busy cards.",
    )
    p.add_argument(
        "--output",
        type=str,
        default=str(RESULTS / "profile_full_model_decode.json"),
    )
    args = p.parse_args()

    if not torch.cuda.is_available():
        print(json.dumps({"error": "CUDA not available"}))
        sys.exit(1)

    max_mem = args.max_memory_gb if args.max_memory_gb > 0 else None
    out = run_profile(
        args.model,
        args.seq_len,
        args.n_decode,
        args.n_warmup,
        args.batch_size,
        max_mem,
    )
    out["device"] = torch.cuda.get_device_name(0)
    out["torch"] = torch.__version__
    out["hip"] = getattr(torch.version, "hip", None)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in (
        "device", "torch", "hip", "seq_len", "n_decode", "batch_size",
        "tokens_per_sec", "sdpa_attention_pct_of_cuda_self",
        "bucket_self_cuda_ms", "interpretation",
    )}, indent=2))
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
