"""
bench_runtime_ratio_all_methods.py — Empirical KV cache ratio on real model cache.

This script measures runtime-observed KV compression ratio from a real prefill cache:

  ratio_observed_runtime = fp16_kv_bytes_materialized / compressed_kv_bytes_materialized

It also records the byte-layout ratio for the same method/bitwidth:

  ratio_calculated_layout = fp16_bytes_per_vector / packed_bytes_per_vector

Unlike pure formula tables, this script materializes compressed K/V buffers for
each method directly from the model's cache tensors and reports actual byte counts.
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
import sys
sys.path.insert(0, str(KERNELS_DIR))

from cache_utils import (
    add_swa_args,
    print_swa_status,
    resolve_swa_window,
    truncate_kv_to_window,
)


def make_prompt_ids(tokenizer, seq_len: int, batch_size: int = 1) -> torch.Tensor:
    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((batch_size, seq_len), pad_id, dtype=torch.long, device="cuda")
    ids[:, 0] = tokenizer.bos_token_id or 1
    return ids


def kv_memory_bytes_from_cache(cache) -> int:
    return sum(
        layer.keys.numel() * layer.keys.element_size()
        + layer.values.numel() * layer.values.element_size()
        for layer in cache.layers
    )


def materialize_compressed_bytes(cache, method: str, bits: int) -> int:
    """Materialize compressed K/V tensors and return total materialized bytes."""
    def tensor_bytes(obj) -> int:
        if torch.is_tensor(obj):
            return int(obj.numel() * obj.element_size())
        if isinstance(obj, dict):
            return sum(tensor_bytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(tensor_bytes(v) for v in obj)
        return 0

    if method == "turbo":
        from turboquant_mi300x import TurboQuantMI300X

        q = TurboQuantMI300X(bits=bits, rotation_seed=42)
        total = 0
        for layer in cache.layers:
            head_dim = layer.keys.shape[-1]
            k_flat = layer.keys.reshape(-1, head_dim).float()
            v_flat = layer.values.reshape(-1, head_dim).float()
            k_comp = q.compress_tensor(k_flat)
            v_comp = q.compress_tensor(v_flat)
            total += tensor_bytes(k_comp) + tensor_bytes(v_comp)
        return int(total)

    from block_quant_rocm import make_quantizer

    q = make_quantizer(method, bits=bits, head_dim=cache.layers[0].keys.shape[-1], device="cuda")
    total = 0
    for layer in cache.layers:
        head_dim = layer.keys.shape[-1]
        k_flat = layer.keys.reshape(-1, head_dim).float()
        v_flat = layer.values.reshape(-1, head_dim).float()
        k_comp = q.compress(k_flat)
        v_comp = q.compress(v_flat)
        total += tensor_bytes(k_comp) + tensor_bytes(v_comp)
    return int(total)


def ratio_calculated_layout(bits: int) -> float:
    from turboquant_mi300x import TQ2_BLOCK_BYTES, TQ3_BLOCK_BYTES, TQ4_BLOCK_BYTES

    bpv = {2: TQ2_BLOCK_BYTES, 3: TQ3_BLOCK_BYTES, 4: TQ4_BLOCK_BYTES}[bits]
    return 256.0 / float(bpv)


def main():
    parser = argparse.ArgumentParser(description="Empirical runtime KV compression ratio benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--bits", type=int, default=3)
    parser.add_argument("--methods", nargs="+", default=["turbo", "planar", "iso", "rotor"])
    parser.add_argument("--output", type=str, default="")
    add_swa_args(parser)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()

    effective_window = resolve_swa_window(args.swa, model, args.window)
    print_swa_status(args.swa, effective_window)

    prompt_ids = make_prompt_ids(tokenizer, args.seq_len, args.batch_size)
    with torch.no_grad():
        out = model(prompt_ids, use_cache=True)
    cache = out.past_key_values

    if effective_window is not None:
        truncate_kv_to_window(cache, effective_window)

    fp16_bytes = kv_memory_bytes_from_cache(cache)
    cache_layers = len(cache.layers)
    cache_kv_heads = int(cache.layers[0].keys.shape[1])
    cache_head_dim = int(cache.layers[0].keys.shape[-1])

    rows: List[Dict] = []
    for method in args.methods:
        compressed_bytes = materialize_compressed_bytes(cache, method=method, bits=args.bits)
        observed = float(fp16_bytes) / float(compressed_bytes)
        calc = ratio_calculated_layout(args.bits)
        rows.append(
            {
                "method": method,
                "bits": args.bits,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "cache_layers": cache_layers,
                "cache_kv_heads": cache_kv_heads,
                "cache_head_dim": cache_head_dim,
                "kv_bytes_fp16": int(fp16_bytes),
                "kv_bytes_compressed_materialized": int(compressed_bytes),
                "ratio_calculated_layout": round(calc, 6),
                "ratio_observed_runtime": round(observed, 6),
                "fp16_bytes_per_vector": 256,
            }
        )

    print("\nEmpirical KV ratio (materialized compressed bytes)")
    print(f"{'method':<10} {'calc_ratio':>10} {'obs_ratio':>10} {'fp16_MB':>10} {'comp_MB':>10}")
    print("-" * 60)
    for r in rows:
        print(
            f"{r['method']:<10} "
            f"{r['ratio_calculated_layout']:>10.3f} "
            f"{r['ratio_observed_runtime']:>10.3f} "
            f"{r['kv_bytes_fp16']/1e6:>10.2f} "
            f"{r['kv_bytes_compressed_materialized']/1e6:>10.2f}"
        )

    result = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "bits": args.bits,
        "swa": args.swa,
        "swa_window": effective_window,
        "results": rows,
    }
    out_path = Path(args.output) if args.output else (RESULTS_DIR / "bench_runtime_ratio_all_methods.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
