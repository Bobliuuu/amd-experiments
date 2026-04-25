"""
bench_measured_cache_memory.py — Screenshot-style measured KV cache memory experiment.

This benchmark measures cache memory from REAL materialized tensors:
  - FP16 cache bytes from model prefill cache tensors
  - Compressed cache bytes from actual compressed outputs produced by each method

No layout-only ratio is used for the headline number.
"""

import argparse
import json
import time
from pathlib import Path

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


def tensor_bytes(obj) -> int:
    if torch.is_tensor(obj):
        return int(obj.numel() * obj.element_size())
    if isinstance(obj, dict):
        return sum(tensor_bytes(v) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(tensor_bytes(v) for v in obj)
    return 0


def compress_cache_measured_bytes(cache, method: str, bits: int) -> tuple[int, float]:
    """Return (materialized compressed bytes, elapsed_ms)."""
    t0 = time.perf_counter()
    total = 0

    if method == "turbo":
        from turboquant_mi300x import TurboQuantMI300X

        q = TurboQuantMI300X(bits=bits, rotation_seed=42)
        for layer in cache.layers:
            hd = layer.keys.shape[-1]
            k_comp = q.compress_tensor(layer.keys.reshape(-1, hd).float())
            v_comp = q.compress_tensor(layer.values.reshape(-1, hd).float())
            total += tensor_bytes(k_comp) + tensor_bytes(v_comp)
    else:
        from block_quant_rocm import make_quantizer

        q = make_quantizer(method, bits=bits, head_dim=cache.layers[0].keys.shape[-1], device="cuda")
        for layer in cache.layers:
            hd = layer.keys.shape[-1]
            k_comp = q.compress(layer.keys.reshape(-1, hd).float())
            v_comp = q.compress(layer.values.reshape(-1, hd).float())
            total += tensor_bytes(k_comp) + tensor_bytes(v_comp)

    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return int(total), float(elapsed_ms)


def main():
    parser = argparse.ArgumentParser(description="Measured KV cache memory experiment")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seq-len", type=int, default=512)
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

    with torch.no_grad():
        prompt_ids = make_prompt_ids(tokenizer, args.seq_len, args.batch_size)
        out = model(prompt_ids, use_cache=True)
    cache = out.past_key_values

    if effective_window is not None:
        truncate_kv_to_window(cache, effective_window)

    layers = len(cache.layers)
    kv_heads = int(cache.layers[0].keys.shape[1])
    head_dim = int(cache.layers[0].keys.shape[-1])
    fp16_bytes = kv_memory_bytes_from_cache(cache)

    rows = []
    for method in args.methods:
        comp_bytes, comp_ms = compress_cache_measured_bytes(cache, method=method, bits=args.bits)
        ratio = float(fp16_bytes) / float(comp_bytes)
        label = f"{method}{args.bits}"

        print(f"\nCompressed {layers} layers x {kv_heads} heads in {comp_ms:.1f} ms  [{label}]")
        print("\nKV cache memory:")
        print(f"  FP16 uncompressed:  {fp16_bytes / 1024.0:8.1f} KB")
        print(f"  {label:<16}:  {comp_bytes / 1024.0:8.1f} KB")
        print(f"  Compression ratio:  {ratio:8.3f}x")

        rows.append(
            {
                "method": method,
                "bits": args.bits,
                "label": label,
                "seq_len": args.seq_len,
                "batch_size": args.batch_size,
                "cache_layers": layers,
                "cache_kv_heads": kv_heads,
                "cache_head_dim": head_dim,
                "compress_ms": round(comp_ms, 3),
                "kv_bytes_fp16": int(fp16_bytes),
                "kv_bytes_compressed_measured": int(comp_bytes),
                "compression_ratio_measured": ratio,
            }
        )

    out_obj = {
        "model": args.model,
        "device": torch.cuda.get_device_name(0),
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "bits": args.bits,
        "swa": args.swa,
        "swa_window": effective_window,
        "results": rows,
    }
    out_path = Path(args.output) if args.output else (RESULTS_DIR / "bench_measured_cache_memory.json")
    with open(out_path, "w") as f:
        json.dump(out_obj, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
