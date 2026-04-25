"""
int4_baseline.py — INT4 KV cache baseline benchmark

Measures decode-phase tokens/sec using symmetric per-tensor INT4 KV cache.
INT4 offers 4× memory reduction vs FP16, at significant quality cost.

Strategy:
  - Prefill in FP16
  - Quantize KV cache to INT4 (symmetric, per-head scale)
  - Each decode step: unpack INT4 → FP16 for attention, re-pack after
  - Storage: 2 values per byte (nibble packing)

Compression: 4× vs FP16 (4-bit vs 16-bit per element)
Expected quality: significant accuracy degradation at short contexts

Usage:
    python3 baselines/int4_baseline.py --model mistralai/Mistral-7B-v0.1
"""

import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
if _hf_token := os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "kernels"))
from cache_utils import (
    add_swa_args,
    print_swa_status,
    resolve_swa_window,
    truncate_kv_to_window,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

os.environ.setdefault("PYTORCH_TUNABLEOP_ENABLED", "0")

# ──────────────────────────────────────────────────────────────────────────────
# INT4 quantization helpers (symmetric per-tensor)
# ──────────────────────────────────────────────────────────────────────────────

INT4_MAX = 7  # symmetric INT4: range [-7, 7]


def quantize_int4(t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Quantize float16 tensor to INT4 (stored as uint8 nibble-packed).

    Returns (packed_uint8, scale) where:
      packed = two INT4 values packed into one uint8 byte
      scale  = per-tensor float32 scale factor

    Dequant: float = clamp(packed_nibble, -7, 7).float() / scale
    """
    t_f = t.float()
    amax = t_f.abs().amax().clamp(min=1e-12)
    scale = INT4_MAX / amax                          # scale: map [-amax, amax] → [-7, 7]

    # Quantize to integers in [-7, 7]
    t_int = (t_f * scale).round().clamp(-INT4_MAX, INT4_MAX).to(torch.int8)

    # Offset to [0, 14] for nibble storage, then pack pairs into uint8
    t_offset = (t_int + INT4_MAX).to(torch.uint8)   # [0, 14]

    flat = t_offset.reshape(-1)
    if flat.numel() % 2 != 0:
        flat = torch.cat([flat, flat.new_zeros(1)])  # pad to even length

    # Pack: lo nibble = even index, hi nibble = odd index
    lo = flat[0::2]
    hi = flat[1::2]
    packed = (lo & 0x0F) | ((hi & 0x0F) << 4)       # (numel//2,) uint8

    return packed, scale


def dequantize_int4(
    packed: torch.Tensor,
    scale: torch.Tensor,
    original_shape: torch.Size,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Unpack INT4 bytes and dequantize to float16."""
    lo = (packed & 0x0F).to(torch.int8)              # lower nibble
    hi = ((packed >> 4) & 0x0F).to(torch.int8)      # upper nibble

    flat_interleaved = torch.stack([lo, hi], dim=-1).reshape(-1)
    flat_int = flat_interleaved[:np.prod(original_shape)]

    # Un-offset: [0, 14] → [-7, 7]
    flat_int = flat_int.to(torch.int8) - INT4_MAX

    return (flat_int.float() / scale).to(dtype).reshape(original_shape)


def quantize_kv_int4(
    kv: Tuple[torch.Tensor, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Size, torch.Size]:
    """Quantize a (K, V) tensor pair to nibble-packed INT4."""
    k, v = kv
    k_packed, k_scale = quantize_int4(k)
    v_packed, v_scale = quantize_int4(v)
    return k_packed, k_scale, v_packed, v_scale, k.shape, v.shape


def dequantize_kv_int4(
    k_packed, k_scale, v_packed, v_scale, k_shape, v_shape,
    dtype=torch.float16,
) -> Tuple[torch.Tensor, torch.Tensor]:
    k = dequantize_int4(k_packed, k_scale, k_shape, dtype)
    v = dequantize_int4(v_packed, v_scale, v_shape, dtype)
    return k, v


# ──────────────────────────────────────────────────────────────────────────────
# Benchmark
# ──────────────────────────────────────────────────────────────────────────────

def make_prompt_ids(tokenizer, seq_len: int) -> torch.Tensor:
    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((1, seq_len), pad_id, dtype=torch.long, device="cuda")
    ids[0, 0] = tokenizer.bos_token_id or 1
    return ids


def benchmark_int4_decode(
    model, tokenizer, seq_len: int,
    n_decode: int = 100, n_runs: int = 5,
    swa_window: int | None = None,
) -> Dict:
    def _int4_roundtrip_cache(cache) -> None:
        """Apply INT4 quant-dequant in-place to all layers (transformers 5.x cache.layers)."""
        for layer in cache.layers:
            k, v = layer.keys, layer.values
            k_packed, k_scale = quantize_int4(k.float())
            v_packed, v_scale = quantize_int4(v.float())
            layer.keys  = dequantize_int4(k_packed, k_scale, k.shape, k.dtype)
            layer.values = dequantize_int4(v_packed, v_scale, v.shape, v.dtype)

    model.eval()
    torch.cuda.reset_peak_memory_stats()

    # Prefill in FP16
    prompt_ids = make_prompt_ids(tokenizer, seq_len)
    with torch.no_grad():
        torch.cuda.synchronize()
        t0_prefill = time.perf_counter()
        prefill_out = model(prompt_ids, use_cache=True)
        torch.cuda.synchronize()
        prefill_ms = (time.perf_counter() - t0_prefill) * 1000

    cache = prefill_out.past_key_values

    # Compute compression ratio: INT4 is 4-bit vs 16-bit → 4×
    fp16_bytes = sum(
        layer.keys.numel() * 2 + layer.values.numel() * 2
        for layer in cache.layers
    )
    int4_bytes = fp16_bytes // 4  # nibble-packed: 4 bits per element
    compression_ratio = fp16_bytes / int4_bytes

    # SWA: truncate prefill cache to window before initial INT4 roundtrip
    if swa_window is not None:
        truncate_kv_to_window(cache, swa_window)

    # Apply initial INT4 round-trip (simulate INT4 storage after prefill)
    _int4_roundtrip_cache(cache)
    del prefill_out
    torch.cuda.empty_cache()

    peak_vram = torch.cuda.max_memory_allocated() / 1e9
    times = []

    for _ in range(n_runs):
        next_token = torch.tensor([[tokenizer.eos_token_id or 1]], device="cuda")
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        for _ in range(n_decode):
            with torch.no_grad():
                out = model(next_token, past_key_values=cache, use_cache=True)
            cache = out.past_key_values
            if swa_window is not None:
                truncate_kv_to_window(cache, swa_window)
            _int4_roundtrip_cache(cache)
            next_token = out.logits[:, -1:, :].argmax(dim=-1)

        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    elapsed_s = float(np.median(times))

    return {
        "tokens_per_sec":    round(n_decode / elapsed_s, 2),
        "latency_ms":        round(elapsed_s / n_decode * 1000, 3),
        "vram_peak_gb":      round(peak_vram, 2),
        "prefill_ms":        round(prefill_ms, 1),
        "seq_len":           seq_len,
        "n_decode":          n_decode,
        "kv_dtype":          "int4_symmetric",
        "compression_ratio": round(compression_ratio, 2),
        "n_runs":            n_runs,
        "swa_window":        swa_window,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="INT4 KV cache baseline benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--seq-lens", nargs="+", type=int,
                        default=[512, 2048, 8192, 32768, 65536, 131072])
    parser.add_argument("--n-decode", type=int, default=100)
    parser.add_argument("--n-runs", type=int, default=5)
    parser.add_argument("--max-seq-len", type=int, default=None)
    parser.add_argument("--output", type=str, default=None)
    add_swa_args(parser)
    args = parser.parse_args()

    print(f"=== INT4 KV Cache Baseline Benchmark ===")
    print(f"Model:  {args.model}")
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"INT4 range: symmetric [-{INT4_MAX}, {INT4_MAX}], stored as nibble-packed uint8")
    print(f"Compression: 4× vs FP16")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16,
        device_map="cuda", attn_implementation="sdpa",
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params\n")

    effective_window = resolve_swa_window(args.swa, model, args.window)
    print_swa_status(args.swa, effective_window)

    results = {
        "model":    args.model,
        "device":   {"name": torch.cuda.get_device_name(0)},
        "kv_config": "int4_symmetric",
        "swa":       args.swa,
        "swa_window": effective_window,
        "benchmarks": [],
    }

    for seq_len in sorted(args.seq_lens):
        if args.max_seq_len and seq_len > args.max_seq_len:
            print(f"  seq_len={seq_len:6d}: SKIPPED")
            continue
        print(f"  seq_len={seq_len:6d} ...", end="", flush=True)
        try:
            r = benchmark_int4_decode(model, tokenizer, seq_len,
                                      n_decode=args.n_decode, n_runs=args.n_runs,
                                      swa_window=effective_window)
            results["benchmarks"].append(r)
            print(f" {r['tokens_per_sec']:7.1f} tok/s | "
                  f"latency {r['latency_ms']:.1f} ms | "
                  f"VRAM {r['vram_peak_gb']:.1f} GB | "
                  f"ratio {r['compression_ratio']:.0f}×")
        except torch.cuda.OutOfMemoryError:
            print(f" OOM")
            break
        except Exception as e:
            print(f" ERROR: {e}")
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    model_slug = args.model.replace("/", "_")
    output_path = args.output or (RESULTS_DIR / f"int4_baseline_{model_slug}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
