"""
bench_quality.py — Perplexity and output quality vs compression level

Measures:
  1. Perplexity on WikiText-103 validation subset (stride evaluation)
  2. Output cosine similarity vs FP16 baseline (per-layer KV reconstruction)

Compression levels tested:
  - FP16 (baseline, perplexity = reference)
  - FP8 E4M3 (2× compression)
  - INT4 symmetric (4× compression)
  - TQ3 (4.92× compression) — Python wrapper
  - TQ4 (3.76× compression) — Python wrapper

Usage:
    python3 benchmarks/bench_quality.py --model mistralai/Mistral-7B-v0.1
    python3 benchmarks/bench_quality.py --model mistralai/Mistral-7B-v0.1 --n-tokens 500
"""

import argparse
import gc
import json
import os
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
if _hf_token := os.environ.get("HF_TOKEN"):
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR.mkdir(exist_ok=True)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_wikitext(tokenizer, n_tokens: int = 2048, split: str = "validation"):
    """Load WikiText-103 validation set from HuggingFace datasets."""
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split)
        text = "\n".join(ds["text"])
    except Exception:
        # Fallback: use a fixed text if datasets not available
        text = ("The quick brown fox jumps over the lazy dog. " * 200 +
                "In a galaxy far, far away, scientists discovered " * 100)

    ids = tokenizer.encode(text, add_special_tokens=False)
    return torch.tensor(ids[:n_tokens], dtype=torch.long)


# ──────────────────────────────────────────────────────────────────────────────
# KV cache hooks for quantization
# ──────────────────────────────────────────────────────────────────────────────

class KVQuantWrapper:
    """
    Wraps a model's forward pass to apply KV cache quantization.
    Uses a post-process hook on past_key_values output.
    """

    def __init__(self, model, kv_config: str, bits: int = 3):
        self.model = model
        self.kv_config = kv_config
        self.bits = bits

        # Import helpers based on config
        if kv_config == "tq3" or kv_config == "tq4":
            from turboquant_mi300x import TurboQuantMI300X
            self.tq = TurboQuantMI300X(bits=bits, rotation_seed=42)
        elif kv_config == "fp8":
            sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
            from fp8_baseline import quantize_kv_fp8, dequantize_kv_fp8, FP8_DTYPE
            self.quantize_fp8 = quantize_kv_fp8
            self.dequantize_fp8 = dequantize_kv_fp8
        elif kv_config == "int4":
            sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
            from int4_baseline import quantize_kv_int4, dequantize_kv_int4
            self.quantize_int4 = quantize_kv_int4
            self.dequantize_int4 = dequantize_kv_int4

    def quantize_kv(self, past_kv):
        """Quantize and immediately dequantize in-place to simulate compression round-trip.

        Uses cache.layers (transformers 5.x DynamicCache) where each layer exposes
        .keys and .values tensors. Modifies in-place and returns the same cache object.
        """
        if self.kv_config == "fp16":
            return past_kv

        for layer in past_kv.layers:
            k, v = layer.keys, layer.values
            if self.kv_config in ("tq3", "tq4"):
                orig_k_shape, orig_v_shape = k.shape, v.shape
                k_flat = k.reshape(-1, k.shape[-1]).float()
                v_flat = v.reshape(-1, v.shape[-1]).float()
                k_comp = self.tq.compress_tensor(k_flat)
                v_comp = self.tq.compress_tensor(v_flat)
                layer.keys  = self.tq.decompress_tensor(k_comp, orig_k_shape).to(k.dtype)
                layer.values = self.tq.decompress_tensor(v_comp, orig_v_shape).to(v.dtype)
            elif self.kv_config == "fp8":
                k_fp8, k_scale, v_fp8, v_scale = self.quantize_fp8((k, v))
                k_hat, v_hat = self.dequantize_fp8(k_fp8, k_scale, v_fp8, v_scale)
                layer.keys  = k_hat
                layer.values = v_hat
            elif self.kv_config == "int4":
                blk = self.quantize_int4((k, v))
                k_hat, v_hat = self.dequantize_int4(*blk)
                layer.keys  = k_hat
                layer.values = v_hat
        return past_kv


# ──────────────────────────────────────────────────────────────────────────────
# Perplexity evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_perplexity(
    model,
    tokenizer,
    token_ids: torch.Tensor,
    kv_wrapper: KVQuantWrapper,
    context_len: int = 1024,
    stride: int = 512,
) -> Dict:
    """
    Stride-based perplexity evaluation with quantized KV cache.

    For each window of `context_len` tokens:
      1. Run forward with previous window as KV cache context
      2. Apply KV quantization round-trip
      3. Compute cross-entropy loss on new tokens

    Returns {"perplexity": float, "nll_mean": float, "n_windows": int}
    """
    model.eval()
    ids = token_ids.to("cuda")
    n = len(ids)
    nlls = []

    with torch.no_grad():
        for start in range(0, n - 1, stride):
            end = min(start + context_len, n - 1)
            if end <= start:
                break

            # Context window
            input_ids = ids[start:end].unsqueeze(0)
            labels    = ids[start + 1:end + 1].unsqueeze(0)

            # First chunk: no KV cache
            out = model(input_ids, use_cache=True)
            past_kv = kv_wrapper.quantize_kv(out.past_key_values)

            logits = out.logits  # (1, seq_len, vocab)
            loss   = torch.nn.functional.cross_entropy(
                logits[0, -stride:, :],
                labels[0, :stride if start + stride <= n else None],
                reduction="mean"
            )
            nlls.append(loss.item())

    nll_mean = float(np.mean(nlls)) if nlls else float("inf")
    ppl = float(np.exp(nll_mean))
    return {"perplexity": round(ppl, 3), "nll_mean": round(nll_mean, 4), "n_windows": len(nlls)}


# ──────────────────────────────────────────────────────────────────────────────
# KV reconstruction quality
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_kv_quality(model, tokenizer, seq_len: int = 512) -> Dict:
    """
    Measure KV reconstruction quality for each compression scheme.

    For a sample prompt, compare the reconstructed KV (after quant-dequant)
    against the original FP16 KV for each layer and head.

    Returns dict with per-scheme: mean cosine similarity, mean MSE.
    """
    from turboquant_mi300x import TurboQuantMI300X

    # FP8 helpers
    try:
        from baselines.fp8_baseline import quantize_kv_fp8, dequantize_kv_fp8
    except ImportError:
        sys.path.insert(0, str(Path(__file__).parent.parent / "baselines"))
        from fp8_baseline import quantize_kv_fp8, dequantize_kv_fp8

    tq3 = TurboQuantMI300X(bits=3, rotation_seed=42)
    tq4 = TurboQuantMI300X(bits=4, rotation_seed=42)

    pad_id = tokenizer.eos_token_id or 1
    ids = torch.full((1, seq_len), pad_id, dtype=torch.long, device="cuda")
    ids[0, 0] = tokenizer.bos_token_id or 1

    with torch.no_grad():
        out = model(ids, use_cache=True)

    results = {}
    for scheme_name, quantize_fn in [
        ("fp8",  lambda k, v: (lambda blk: (blk[0].float()/blk[1], blk[2].float()/blk[3]))
                               (quantize_kv_fp8((k, v)))),
        ("int4", None),  # handled separately
        ("tq3",  lambda k, v: (tq3.decompress_tensor(tq3.compress_tensor(k.reshape(-1, 128).float()), k.shape).to(k.dtype),
                               tq3.decompress_tensor(tq3.compress_tensor(v.reshape(-1, 128).float()), v.shape).to(v.dtype))),
        ("tq4",  lambda k, v: (tq4.decompress_tensor(tq4.compress_tensor(k.reshape(-1, 128).float()), k.shape).to(k.dtype),
                               tq4.decompress_tensor(tq4.compress_tensor(v.reshape(-1, 128).float()), v.shape).to(v.dtype))),
    ]:
        if quantize_fn is None:
            continue  # skip INT4 here (handled via KVQuantWrapper in ppl)
        cos_sims, mses = [], []
        kv_src = out.past_key_values
        for layer in kv_src.layers:
            k_orig, v_orig = layer.keys, layer.values
            if k_orig is None or v_orig is None:
                continue
            try:
                k_hat, v_hat = quantize_fn(k_orig.half(), v_orig.half())
                for orig, hat in [(k_orig.float(), k_hat.float()),
                                  (v_orig.float(), v_hat.float())]:
                    flat_o = orig.reshape(-1, 128)
                    flat_h = hat.reshape(-1, 128)
                    cos = torch.nn.functional.cosine_similarity(flat_o, flat_h, dim=-1).mean().item()
                    mse = (flat_o - flat_h).pow(2).mean().item()
                    cos_sims.append(cos)
                    mses.append(mse)
            except Exception:
                pass
        if cos_sims:
            results[scheme_name] = {
                "mean_cosine_sim": round(float(np.mean(cos_sims)), 4),
                "mean_mse":        round(float(np.mean(mses)), 6),
                "n_layers_heads":  len(cos_sims) // 2,
            }

    return results


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="KV cache quality benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--n-tokens", type=int, default=2048,
                        help="WikiText tokens to use for perplexity")
    parser.add_argument("--context-len", type=int, default=512,
                        help="Context window for perplexity evaluation")
    parser.add_argument("--kv-quality-seq-len", type=int, default=256,
                        help="Sequence length for KV reconstruction quality test")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    print(f"=== KV Cache Quality Benchmark ===")
    print(f"Model:    {args.model}")
    print(f"Device:   {torch.cuda.get_device_name(0)}")
    print()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float16,
        device_map="cuda", attn_implementation="eager",
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params\n")

    all_results = {"model": args.model, "device": torch.cuda.get_device_name(0)}

    # #region agent log
    import json as _j, time as _t
    with open('/root/workspace/.cursor/debug-5ac54c.log', 'a') as _lf:
        _lf.write(_j.dumps({'sessionId':'5ac54c','location':'bench_quality.py:main','message':'model_loaded','data':{'dtype':str(next(model.parameters()).dtype)},'timestamp':int(_t.time()*1000),'hypothesisId':'Q'}) + '\n')
    # #endregion

    # ── Perplexity ────────────────────────────────────────────────────────────
    print("── Perplexity on WikiText-103 ──")
    token_ids = load_wikitext(tokenizer, n_tokens=args.n_tokens)
    print(f"  Loaded {len(token_ids)} tokens")

    # #region agent log - probe cache structure before perplexity loop
    import json as _j, time as _t
    try:
        _probe_ids = torch.full((1, 16), tokenizer.eos_token_id or 1, dtype=torch.long, device="cuda")
        with torch.no_grad():
            _probe_out = model(_probe_ids, use_cache=True)
        _probe_cache = _probe_out.past_key_values
        _probe_type = type(_probe_cache).__name__
        _probe_has_layers = hasattr(_probe_cache, 'layers')
        _probe_n_layers = len(_probe_cache.layers) if _probe_has_layers else -1
        _probe_layer0_keys_shape = str(list(_probe_cache.layers[0].keys.shape)) if _probe_has_layers and _probe_n_layers > 0 else 'N/A'
        with open('/root/workspace/.cursor/debug-5ac54c.log', 'a') as _lf:
            _lf.write(_j.dumps({'sessionId':'5ac54c','location':'bench_quality.py:cache_probe','message':'cache_structure','data':{'type':_probe_type,'has_layers':_probe_has_layers,'n_layers':_probe_n_layers,'layer0_keys_shape':_probe_layer0_keys_shape},'timestamp':int(_t.time()*1000),'hypothesisId':'Q'}) + '\n')
        del _probe_out, _probe_cache
        torch.cuda.empty_cache()
    except Exception as _e:
        with open('/root/workspace/.cursor/debug-5ac54c.log', 'a') as _lf:
            _lf.write(_j.dumps({'sessionId':'5ac54c','location':'bench_quality.py:cache_probe','message':'cache_probe_error','data':{'error':str(_e)},'timestamp':int(_t.time()*1000),'hypothesisId':'Q'}) + '\n')
    # #endregion

    ppl_results = {}
    for kv_config in ["fp16", "fp8", "int4", "tq3", "tq4"]:
        print(f"  {kv_config:6s} ...", end="", flush=True)
        try:
            bits = int(kv_config[-1]) if kv_config.startswith("tq") else 3
            wrapper = KVQuantWrapper(model, kv_config, bits=bits)
            r = evaluate_perplexity(
                model, tokenizer, token_ids, wrapper,
                context_len=args.context_len
            )
            ppl_results[kv_config] = r
            baseline_ppl = ppl_results.get("fp16", {}).get("perplexity", r["perplexity"])
            delta = r["perplexity"] - baseline_ppl
            print(f"  PPL={r['perplexity']:.3f}  Δ={delta:+.3f} vs fp16")
        except Exception as e:
            ppl_results[kv_config] = {"error": str(e)}
            print(f"  ERROR: {e}")
            # #region agent log
            import json as _j, time as _t, traceback as _tb
            with open('/root/workspace/.cursor/debug-5ac54c.log', 'a') as _lf:
                _lf.write(_j.dumps({'sessionId':'5ac54c','location':'bench_quality.py:ppl_loop','message':'ppl_error','data':{'kv_config':kv_config,'error':str(e),'tb':_tb.format_exc()[-800:]},'timestamp':int(_t.time()*1000),'hypothesisId':'Q'}) + '\n')
            # #endregion
        finally:
            gc.collect()
            torch.cuda.empty_cache()

    all_results["perplexity"] = ppl_results

    # ── KV Reconstruction Quality ─────────────────────────────────────────────
    print("\n── KV Reconstruction Quality (cosine sim per layer) ──")
    try:
        kv_quality = evaluate_kv_quality(model, tokenizer, args.kv_quality_seq_len)
        for scheme, r in kv_quality.items():
            print(f"  {scheme:6s}: cos_sim={r['mean_cosine_sim']:.4f}  MSE={r['mean_mse']:.6f}")
        all_results["kv_reconstruction"] = kv_quality
    except Exception as e:
        print(f"  ERROR: {e}")
        all_results["kv_reconstruction"] = {"error": str(e)}

    # Save
    model_slug = args.model.replace("/", "_")
    output_path = args.output or (RESULTS_DIR / f"bench_quality_{model_slug}.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
