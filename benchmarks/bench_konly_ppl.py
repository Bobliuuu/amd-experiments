"""
bench_konly_ppl.py — K-Only vs K+V Compression PPL & Bandwidth Analysis

Focused benchmark measuring the PPL cost of compressing V alongside K,
versus K-only compression. Helps decide if the bandwidth saved by V
compression is worth the quality penalty.

Schemes evaluated:
  fp16       — baseline, no compression
  tq3        — TQ3 on both K and V  (K+V compressed)
  tq3_k_only — TQ3 on K only, V stored as FP16
  tq4_k_only — TQ4 on K only, V stored as FP16

Bandwidth model (per token, per layer, head_dim=128):
  FP16 baseline    : K=256 B + V=256 B = 512 B/token
  TQ3 K+V          : K=52 B  + V=52 B  = 104 B/token   (compression_ratio ≈ 4.92×)
  TQ3 K-only       : K=52 B  + V=256 B = 308 B/token   (compression_ratio ≈ 1.66×)
  TQ4 K-only       : K=68 B  + V=256 B = 324 B/token   (compression_ratio ≈ 1.58×)

(TQ3: 52 bytes per 128-dim vector; TQ4: 68 bytes per 128-dim vector)

Usage:
    python3 benchmarks/bench_konly_ppl.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --n-tokens 2048 \\
        --context-len 512
"""

import argparse
import json
import math
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn.functional as F
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
if (_hf_token := os.environ.get("HF_TOKEN")) and _hf_token != "your_hf_token_here":
    from huggingface_hub import login
    login(token=_hf_token, add_to_git_credential=False)

KERNELS_DIR   = Path(__file__).parent.parent / "kernels"
BASELINES_DIR = Path(__file__).parent.parent / "baselines"
RESULTS_DIR   = Path(__file__).parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))
sys.path.insert(0, str(BASELINES_DIR))
RESULTS_DIR.mkdir(exist_ok=True)

# ──────────────────────────────────────────────────────────────────────────────
# Bandwidth constants (bytes per token, per layer)
# head_dim=128 → FP16 vector = 256 bytes
# TQ3 compressed vector = 52 bytes
# TQ4 compressed vector = 68 bytes
# ──────────────────────────────────────────────────────────────────────────────

_FP16_BYTES   = 256   # 128 elements × 2 bytes
_TQ3_BYTES    = 52    # TQ3 compressed: (128 * 3 bits) packed → 48 B + 4 B overhead
_TQ4_BYTES    = 68    # TQ4 compressed: (128 * 4 bits) packed → 64 B + 4 B overhead

BANDWIDTH_BYTES: Dict[str, Dict[str, int]] = {
    "fp16": {
        "k_bytes": _FP16_BYTES,
        "v_bytes": _FP16_BYTES,
        "total":   _FP16_BYTES * 2,
    },
    "tq3": {
        "k_bytes": _TQ3_BYTES,
        "v_bytes": _TQ3_BYTES,
        "total":   _TQ3_BYTES * 2,
    },
    "tq3_k_only": {
        "k_bytes": _TQ3_BYTES,
        "v_bytes": _FP16_BYTES,
        "total":   _TQ3_BYTES + _FP16_BYTES,
    },
    "tq4_k_only": {
        "k_bytes": _TQ4_BYTES,
        "v_bytes": _FP16_BYTES,
        "total":   _TQ4_BYTES + _FP16_BYTES,
    },
}


def compression_ratio(scheme: str) -> float:
    """Compression ratio relative to FP16 K+V baseline."""
    baseline = _FP16_BYTES * 2
    scheme_bytes = BANDWIDTH_BYTES.get(scheme, {}).get("total", baseline)
    return round(baseline / scheme_bytes, 4)


# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_wikitext2(tokenizer, n_tokens: int = 2048) -> torch.Tensor:
    """
    Load WikiText-2 test split. Raises RuntimeError if datasets is unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise RuntimeError(
            "The 'datasets' package is required but not installed.\n"
            "Install it with:  pip install datasets"
        )

    try:
        data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    except Exception as exc:
        raise RuntimeError(
            f"Failed to load wikitext-2-raw-v1 from HuggingFace datasets: {exc}\n"
            "Ensure you have internet access or a local cache."
        ) from exc

    text = "\n".join(data["text"])
    token_ids = tokenizer(text, return_tensors="pt")["input_ids"].squeeze(0)

    if n_tokens > 0:
        token_ids = token_ids[:n_tokens]

    print(f"  WikiText-2 test: {token_ids.shape[0]:,} tokens loaded")
    return token_ids


# ──────────────────────────────────────────────────────────────────────────────
# SDPA Patcher
# ──────────────────────────────────────────────────────────────────────────────

class SDPAKVPatcher:
    """
    Context manager that monkey-patches torch.nn.functional.scaled_dot_product_attention
    to apply KV compression round-trips for the given scheme.

    Only intercepts tensors with head_dim==128 (Mistral KV head identifier).

    Schemes
    -------
    fp16        : no-op
    tq3         : TQ3 compress→decompress on both K and V
    tq3_k_only  : TQ3 on K only, V untouched
    tq4_k_only  : TQ4 on K only, V untouched
    """

    _HEAD_DIM = 128

    def __init__(self, scheme: str, use_fused_tq: bool = True):
        valid = {"fp16", "tq3", "tq3_k_only", "tq4_k_only"}
        if scheme not in valid:
            raise ValueError(f"Unknown scheme '{scheme}'. Valid: {valid}")
        self.scheme = scheme
        self.use_fused_tq = use_fused_tq
        self._orig_sdpa = None
        self._tq = None
        self._fused_tq_fn = None
        self._fused_tq_pack = None

    def _load_tq(self, bits: int):
        from turboquant_mi300x import TurboQuantMI300X
        return TurboQuantMI300X(bits=bits, rotation_seed=42)

    def _make_patched_sdpa(self):
        orig_sdpa = self._orig_sdpa
        scheme    = self.scheme
        tq        = self._tq
        head_dim  = self._HEAD_DIM
        use_fused_tq = self.use_fused_tq
        fused_tq_fn = self._fused_tq_fn
        fused_tq_pack = self._fused_tq_pack

        def _tq_roundtrip(x: torch.Tensor) -> torch.Tensor:
            orig_shape = x.shape
            orig_dtype = x.dtype
            flat = x.reshape(-1, head_dim).float()
            compressed = tq.compress_tensor(flat)
            reconstructed = tq.decompress_tensor(compressed, flat.shape)
            return reconstructed.reshape(orig_shape).to(orig_dtype)

        def patched_sdpa(query, key, value,
                         attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, **kwargs):
            if key.shape[-1] == head_dim:
                if scheme == "fp16":
                    pass

                elif scheme == "tq3":
                    can_use_fused = (
                        use_fused_tq
                        and fused_tq_fn is not None
                        and fused_tq_pack is not None
                        and attn_mask is None
                        and query.ndim == 4
                        and key.ndim == 4
                        and value.ndim == 4
                        and query.shape[-1] == key.shape[-1] == value.shape[-1] == head_dim
                        and query.shape[1] == key.shape[1] == value.shape[1]
                        and (dropout_p is None or float(dropout_p) == 0.0)
                        and not bool(is_causal)
                    )
                    if can_use_fused:
                        q_in = query if query.dtype == torch.float16 else query.to(torch.float16)
                        q_rot = tq.rotate_queries(q_in.float()).to(torch.float16)
                        k_planes, k_norms, v_planes, v_norms = fused_tq_pack(key, value, tq)
                        sm_scale = scale if scale is not None else (head_dim ** -0.5)
                        return fused_tq_fn(
                            q_rot, k_planes, k_norms, v_planes, v_norms,
                            rotation=tq.rotation, sm_scale=float(sm_scale),
                        )
                    key = _tq_roundtrip(key)
                    value = _tq_roundtrip(value)

                elif scheme == "tq3_k_only":
                    key = _tq_roundtrip(key)

                elif scheme == "tq4_k_only":
                    key = _tq_roundtrip(key)

            return orig_sdpa(query, key, value,
                             attn_mask=attn_mask, dropout_p=dropout_p,
                             is_causal=is_causal, scale=scale, **kwargs)

        return patched_sdpa

    def __enter__(self):
        if self.scheme == "tq3":
            self._tq = self._load_tq(bits=3)
        elif self.scheme == "tq3_k_only":
            self._tq = self._load_tq(bits=3)
        elif self.scheme == "tq4_k_only":
            self._tq = self._load_tq(bits=4)

        if self.use_fused_tq and self.scheme in ("tq3", "tq3_k_only"):
            try:
                from tq_triton import turboquant_attention_fwd, compress_kv_for_triton
                self._fused_tq_fn = turboquant_attention_fwd
                self._fused_tq_pack = compress_kv_for_triton
            except Exception:
                self._fused_tq_fn = None
                self._fused_tq_pack = None

        self._orig_sdpa = F.scaled_dot_product_attention
        F.scaled_dot_product_attention = self._make_patched_sdpa()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._orig_sdpa is not None:
            F.scaled_dot_product_attention = self._orig_sdpa
            self._orig_sdpa = None
        return False


# ──────────────────────────────────────────────────────────────────────────────
# PPL computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_ppl(model, token_ids: torch.Tensor, patcher: SDPAKVPatcher,
                context_len: int = 512, device: str = "cuda") -> Dict:
    """
    Strided perplexity evaluation with the given SDPA patcher active.
    Uses use_cache=False to avoid DynamicCache API issues.

    Returns dict: perplexity, nll_mean, n_tokens, n_windows.
    """
    model.eval()
    total_nll = 0.0
    n_tokens_counted = 0
    n_windows = 0

    with torch.no_grad(), patcher:
        for start in range(0, token_ids.shape[0] - 1, context_len):
            chunk = token_ids[start : start + context_len + 1].to(device)
            if chunk.shape[0] < 2:
                break

            input_ids = chunk[:-1].unsqueeze(0)
            labels    = chunk[1:].unsqueeze(0)

            out = model(input_ids, use_cache=False)
            logits = out.logits

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll        += loss.item()
            n_tokens_counted += labels.numel()
            n_windows        += 1

    nll_mean   = total_nll / max(n_tokens_counted, 1)
    perplexity = math.exp(nll_mean)

    return {
        "perplexity": round(perplexity, 4),
        "nll_mean":   round(nll_mean, 6),
        "n_tokens":   n_tokens_counted,
        "n_windows":  n_windows,
    }


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="K-only vs K+V compression PPL & bandwidth benchmark"
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model name or local path")
    parser.add_argument("--n-tokens", type=int, default=2048,
                        help="Number of WikiText-2 tokens to evaluate (0 = all)")
    parser.add_argument("--context-len", type=int, default=512,
                        help="Context window size for strided PPL evaluation")
    parser.add_argument(
        "--disable-fused-tq",
        action="store_true",
        help="Force TQ paths to use compress+decompress emulation.",
    )
    args = parser.parse_args()

    schemes = ["fp16", "tq3", "tq3_k_only", "tq4_k_only"]
    device  = "cuda"

    print(f"\n{'='*70}")
    print(f"K-Only vs K+V Compression: PPL & Bandwidth")
    print(f"Model:       {args.model}")
    print(f"n_tokens:    {args.n_tokens}")
    print(f"context_len: {args.context_len}")
    print(f"{'='*70}\n")

    # Bandwidth summary
    print("Bandwidth model (bytes/token/layer, head_dim=128):")
    print(f"  {'Scheme':<16} {'K bytes':>10} {'V bytes':>10} {'Total':>10} {'Ratio vs FP16':>15}")
    print(f"  {'-'*60}")
    for s in schemes:
        bw = BANDWIDTH_BYTES[s]
        ratio = compression_ratio(s)
        print(f"  {s:<16} {bw['k_bytes']:>10} {bw['v_bytes']:>10} "
              f"{bw['total']:>10} {ratio:>14.2f}×")
    print()

    from transformers import AutoTokenizer, AutoModelForCausalLM

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    print("Loading model (dtype=float16, attn_implementation=sdpa)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda",
        attn_implementation="sdpa",
    )
    model.eval()
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model loaded: {param_count/1e9:.2f}B params, dtype={next(model.parameters()).dtype}\n")

    print("Loading WikiText-2...")
    token_ids = load_wikitext2(tokenizer, n_tokens=args.n_tokens)
    print()

    all_results: List[Dict] = []
    fp16_ppl: Optional[float] = None

    header = (f"{'Scheme':<16} {'PPL':>10} {'NLL':>10} "
              f"{'Δ PPL':>10} {'BW Ratio':>10} {'n_windows':>10}")
    print(header)
    print("-" * len(header))

    for scheme in schemes:
        patcher = SDPAKVPatcher(
            scheme=scheme,
            use_fused_tq=not args.disable_fused_tq,
        )

        try:
            result = compute_ppl(
                model, token_ids, patcher,
                context_len=args.context_len,
                device=device,
            )
        except Exception as exc:
            warnings.warn(f"[bench_konly_ppl] Scheme '{scheme}' failed: {exc}")
            continue

        ppl   = result["perplexity"]
        ratio = compression_ratio(scheme)
        bw    = BANDWIDTH_BYTES[scheme]

        if scheme == "fp16":
            fp16_ppl  = ppl
            delta_ppl = 0.0
        else:
            delta_ppl = round(ppl - fp16_ppl, 4) if fp16_ppl is not None else float("nan")

        delta_str = f"{delta_ppl:+.4f}" if scheme != "fp16" else "—"

        print(f"  {scheme:<14} {ppl:>10.4f} {result['nll_mean']:>10.6f} "
              f"{delta_str:>10} {ratio:>9.2f}× {result['n_windows']:>10}")

        all_results.append({
            "scheme":            scheme,
            "perplexity":        ppl,
            "nll_mean":          result["nll_mean"],
            "delta_ppl":         delta_ppl,
            "n_tokens":          result["n_tokens"],
            "n_windows":         result["n_windows"],
            "compression_ratio": ratio,
            "k_bytes_per_token": bw["k_bytes"],
            "v_bytes_per_token": bw["v_bytes"],
            "total_bytes_per_token": bw["total"],
        })

    print()

    # Analysis summary
    if fp16_ppl is not None and len(all_results) >= 2:
        print("Analysis: PPL cost vs bandwidth saved")
        print(f"  {'Scheme':<16} {'ΔPPL':>8} {'BW saved vs FP16':>18} {'PPL/BW tradeoff':>18}")
        print(f"  {'-'*65}")
        fp16_bw = BANDWIDTH_BYTES["fp16"]["total"]
        for r in all_results:
            if r["scheme"] == "fp16":
                continue
            bw_saved_pct = (fp16_bw - r["total_bytes_per_token"]) / fp16_bw * 100
            # PPL degradation per % bandwidth saved (lower = better tradeoff)
            if bw_saved_pct > 0:
                tradeoff = r["delta_ppl"] / bw_saved_pct
            else:
                tradeoff = float("nan")
            print(f"  {r['scheme']:<16} {r['delta_ppl']:>+8.4f} "
                  f"{bw_saved_pct:>17.1f}% "
                  f"{tradeoff:>17.6f}")
        print()

    # Save results
    model_slug = args.model.replace("/", "_")
    out_path = RESULTS_DIR / f"bench_konly_ppl_{model_slug}.json"
    payload = {
        "model":       args.model,
        "n_tokens":    args.n_tokens,
        "context_len": args.context_len,
        "fp16_ppl":    fp16_ppl,
        "bandwidth_model": BANDWIDTH_BYTES,
        "results":     all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
