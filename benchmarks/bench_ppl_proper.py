"""
bench_ppl_proper.py — WikiText-2 Perplexity Benchmark via SDPA Patching

Measures perplexity for KV cache compression methods by globally monkey-patching
torch.nn.functional.scaled_dot_product_attention to intercept K/V tensors,
apply compress→decompress, then call the original SDPA.

Schemes: fp16, fp8, int4, tq3, tq4, tq3_k_only, tq4_k_only

Usage:
    python3 benchmarks/bench_ppl_proper.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --n-tokens 2048 \\
        --schemes fp16 fp8 int4 tq3 tq4
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
# Dataset
# ──────────────────────────────────────────────────────────────────────────────

def load_wikitext2(tokenizer, n_tokens: int = 2048) -> torch.Tensor:
    """
    Load WikiText-2 test split from HuggingFace datasets.
    Tokenizes the full text and truncates to n_tokens.
    Raises RuntimeError if datasets is unavailable — no silent fallback.
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
# SDPA Patcher context manager
# ──────────────────────────────────────────────────────────────────────────────

class SDPAKVPatcher:
    """
    Context manager that monkey-patches torch.nn.functional.scaled_dot_product_attention
    to intercept K/V tensors and apply KV cache compression round-trips.

    Only tensors with head_dim==128 are intercepted (Mistral KV heads).
    All other SDPA calls pass through unmodified.

    Schemes
    -------
    fp16        : no-op, call original SDPA
    tq3 / tq4  : TurboQuantMI300X compress→decompress on both K and V
    fp8         : quantize_kv_fp8 / dequantize_kv_fp8
    int4        : quantize_kv_int4 / dequantize_kv_int4
    tq3_k_only  : TQ3 on K only, V left as-is
    tq4_k_only  : TQ4 on K only, V left as-is
    """

    _HEAD_DIM = 128  # Mistral KV head_dim; used as selector

    def __init__(self, scheme: str, bits: int = 3, use_fused_tq: bool = True):
        self.scheme = scheme
        self.bits = bits
        self.use_fused_tq = use_fused_tq
        self._orig_sdpa = None
        self._tq = None
        self._fp8_available = False
        self._fused_tq_fn = None
        self._fused_tq_pack = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_tq(self, bits: int):
        from turboquant_mi300x import TurboQuantMI300X
        return TurboQuantMI300X(bits=bits, rotation_seed=42)

    def _compress_decompress_tq(self, x: torch.Tensor, tq) -> torch.Tensor:
        """Round-trip a (..., head_dim) float tensor through TurboQuant."""
        orig_shape = x.shape
        orig_dtype = x.dtype
        flat = x.reshape(-1, self._HEAD_DIM).float()
        compressed = tq.compress_tensor(flat)
        reconstructed = tq.decompress_tensor(compressed, flat.shape)
        return reconstructed.reshape(orig_shape).to(orig_dtype)

    def _make_patched_sdpa(self):
        orig_sdpa = self._orig_sdpa
        scheme    = self.scheme
        tq        = self._tq
        head_dim  = self._HEAD_DIM
        use_fused_tq = self.use_fused_tq
        fused_tq_fn = self._fused_tq_fn
        fused_tq_pack = self._fused_tq_pack

        # Capture baseline/fp8/int4 callables at patch time to avoid repeated imports
        fp8_quant   = None
        fp8_dequant = None
        int4_quant  = None
        int4_dequant = None

        if scheme == "fp8":
            try:
                from fp8_baseline import quantize_kv_fp8, dequantize_kv_fp8, FP8_DTYPE
                if FP8_DTYPE is None:
                    raise ImportError("FP8_DTYPE is None — FP8 not available on this platform")
                fp8_quant   = quantize_kv_fp8
                fp8_dequant = dequantize_kv_fp8
            except ImportError as exc:
                warnings.warn(f"[SDPAKVPatcher] scheme='fp8' unavailable: {exc}. Falling back to fp16.")
                scheme = "fp16"

        if scheme == "int4":
            from int4_baseline import quantize_kv_int4, dequantize_kv_int4
            int4_quant   = quantize_kv_int4
            int4_dequant = dequantize_kv_int4

        def _compress_decompress_tq_local(x: torch.Tensor) -> torch.Tensor:
            orig_shape = x.shape
            orig_dtype = x.dtype
            flat = x.reshape(-1, head_dim).float()
            compressed = tq.compress_tensor(flat)
            reconstructed = tq.decompress_tensor(compressed, flat.shape)
            return reconstructed.reshape(orig_shape).to(orig_dtype)

        def patched_sdpa(query, key, value,
                         attn_mask=None, dropout_p=0.0,
                         is_causal=False, scale=None, **kwargs):
            # Only intercept KV tensors with the expected head_dim
            if key.shape[-1] == head_dim:
                if scheme == "fp16":
                    pass  # no-op

                elif scheme in ("tq3", "tq4"):
                    can_use_fused = (
                        use_fused_tq
                        and scheme == "tq3"
                        and fused_tq_fn is not None
                        and fused_tq_pack is not None
                        and attn_mask is None
                        and query.ndim == 4
                        and key.ndim == 4
                        and value.ndim == 4
                        and query.shape[0] == key.shape[0] == value.shape[0]
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
                    key = _compress_decompress_tq_local(key)
                    value = _compress_decompress_tq_local(value)

                elif scheme in ("tq3_k_only", "tq4_k_only"):
                    key = _compress_decompress_tq_local(key)
                    # value left as-is

                elif scheme == "fp8":
                    k_fp8, k_scale, v_fp8, v_scale = fp8_quant((key, value))
                    key, value = fp8_dequant(k_fp8, k_scale, v_fp8, v_scale)

                elif scheme == "int4":
                    block = int4_quant((key, value))
                    key, value = int4_dequant(*block)

            return orig_sdpa(query, key, value,
                             attn_mask=attn_mask, dropout_p=dropout_p,
                             is_causal=is_causal, scale=scale, **kwargs)

        return patched_sdpa

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self):
        # Initialise TQ if needed
        if self.scheme in ("tq3", "tq3_k_only"):
            self._tq = self._load_tq(bits=3)
        elif self.scheme in ("tq4", "tq4_k_only"):
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
        return False  # do not suppress exceptions


# ──────────────────────────────────────────────────────────────────────────────
# PPL computation
# ──────────────────────────────────────────────────────────────────────────────

def compute_ppl(model, token_ids: torch.Tensor, patcher: SDPAKVPatcher,
                context_len: int = 512, device: str = "cuda") -> Dict:
    """
    Compute WikiText-2 perplexity under the given SDPA patcher using strided
    context windows.

    Returns dict with keys: perplexity, nll_mean, n_tokens, n_windows.
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

            input_ids = chunk[:-1].unsqueeze(0)   # (1, seq_len)
            labels    = chunk[1:].unsqueeze(0)     # (1, seq_len)

            out = model(input_ids, use_cache=False)
            logits = out.logits  # (1, seq_len, vocab_size)

            loss = F.cross_entropy(
                logits.reshape(-1, logits.shape[-1]),
                labels.reshape(-1),
                reduction="sum",
            )
            total_nll          += loss.item()
            n_tokens_counted   += labels.numel()
            n_windows          += 1

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
        description="WikiText-2 PPL benchmark for KV cache compression via SDPA patching"
    )
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1",
                        help="HuggingFace model name or local path")
    parser.add_argument("--n-tokens", type=int, default=2048,
                        help="Number of WikiText-2 tokens to evaluate (0 = all)")
    parser.add_argument("--context-len", type=int, default=512,
                        help="Context window size for strided PPL evaluation")
    parser.add_argument("--schemes", nargs="+",
                        default=["fp16", "fp8", "int4", "tq3", "tq4",
                                 "tq3_k_only", "tq4_k_only"],
                        help="Compression schemes to benchmark")
    parser.add_argument(
        "--disable-fused-tq",
        action="store_true",
        help="Force TQ schemes to use compress+decompress path (disable fused Triton path).",
    )
    args = parser.parse_args()

    device = "cuda"

    print(f"\n{'='*70}")
    print(f"WikiText-2 PPL Benchmark (SDPA patching)")
    print(f"Model:       {args.model}")
    print(f"n_tokens:    {args.n_tokens}")
    print(f"context_len: {args.context_len}")
    print(f"schemes:     {args.schemes}")
    print(f"{'='*70}\n")

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

    all_results = []
    fp16_ppl: Optional[float] = None

    header = f"{'Scheme':<16} {'PPL':>10} {'NLL':>10} {'Δ vs FP16':>12} {'n_windows':>10}"
    print(header)
    print("-" * len(header))

    for scheme in args.schemes:
        # Determine bits for TQ schemes
        bits = 3
        if "4" in scheme:
            bits = 4

        patcher = SDPAKVPatcher(
            scheme=scheme,
            bits=bits,
            use_fused_tq=not args.disable_fused_tq,
        )

        try:
            result = compute_ppl(
                model, token_ids, patcher,
                context_len=args.context_len,
                device=device,
            )
        except ImportError as exc:
            warnings.warn(f"[bench_ppl_proper] Skipping scheme '{scheme}': {exc}")
            continue

        ppl = result["perplexity"]

        if scheme == "fp16":
            fp16_ppl = ppl

        delta_str = "—"
        if fp16_ppl is not None and scheme != "fp16":
            delta = ppl - fp16_ppl
            delta_str = f"{delta:+.4f}"

        print(f"  {scheme:<14} {ppl:>10.4f} {result['nll_mean']:>10.6f} "
              f"{delta_str:>12} {result['n_windows']:>10}")

        all_results.append({
            "scheme":     scheme,
            "bits":       bits if scheme != "fp16" else 16,
            **result,
            "delta_ppl":  round(ppl - fp16_ppl, 4) if fp16_ppl and scheme != "fp16" else 0.0,
        })

    print()

    # Save results
    model_slug = args.model.replace("/", "_")
    out_path = RESULTS_DIR / f"bench_ppl_proper_{model_slug}.json"
    payload = {
        "model":       args.model,
        "n_tokens":    args.n_tokens,
        "context_len": args.context_len,
        "fp16_ppl":    fp16_ppl,
        "results":     all_results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
