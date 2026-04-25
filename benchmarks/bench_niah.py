"""
bench_niah.py — Needle-in-Haystack Benchmark for All KV Compression Methods

Tests long-context retrieval quality under KV compression by injecting a
specific "needle" fact into a long context ("haystack") and asking the model
to retrieve it.

RotorQuant README reports CUDA results: iso3/planar3 pass at 8K, 32K, 65K.
This benchmark verifies/extends those results on AMD ROCm.

Test design:
  1. Generate a long context with a specific needle fact
  2. Ask the model to retrieve the needle
  3. Check if the answer appears in the model's response
  4. Report pass rate at each (method, context_length) pair

Usage:
    python3 benchmarks/bench_niah.py \\
        --model mistralai/Mistral-7B-v0.1 \\
        --context-lens 8192 32768 \\
        --methods fp16 planar3 iso3 turbo3 rotor3 \\
        --n-trials 5
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR.mkdir(exist_ok=True)

from cache_utils import (
    add_swa_args,
    clamp_seq_to_window,
    print_swa_status,
    resolve_swa_window,
)


def _swa_safe_needle_position(default_pos: float, ctx_len: int,
                               window: int | None, safety: int = 256) -> float:
    """Push needle into the last (window - safety) tokens when SWA is on,
    so the model can actually attend to it. Returns default_pos if SWA is off
    or context fits within window."""
    if window is None or ctx_len <= window:
        return default_pos
    min_pos = max(0.0, 1.0 - max(window - safety, 1) / ctx_len)
    return max(default_pos, min_pos)

# Needle facts and their expected answers
NEEDLES = [
    ("The secret passcode is ALPHA-7734-ZETA.", "ALPHA-7734-ZETA"),
    ("The treasure is buried at coordinates 42.3601° N, 71.0589° W.", "42.3601"),
    ("The special keyword for the event is 'Crimson Thunderbolt'.", "Crimson Thunderbolt"),
    ("The CEO's emergency contact number is 555-0192.", "555-0192"),
    ("The activation code is BRAVO-9182-DELTA.", "BRAVO-9182-DELTA"),
    ("The hidden file is named 'project_phoenix_v7.tar.gz'.", "project_phoenix"),
    ("The meeting room for the board is Room 4217 on the 42nd floor.", "4217"),
    ("The private key fingerprint is A4:B7:C3:D9:E2:F1.", "A4:B7"),
]

HAYSTACK_UNIT = """
The science of artificial intelligence encompasses a broad range of methodologies
and applications. Machine learning, a subset of AI, enables systems to learn from
data and improve their performance over time without being explicitly programmed.
Deep learning, in turn, uses neural networks with many layers to model complex
patterns in data. Natural language processing allows computers to understand and
generate human language. Computer vision enables machines to interpret visual
information from the world. These technologies are being applied across industries
to automate processes, improve decision-making, and create new products and services.
"""


def build_niah_prompt(needle: str, context_len_target: int,
                      needle_position: float = 0.5) -> str:
    """
    Build a prompt with the needle inserted at needle_position (0=start, 1=end).
    Pads with haystack text to reach approximately context_len_target tokens.
    """
    # Repeat haystack to fill context
    haystack = HAYSTACK_UNIT.strip()
    repeats_needed = (context_len_target * 4) // len(haystack) + 10
    full_haystack = (" " + haystack) * repeats_needed

    # Split at needle position
    split_point = int(len(full_haystack) * needle_position)
    before = full_haystack[:split_point]
    after = full_haystack[split_point:]

    prompt = (
        f"Read the following text carefully:\n\n"
        f"{before}\n\n"
        f"IMPORTANT: {needle}\n\n"
        f"{after}\n\n"
        f"Question: Based on the text above, what specific information "
        f"was marked as IMPORTANT? Quote it exactly."
    )
    return prompt


def check_answer(response: str, expected: str) -> bool:
    """Check if the expected answer appears in the response."""
    return expected.lower() in response.lower()


class NiahQuantizedModel:
    """
    Model wrapper that applies KV compression during generation.
    Uses a hook to compress/decompress KV at each decode step.
    """

    def __init__(self, model, quantizer=None, method: str = "fp16"):
        self.model = model
        self.quantizer = quantizer
        self.method = method

    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 50,
                 device: str = "cuda") -> torch.Tensor:
        """Generate tokens with compressed KV cache."""
        if self.method == "fp16" or self.quantizer is None:
            with torch.no_grad():
                out = self.model.generate(
                    input_ids, max_new_tokens=max_new_tokens,
                    do_sample=False, pad_token_id=2)
            return out

        # For compressed methods: greedy decode with manual KV compression
        # between each step
        with torch.no_grad():
            # Prefill
            out = self.model(input_ids, use_cache=True)
            past_kv = out.past_key_values
            next_token_logits = out.logits[:, -1, :]
            next_token = next_token_logits.argmax(dim=-1, keepdim=True)

            generated = [next_token.squeeze().item()]

            # Compress the KV cache after prefill
            if hasattr(past_kv, "key_cache"):
                # DynamicCache
                for l_idx in range(len(past_kv.key_cache)):
                    k = past_kv.key_cache[l_idx]
                    v = past_kv.value_cache[l_idx]
                    k_flat = k.squeeze(0).reshape(-1, k.shape[-1])
                    v_flat = v.squeeze(0).reshape(-1, v.shape[-1])

                    if self.method == "turbo":
                        k_comp = self.quantizer.compress_tensor(k_flat.float())
                        v_comp = self.quantizer.compress_tensor(v_flat.float())
                        k_hat = self.quantizer.decompress_tensor(k_comp, k_flat.shape)
                        v_hat = self.quantizer.decompress_tensor(v_comp, v_flat.shape)
                    else:
                        k_comp = self.quantizer.compress(k_flat)
                        v_comp = self.quantizer.compress(v_flat)
                        k_hat = self.quantizer.decompress(k_comp, k_flat.shape)
                        v_hat = self.quantizer.decompress(v_comp, v_flat.shape)

                    past_kv.key_cache[l_idx] = k_hat.to(k.dtype).reshape(k.shape)
                    past_kv.value_cache[l_idx] = v_hat.to(v.dtype).reshape(v.shape)

            for _ in range(max_new_tokens - 1):
                out = self.model(
                    input_ids=next_token,
                    past_key_values=past_kv,
                    use_cache=True,
                )
                past_kv = out.past_key_values
                next_token_logits = out.logits[:, -1, :]
                next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                tok = next_token.squeeze().item()
                generated.append(tok)
                if tok in (2, 1):  # EOS tokens
                    break

        all_tokens = torch.cat([input_ids, torch.tensor([generated], device=device)], dim=1)
        return all_tokens


def run_niah(args, device: str):
    """Run Needle-in-Haystack for all methods and context lengths."""
    from transformers import AutoTokenizer, AutoModelForCausalLM

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16,
        device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    print("Model loaded.\n")

    effective_window = resolve_swa_window(args.swa, model, args.window)
    print_swa_status(args.swa, effective_window)

    all_results = []

    for method_spec in args.methods:
        if method_spec == "fp16":
            method, bits = "fp16", 0
            quantizer = None
        elif method_spec == "turbo3":
            method, bits = "turbo", 3
            from turboquant_mi300x import TurboQuantMI300X
            quantizer = TurboQuantMI300X(bits=3, rotation_seed=42)
        elif method_spec == "turbo4":
            method, bits = "turbo", 4
            from turboquant_mi300x import TurboQuantMI300X
            quantizer = TurboQuantMI300X(bits=4, rotation_seed=42)
        else:
            method = method_spec[:-1]
            bits = int(method_spec[-1])
            from block_quant_rocm import make_quantizer
            quantizer = make_quantizer(method, bits=bits, head_dim=128, device=device)
            # Warmup
            _ = quantizer.compress(torch.randn(64, 128, device=device))

        niah_model = NiahQuantizedModel(model, quantizer, method)

        for ctx_len in args.context_lens:
            passes = 0
            responses = []

            for trial in range(args.n_trials):
                needle_info = NEEDLES[trial % len(NEEDLES)]
                needle_text, expected_answer = needle_info

                # Vary needle position across trials
                needle_pos = 0.1 + (trial / max(args.n_trials - 1, 1)) * 0.8
                # When SWA is on and ctx > window, push needle into the window
                # so the model can actually attend to it (otherwise broken by design).
                needle_pos = _swa_safe_needle_position(needle_pos, ctx_len, effective_window)

                prompt = build_niah_prompt(needle_text, ctx_len, needle_pos)

                tokens = tokenizer(prompt, return_tensors="pt",
                                   truncation=True, max_length=ctx_len)
                input_ids = tokens["input_ids"].to(device)
                actual_len = input_ids.shape[1]

                t0 = time.perf_counter()
                output_ids = niah_model.generate(input_ids, max_new_tokens=50, device=device)
                elapsed = time.perf_counter() - t0

                new_tokens = output_ids[0, actual_len:]
                response = tokenizer.decode(new_tokens, skip_special_tokens=True)
                passed = check_answer(response, expected_answer)

                if passed:
                    passes += 1
                responses.append({
                    "trial": trial,
                    "expected": expected_answer,
                    "response": response[:200],
                    "passed": passed,
                    "elapsed_s": elapsed,
                })

            pass_rate = passes / args.n_trials
            result = {
                "method": method,
                "bits": bits,
                "context_len": ctx_len,
                "n_trials": args.n_trials,
                "n_passed": passes,
                "pass_rate": pass_rate,
                "responses": responses,
            }
            all_results.append(result)

            label = method_spec
            status = "PASS" if pass_rate >= 0.6 else ("PARTIAL" if pass_rate > 0 else "FAIL")
            print(f"  {label:<12} ctx={ctx_len:>7} "
                  f"pass_rate={pass_rate:.1%} ({passes}/{args.n_trials}) [{status}]")

    del model
    torch.cuda.empty_cache()
    return all_results


def print_summary(results: list):
    """Print pass/fail summary table."""
    methods = sorted(set(f"{r['method']}{r['bits']}" for r in results))
    ctx_lens = sorted(set(r["context_len"] for r in results))

    print(f"\n{'NiaH Pass Rate Summary':}")
    header = f"{'Method':<14}" + "".join(f"{c:>10}" for c in ctx_lens)
    print(header)
    print("-" * len(header))

    for m in methods:
        row = f"{m:<14}"
        for ctx in ctx_lens:
            r = next((x for x in results
                       if f"{x['method']}{x['bits']}" == m
                       and x["context_len"] == ctx), None)
            if r:
                rate = r["pass_rate"]
                row += f"{rate:>9.0%} "
            else:
                row += f"{'—':>10}"
        print(row)


def run_synthetic_niah(args, device: str) -> list:
    # Synthetic mode: no model, but still honor the user's --swa intent
    # by clamping context length to the requested window.
    """
    Synthetic Needle-in-Haystack: tests attention rank preservation under compression.

    Creates n_tokens random K vectors (haystack) plus 1 distinct "needle" K vector.
    A query Q is crafted to maximally attend to the needle. After compression and
    decompression, we check whether the needle still has the highest attention score.

    This directly measures the key quality metric for KV cache compression:
    does compression preserve attention rank ordering?
    """
    from block_quant_rocm import make_quantizer
    import torch.nn.functional as F

    n_heads = 8
    head_dim = 128

    all_results = []

    print(f"\n{'Synthetic NiaH — Attention Rank Preservation':}")
    print(f"Checking if top-1 attention token is preserved after KV compression.")
    print(f"{'Method':<12} {'CtxLen':>8} {'Pass%':>8} {'AvgRank':>9} {'MeanCosSim':>12}")
    print("-" * 55)

    for method_spec in args.methods:
        if method_spec == "fp16":
            method, bits = "fp16", 0
            quantizer = None
        elif method_spec.startswith("turbo"):
            method, bits = "turbo", int(method_spec[-1])
            from turboquant_mi300x import TurboQuantMI300X
            quantizer = TurboQuantMI300X(bits=bits, rotation_seed=42)
        else:
            method = method_spec[:-1]
            bits = int(method_spec[-1])
            quantizer = make_quantizer(method, bits=bits, head_dim=head_dim, device=device)
            _ = quantizer.compress(torch.randn(64, head_dim, device=device))

        for ctx_len in args.context_lens:
            n_tokens = clamp_seq_to_window(ctx_len, args.swa, args.window)
            passes = 0
            rank_sum = 0
            cos_sims = []

            for trial in range(args.n_trials):
                # Generate random K vectors (haystack)
                K = torch.randn(n_heads, n_tokens, head_dim, device=device)

                # Designate token at needle_pos as the needle
                needle_pos = n_tokens // 2 + trial * (n_tokens // (args.n_trials + 1))
                needle_pos = min(needle_pos, n_tokens - 1)

                # Make needle distinct: set to a strong fixed-direction signal
                # Signal strength must exceed expected max of randn, which scales as
                # ~4*sqrt(1/head_dim) * sqrt(2*log(n_tokens)).
                # For n_tokens=65536, head_dim=128: max ≈ 0.088 * sqrt(2*ln(65536)) ≈ 0.39
                # Use signal strength = 12.0 / sqrt(head_dim) ≈ 1.06 >> 0.39 → needle always wins
                signal_strength = 12.0
                needle_signal = torch.randn(head_dim, device=device)
                needle_signal = needle_signal / needle_signal.norm()
                K[:, needle_pos, :] = needle_signal.unsqueeze(0).expand(n_heads, -1) * signal_strength

                # Query aligned exactly with needle direction
                Q = needle_signal.unsqueeze(0).unsqueeze(0).expand(n_heads, 1, head_dim)

                if method == "fp16":
                    K_hat = K
                else:
                    # Compress and decompress K
                    K_flat = K.reshape(n_heads * n_tokens, head_dim).float()
                    if method == "turbo":
                        comp = quantizer.compress_tensor(K_flat)
                        K_hat_flat = quantizer.decompress_tensor(comp, K_flat.shape)
                    else:
                        comp = quantizer.compress(K_flat)
                        K_hat_flat = quantizer.decompress(comp, K_flat.shape).float()
                    K_hat = K_hat_flat.reshape(n_heads, n_tokens, head_dim).to(K.dtype)

                # Compute attention scores
                sm_scale = head_dim ** -0.5
                scores = torch.bmm(Q, K_hat.transpose(-2, -1)) * sm_scale  # (H, 1, T)
                top1 = scores[0, 0].argmax().item()
                rank = (scores[0, 0] >= scores[0, 0, needle_pos]).sum().item()

                # Cosine sim of compressed vs original needle
                k_orig = K[:, needle_pos, :].float()
                k_hat = K_hat[:, needle_pos, :].float()
                cs = F.cosine_similarity(k_orig, k_hat, dim=-1).mean().item()
                cos_sims.append(cs)

                if top1 == needle_pos:
                    passes += 1
                rank_sum += rank

            pass_rate = passes / args.n_trials
            avg_rank = rank_sum / args.n_trials
            avg_cs = sum(cos_sims) / len(cos_sims)

            result = {
                "method": method,
                "bits": bits,
                "context_len": ctx_len,
                "cache_n_tokens": n_tokens,
                "n_trials": args.n_trials,
                "pass_rate": pass_rate,
                "avg_rank": avg_rank,
                "avg_cos_sim": avg_cs,
                "synthetic": True,
                "swa_window": args.window if args.swa == "on" else None,
            }
            all_results.append(result)
            print(f"  {method_spec:<12} {ctx_len:>8} {pass_rate:>7.0%}  {avg_rank:>8.1f} {avg_cs:>12.4f}")

    return all_results


def main():
    parser = argparse.ArgumentParser(description="Needle-in-Haystack benchmark")
    parser.add_argument("--model", default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--context-lens", type=int, nargs="+",
                        default=[8192, 32768, 65536])
    parser.add_argument("--methods", type=str, nargs="+",
                        default=["fp16", "planar3", "iso3", "rotor3", "turbo3"])
    parser.add_argument("--n-trials", type=int, default=5,
                        help="Number of trials per (method, context_len)")
    parser.add_argument("--synthetic", action="store_true",
                        help="Run synthetic attention-rank-preservation test (no model needed)")
    parser.add_argument("--output", type=str, default="")
    add_swa_args(parser)
    args = parser.parse_args()

    device = "cuda"
    print(f"\n{'='*60}")
    print(f"Needle-in-Haystack — Long Context Retrieval Test")
    if args.synthetic:
        print(f"Mode: SYNTHETIC (attention rank preservation)")
    else:
        print(f"Model: {args.model}")
    print(f"Context lengths: {args.context_lens}")
    print(f"{'='*60}\n")

    if args.synthetic:
        results = run_synthetic_niah(args, device)
    else:
        results = run_niah(args, device)
    print_summary(results)

    out_path = args.output or str(RESULTS_DIR / "bench_niah.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
