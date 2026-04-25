"""Cache utilities for benchmark scripts.

truncate_kv_to_window  — shrink an HF DynamicCache to the last `window` tokens (in-place).
get_swa_window         — read sliding_window from a model config.
resolve_swa_window     — resolve --swa/--window CLI args to an effective window or None.
clamp_seq_to_window    — for synthetic benches; returns min(seq_len, window) when --swa on.
add_swa_args           — register --swa/--window on an argparse parser with the standard flags.
print_swa_status       — print the resolved SWA mode and effective window.
"""

from __future__ import annotations

from typing import Optional


def truncate_kv_to_window(cache, window: int) -> None:
    """In-place: keep only the last `window` tokens in each layer's K/V tensor.

    Works on transformers >=5.x DynamicCache (cache.layers[i].keys/.values).
    K/V shape per layer: (batch, num_kv_heads, seq, head_dim). Truncates the seq axis.
    No-op for layers shorter than `window`.
    """
    for layer in cache.layers:
        if layer.keys.shape[-2] > window:
            layer.keys = layer.keys[..., -window:, :].contiguous()
            layer.values = layer.values[..., -window:, :].contiguous()


def get_swa_window(model) -> Optional[int]:
    """Return model.config.sliding_window if set and positive, else None."""
    sw = getattr(model.config, "sliding_window", None)
    return int(sw) if sw and sw > 0 else None


def resolve_swa_window(swa: str, model=None, window: int = 0) -> Optional[int]:
    """Resolve --swa / --window CLI args to an effective window length.

    swa='off' -> None (truncation disabled).
    swa='on'  -> `window` if >0, else model.config.sliding_window. Raises if neither.
    """
    if swa == "off":
        return None
    if swa != "on":
        raise ValueError(f"--swa must be 'on' or 'off', got {swa!r}")
    if window > 0:
        return window
    if model is not None:
        w = get_swa_window(model)
        if w is not None:
            return w
    raise ValueError(
        "--swa on requires --window N or a model.config.sliding_window value"
    )


def clamp_seq_to_window(seq_len: int, swa: str, window: int = 0) -> int:
    """For synthetic benches: returns min(seq_len, window) when --swa on, else seq_len."""
    if swa == "off":
        return seq_len
    if swa != "on":
        raise ValueError(f"--swa must be 'on' or 'off', got {swa!r}")
    if window <= 0:
        raise ValueError("--swa on requires --window N for synthetic benches")
    return min(seq_len, window)


def add_swa_args(parser) -> None:
    """Register the standard --swa / --window CLI args on an argparse parser."""
    parser.add_argument(
        "--swa",
        choices=["on", "off"],
        default="off",
        help="Sliding window attention mode (default: off). When 'on', KV cache is "
             "truncated to --window each decode step (HF benches) or synthetic "
             "seq_len is clamped to --window (synthetic benches).",
    )
    parser.add_argument(
        "--window",
        type=int,
        default=0,
        help="Sliding window size in tokens. With --swa on: defaults to "
             "model.config.sliding_window for HF benches; required for synthetic benches.",
    )


def print_swa_status(swa: str, effective_window: Optional[int]) -> None:
    """One-line status print describing the resolved SWA mode."""
    if swa == "off":
        print("SWA: off (full-length KV cache)")
    else:
        print(f"SWA: on (cache truncated to last {effective_window} tokens)")


def vllm_swa_warn(swa: str, max_model_len: int, mistral_window: int = 4096) -> None:
    """Operator reminders when running vLLM benches with --swa on.

    Reminds the operator to (a) apply scripts/patch_vllm_rocm_sliding_window_custom_paged.py
    so vLLM uses the custom paged-attention path instead of falling back to Triton, and
    (b) raise --max-model-len above the model's window so SWA actually fires.
    """
    if swa != "on":
        return
    print(
        "[SWA on] vLLM Mistral SWA path requires "
        "scripts/patch_vllm_rocm_sliding_window_custom_paged.py "
        "to be applied to your vLLM install. Without it, the engine falls "
        "back to the Triton path."
    )
    if max_model_len <= mistral_window:
        print(
            f"[SWA on] WARNING: max_model_len={max_model_len} ≤ {mistral_window}. "
            "SWA will not fire — raise --max-model-len to test it."
        )
