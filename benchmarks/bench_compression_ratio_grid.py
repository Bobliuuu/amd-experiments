"""
bench_compression_ratio_grid.py — Screenshot-style compression ratio table experiment.

Measures compressed bytes by actually running TurboQuant compression on synthetic KV
vectors and comparing against FP16 bytes across multiple context lengths and bit widths.
"""

import argparse
import json
from pathlib import Path

import torch

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)
import sys

sys.path.insert(0, str(KERNELS_DIR))


def main():
    parser = argparse.ArgumentParser(description="Compression-ratio grid benchmark")
    parser.add_argument("--bits", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--ctx", nargs="+", type=int, default=[2048, 4096, 8192, 16384])
    parser.add_argument("--layers", type=int, default=28)
    parser.add_argument("--kv-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    from turboquant_mi300x import TurboQuantMI300X

    torch.manual_seed(0)
    rows = []

    print("\n12. Compression ratios (measured compressed tensors)\n")
    for bits in args.bits:
        engine = TurboQuantMI300X(head_dim=args.head_dim, bits=bits, rotation_seed=42, device=args.device)
        for ctx in args.ctx:
            n_vec = args.layers * args.kv_heads * ctx
            x = torch.randn(n_vec, args.head_dim, device=args.device, dtype=torch.float32)
            fp16_bytes = int(n_vec * args.head_dim * 2)
            comp = engine.compress_tensor(x)
            comp_bytes = int(comp.numel() * comp.element_size())
            ratio = float(fp16_bytes) / float(comp_bytes)
            print(
                f"{bits}-bit @ {ctx:>5} tokens: {comp_bytes/1024:.0f} KB vs "
                f"{fp16_bytes/1024:.0f} KB FP16  ({ratio:.2f}x)"
            )
            rows.append(
                {
                    "bits": bits,
                    "ctx_tokens": ctx,
                    "layers": args.layers,
                    "kv_heads": args.kv_heads,
                    "head_dim": args.head_dim,
                    "fp16_bytes": fp16_bytes,
                    "compressed_bytes_measured": comp_bytes,
                    "compression_ratio_measured": ratio,
                }
            )
        print()

    out = {
        "device": torch.cuda.get_device_name(0) if args.device == "cuda" else args.device,
        "results": rows,
    }
    out_path = Path(args.output) if args.output else (RESULTS_DIR / "bench_compression_ratio_grid.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
