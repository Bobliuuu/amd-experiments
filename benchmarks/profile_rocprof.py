"""
profile_rocprof.py — ROCm Hardware Counter Collection for All Methods

Generates rocprofv2 collection commands and helper scripts for profiling
the compress/decompress kernels with hardware counters on MI300X.

This script:
  1. Generates a minimal benchmark script that runs exactly one kernel invocation
     per method (suitable for rocprofv2 wrapping)
  2. Prints the rocprofv2 commands to collect key hardware counters
  3. Parses and summarizes counter CSV output if available

Key counters targeted:
  FETCH_SIZE       — bytes read from HBM per kernel (actual bandwidth)
  WRITE_SIZE       — bytes written to HBM per kernel
  SQ_INSTS_VMEM_RD — vector memory read instruction count (gather bottleneck proxy)
  SQ_INSTS_VALU    — vector ALU instruction count (rotation work)
  WAVE_CYCLES      — total wave execution cycles
  WAVES_EXECUTED   — number of wavefronts launched

Note on MI300X VF (virtual function) environment:
  Hardware counters may be restricted in VF mode. The timeline trace
  (kernel names + durations) is always available. Counter availability
  should be verified before collection.

Usage:
    python3 benchmarks/profile_rocprof.py --generate-scripts
    python3 benchmarks/profile_rocprof.py --parse-csv rocprof_output.csv
    python3 benchmarks/profile_rocprof.py --check-availability
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

KERNELS_DIR = Path(__file__).parent.parent / "kernels"
RESULTS_DIR = Path(__file__).parent.parent / "results"
SCRIPTS_DIR = Path(__file__).parent.parent / "profiling"
sys.path.insert(0, str(KERNELS_DIR))
RESULTS_DIR.mkdir(exist_ok=True)
SCRIPTS_DIR.mkdir(exist_ok=True)

# rocprofv2 hardware counters for MI300X (gfx942)
ROCPROF_COUNTERS = [
    "FETCH_SIZE",        # bytes fetched from HBM (L2 misses → HBM)
    "WRITE_SIZE",        # bytes written to HBM
    "SQ_INSTS_VMEM_RD",  # vector memory read instructions
    "SQ_INSTS_VMEM_WR",  # vector memory write instructions
    "SQ_INSTS_VALU",     # vector ALU instructions (includes FMAs)
    "WAVE_CYCLES",       # total cycles executed by wavefronts
    "WAVES_EXECUTED",    # number of wavefronts dispatched
]

# Counters available without VF restrictions (usually safe)
SAFE_COUNTERS = ["FETCH_SIZE", "WRITE_SIZE", "WAVE_CYCLES"]

# Counters that may require bare-metal or SRIOV access
RESTRICTED_COUNTERS = ["SQ_INSTS_VMEM_RD", "SQ_INSTS_VALU", "WAVES_EXECUTED"]

KERNEL_SINGLE_BENCH_SCRIPT = '''"""
rocprof_single_kernel.py — Single-kernel benchmark for rocprofv2 wrapping.
Run this script wrapped in rocprofv2 to collect hardware counters.

Usage:
    rocprofv2 --pmc {counters} --output-format csv \\
              python3 profiling/rocprof_single_kernel.py --method {method} --bits {bits}
"""
import sys
import torch
sys.path.insert(0, "{kernels_dir}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", default="planar")
parser.add_argument("--bits", type=int, default=3)
parser.add_argument("--n-vectors", type=int, default=65536)
parser.add_argument("--n-iters", type=int, default=10)
args = parser.parse_args()

device = "cuda"
import numpy as np
rng = np.random.default_rng(42)
x = torch.from_numpy(rng.standard_normal((args.n_vectors, 128)).astype("float32")).to(device)
torch.cuda.synchronize()

if args.method == "turbo":
    from turboquant_mi300x import TurboQuantMI300X
    tq = TurboQuantMI300X(bits=args.bits, rotation_seed=42)
    for _ in range(3):  # warmup
        tq.compress_tensor(x)
    torch.cuda.synchronize()
    for _ in range(args.n_iters):
        tq.compress_tensor(x)
    torch.cuda.synchronize()
    print(f"TurboQuant{args.bits} compress: {args.n_iters} iters, {args.n_vectors} vectors")
else:
    from block_quant_rocm import make_quantizer
    q = make_quantizer(args.method, bits=args.bits, device=device)
    for _ in range(3):  # warmup
        q.compress(x)
    torch.cuda.synchronize()
    for _ in range(args.n_iters):
        q.compress(x)
    torch.cuda.synchronize()
    print(f"{args.method}{args.bits} compress: {args.n_iters} iters, {args.n_vectors} vectors")
'''

TIMELINE_SCRIPT = '''"""
rocprof_timeline.py — One decode step for kernel timeline collection.

Usage:
    rocprofv2 --sys-trace --output-format json \\
              python3 profiling/rocprof_timeline.py --method {method}
"""
import sys
import torch
sys.path.insert(0, "{kernels_dir}")

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--method", default="planar3")
parser.add_argument("--n-layers", type=int, default=32)
parser.add_argument("--n-kv-heads", type=int, default=8)
parser.add_argument("--seq-len", type=int, default=32768)
parser.add_argument("--head-dim", type=int, default=128)
args = parser.parse_args()

device = "cuda"
import time
import numpy as np

n_vectors = args.n_kv_heads * args.seq_len

method = args.method[:-1] if args.method[-1].isdigit() else args.method
bits = int(args.method[-1]) if args.method[-1].isdigit() else 3

if method == "fp16":
    K = torch.randn(args.n_kv_heads, args.seq_len, args.head_dim, device=device)
    V = torch.randn_like(K)
    q_tok = torch.randn(args.n_kv_heads, 1, args.head_dim, device=device)
    sm_scale = args.head_dim ** -0.5
    torch.cuda.synchronize()
    # Timeline: one decode step
    for l in range(args.n_layers):
        scores = torch.bmm(q_tok, K.transpose(-2, -1)) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        _ = torch.bmm(weights, V)
    torch.cuda.synchronize()
    print("FP16 decode step complete")
else:
    from block_quant_rocm import make_quantizer
    q_obj = make_quantizer(method, bits=bits, device=device)
    K_raw = torch.randn(n_vectors, args.head_dim, device=device)
    K_comp = q_obj.compress(K_raw)
    V_comp = q_obj.compress(K_raw)
    q_tok = torch.randn(args.n_kv_heads, 1, args.head_dim, device=device)
    sm_scale = args.head_dim ** -0.5
    torch.cuda.synchronize()
    # Timeline: one decode step (decompress + attention)
    for l in range(args.n_layers):
        K_fp = q_obj.decompress(K_comp, (n_vectors, args.head_dim)).reshape(args.n_kv_heads, args.seq_len, args.head_dim)
        V_fp = q_obj.decompress(V_comp, (n_vectors, args.head_dim)).reshape(args.n_kv_heads, args.seq_len, args.head_dim)
        scores = torch.bmm(q_tok, K_fp.transpose(-2, -1)) * sm_scale
        weights = torch.softmax(scores, dim=-1)
        _ = torch.bmm(weights, V_fp)
    torch.cuda.synchronize()
    print(f"{args.method} decode step complete")
'''


def generate_scripts(kernels_dir: str):
    """Write profiling scripts to the profiling/ directory."""
    print(f"Writing profiling scripts to {SCRIPTS_DIR}/")

    # Single kernel benchmark script
    script_path = SCRIPTS_DIR / "rocprof_single_kernel.py"
    script_path.write_text(
        KERNEL_SINGLE_BENCH_SCRIPT.format(kernels_dir=kernels_dir))
    print(f"  Created: {script_path}")

    # Timeline script
    timeline_path = SCRIPTS_DIR / "rocprof_timeline.py"
    timeline_path.write_text(
        TIMELINE_SCRIPT.format(kernels_dir=kernels_dir))
    print(f"  Created: {timeline_path}")

    # Shell script for all methods
    methods = [
        ("fp16", None), ("turbo", 3), ("turbo", 4),
        ("planar", 3), ("planar", 4),
        ("iso", 3), ("iso", 4),
        ("rotor", 3), ("rotor", 4),
    ]

    sh_lines = [
        "#!/bin/bash",
        "# rocprof_collect_all.sh — Collect hardware counters for all methods",
        "# Generated by profile_rocprof.py",
        "",
        "set -e",
        f"KERNELS_DIR={kernels_dir}",
        f"RESULTS_DIR={RESULTS_DIR}",
        f"SCRIPTS_DIR={SCRIPTS_DIR}",
        "",
        "COUNTERS='FETCH_SIZE,WRITE_SIZE,SQ_INSTS_VMEM_RD,SQ_INSTS_VALU,WAVE_CYCLES'",
        "SAFE_COUNTERS='FETCH_SIZE,WRITE_SIZE,WAVE_CYCLES'",
        "",
        "echo 'Collecting hardware counters (use SAFE_COUNTERS if COUNTERS fails on VF)'",
        "",
    ]

    for method, bits in methods:
        if method == "fp16":
            continue
        label = f"{method}{bits}"
        sh_lines.extend([
            f"echo 'Profiling {label}...'",
            f"rocprofv2 --pmc $COUNTERS --output-file $RESULTS_DIR/rocprof_{label}.csv \\",
            f"    python3 $SCRIPTS_DIR/rocprof_single_kernel.py --method {method} --bits {bits} \\",
            f"    || rocprofv2 --pmc $SAFE_COUNTERS --output-file $RESULTS_DIR/rocprof_{label}_safe.csv \\",
            f"       python3 $SCRIPTS_DIR/rocprof_single_kernel.py --method {method} --bits {bits}",
            "",
        ])

    sh_lines.extend([
        "echo ''",
        "echo 'Timeline traces:'",
    ])
    for method_spec in ["fp16", "planar3", "iso3", "rotor3", "turbo3"]:
        sh_lines.extend([
            f"rocprofv2 --sys-trace --output-format json \\",
            f"    --output-file $RESULTS_DIR/timeline_{method_spec}.json \\",
            f"    python3 $SCRIPTS_DIR/rocprof_timeline.py --method {method_spec}",
        ])

    sh_lines.append("")
    sh_lines.append("echo 'Done. Parse results with: python3 benchmarks/profile_rocprof.py --parse-csv'")

    sh_path = SCRIPTS_DIR / "rocprof_collect_all.sh"
    sh_path.write_text("\n".join(sh_lines))
    sh_path.chmod(0o755)
    print(f"  Created: {sh_path}")

    # Print usage instructions
    print(f"""
Collection commands:
  # Full collection (may fail in VF mode — try SAFE_COUNTERS if so):
  rocprofv2 --pmc FETCH_SIZE,WRITE_SIZE,SQ_INSTS_VMEM_RD,SQ_INSTS_VALU \\
            --output-format csv \\
            python3 {script_path} --method planar --bits 3

  # Check counter availability:
  rocprofv2 --list-metrics gfx942

  # Timeline trace (always available):
  rocprofv2 --sys-trace --output-format json \\
            python3 {timeline_path} --method planar3

  # Run all at once:
  bash {sh_path}

  # Parse results:
  python3 {Path(__file__)} --parse-csv {RESULTS_DIR}/rocprof_planar3.csv
""")


def check_availability():
    """Check if rocprofv2 is available and what counters are accessible."""
    print("Checking rocprofv2 availability...")

    # Check if rocprofv2 exists
    result = subprocess.run(["which", "rocprofv2"],
                            capture_output=True, text=True)
    if result.returncode != 0:
        print("  rocprofv2 NOT FOUND — install ROCm profiling tools")
        print("  On ROCm: sudo apt install rocprofiler-v2")
        return False

    print(f"  rocprofv2 found: {result.stdout.strip()}")

    # List available metrics
    result = subprocess.run(["rocprofv2", "--list-metrics", "gfx942"],
                            capture_output=True, text=True)
    if result.returncode == 0:
        output = result.stdout + result.stderr
        print("\n  Available counters on gfx942:")

        for counter in ROCPROF_COUNTERS:
            found = counter in output
            status = "available" if found else "NOT FOUND"
            restricted = counter in RESTRICTED_COUNTERS
            note = " (may need bare-metal)" if restricted else ""
            print(f"    {counter:<25} {status}{note}")
    else:
        print(f"  Could not list metrics: {result.stderr[:200]}")
        print("  This may be a VF restriction — try bare-metal access")

    return True


def parse_csv(csv_path: str) -> dict:
    """Parse rocprofv2 CSV output and print a summary."""
    import csv

    if not Path(csv_path).exists():
        print(f"CSV file not found: {csv_path}")
        return {}

    results = {}
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            kernel_name = row.get("Kernel_Name", row.get("KernelName", "unknown"))
            for counter in ROCPROF_COUNTERS:
                if counter in row:
                    if kernel_name not in results:
                        results[kernel_name] = {}
                    results[kernel_name][counter] = float(row[counter])

    if not results:
        print(f"No counter data found in {csv_path}")
        return {}

    print(f"\nrocprof Counter Summary: {csv_path}")
    print(f"{'Kernel':<50} {'FETCH_SIZE':>12} {'SQ_VMEM_RD':>12} {'SQ_VALU':>10}")
    print("-" * 90)
    for kernel, counters in results.items():
        fetch = counters.get("FETCH_SIZE", 0)
        vmem = counters.get("SQ_INSTS_VMEM_RD", 0)
        valu = counters.get("SQ_INSTS_VALU", 0)
        short_name = kernel[-50:] if len(kernel) > 50 else kernel
        print(f"{short_name:<50} {fetch:>12.0f} {vmem:>12.0f} {valu:>10.0f}")

    # VMEM/VALU ratio (key indicator of gather bottleneck)
    for kernel, counters in results.items():
        vmem = counters.get("SQ_INSTS_VMEM_RD", 0)
        valu = counters.get("SQ_INSTS_VALU", 1)
        ratio = vmem / valu if valu > 0 else 0
        if ratio > 0.1:
            print(f"\n  Note: {kernel[-40:]} VMEM/VALU ratio = {ratio:.3f}")
            if ratio > 0.5:
                print(f"    HIGH ratio → likely gather-bottlenecked (like TQ v1 before fix)")

    return results


def print_interpretation_guide():
    """Print guide for interpreting rocprof results."""
    print("""
Counter Interpretation Guide for KV Compression Methods:
═══════════════════════════════════════════════════════════

FETCH_SIZE / kernel_duration_us = Achieved HBM bandwidth (GB/s)
  - MI300X theoretical peak: 5,300 GB/s
  - FP16 SDPA achieves ~350 GB/s in practice
  - TQ3 decompress (HIP): ~198 GB/s
  - Triton kernels: typically 20-80 GB/s (VMEM gather overhead)

SQ_INSTS_VMEM_RD: Number of vector memory read instructions
  - HIGH value = gather bottleneck (like TQ3 v1 with centroid scatter)
  - LOW value (after tl.where fix) = computation moved to VALU
  - Expected ratio VMEM_RD/VALU < 0.1 for well-optimized kernels

SQ_INSTS_VALU: Number of vector ALU instructions
  - PlanarQuant compress: ~256 FMAs/vec → ~16M VALU for 65K vectors
  - IsoQuant compress: ~512 FMAs/vec → ~33M VALU for 65K vectors
  - RotorQuant compress: ~1194 FMAs/vec → ~78M VALU for 65K vectors
  - TurboQuant compress: ~16384 FMAs/vec → ~1073M VALU for 65K vectors
  Ratio Rotor/Planar ≈ 4.7× more compute for same compression ratio

WAVE_CYCLES: Total GPU cycles used
  - Divide by WAVES_EXECUTED → cycles per wavefront
  - Compare against occupancy target: MI300X CDNA3 = up to 8 waves/CU
  - Low cycles/wave with many waves = good occupancy

Key diagnostic questions:
  1. Is FETCH_SIZE/duration close to peak bandwidth? If not, compute-bound
  2. Is SQ_INSTS_VMEM_RD >> SQ_INSTS_VALU? If yes, gather-bottlenecked
  3. Does WAVE_CYCLES/WAVES_EXECUTED differ across methods? Shows rotation cost
""")


def main():
    parser = argparse.ArgumentParser(description="ROCm profiling helper")
    parser.add_argument("--generate-scripts", action="store_true",
                        help="Generate profiling scripts and collection commands")
    parser.add_argument("--check-availability", action="store_true",
                        help="Check rocprofv2 availability and accessible counters")
    parser.add_argument("--parse-csv", type=str, metavar="FILE",
                        help="Parse rocprofv2 CSV output file")
    parser.add_argument("--guide", action="store_true",
                        help="Print counter interpretation guide")
    args = parser.parse_args()

    if args.guide:
        print_interpretation_guide()
        return

    if args.check_availability:
        check_availability()
        return

    if args.parse_csv:
        parse_csv(args.parse_csv)
        return

    if args.generate_scripts:
        generate_scripts(str(KERNELS_DIR))
        return

    # Default: generate scripts + show guide
    generate_scripts(str(KERNELS_DIR))
    print_interpretation_guide()


if __name__ == "__main__":
    main()
