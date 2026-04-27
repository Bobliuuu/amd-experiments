"""
tq_triton.py — Fused TurboQuant Dequant-Attention Kernel (Triton, ROCm/MI300X)

Implements a Triton kernel that fuses TQ3 decompression with Flash-Attention-2
style attention score computation, reading 52 bytes/token vs 256 bytes/token
for FP16 — a 4.92× bandwidth reduction.

Algorithm (Flash Attention 2 style):
  For each KV block of BLOCK_N tokens:
    1. Load 3 bit-planes (plane-0=LSB, plane-2=MSB), each [BLOCK_N, 16 bytes].
    2. Extract bit0/bit1/bit2 per (token, dim) by shifting the byte and masking.
    3. Reconstruct centroid value via cascaded tl.where — no gather/table-lookup.
       (Only 6 VALU instructions vs a scatter VMEM load per element.)
    4. Compute Q·K^T matmul (FP16 MFMA), then fold norms *after* the dot product:
         scores = dot(q, k_centroids^T) * (k_norms * sm_scale)
       This avoids materializing a full [BLOCK_N, head_dim] FP32 tensor.
    5. Online softmax update.
    6. For V: scale the softmax weights p by v_norms *before* the dot product:
         acc += dot(p * v_norms, v_centroids)
       Again, p is [BLOCK_M, BLOCK_N] (tiny) so the norm multiply is O(BLOCK_N).
  Normalize and write output.
  Apply inverse rotation outside the kernel (one 128×128 matmul, negligible).

Bit-plane format (matches turboquant_mi300x.py/_pack_bitplanes):
  Block = [norm:4 bytes][plane0:16 bytes][plane1:16 bytes][plane2:16 bytes]
  Plane b (b=0=LSB, b=2=MSB): bit b of index[j] at byte (b*16+j//8), bit (j%8)

Performance on MI300X (gfx942):
  Original kernel: effective 12 GB/s (compute-bound on gather + bit-extract).
  This kernel: replaces scatter-VMEM gather with pure VALU (tl.where cascades),
  folds norms to skip two [BLOCK_N, D] tensor materializations.
  Autotuning finds the best (BLOCK_M, BLOCK_N) for the decode-step workload.

Usage:
    from tq_triton import turboquant_attention_fwd
    # q: (B, H, S_q, D) float16 — PRE-ROTATED (q @ R^T)
    # k_planes: (B, H, S_k, 48) uint8
    # k_norms:  (B, H, S_k)     float32
    # rotation: (D, D)           float32 — the rotation matrix R
    out = turboquant_attention_fwd(q, k_planes, k_norms, v_planes, v_norms,
                                   rotation=rotation)
"""

import torch
import triton
import triton.language as tl
from typing import Optional


# ──────────────────────────────────────────────────────────────────────────────
# Centroid constants (compile-time, avoids any runtime pointer)
# ──────────────────────────────────────────────────────────────────────────────

# TQ3 Lloyd-Max centroids for N(0,1) after rotation:
#   index 0-3 → negative; index 4-7 → positive (symmetric)
# Magnitudes for each (b1, b0) pair — same for positive and negative side,
# just mirrored: for negative, bit pattern *inverts* before the mag lookup.
# Triton 3.6+ (ROCm 7.2 containers): use tl.constexpr(...) — annotated globals
# are not always treated as kernel-visible constexprs.
_C00 = tl.constexpr(0.02174971334976657)   # |idx|=0 in {4,3} → smallest
_C01 = tl.constexpr(0.06702922184405663)   # |idx|=1 in {5,2}
_C10 = tl.constexpr(0.11879501670185091)   # |idx|=2 in {6,1}
_C11 = tl.constexpr(0.18904037194348838)   # |idx|=3 in {7,0} → largest


# ──────────────────────────────────────────────────────────────────────────────
# Inline centroid decode — no pointer, no gather, pure VALU
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _bits_to_centroid(b0, b1, b2):
    """
    Convert three [BLOCK_N, head_dim] int32 bit tensors (each element 0 or 1)
    to float32 TQ3 centroid values.  No memory access — pure ALU.

    Encoding (index = b2<<2 | b1<<1 | b0):
      Positive (b2=1): b1=0,b0=0 → 0.02175; b1=1,b0=1 → 0.18904
      Negative (b2=0): b1=0,b0=0 → -0.18904; b1=1,b0=1 → -0.02175
    The magnitude table is the same for both signs; for negative, (b1,b0) is
    complemented before the lookup so the ordering inverts correctly.

    6 VALU instructions total (2 flip-selects + 2 magnitude-selects + sign + mul).
    """
    b0f = b0.to(tl.float32)
    b1f = b1.to(tl.float32)
    b2f = b2.to(tl.float32)
    # For negative centroids (b2=0), flip b0 and b1 so the same table gives the
    # right magnitude (high bit pattern → small |value| on the negative side).
    eb0 = tl.where(b2f > 0.5, b0f, 1.0 - b0f)
    eb1 = tl.where(b2f > 0.5, b1f, 1.0 - b1f)
    mag = tl.where(eb1 > 0.5,
                   tl.where(eb0 > 0.5, _C11, _C10),
                   tl.where(eb0 > 0.5, _C01, _C00))
    sign = 2.0 * b2f - 1.0   # +1.0 for positive (b2=1), -1.0 for negative (b2=0)
    return sign * mag


# ──────────────────────────────────────────────────────────────────────────────
# Triton kernel: fused TQ3 dequantize + Flash Attention 2
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # (BLOCK_M, BLOCK_N): BLOCK_M=16 is optimal for decode (S_q≈1)
        # Smaller BLOCK_N → less register pressure → more concurrent wavefronts
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32},  num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64},  num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 32},  num_warps=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64},  num_warps=4),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64},  num_warps=4),
    ],
    key=["seq_q", "seq_k", "head_dim"],
)
@triton.jit
def _tq3_attention_kernel(
    # Query (pre-rotated: q_rot = q @ R^T), float16
    Q_ptr, stride_qb, stride_qh, stride_qm, stride_qd,
    # Compressed K: bit-planes (uint8, 48 bytes per token)
    K_planes_ptr, stride_kb, stride_kh, stride_kn,
    # K norms (float32, one per token)
    K_norms_ptr, stride_knb, stride_knh, stride_knn,
    # Compressed V: bit-planes (uint8, 48 bytes per token)
    V_planes_ptr, stride_vb, stride_vh, stride_vn,
    # V norms (float32, one per token)
    V_norms_ptr, stride_vnb, stride_vnh, stride_vnn,
    # Output (float16, in ROTATED space — caller applies @R for original space)
    O_ptr, stride_ob, stride_oh, stride_om, stride_od,
    # Shape / scale
    batch: int, heads: int, seq_q: int, seq_k: int,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: float,
):
    """
    Fused TQ3 dequantize + attention (autotuned BLOCK_M / BLOCK_N).

    Grid: (ceil(S_q/BLOCK_M), B×H)

    Key optimizations vs prior version:
      1. Centroid lookup via tl.where cascade (6 VALU instr) instead of
         scatter-gather (VMEM per element, 400-cycle latency on CDNA3).
      2. K norms folded into scores AFTER dot product:
           scores = dot(q, k_centroids^T) * (norms * sm_scale)
         Avoids materializing k_fp32 = k_centroids * norms[:, None] in registers.
      3. V norms folded into p BEFORE accumulate dot product:
           acc += dot(p * v_norms[None, :], v_centroids)
         p is [BLOCK_M, BLOCK_N] — scaling it is O(BLOCK_M * BLOCK_N), not
         O(BLOCK_N * D), saving D/BLOCK_M multiplies.
      4. FP16 inputs to tl.dot → uses MFMA fp16 accumulate (faster on gfx942).
      5. Autotuning picks best (BLOCK_M, BLOCK_N) per seq shape.
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)

    batch_idx = pid_bh // heads
    head_idx  = pid_bh %  heads

    # ── Load Q block [BLOCK_M, head_dim] ──────────────────────────────────────
    q_off  = (batch_idx * stride_qb + head_idx * stride_qh
              + pid_m * BLOCK_M * stride_qm)
    q_ptrs = (Q_ptr + q_off
              + tl.arange(0, BLOCK_M)[:, None] * stride_qm
              + tl.arange(0, head_dim)[None, :] * stride_qd)
    m_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < seq_q
    # Load in FP16 — keeps it in FP16 for MFMA; accumulate scores in FP32
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float16)

    # ── Online softmax state ───────────────────────────────────────────────────
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # Base offsets for this batch/head
    k_base  = batch_idx * stride_kb  + head_idx * stride_kh
    kn_base = batch_idx * stride_knb + head_idx * stride_knh
    v_base  = batch_idx * stride_vb  + head_idx * stride_vh
    vn_base = batch_idx * stride_vnb + head_idx * stride_vnh

    # Per-dimension byte/bit offsets (constexpr, computed once, no runtime cost)
    n_range = tl.arange(0, BLOCK_N)   # [BLOCK_N]
    d_range = tl.arange(0, head_dim)  # [head_dim]
    byte_in_plane = d_range // 8      # [head_dim] which byte within a 16-byte plane
    bit_in_byte   = d_range % 8       # [head_dim] which bit within that byte

    # ── Iterate over KV blocks ─────────────────────────────────────────────────
    for block_n_start in range(0, seq_k, BLOCK_N):
        n_mask    = (block_n_start + n_range) < seq_k          # [BLOCK_N]
        n_mask_2d = n_mask[:, None]                             # [BLOCK_N, 1] → broadcast

        # ── Decode K ──────────────────────────────────────────────────────────
        # Memory layout: [B, H, S_k, 48 bytes] with stride_kn=48.
        # 3 vectorized 2D loads, one per bit-plane (each 16 bytes wide).
        # Each (n, d) element selects byte = n*48 + plane*16 + d//8.
        # Loading each unique byte 8 times (once per d sharing that byte),
        # but L1 cache (64KB on MI300X) absorbs the repetition — actual
        # HBM traffic = 48 bytes/token as designed.
        k_block_base = K_planes_ptr + k_base + block_n_start * stride_kn
        k_byte_ptrs  = (k_block_base
                        + n_range[:, None] * stride_kn
                        + byte_in_plane[None, :])        # [BLOCK_N, head_dim]
        b0k_raw = tl.load(k_byte_ptrs,      mask=n_mask_2d, other=0).to(tl.int32)
        b1k_raw = tl.load(k_byte_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2k_raw = tl.load(k_byte_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)

        # Extract individual bits for each (token, dim) — [BLOCK_N, head_dim]
        bit_shift = bit_in_byte[None, :]   # [1, head_dim] broadcasts over BLOCK_N
        b0k = (b0k_raw >> bit_shift) & 1
        b1k = (b1k_raw >> bit_shift) & 1
        b2k = (b2k_raw >> bit_shift) & 1

        # Centroid values via pure VALU (no gather, no Centroids_ptr)
        k_centroids = _bits_to_centroid(b0k, b1k, b2k)  # [BLOCK_N, head_dim] fp32

        # ── Attention scores [BLOCK_M, BLOCK_N] ───────────────────────────────
        # dot(q, k_centroids^T) using FP16 MFMA, accumulate in FP32.
        # Norms folded in AFTER the matmul → no [BLOCK_N, D] norm-scaled tensor.
        raw_scores = tl.dot(q, k_centroids.to(tl.float16).T,
                            out_dtype=tl.float32)       # [BLOCK_M, BLOCK_N]

        k_norms_block = tl.load(
            K_norms_ptr + kn_base + (block_n_start + n_range) * stride_knn,
            mask=n_mask, other=0.0)                      # [BLOCK_N] fp32
        # Scale each KV-token's score column by its norm and softmax scale
        scores = raw_scores * (k_norms_block * sm_scale)[None, :]
        scores = tl.where(n_mask[None, :], scores, -1e9)

        # ── Online softmax update ──────────────────────────────────────────────
        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])          # [BLOCK_M, BLOCK_N]
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        m_i   = m_new

        # ── Decode V (identical bit-extraction pattern) ────────────────────────
        v_block_base = V_planes_ptr + v_base + block_n_start * stride_vn
        v_byte_ptrs  = (v_block_base
                        + n_range[:, None] * stride_vn
                        + byte_in_plane[None, :])
        b0v_raw = tl.load(v_byte_ptrs,      mask=n_mask_2d, other=0).to(tl.int32)
        b1v_raw = tl.load(v_byte_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2v_raw = tl.load(v_byte_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)

        b0v = (b0v_raw >> bit_shift) & 1
        b1v = (b1v_raw >> bit_shift) & 1
        b2v = (b2v_raw >> bit_shift) & 1
        v_centroids = _bits_to_centroid(b0v, b1v, b2v)  # [BLOCK_N, head_dim] fp32

        # ── Accumulate output ──────────────────────────────────────────────────
        # Fold V norms into p (tiny [BLOCK_M, BLOCK_N] tensor) instead of
        # expanding norms across [BLOCK_N, head_dim].
        v_norms_block = tl.load(
            V_norms_ptr + vn_base + (block_n_start + n_range) * stride_vnn,
            mask=n_mask, other=0.0)                      # [BLOCK_N] fp32
        p_scaled = p * v_norms_block[None, :]            # [BLOCK_M, BLOCK_N]

        acc = acc * alpha[:, None] + tl.dot(p_scaled.to(tl.float16),
                                            v_centroids.to(tl.float16),
                                            out_dtype=tl.float32)

    # ── Normalize and write output ─────────────────────────────────────────────
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    out_off  = (batch_idx * stride_ob + head_idx * stride_oh
                + pid_m * BLOCK_M * stride_om)
    out_ptrs = (O_ptr + out_off
                + tl.arange(0, BLOCK_M)[:, None] * stride_om
                + tl.arange(0, head_dim)[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# v3 kernel: compact 16-byte plane loads (8× fewer VMEM instructions)
#
# Root cause of v2's 25 GB/s ceiling (vs 5300 GB/s HBM peak):
#   v2 loads k_byte_ptrs = [BLOCK_N, head_dim=128] where byte_in_plane = d//8
#   maps 8 consecutive columns to the same byte.  L1 caches the 7 redundant
#   hits, so HBM traffic is correct (48 B/token), but 384 VMEM *instructions*
#   are issued per KV block (128/plane × 3 planes) — at 1 instr / 4 cycles on
#   CDNA3 that is 1536 cycles just for instruction issue, competing with MFMA.
#
# Fix: load [BLOCK_N, 16] compact bytes (one per unique byte in the plane),
# then expand to [BLOCK_N, 128] in registers via Triton 3.1 3D reshape+shift:
#
#   b_cmp : [BLOCK_N, 16]
#   b_3d  = reshape(b_cmp, (BLOCK_N, 16, 1)) >> reshape(arange(8), (1, 1, 8))
#           → [BLOCK_N, 16, 8]  (pure VALU: 8 shift+AND instructions)
#   b     = reshape(b_3d & 1,  (BLOCK_N, 128))
#           C-order: element [n, 8*byte+bit] = correct — matches dim j → byte j//8, bit j%8.
#
# VMEM instruction count: 48 total (16/plane × 3 planes) vs 384 in v2 → 8× drop.
# The expansion is 3 × BLOCK_N pure-VALU shift+AND ops — negligible vs MFMA.
#
# Additional v3 changes:
#   • num_stages=2 on all configs — software-pipeline the KV load loop to hide
#     the remaining VMEM latency behind MFMA execution.
#   • Wider BLOCK_N options (128, 256) probed in autotune — larger tiles amortise
#     the per-block overhead (norm loads, softmax bookkeeping).
#   • K and V planes share one base-pointer computation per KV block.
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        # num_stages=2: double-buffer the KV load loop (hides ~400-cycle VMEM latency)
        triton.Config({"BLOCK_M": 16, "BLOCK_N":  32}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N":  64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N":  64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N":  64}, num_warps=4, num_stages=2),
        # num_stages=1 variants in case the register pressure from larger tiles
        # prevents double buffering (CDNA3 has 512 VGPRs / wavefront):
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=1),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 256}, num_warps=8, num_stages=1),
    ],
    key=["seq_q", "seq_k", "head_dim"],
)
@triton.jit
def _tq3v3_attention_kernel(
    # Query (pre-rotated), float16
    Q_ptr, stride_qb, stride_qh, stride_qm, stride_qd,
    # Compressed K: bit-planes (uint8, 48 bytes per token, stride_kn=48)
    K_planes_ptr, stride_kb, stride_kh, stride_kn,
    # K norms (float32, one per token)
    K_norms_ptr, stride_knb, stride_knh, stride_knn,
    # Compressed V: bit-planes (uint8, 48 bytes per token)
    V_planes_ptr, stride_vb, stride_vh, stride_vn,
    # V norms (float32, one per token)
    V_norms_ptr, stride_vnb, stride_vnh, stride_vnn,
    # Output (float16)
    O_ptr, stride_ob, stride_oh, stride_om, stride_od,
    # Shape / scale
    batch: int, heads: int, seq_q: int, seq_k: int,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: float,
):
    """
    v3: fused TQ3 dequantize + Flash-Attention-2 attention.

    Key change vs v2: compact plane loads — [BLOCK_N, 16] bytes per plane instead
    of [BLOCK_N, 128] — reduce VMEM instruction count 8× (768 → 96 per KV block).
    Expansion to [BLOCK_N, 128] is done in registers via 3D tl.reshape+bitshift.

    Measured result on gfx942 (batch=1, H=8 KV heads, seq_k=65536):
      v2: 8.23 ms   v3: 9.06 ms  (v3 is ~10% slower)

    The VMEM savings are real and the bit ordering is verified correct, but they
    don't help because the real bottleneck is grid parallelism: with batch=1 the
    kernel launches only 8 wavefronts (= 0.7% of 1216 SIMD units on MI300X).
    The sequential KV loop of 1024 iterations per wavefront dominates, and the
    3D reshape VALU overhead outweighs the latency reduction from fewer VMEM
    instructions on so few active wavefronts.

    The pattern IS useful as a reference and for larger batch sizes where VMEM
    instruction throughput becomes the binding constraint.  The true fix for
    batch=1 decode is sequence-parallel (Split-K) attention: exposing S_k in the
    grid to saturate all 304 CUs — see summary.md for the proposed v4 design.

    Requires Triton ≥ 3.1 (for tl.reshape on 3D shapes and element-wise 3D ops).
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // heads
    head_idx  = pid_bh %  heads

    # ── Load Q block [BLOCK_M, head_dim] ─────────────────────────────────────
    q_off  = (batch_idx * stride_qb + head_idx * stride_qh
              + pid_m * BLOCK_M * stride_qm)
    q_ptrs = (Q_ptr + q_off
              + tl.arange(0, BLOCK_M)[:, None] * stride_qm
              + tl.arange(0, head_dim)[None, :] * stride_qd)
    m_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < seq_q
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float16)

    # ── Online softmax state ──────────────────────────────────────────────────
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    k_base  = batch_idx * stride_kb  + head_idx * stride_kh
    kn_base = batch_idx * stride_knb + head_idx * stride_knh
    v_base  = batch_idx * stride_vb  + head_idx * stride_vh
    vn_base = batch_idx * stride_vnb + head_idx * stride_vnh

    n_range = tl.arange(0, BLOCK_N)   # [BLOCK_N]
    # Compact byte range: 16 bytes per plane (one byte per 8 consecutive dims)
    p16     = tl.arange(0, 16)        # [16]
    # Bit positions 0..7 in a 3D layout for the expand step — shape (1, 1, 8)
    # CDNA3 shift instructions are free when the shift amount is a constexpr lane offset.
    bit8    = tl.reshape(tl.arange(0, 8), (1, 1, 8))  # [1, 1, 8]

    # ── Iterate over KV blocks ────────────────────────────────────────────────
    for block_n_start in range(0, seq_k, BLOCK_N):
        n_mask    = (block_n_start + n_range) < seq_k   # [BLOCK_N]
        n_mask_2d = n_mask[:, None]                      # [BLOCK_N, 1]

        # ── Load K planes: [BLOCK_N, 16] compact bytes per plane ─────────────
        # vs v2's [BLOCK_N, 128] which issued 8× duplicate pointer dereferences.
        # 3 loads of [BLOCK_N, 16] = 48 VMEM instructions total (was 384).
        k_blk = K_planes_ptr + k_base + block_n_start * stride_kn
        k_ptrs = k_blk + n_range[:, None] * stride_kn + p16[None, :]  # [BLOCK_N, 16]
        b0k_c  = tl.load(k_ptrs,      mask=n_mask_2d, other=0).to(tl.int32)
        b1k_c  = tl.load(k_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2k_c  = tl.load(k_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)

        # ── Expand [BLOCK_N, 16] → [BLOCK_N, 128] via 3D reshape ─────────────
        # reshape(b_c, (N,16,1)) >> [[[ 0,1,2,...,7 ]]]  →  [N, 16, 8]
        # reshape(_, (N, 128))  gives element [n, 8*byte+bit] = (byte_val>>bit)&1
        # C-order ensures dim j maps to byte j//8, bit j%8 — matching v2's
        # byte_in_plane = d//8, bit_in_byte = d%8 convention.
        b0k_3d = tl.reshape(b0k_c, (BLOCK_N, 16, 1))
        b1k_3d = tl.reshape(b1k_c, (BLOCK_N, 16, 1))
        b2k_3d = tl.reshape(b2k_c, (BLOCK_N, 16, 1))

        b0k = tl.reshape((b0k_3d >> bit8) & 1, (BLOCK_N, head_dim))
        b1k = tl.reshape((b1k_3d >> bit8) & 1, (BLOCK_N, head_dim))
        b2k = tl.reshape((b2k_3d >> bit8) & 1, (BLOCK_N, head_dim))

        k_centroids = _bits_to_centroid(b0k, b1k, b2k)  # [BLOCK_N, head_dim] fp32

        # ── Attention scores [BLOCK_M, BLOCK_N] ──────────────────────────────
        raw_scores = tl.dot(q, k_centroids.to(tl.float16).T, out_dtype=tl.float32)

        k_norms_block = tl.load(
            K_norms_ptr + kn_base + (block_n_start + n_range) * stride_knn,
            mask=n_mask, other=0.0)
        scores = raw_scores * (k_norms_block * sm_scale)[None, :]
        scores = tl.where(n_mask[None, :], scores, -1e9)

        # ── Online softmax update ─────────────────────────────────────────────
        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        m_i   = m_new

        # ── Load V planes: same compact pattern ───────────────────────────────
        v_blk  = V_planes_ptr + v_base + block_n_start * stride_vn
        v_ptrs = v_blk + n_range[:, None] * stride_vn + p16[None, :]
        b0v_c  = tl.load(v_ptrs,      mask=n_mask_2d, other=0).to(tl.int32)
        b1v_c  = tl.load(v_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2v_c  = tl.load(v_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)

        b0v_3d = tl.reshape(b0v_c, (BLOCK_N, 16, 1))
        b1v_3d = tl.reshape(b1v_c, (BLOCK_N, 16, 1))
        b2v_3d = tl.reshape(b2v_c, (BLOCK_N, 16, 1))

        b0v = tl.reshape((b0v_3d >> bit8) & 1, (BLOCK_N, head_dim))
        b1v = tl.reshape((b1v_3d >> bit8) & 1, (BLOCK_N, head_dim))
        b2v = tl.reshape((b2v_3d >> bit8) & 1, (BLOCK_N, head_dim))

        v_centroids = _bits_to_centroid(b0v, b1v, b2v)

        # ── Accumulate output ─────────────────────────────────────────────────
        v_norms_block = tl.load(
            V_norms_ptr + vn_base + (block_n_start + n_range) * stride_vnn,
            mask=n_mask, other=0.0)
        p_scaled = p * v_norms_block[None, :]

        acc = acc * alpha[:, None] + tl.dot(p_scaled.to(tl.float16),
                                            v_centroids.to(tl.float16),
                                            out_dtype=tl.float32)

    # ── Normalize and write output ────────────────────────────────────────────
    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    out_off  = (batch_idx * stride_ob + head_idx * stride_oh
                + pid_m * BLOCK_M * stride_om)
    out_ptrs = (O_ptr + out_off
                + tl.arange(0, BLOCK_M)[:, None] * stride_om
                + tl.arange(0, head_dim)[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# Fast-decode variant: nibble-packed format (4 bits per index, 64 bytes/token)
# ──────────────────────────────────────────────────────────────────────────────
# Trade 23% compression (64B vs 52B) for trivially cheap bit extraction:
#   byte i = (idx[2i] << 4) | idx[2i+1]   (both in range 0-7, upper nibble used)
# Decode: idx_even = (byte >> 4) & 7,  idx_odd = byte & 7
# Then use _bits_to_centroid as before, or a direct 8-entry tl.where chain.

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 32},  num_warps=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 64},  num_warps=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4),
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 64},  num_warps=4),
    ],
    key=["seq_q", "seq_k", "head_dim"],
)
@triton.jit
def _tq3_nibble_attention_kernel(
    Q_ptr, stride_qb, stride_qh, stride_qm, stride_qd,
    # Nibble-packed K: [B, H, S_k, 64] uint8 (64 bytes = 128 nibbles = 128 indices)
    K_nibbles_ptr, stride_kb, stride_kh, stride_kn,
    K_norms_ptr,   stride_knb, stride_knh, stride_knn,
    V_nibbles_ptr, stride_vb, stride_vh, stride_vn,
    V_norms_ptr,   stride_vnb, stride_vnh, stride_vnn,
    O_ptr, stride_ob, stride_oh, stride_om, stride_od,
    batch: int, heads: int, seq_q: int, seq_k: int,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: float,
):
    """
    Like _tq3_attention_kernel but reads nibble-packed KV (64 bytes/token).

    Bit extraction is 2 ops per pair of dimensions (>> 4 and & 7) instead of
    3 loads + 3 shifts + 2 ORs for bit-plane layout.  About 3× fewer operations
    for the decode step, at the cost of 12 extra bytes per token (23% less compression).
    """
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // heads
    head_idx  = pid_bh %  heads

    q_off  = (batch_idx * stride_qb + head_idx * stride_qh
              + pid_m * BLOCK_M * stride_qm)
    q_ptrs = (Q_ptr + q_off
              + tl.arange(0, BLOCK_M)[:, None] * stride_qm
              + tl.arange(0, head_dim)[None, :] * stride_qd)
    m_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < seq_q
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    k_base  = batch_idx * stride_kb  + head_idx * stride_kh
    kn_base = batch_idx * stride_knb + head_idx * stride_knh
    v_base  = batch_idx * stride_vb  + head_idx * stride_vh
    vn_base = batch_idx * stride_vnb + head_idx * stride_vnh

    n_range = tl.arange(0, BLOCK_N)
    # 64 nibble-bytes cover 128 dims; load even-indexed bytes for even dims,
    # same bytes for odd dims (high nibble → even dim, low nibble → odd dim).
    # d_range = [0,1,2,...,127]; the nibble byte for dim d is d//2.
    d_range   = tl.arange(0, head_dim)         # [head_dim=128]
    nibble_byte = d_range // 2                  # byte index in 64-byte block
    is_even_dim = (d_range % 2 == 0)            # True for dims 0,2,4,...

    for block_n_start in range(0, seq_k, BLOCK_N):
        n_mask    = (block_n_start + n_range) < seq_k
        n_mask_2d = n_mask[:, None]

        # ── Decode K ──────────────────────────────────────────────────────────
        # Load [BLOCK_N, 64] nibble bytes then unpack to [BLOCK_N, 128] indices.
        k_nibble_base = K_nibbles_ptr + k_base + block_n_start * stride_kn
        # Each row in memory: 64 bytes covering 128 dims (2 dims per byte).
        k_nibble_ptrs = (k_nibble_base
                         + n_range[:, None] * stride_kn
                         + nibble_byte[None, :])        # [BLOCK_N, head_dim]
        k_bytes = tl.load(k_nibble_ptrs, mask=n_mask_2d, other=0).to(tl.int32)
        # High nibble → even dimensions, low nibble → odd dimensions
        k_idx = tl.where(is_even_dim[None, :],
                         (k_bytes >> 4) & 7,
                         k_bytes & 7)                   # [BLOCK_N, head_dim] in 0-7

        # Centroid from idx using bit decomposition (no pointer needed)
        b0k = k_idx & 1
        b1k = (k_idx >> 1) & 1
        b2k = (k_idx >> 2) & 1
        k_centroids = _bits_to_centroid(b0k, b1k, b2k)

        raw_scores = tl.dot(q, k_centroids.to(tl.float16).T, out_dtype=tl.float32)
        k_norms_block = tl.load(
            K_norms_ptr + kn_base + (block_n_start + n_range) * stride_knn,
            mask=n_mask, other=0.0)
        scores = raw_scores * (k_norms_block * sm_scale)[None, :]
        scores = tl.where(n_mask[None, :], scores, -1e9)

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        m_i   = m_new

        # ── Decode V ──────────────────────────────────────────────────────────
        v_nibble_base = V_nibbles_ptr + v_base + block_n_start * stride_vn
        v_nibble_ptrs = (v_nibble_base
                         + n_range[:, None] * stride_vn
                         + nibble_byte[None, :])
        v_bytes = tl.load(v_nibble_ptrs, mask=n_mask_2d, other=0).to(tl.int32)
        v_idx = tl.where(is_even_dim[None, :],
                         (v_bytes >> 4) & 7,
                         v_bytes & 7)
        b0v = v_idx & 1
        b1v = (v_idx >> 1) & 1
        b2v = (v_idx >> 2) & 1
        v_centroids = _bits_to_centroid(b0v, b1v, b2v)

        v_norms_block = tl.load(
            V_norms_ptr + vn_base + (block_n_start + n_range) * stride_vnn,
            mask=n_mask, other=0.0)
        p_scaled = p * v_norms_block[None, :]

        acc = acc * alpha[:, None] + tl.dot(p_scaled.to(tl.float16),
                                            v_centroids.to(tl.float16),
                                            out_dtype=tl.float32)

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    out_off  = (batch_idx * stride_ob + head_idx * stride_oh
                + pid_m * BLOCK_M * stride_om)
    out_ptrs = (O_ptr + out_off
                + tl.arange(0, BLOCK_M)[:, None] * stride_om
                + tl.arange(0, head_dim)[None, :] * stride_od)
    tl.store(out_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# Split-K (sequence-parallel) kernels for decode-heavy workloads
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _tq3_splitk_partial_kernel(
    Q_ptr, stride_qb, stride_qh, stride_qm, stride_qd,
    K_planes_ptr, stride_kb, stride_kh, stride_kn,
    K_norms_ptr, stride_knb, stride_knh, stride_knn,
    V_planes_ptr, stride_vb, stride_vh, stride_vn,
    V_norms_ptr, stride_vnb, stride_vnh, stride_vnn,
    Partial_m_ptr, stride_pmb, stride_pmh, stride_pmm, stride_pms,
    Partial_l_ptr, stride_plb, stride_plh, stride_plm, stride_pls,
    Partial_acc_ptr, stride_pab, stride_pah, stride_pam, stride_pas, stride_pae, stride_pad,
    batch: int, heads: int, seq_q: int, seq_k: int,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    sm_scale: float,
):
    """
    One program computes a single (query-block, batch-head, split) partial result.
    Grid: (ceil(S_q/BLOCK_M), B*H, ceil(S_k/BLOCK_N)).
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    pid_split = tl.program_id(2)

    batch_idx = pid_bh // heads
    head_idx = pid_bh % heads

    q_off = (batch_idx * stride_qb + head_idx * stride_qh + pid_m * BLOCK_M * stride_qm)
    q_ptrs = (
        Q_ptr
        + q_off
        + tl.arange(0, BLOCK_M)[:, None] * stride_qm
        + tl.arange(0, head_dim)[None, :] * stride_qd
    )
    m_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < seq_q
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    k_base = batch_idx * stride_kb + head_idx * stride_kh
    kn_base = batch_idx * stride_knb + head_idx * stride_knh
    v_base = batch_idx * stride_vb + head_idx * stride_vh
    vn_base = batch_idx * stride_vnb + head_idx * stride_vnh

    n_range = tl.arange(0, BLOCK_N)
    d_range = tl.arange(0, head_dim)
    byte_in_plane = d_range // 8
    bit_in_byte = d_range % 8

    bit_shift = bit_in_byte[None, :]
    split_start = pid_split * BLOCK_KV
    split_end = tl.minimum(split_start + BLOCK_KV, seq_k)
    for rel_n in range(0, BLOCK_KV, BLOCK_N):
        block_n_start = split_start + rel_n
        n_mask = (block_n_start + n_range) < split_end
        n_mask_2d = n_mask[:, None]

        k_block_base = K_planes_ptr + k_base + block_n_start * stride_kn
        k_byte_ptrs = (
            k_block_base
            + n_range[:, None] * stride_kn
            + byte_in_plane[None, :]
        )
        b0k_raw = tl.load(k_byte_ptrs, mask=n_mask_2d, other=0).to(tl.int32)
        b1k_raw = tl.load(k_byte_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2k_raw = tl.load(k_byte_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)

        b0k = (b0k_raw >> bit_shift) & 1
        b1k = (b1k_raw >> bit_shift) & 1
        b2k = (b2k_raw >> bit_shift) & 1
        k_centroids = _bits_to_centroid(b0k, b1k, b2k)

        raw_scores = tl.dot(q, k_centroids.to(tl.float16).T, out_dtype=tl.float32)
        k_norms_block = tl.load(
            K_norms_ptr + kn_base + (block_n_start + n_range) * stride_knn,
            mask=n_mask,
            other=0.0,
        )
        scores = raw_scores * (k_norms_block * sm_scale)[None, :]
        scores = tl.where(n_mask[None, :], scores, -1e9)

        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        v_block_base = V_planes_ptr + v_base + block_n_start * stride_vn
        v_byte_ptrs = (
            v_block_base
            + n_range[:, None] * stride_vn
            + byte_in_plane[None, :]
        )
        b0v_raw = tl.load(v_byte_ptrs, mask=n_mask_2d, other=0).to(tl.int32)
        b1v_raw = tl.load(v_byte_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2v_raw = tl.load(v_byte_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)
        b0v = (b0v_raw >> bit_shift) & 1
        b1v = (b1v_raw >> bit_shift) & 1
        b2v = (b2v_raw >> bit_shift) & 1
        v_centroids = _bits_to_centroid(b0v, b1v, b2v)

        v_norms_block = tl.load(
            V_norms_ptr + vn_base + (block_n_start + n_range) * stride_vnn,
            mask=n_mask,
            other=0.0,
        )
        p_scaled = p * v_norms_block[None, :]
        acc = acc * alpha[:, None] + tl.dot(
            p_scaled.to(tl.float16), v_centroids.to(tl.float16), out_dtype=tl.float32
        )

    m_ptr = (
        Partial_m_ptr
        + batch_idx * stride_pmb
        + head_idx * stride_pmh
        + pid_m * stride_pmm
        + pid_split * stride_pms
    )
    l_ptr = (
        Partial_l_ptr
        + batch_idx * stride_plb
        + head_idx * stride_plh
        + pid_m * stride_plm
        + pid_split * stride_pls
    )
    tl.store(m_ptr + tl.arange(0, BLOCK_M), m_i, mask=m_mask)
    tl.store(l_ptr + tl.arange(0, BLOCK_M), l_i, mask=m_mask)

    acc_ptrs = (
        Partial_acc_ptr
        + batch_idx * stride_pab
        + head_idx * stride_pah
        + pid_m * stride_pam
        + pid_split * stride_pas
        + tl.arange(0, BLOCK_M)[:, None] * stride_pae
        + tl.arange(0, head_dim)[None, :] * stride_pad
    )
    tl.store(acc_ptrs, acc, mask=m_mask[:, None])


@triton.jit
def _tq3_splitk_reduce_kernel(
    Partial_m_ptr, stride_pmb, stride_pmh, stride_pmm, stride_pms, stride_pme,
    Partial_l_ptr, stride_plb, stride_plh, stride_plm, stride_pls, stride_ple,
    Partial_acc_ptr, stride_pab, stride_pah, stride_pam, stride_pas, stride_pae, stride_pad,
    O_ptr, stride_ob, stride_oh, stride_om, stride_od,
    heads: int, seq_q: int, n_splits: int,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """
    Merge split partials using log-sum-exp algebra:
      m = max_s m_s
      l = sum_s exp(m_s - m) * l_s
      acc = sum_s exp(m_s - m) * acc_s
      out = acc / l
    Grid: (ceil(S_q/BLOCK_M), B*H)
    """
    pid_m = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx = pid_bh // heads
    head_idx = pid_bh % heads

    m_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < seq_q

    m_max = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_sum = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc_sum = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    for s in range(0, n_splits):
        m_ptr = (
            Partial_m_ptr
            + batch_idx * stride_pmb
            + head_idx * stride_pmh
            + pid_m * stride_pmm
            + s * stride_pms
        )
        l_ptr = (
            Partial_l_ptr
            + batch_idx * stride_plb
            + head_idx * stride_plh
            + pid_m * stride_plm
            + s * stride_pls
        )
        m_s = tl.load(m_ptr + tl.arange(0, BLOCK_M) * stride_pme, mask=m_mask, other=-float("inf"))
        l_s = tl.load(l_ptr + tl.arange(0, BLOCK_M) * stride_ple, mask=m_mask, other=0.0)

        acc_ptrs = (
            Partial_acc_ptr
            + batch_idx * stride_pab
            + head_idx * stride_pah
            + pid_m * stride_pam
            + s * stride_pas
            + tl.arange(0, BLOCK_M)[:, None] * stride_pae
            + tl.arange(0, head_dim)[None, :] * stride_pad
        )
        acc_s = tl.load(acc_ptrs, mask=m_mask[:, None], other=0.0)

        m_new = tl.maximum(m_max, m_s)
        a = tl.exp(m_max - m_new)
        b = tl.exp(m_s - m_new)
        l_sum = l_sum * a + l_s * b
        acc_sum = acc_sum * a[:, None] + acc_s * b[:, None]
        m_max = m_new

    l_safe = tl.where(l_sum > 0, l_sum, 1.0)
    out = acc_sum / l_safe[:, None]

    out_off = batch_idx * stride_ob + head_idx * stride_oh + pid_m * BLOCK_M * stride_om
    out_ptrs = (
        O_ptr
        + out_off
        + tl.arange(0, BLOCK_M)[:, None] * stride_om
        + tl.arange(0, head_dim)[None, :] * stride_od
    )
    tl.store(out_ptrs, out.to(tl.float16), mask=m_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# Python entry points
# ──────────────────────────────────────────────────────────────────────────────

def turboquant_attention_fwd(
    q: torch.Tensor,
    k_planes: torch.Tensor,
    k_norms: torch.Tensor,
    v_planes: torch.Tensor,
    v_norms: torch.Tensor,
    rotation: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    BLOCK_M: Optional[int] = None,
    BLOCK_N: Optional[int] = None,
    BLOCK_KV: Optional[int] = None,
    use_split_k: bool = True,
) -> torch.Tensor:
    """
    Fused TQ3 dequantize + attention forward pass (bit-plane format, 52 B/token).

    Parameters
    ----------
    q        : (B, H, S_q, D) float16 — queries, PRE-ROTATED: q @ R^T
    k_planes : (B, H, S_k, 48) uint8 — TQ3 bit-planes for keys
    k_norms  : (B, H, S_k) float32 — key norms
    v_planes : (B, H, S_k, 48) uint8 — TQ3 bit-planes for values
    v_norms  : (B, H, S_k) float32 — value norms
    rotation : (D, D) float32 — rotation matrix R used during compression.
               If provided, applies the inverse rotation (@R) so output is in
               the original space (matching scaled_dot_product_attention).
               If None, output stays in the rotated space (slightly faster).
    sm_scale : float — softmax scale, defaults to 1/sqrt(D)
    BLOCK_M, BLOCK_N : int — override autotuned block sizes (for testing only)
    use_split_k : bool — enable Split-K path for decode-heavy shapes

    Returns
    -------
    output : (B, H, S_q, D) float16
    """
    B, H, S_q, D = q.shape
    _, _, S_k, _  = k_planes.shape

    assert D == 128, f"Only head_dim=128 supported, got {D}"
    assert k_planes.shape[-1] == 48, "k_planes must have 48 bytes per token (bit-plane format)"
    assert k_planes.dtype == torch.uint8
    assert k_norms.dtype == torch.float32

    if sm_scale is None:
        sm_scale = float(D ** -0.5)

    # Ensure contiguous layout for pointer-arithmetic correctness
    q        = q.contiguous()
    k_planes = k_planes.contiguous()
    k_norms  = k_norms.contiguous()
    v_planes = v_planes.contiguous()
    v_norms  = v_norms.contiguous()

    out = torch.empty(B, H, S_q, D, dtype=torch.float16, device=q.device)

    use_splitk_path = use_split_k and S_q <= 16 and S_k >= 4096

    if use_splitk_path:
        split_block_n = BLOCK_N if BLOCK_N is not None else 64
        split_block_m = BLOCK_M if BLOCK_M is not None else 16
        split_kv = BLOCK_KV if BLOCK_KV is not None else 2048
        split_kv = max(split_kv, split_block_n)
        n_splits = triton.cdiv(S_k, split_kv)

        partial_m = torch.empty((B, H, triton.cdiv(S_q, split_block_m), n_splits, split_block_m),
                                dtype=torch.float32, device=q.device)
        partial_l = torch.empty_like(partial_m)
        partial_acc = torch.empty(
            (B, H, triton.cdiv(S_q, split_block_m), n_splits, split_block_m, D),
            dtype=torch.float32, device=q.device
        )

        grid_partial = (triton.cdiv(S_q, split_block_m), B * H, n_splits)
        _tq3_splitk_partial_kernel[grid_partial](
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_planes, k_planes.stride(0), k_planes.stride(1), k_planes.stride(2),
            k_norms, k_norms.stride(0), k_norms.stride(1), k_norms.stride(2),
            v_planes, v_planes.stride(0), v_planes.stride(1), v_planes.stride(2),
            v_norms, v_norms.stride(0), v_norms.stride(1), v_norms.stride(2),
            partial_m, partial_m.stride(0), partial_m.stride(1), partial_m.stride(2), partial_m.stride(3),
            partial_l, partial_l.stride(0), partial_l.stride(1), partial_l.stride(2), partial_l.stride(3),
            partial_acc, partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3), partial_acc.stride(4), partial_acc.stride(5),
            B, H, S_q, S_k,
            head_dim=D,
            BLOCK_M=split_block_m,
            BLOCK_N=split_block_n,
            BLOCK_KV=split_kv,
            sm_scale=sm_scale,
        )

        # Merge partials in PyTorch (GPU) using the stable softmax combine rule.
        # This avoids runtime-trip-count constraints in Triton reduction loops.
        m = partial_m.max(dim=3).values
        w = torch.exp(partial_m - m.unsqueeze(3))
        l = (w * partial_l).sum(dim=3)
        acc = (w.unsqueeze(-1) * partial_acc).sum(dim=3)
        out_blocks = (acc / l.clamp_min(1e-12).unsqueeze(-1)).to(torch.float16)
        out_reshaped = out_blocks.reshape(B, H, -1, D)
        out.copy_(out_reshaped[:, :, :S_q, :])
    elif BLOCK_M is not None and BLOCK_N is not None:
        # Manual block size — bypass autotuning (useful for ablation)
        grid = (triton.cdiv(S_q, BLOCK_M), B * H)
        _tq3_attention_kernel[grid](
            q,        q.stride(0),       q.stride(1),       q.stride(2),       q.stride(3),
            k_planes, k_planes.stride(0), k_planes.stride(1), k_planes.stride(2),
            k_norms,  k_norms.stride(0),  k_norms.stride(1),  k_norms.stride(2),
            v_planes, v_planes.stride(0), v_planes.stride(1), v_planes.stride(2),
            v_norms,  v_norms.stride(0),  v_norms.stride(1),  v_norms.stride(2),
            out,      out.stride(0),      out.stride(1),      out.stride(2),      out.stride(3),
            B, H, S_q, S_k,
            head_dim=D,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            sm_scale=sm_scale,
        )
    else:
        # Let Triton autotune choose (BLOCK_M, BLOCK_N) — grid uses a dummy shape
        # that the autotune key maps to; the actual grid is recomputed inside configs.
        def grid_fn(meta):
            return (triton.cdiv(S_q, meta["BLOCK_M"]), B * H)
        _tq3_attention_kernel[grid_fn](
            q,        q.stride(0),       q.stride(1),       q.stride(2),       q.stride(3),
            k_planes, k_planes.stride(0), k_planes.stride(1), k_planes.stride(2),
            k_norms,  k_norms.stride(0),  k_norms.stride(1),  k_norms.stride(2),
            v_planes, v_planes.stride(0), v_planes.stride(1), v_planes.stride(2),
            v_norms,  v_norms.stride(0),  v_norms.stride(1),  v_norms.stride(2),
            out,      out.stride(0),      out.stride(1),      out.stride(2),      out.stride(3),
            B, H, S_q, S_k,
            head_dim=D,
            sm_scale=sm_scale,
        )

    if rotation is not None:
        R = rotation.to(q.device).float()
        n_q = B * H * S_q
        out = (out.float().reshape(n_q, D) @ R).reshape(B, H, S_q, D).half()

    return out


def turboquant_attention_v3(
    q: torch.Tensor,
    k_planes: torch.Tensor,
    k_norms: torch.Tensor,
    v_planes: torch.Tensor,
    v_norms: torch.Tensor,
    rotation: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    v3 fused TQ3 dequantize + attention — compact plane loads (8× fewer VMEM instr).

    Same interface as turboquant_attention_fwd; drop-in replacement.
    Uses _tq3v3_attention_kernel which loads [BLOCK_N, 16] bytes per plane
    instead of [BLOCK_N, 128], eliminating the 8× VMEM instruction duplication
    that caused v2 to stall at ~25 GB/s effective bandwidth.

    Requires Triton ≥ 3.1 for 3D tl.reshape support.

    Parameters
    ----------
    q        : (B, H, S_q, D) float16 — queries PRE-ROTATED (q @ R^T)
    k_planes : (B, H, S_k, 48) uint8 — TQ3 bit-planes for keys
    k_norms  : (B, H, S_k) float32
    v_planes : (B, H, S_k, 48) uint8 — TQ3 bit-planes for values
    v_norms  : (B, H, S_k) float32
    rotation : (D, D) float32 — if provided, output is rotated back to original space
    sm_scale : float — defaults to 1/sqrt(D)

    Returns
    -------
    output : (B, H, S_q, D) float16
    """
    B, H, S_q, D = q.shape
    _, _, S_k, _  = k_planes.shape

    assert D == 128, f"Only head_dim=128 supported, got {D}"
    assert k_planes.shape[-1] == 48
    assert k_planes.dtype == torch.uint8
    assert k_norms.dtype == torch.float32

    if sm_scale is None:
        sm_scale = float(D ** -0.5)

    q        = q.contiguous()
    k_planes = k_planes.contiguous()
    k_norms  = k_norms.contiguous()
    v_planes = v_planes.contiguous()
    v_norms  = v_norms.contiguous()

    out = torch.empty(B, H, S_q, D, dtype=torch.float16, device=q.device)

    def grid_fn(meta):
        return (triton.cdiv(S_q, meta["BLOCK_M"]), B * H)

    _tq3v3_attention_kernel[grid_fn](
        q,        q.stride(0),        q.stride(1),        q.stride(2),        q.stride(3),
        k_planes, k_planes.stride(0), k_planes.stride(1), k_planes.stride(2),
        k_norms,  k_norms.stride(0),  k_norms.stride(1),  k_norms.stride(2),
        v_planes, v_planes.stride(0), v_planes.stride(1), v_planes.stride(2),
        v_norms,  v_norms.stride(0),  v_norms.stride(1),  v_norms.stride(2),
        out,      out.stride(0),      out.stride(1),       out.stride(2),       out.stride(3),
        B, H, S_q, S_k,
        head_dim=D,
        sm_scale=sm_scale,
    )

    if rotation is not None:
        R = rotation.to(q.device).float()
        n_q = B * H * S_q
        out = (out.float().reshape(n_q, D) @ R).reshape(B, H, S_q, D).half()

    return out


def turboquant_nibble_attention_fwd(
    q: torch.Tensor,
    k_nibbles: torch.Tensor,
    k_norms: torch.Tensor,
    v_nibbles: torch.Tensor,
    v_norms: torch.Tensor,
    rotation: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused TQ3 attention with nibble-packed KV cache (64 bytes/token, 4× compression).

    Nibble format: k_nibbles[..., i] = (idx[2i] << 4) | idx[2i+1]
    Easier to decode than bit-plane format (~3× fewer bit-ops per element),
    at the cost of 12 extra bytes per token vs bit-plane (52B → 64B).

    Parameters
    ----------
    q         : (B, H, S_q, D) float16 — queries, PRE-ROTATED: q @ R^T
    k_nibbles : (B, H, S_k, 64) uint8
    k_norms   : (B, H, S_k) float32
    v_nibbles : (B, H, S_k, 64) uint8
    v_norms   : (B, H, S_k) float32
    rotation  : (D, D) float32 — optional inverse rotation
    sm_scale  : float

    Returns
    -------
    output : (B, H, S_q, D) float16
    """
    B, H, S_q, D = q.shape
    _, _, S_k, _  = k_nibbles.shape

    assert D == 128
    assert k_nibbles.shape[-1] == 64, "k_nibbles must have 64 bytes per token (nibble format)"
    assert k_nibbles.dtype == torch.uint8
    assert k_norms.dtype == torch.float32

    if sm_scale is None:
        sm_scale = float(D ** -0.5)

    q        = q.contiguous()
    k_nibbles = k_nibbles.contiguous()
    k_norms  = k_norms.contiguous()
    v_nibbles = v_nibbles.contiguous()
    v_norms  = v_norms.contiguous()

    out = torch.empty(B, H, S_q, D, dtype=torch.float16, device=q.device)

    def grid_fn(meta):
        return (triton.cdiv(S_q, meta["BLOCK_M"]), B * H)

    _tq3_nibble_attention_kernel[grid_fn](
        q,         q.stride(0),         q.stride(1),         q.stride(2),         q.stride(3),
        k_nibbles, k_nibbles.stride(0), k_nibbles.stride(1), k_nibbles.stride(2),
        k_norms,   k_norms.stride(0),   k_norms.stride(1),   k_norms.stride(2),
        v_nibbles, v_nibbles.stride(0), v_nibbles.stride(1), v_nibbles.stride(2),
        v_norms,   v_norms.stride(0),   v_norms.stride(1),   v_norms.stride(2),
        out,       out.stride(0),       out.stride(1),        out.stride(2),        out.stride(3),
        B, H, S_q, S_k,
        head_dim=D,
        sm_scale=sm_scale,
    )

    if rotation is not None:
        R = rotation.to(q.device).float()
        n_q = B * H * S_q
        out = (out.float().reshape(n_q, D) @ R).reshape(B, H, S_q, D).half()

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Utilities: compress KV to the bit-plane format expected by this kernel
# ──────────────────────────────────────────────────────────────────────────────

def compress_kv_for_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    tq_engine,
) -> tuple:
    """
    Compress (B, H, S, D) KV tensors into the bit-plane format for
    turboquant_attention_fwd.

    Returns (k_planes, k_norms, v_planes, v_norms) where:
      k_planes : (B, H, S, 48) uint8 — bit-plane format (52 B/token with norm)
      k_norms  : (B, H, S)     float32
      v_planes : (B, H, S, 48) uint8
      v_norms  : (B, H, S)     float32
    """
    B, H, S, D = k.shape
    n = B * H * S

    k_fp32 = k.reshape(n, D).float().contiguous()
    v_fp32 = v.reshape(n, D).float().contiguous()

    # compress_tensor returns (n, 52) uint8: [norm:4 bytes][planes:48 bytes]
    k_comp = tq_engine.compress_tensor(k_fp32)
    v_comp = tq_engine.compress_tensor(v_fp32)

    k_norms_raw  = k_comp[:, :4].contiguous()
    k_planes_raw = k_comp[:, 4:]
    v_norms_raw  = v_comp[:, :4].contiguous()
    v_planes_raw = v_comp[:, 4:]

    k_norms_f32 = k_norms_raw.view(-1).view(torch.float32).view(n)
    v_norms_f32 = v_norms_raw.view(-1).view(torch.float32).view(n)

    return (
        k_planes_raw.view(B, H, S, 48),
        k_norms_f32.view(B, H, S),
        v_planes_raw.view(B, H, S, 48),
        v_norms_f32.view(B, H, S),
    )


def compress_kv_nibble(
    k: torch.Tensor,
    v: torch.Tensor,
    tq_engine,
) -> tuple:
    """
    Compress (B, H, S, D) KV tensors into nibble-packed format for
    turboquant_nibble_attention_fwd.

    The nibble format packs two 3-bit indices per byte:
      byte[i] = (idx[2i] << 4) | idx[2i+1]

    Returns (k_nibbles, k_norms, v_nibbles, v_norms) where:
      k_nibbles : (B, H, S, 64) uint8
      k_norms   : (B, H, S)     float32
      v_nibbles : (B, H, S, 64) uint8
      v_norms   : (B, H, S)     float32
    """
    B, H, S, D = k.shape
    n = B * H * S

    k_fp32 = k.reshape(n, D).float().contiguous()
    v_fp32 = v.reshape(n, D).float().contiguous()

    k_comp = tq_engine.compress_tensor(k_fp32)  # (n, 52) uint8
    v_comp = tq_engine.compress_tensor(v_fp32)

    def _to_nibbles(comp: torch.Tensor) -> tuple:
        """Convert (n, 52) bit-plane compressed to (n, 64) nibble-packed + norms."""
        norms_raw  = comp[:, :4].contiguous()
        planes_raw = comp[:, 4:]              # (n, 48) bit-planes

        norms_f32 = norms_raw.view(-1).view(torch.float32).view(n)

        # Reconstruct 128 3-bit indices from bit-planes
        # planes_raw[i, p*16 + byte_pos]: plane p, byte at position byte_pos
        p0 = planes_raw[:, :16]              # (n, 16) bytes — plane 0 (LSB)
        p1 = planes_raw[:, 16:32]            # plane 1
        p2 = planes_raw[:, 32:]              # plane 2 (MSB)

        # For each byte, extract 8 bits → 8 indices, then pack as 4 nibble-pairs
        # Each plane byte covers 8 dimensions; we have 16 bytes × 3 planes = 128 dims
        # nibble_bytes[i, j] packs dims 2j and 2j+1 into one byte (64 bytes total)
        idx_all = torch.zeros(n, D, dtype=torch.uint8, device=comp.device)
        for byte_pos in range(16):          # 16 bytes per plane
            for bit in range(8):            # 8 bits per byte
                d = byte_pos * 8 + bit
                bit0 = (p0[:, byte_pos].int() >> bit) & 1
                bit1 = (p1[:, byte_pos].int() >> bit) & 1
                bit2 = (p2[:, byte_pos].int() >> bit) & 1
                idx_all[:, d] = ((bit2 << 2) | (bit1 << 1) | bit0).byte()

        # Pack pairs into nibbles: byte i = (idx[2i] << 4) | idx[2i+1]
        idx_even = idx_all[:, 0::2].int()    # (n, 64)
        idx_odd  = idx_all[:, 1::2].int()    # (n, 64)
        nibbles  = ((idx_even << 4) | idx_odd).byte()  # (n, 64) uint8

        return nibbles, norms_f32

    k_nibbles, k_norms_f32 = _to_nibbles(k_comp)
    v_nibbles, v_norms_f32 = _to_nibbles(v_comp)

    return (
        k_nibbles.view(B, H, S, 64),
        k_norms_f32.view(B, H, S),
        v_nibbles.view(B, H, S, 64),
        v_norms_f32.view(B, H, S),
    )


# ──────────────────────────────────────────────────────────────────────────────
# GQA-aware kernel: index KV by H_kv, fan out to gqa_ratio Q heads inside the tile
#
# Q is (B, H_q, S_q, D); K/V are (B, H_kv, S_k, 48) with H_q = H_kv * gqa_ratio.
# Each program handles one (B, H_kv) pair and a tile of BLOCK_M rows drawn from
# the gqa_ratio * S_q rows that share that KV head — K is loaded once per
# BLOCK_N and reused across all gqa_ratio Q heads in the group, dropping HBM
# traffic by gqa_ratio× vs the expand+MHA path.
#
# M-mapping: m_off = pid_m * BLOCK_M + arange(BLOCK_M)
#            s_q_off  = m_off // gqa_ratio   (which query position)
#            g_off    = m_off %  gqa_ratio   (which Q in group)
#            q_head   = head_kv_idx * gqa_ratio + g_off
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M":  4, "BLOCK_N":  64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M":  4, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M":  8, "BLOCK_N":  64}, num_warps=2, num_stages=2),
        triton.Config({"BLOCK_M":  8, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N":  64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 128}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 32, "BLOCK_N":  64}, num_warps=4, num_stages=2),
    ],
    key=["seq_q", "seq_k", "head_dim", "gqa_ratio"],
)
@triton.jit
def _tq3_gqa_attention_kernel(
    Q_ptr, stride_qb, stride_qh, stride_qm, stride_qd,
    K_planes_ptr, stride_kb, stride_kh, stride_kn,
    K_norms_ptr, stride_knb, stride_knh, stride_knn,
    V_planes_ptr, stride_vb, stride_vh, stride_vn,
    V_norms_ptr, stride_vnb, stride_vnh, stride_vnn,
    O_ptr, stride_ob, stride_oh, stride_om, stride_od,
    batch: int, h_kv: int, seq_q: int, seq_k: int,
    head_dim: tl.constexpr,
    gqa_ratio: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: float,
):
    pid_m  = tl.program_id(0)
    pid_bh = tl.program_id(1)
    batch_idx   = pid_bh // h_kv
    head_kv_idx = pid_bh %  h_kv

    m_off    = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    M_total  = gqa_ratio * seq_q
    m_mask   = m_off < M_total
    g_off    = m_off % gqa_ratio
    s_q_off  = m_off // gqa_ratio
    q_head   = head_kv_idx * gqa_ratio + g_off

    q_ptrs = (Q_ptr
              + batch_idx * stride_qb
              + q_head[:, None] * stride_qh
              + s_q_off[:, None] * stride_qm
              + tl.arange(0, head_dim)[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    k_base  = batch_idx * stride_kb  + head_kv_idx * stride_kh
    kn_base = batch_idx * stride_knb + head_kv_idx * stride_knh
    v_base  = batch_idx * stride_vb  + head_kv_idx * stride_vh
    vn_base = batch_idx * stride_vnb + head_kv_idx * stride_vnh

    n_range = tl.arange(0, BLOCK_N)
    d_range = tl.arange(0, head_dim)
    byte_in_plane = d_range // 8
    bit_in_byte   = d_range % 8
    bit_shift     = bit_in_byte[None, :]

    for block_n_start in range(0, seq_k, BLOCK_N):
        n_mask    = (block_n_start + n_range) < seq_k
        n_mask_2d = n_mask[:, None]

        k_block_base = K_planes_ptr + k_base + block_n_start * stride_kn
        k_byte_ptrs  = (k_block_base
                        + n_range[:, None] * stride_kn
                        + byte_in_plane[None, :])
        b0k_raw = tl.load(k_byte_ptrs,      mask=n_mask_2d, other=0).to(tl.int32)
        b1k_raw = tl.load(k_byte_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2k_raw = tl.load(k_byte_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)
        b0k = (b0k_raw >> bit_shift) & 1
        b1k = (b1k_raw >> bit_shift) & 1
        b2k = (b2k_raw >> bit_shift) & 1
        k_centroids = _bits_to_centroid(b0k, b1k, b2k)

        raw_scores = tl.dot(q, k_centroids.to(tl.float16).T, out_dtype=tl.float32)

        k_norms_block = tl.load(
            K_norms_ptr + kn_base + (block_n_start + n_range) * stride_knn,
            mask=n_mask, other=0.0)
        scores = raw_scores * (k_norms_block * sm_scale)[None, :]
        scores = tl.where(n_mask[None, :], scores, -1e9)

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        m_i   = m_new

        v_block_base = V_planes_ptr + v_base + block_n_start * stride_vn
        v_byte_ptrs  = (v_block_base
                        + n_range[:, None] * stride_vn
                        + byte_in_plane[None, :])
        b0v_raw = tl.load(v_byte_ptrs,      mask=n_mask_2d, other=0).to(tl.int32)
        b1v_raw = tl.load(v_byte_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2v_raw = tl.load(v_byte_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)
        b0v = (b0v_raw >> bit_shift) & 1
        b1v = (b1v_raw >> bit_shift) & 1
        b2v = (b2v_raw >> bit_shift) & 1
        v_centroids = _bits_to_centroid(b0v, b1v, b2v)

        v_norms_block = tl.load(
            V_norms_ptr + vn_base + (block_n_start + n_range) * stride_vnn,
            mask=n_mask, other=0.0)
        p_scaled = p * v_norms_block[None, :]
        acc = acc * alpha[:, None] + tl.dot(p_scaled.to(tl.float16),
                                            v_centroids.to(tl.float16),
                                            out_dtype=tl.float32)

    l_safe = tl.where(l_i > 0, l_i, 1.0)
    acc = acc / l_safe[:, None]

    o_ptrs = (O_ptr
              + batch_idx * stride_ob
              + q_head[:, None] * stride_oh
              + s_q_off[:, None] * stride_om
              + tl.arange(0, head_dim)[None, :] * stride_od)
    tl.store(o_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


@triton.jit
def _tq3_gqa_splitk_partial_kernel(
    Q_ptr, stride_qb, stride_qh, stride_qm, stride_qd,
    K_planes_ptr, stride_kb, stride_kh, stride_kn,
    K_norms_ptr, stride_knb, stride_knh, stride_knn,
    V_planes_ptr, stride_vb, stride_vh, stride_vn,
    V_norms_ptr, stride_vnb, stride_vnh, stride_vnn,
    Partial_m_ptr,   stride_pmb, stride_pmh, stride_pmm, stride_pms,
    Partial_l_ptr,   stride_plb, stride_plh, stride_plm, stride_pls,
    Partial_acc_ptr, stride_pab, stride_pah, stride_pam, stride_pas, stride_pae, stride_pad,
    batch: int, h_kv: int, seq_q: int, seq_k: int,
    head_dim: tl.constexpr,
    gqa_ratio: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    sm_scale: float,
):
    """
    GQA Split-K partial. Grid: (cdiv(gqa_ratio*S_q, BLOCK_M), B*H_kv, n_splits).
    Partial layout: (B, H_kv, M_tiles, n_splits, BLOCK_M[, D]) — reduce in PyTorch
    then permute to (B, H_q, S_q, D).
    """
    pid_m     = tl.program_id(0)
    pid_bh    = tl.program_id(1)
    pid_split = tl.program_id(2)
    batch_idx   = pid_bh // h_kv
    head_kv_idx = pid_bh %  h_kv

    m_off    = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    M_total  = gqa_ratio * seq_q
    m_mask   = m_off < M_total
    g_off    = m_off % gqa_ratio
    s_q_off  = m_off // gqa_ratio
    q_head   = head_kv_idx * gqa_ratio + g_off

    q_ptrs = (Q_ptr
              + batch_idx * stride_qb
              + q_head[:, None] * stride_qh
              + s_q_off[:, None] * stride_qm
              + tl.arange(0, head_dim)[None, :] * stride_qd)
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float16)

    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    k_base  = batch_idx * stride_kb  + head_kv_idx * stride_kh
    kn_base = batch_idx * stride_knb + head_kv_idx * stride_knh
    v_base  = batch_idx * stride_vb  + head_kv_idx * stride_vh
    vn_base = batch_idx * stride_vnb + head_kv_idx * stride_vnh

    n_range = tl.arange(0, BLOCK_N)
    d_range = tl.arange(0, head_dim)
    byte_in_plane = d_range // 8
    bit_in_byte   = d_range % 8
    bit_shift     = bit_in_byte[None, :]

    split_start = pid_split * BLOCK_KV
    split_end   = tl.minimum(split_start + BLOCK_KV, seq_k)
    for rel_n in range(0, BLOCK_KV, BLOCK_N):
        block_n_start = split_start + rel_n
        n_mask    = (block_n_start + n_range) < split_end
        n_mask_2d = n_mask[:, None]

        k_block_base = K_planes_ptr + k_base + block_n_start * stride_kn
        k_byte_ptrs  = (k_block_base
                        + n_range[:, None] * stride_kn
                        + byte_in_plane[None, :])
        b0k_raw = tl.load(k_byte_ptrs,      mask=n_mask_2d, other=0).to(tl.int32)
        b1k_raw = tl.load(k_byte_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2k_raw = tl.load(k_byte_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)
        b0k = (b0k_raw >> bit_shift) & 1
        b1k = (b1k_raw >> bit_shift) & 1
        b2k = (b2k_raw >> bit_shift) & 1
        k_centroids = _bits_to_centroid(b0k, b1k, b2k)

        raw_scores = tl.dot(q, k_centroids.to(tl.float16).T, out_dtype=tl.float32)
        k_norms_block = tl.load(
            K_norms_ptr + kn_base + (block_n_start + n_range) * stride_knn,
            mask=n_mask, other=0.0)
        scores = raw_scores * (k_norms_block * sm_scale)[None, :]
        scores = tl.where(n_mask[None, :], scores, -1e9)

        m_ij  = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p     = tl.exp(scores - m_new[:, None])
        l_i   = l_i * alpha + tl.sum(p, axis=1)
        m_i   = m_new

        v_block_base = V_planes_ptr + v_base + block_n_start * stride_vn
        v_byte_ptrs  = (v_block_base
                        + n_range[:, None] * stride_vn
                        + byte_in_plane[None, :])
        b0v_raw = tl.load(v_byte_ptrs,      mask=n_mask_2d, other=0).to(tl.int32)
        b1v_raw = tl.load(v_byte_ptrs + 16, mask=n_mask_2d, other=0).to(tl.int32)
        b2v_raw = tl.load(v_byte_ptrs + 32, mask=n_mask_2d, other=0).to(tl.int32)
        b0v = (b0v_raw >> bit_shift) & 1
        b1v = (b1v_raw >> bit_shift) & 1
        b2v = (b2v_raw >> bit_shift) & 1
        v_centroids = _bits_to_centroid(b0v, b1v, b2v)

        v_norms_block = tl.load(
            V_norms_ptr + vn_base + (block_n_start + n_range) * stride_vnn,
            mask=n_mask, other=0.0)
        p_scaled = p * v_norms_block[None, :]
        acc = acc * alpha[:, None] + tl.dot(p_scaled.to(tl.float16),
                                            v_centroids.to(tl.float16),
                                            out_dtype=tl.float32)

    m_ptr = (Partial_m_ptr
             + batch_idx * stride_pmb
             + head_kv_idx * stride_pmh
             + pid_m * stride_pmm
             + pid_split * stride_pms)
    l_ptr = (Partial_l_ptr
             + batch_idx * stride_plb
             + head_kv_idx * stride_plh
             + pid_m * stride_plm
             + pid_split * stride_pls)
    tl.store(m_ptr + tl.arange(0, BLOCK_M), m_i, mask=m_mask)
    tl.store(l_ptr + tl.arange(0, BLOCK_M), l_i, mask=m_mask)

    acc_ptrs = (Partial_acc_ptr
                + batch_idx * stride_pab
                + head_kv_idx * stride_pah
                + pid_m * stride_pam
                + pid_split * stride_pas
                + tl.arange(0, BLOCK_M)[:, None] * stride_pae
                + tl.arange(0, head_dim)[None, :] * stride_pad)
    tl.store(acc_ptrs, acc, mask=m_mask[:, None])


def turboquant_gqa_attention_fwd(
    q: torch.Tensor,
    k_planes: torch.Tensor,
    k_norms: torch.Tensor,
    v_planes: torch.Tensor,
    v_norms: torch.Tensor,
    gqa_ratio: int,
    rotation: Optional[torch.Tensor] = None,
    sm_scale: Optional[float] = None,
    use_split_k: bool = True,
) -> torch.Tensor:
    """
    GQA-aware fused TQ3 dequantize + attention.

    Q is (B, H_q, S_q, D); K/V planes are (B, H_kv, S_k, 48) with H_q = H_kv * gqa_ratio.
    The kernel reads each KV head once and fans out across the gqa_ratio Q heads
    in its group — vs the expand+MHA path which duplicates compressed bytes
    gqa_ratio× along the head axis.

    gqa_ratio == 1 forwards to turboquant_attention_fwd (the MHA-tuned kernel),
    so this is a strict superset.
    """
    B, H_q, S_q, D = q.shape
    _, H_kv, S_k, _ = k_planes.shape

    assert D == 128, f"Only head_dim=128 supported, got {D}"
    assert H_q == H_kv * gqa_ratio, (
        f"H_q ({H_q}) must equal H_kv ({H_kv}) * gqa_ratio ({gqa_ratio})"
    )
    assert k_planes.shape[-1] == 48
    assert k_planes.dtype == torch.uint8
    assert k_norms.dtype == torch.float32

    if gqa_ratio == 1:
        return turboquant_attention_fwd(
            q, k_planes, k_norms, v_planes, v_norms,
            rotation=rotation, sm_scale=sm_scale, use_split_k=use_split_k,
        )

    if sm_scale is None:
        sm_scale = float(D ** -0.5)

    q        = q.contiguous()
    k_planes = k_planes.contiguous()
    k_norms  = k_norms.contiguous()
    v_planes = v_planes.contiguous()
    v_norms  = v_norms.contiguous()

    out = torch.empty(B, H_q, S_q, D, dtype=torch.float16, device=q.device)
    M_total = gqa_ratio * S_q
    use_splitk_path = use_split_k and S_q <= 16 and S_k >= 4096

    if use_splitk_path:
        split_block_n = 64
        split_block_m = max(gqa_ratio, 4)
        split_kv = max(2048, split_block_n)
        n_splits = triton.cdiv(S_k, split_kv)
        m_tiles  = triton.cdiv(M_total, split_block_m)

        partial_m = torch.empty(
            (B, H_kv, m_tiles, n_splits, split_block_m),
            dtype=torch.float32, device=q.device,
        )
        partial_l = torch.empty_like(partial_m)
        partial_acc = torch.empty(
            (B, H_kv, m_tiles, n_splits, split_block_m, D),
            dtype=torch.float32, device=q.device,
        )

        grid_partial = (m_tiles, B * H_kv, n_splits)
        _tq3_gqa_splitk_partial_kernel[grid_partial](
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_planes, k_planes.stride(0), k_planes.stride(1), k_planes.stride(2),
            k_norms,  k_norms.stride(0),  k_norms.stride(1),  k_norms.stride(2),
            v_planes, v_planes.stride(0), v_planes.stride(1), v_planes.stride(2),
            v_norms,  v_norms.stride(0),  v_norms.stride(1),  v_norms.stride(2),
            partial_m,   partial_m.stride(0),   partial_m.stride(1),   partial_m.stride(2),   partial_m.stride(3),
            partial_l,   partial_l.stride(0),   partial_l.stride(1),   partial_l.stride(2),   partial_l.stride(3),
            partial_acc, partial_acc.stride(0), partial_acc.stride(1), partial_acc.stride(2), partial_acc.stride(3), partial_acc.stride(4), partial_acc.stride(5),
            B, H_kv, S_q, S_k,
            head_dim=D,
            gqa_ratio=gqa_ratio,
            BLOCK_M=split_block_m,
            BLOCK_N=split_block_n,
            BLOCK_KV=split_kv,
            sm_scale=sm_scale,
        )

        # Stable softmax merge across the n_splits axis (dim=3).
        m = partial_m.max(dim=3).values
        w = torch.exp(partial_m - m.unsqueeze(3))
        l = (w * partial_l).sum(dim=3)
        acc = (w.unsqueeze(-1) * partial_acc).sum(dim=3)
        out_blocks = (acc / l.clamp_min(1e-12).unsqueeze(-1)).to(torch.float16)
        # (B, H_kv, m_tiles, BLOCK_M, D) -> (B, H_kv, m_tiles*BLOCK_M, D), trim, then
        # reshape to (B, H_kv, S_q, gqa_ratio, D) — m flat layout is s_q*gqa_ratio + g.
        out_flat = out_blocks.reshape(B, H_kv, m_tiles * split_block_m, D)[:, :, :M_total, :]
        out_5d = out_flat.reshape(B, H_kv, S_q, gqa_ratio, D)
        # Permute to (B, H_kv, gqa_ratio, S_q, D) so merging dims 1,2 yields q_head = h_kv*ratio + g.
        out.copy_(out_5d.permute(0, 1, 3, 2, 4).reshape(B, H_q, S_q, D))
    else:
        def grid_fn(meta):
            return (triton.cdiv(M_total, meta["BLOCK_M"]), B * H_kv)
        _tq3_gqa_attention_kernel[grid_fn](
            q, q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k_planes, k_planes.stride(0), k_planes.stride(1), k_planes.stride(2),
            k_norms,  k_norms.stride(0),  k_norms.stride(1),  k_norms.stride(2),
            v_planes, v_planes.stride(0), v_planes.stride(1), v_planes.stride(2),
            v_norms,  v_norms.stride(0),  v_norms.stride(1),  v_norms.stride(2),
            out,      out.stride(0),      out.stride(1),       out.stride(2),       out.stride(3),
            B, H_kv, S_q, S_k,
            head_dim=D,
            gqa_ratio=gqa_ratio,
            sm_scale=sm_scale,
        )

    if rotation is not None:
        R = rotation.to(q.device).float()
        n_q = B * H_q * S_q
        out = (out.float().reshape(n_q, D) @ R).reshape(B, H_q, S_q, D).half()

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Self-test: verify output, then benchmark throughput
# ──────────────────────────────────────────────────────────────────────────────

def _test_triton_attention():
    """Verify fused attention output matches reference (cosine sim > 0.90)."""
    if not torch.cuda.is_available():
        print("No GPU available — skipping Triton test.")
        return

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from turboquant_mi300x import TurboQuantMI300X

    print("=" * 70)
    print("Testing Triton fused TQ3 attention (bit-plane + nibble formats)...")
    B, H, S_q, S_k, D = 1, 4, 64, 512, 128

    tq = TurboQuantMI300X(bits=3, device="cuda")
    sm_scale = D ** -0.5

    torch.manual_seed(0)
    q = torch.randn(B, H, S_q, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S_k, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S_k, D, device="cuda", dtype=torch.float16)

    # ── Bit-plane format ───────────────────────────────────────────────────────
    k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k, v, tq)

    k_norms_ref = k.float().reshape(-1, D).norm(dim=-1).reshape(B, H, S_k)
    norm_err = (k_norms - k_norms_ref).abs().max().item()
    print(f"  Norm extraction error (bit-plane):  {norm_err:.6f}")

    q_rot = tq.rotate_queries(q.float()).half()

    out_fused = turboquant_attention_fwd(
        q_rot, k_planes, k_norms, v_planes, v_norms,
        rotation=tq.rotation, sm_scale=sm_scale,
    )

    from torch.nn.functional import scaled_dot_product_attention
    k_decomp = tq.decompress_tensor(
        torch.cat([k_norms.reshape(-1, 1).view(torch.uint8).view(-1, 4),
                   k_planes.reshape(-1, 48)], dim=1),
        (B, H, S_k, D),
    ).half()
    v_decomp = tq.decompress_tensor(
        torch.cat([v_norms.reshape(-1, 1).view(torch.uint8).view(-1, 4),
                   v_planes.reshape(-1, 48)], dim=1),
        (B, H, S_k, D),
    ).half()
    out_ref = scaled_dot_product_attention(q, k_decomp, v_decomp, scale=sm_scale)

    if out_fused.isnan().any():
        print("  FAIL (bit-plane): output contains NaN!")
        return
    cos_bp = torch.nn.functional.cosine_similarity(
        out_fused.reshape(-1).float(), out_ref.reshape(-1).float(), dim=0).item()
    mse_bp = (out_fused.float() - out_ref.float()).pow(2).mean().item()
    print(f"  Bit-plane cosine sim: {cos_bp:.4f}  MSE: {mse_bp:.6f}",
          "✓" if cos_bp > 0.90 else "FAIL")

    # ── Nibble format ──────────────────────────────────────────────────────────
    k_nibbles, k_norms_n, v_nibbles, v_norms_n = compress_kv_nibble(k, v, tq)
    out_nibble = turboquant_nibble_attention_fwd(
        q_rot, k_nibbles, k_norms_n, v_nibbles, v_norms_n,
        rotation=tq.rotation, sm_scale=sm_scale,
    )
    if out_nibble.isnan().any():
        print("  FAIL (nibble): output contains NaN!")
        return
    cos_nb = torch.nn.functional.cosine_similarity(
        out_nibble.reshape(-1).float(), out_ref.reshape(-1).float(), dim=0).item()
    mse_nb = (out_nibble.float() - out_ref.float()).pow(2).mean().item()
    print(f"  Nibble cosine sim:    {cos_nb:.4f}  MSE: {mse_nb:.6f}",
          "✓" if cos_nb > 0.90 else "FAIL")

    print()
    return cos_bp, cos_nb


def _benchmark_throughput():
    """
    Compare FP16 SDPA, old Python TQ3, and new Triton kernels (bit-plane + nibble).
    Reports effective HBM bandwidth and speedup ratios.
    """
    if not torch.cuda.is_available():
        print("No GPU — skipping benchmark.")
        return

    import sys, time
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from turboquant_mi300x import TurboQuantMI300X
    from torch.nn.functional import scaled_dot_product_attention

    print("\n" + "=" * 80)
    print("Throughput: FP16 SDPA  vs  Triton TQ3 (bit-plane)  vs  Triton TQ3 (nibble)")
    print("=" * 80)
    print("Why TQ3 was ~6× slower: centroid gather compiled to scattered VMEM loads")
    print("  (400-cycle latency on CDNA3 vs 4 cycles for VALU).  Fixes applied:")
    print("  1. tl.where cascade replaces gather (6 VALU ops, no memory access)")
    print("  2. K norms folded into scores after dot (avoids [BN,D] temp tensor)")
    print("  3. V norms folded into p before dot   (scales [1,BN], not [BN,D])")
    print("  4. FP16 inputs to tl.dot → MFMA fp16 units on gfx942")
    print("  5. Autotuned (BLOCK_M, BLOCK_N) across 6 configs")
    print("  Nibble kernel: 2 ops per pair vs 9 ops for bit-planes (3× less decode work)")
    print()

    tq = TurboQuantMI300X(bits=3, device="cuda")
    B, H, D = 1, 32, 128
    S_q = 1
    sm_scale = D ** -0.5
    WARMUP, REPS = 5, 50

    fmt = (f"{'seq_k':>8}  {'FP16':>8}  {'PyTQ3':>8}"
           f"  {'Triton-BP':>10}  {'vs FP16':>8}  {'Triton-Nb':>10}  {'vs FP16':>8}")
    print(fmt)
    print("-" * len(fmt))

    results = []
    for seq_k in [1024, 4096, 16384, 32768, 65536, 131072]:
        torch.manual_seed(42)
        q    = torch.randn(B, H, S_q, D, device="cuda", dtype=torch.float16)
        k_fp = torch.randn(B, H, seq_k, D, device="cuda", dtype=torch.float16)
        v_fp = torch.randn(B, H, seq_k, D, device="cuda", dtype=torch.float16)

        k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k_fp, v_fp, tq)
        k_nibbles, k_norms_n, v_nibbles, v_norms_n = compress_kv_nibble(k_fp, v_fp, tq)
        q_rot = tq.rotate_queries(q.float()).half()

        def _bench(fn):
            for _ in range(WARMUP):
                fn()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(REPS):
                fn()
            torch.cuda.synchronize()
            return (time.perf_counter() - t0) * 1000 / REPS

        ms_fp16 = _bench(lambda: scaled_dot_product_attention(q, k_fp, v_fp, scale=sm_scale))
        ms_bp   = _bench(lambda: turboquant_attention_fwd(
            q_rot, k_planes, k_norms, v_planes, v_norms,
            rotation=tq.rotation, sm_scale=sm_scale))
        ms_nb   = _bench(lambda: turboquant_nibble_attention_fwd(
            q_rot, k_nibbles, k_norms_n, v_nibbles, v_norms_n,
            rotation=tq.rotation, sm_scale=sm_scale))

        # Effective HBM bandwidth: bytes actually read / wall time
        kv_bytes_fp16 = seq_k * D * 2 * 2 * H   # K+V, FP16, all heads
        kv_bytes_bp   = seq_k * 52       * 2 * H   # K+V, 52B/token
        kv_bytes_nb   = seq_k * 68       * 2 * H   # K+V, 64B nibble + 4B norm

        bw_fp16 = kv_bytes_fp16 / (ms_fp16 * 1e-3) / 1e9
        bw_bp   = kv_bytes_bp   / (ms_bp   * 1e-3) / 1e9
        bw_nb   = kv_bytes_nb   / (ms_nb   * 1e-3) / 1e9

        print(f"{seq_k:>8}  {ms_fp16:>7.3f}ms  {'—':>7}  "
              f"{ms_bp:>9.3f}ms  {ms_fp16/ms_bp:>7.2f}×  "
              f"{ms_nb:>9.3f}ms  {ms_fp16/ms_nb:>7.2f}×")
        print(f"{'':>8}  BW: {bw_fp16:>5.1f}GB/s               "
              f"{bw_bp:>5.1f}GB/s              {bw_nb:>5.1f}GB/s")

        results.append({
            "seq_k": seq_k,
            "fp16_ms": ms_fp16,
            "triton_bitplane_ms": ms_bp,
            "triton_nibble_ms": ms_nb,
            "speedup_bp_vs_fp16": ms_fp16 / ms_bp,
            "speedup_nb_vs_fp16": ms_fp16 / ms_nb,
            "bw_fp16_GBs": bw_fp16,
            "bw_bp_GBs": bw_bp,
            "bw_nb_GBs": bw_nb,
        })

    print()
    print("Note: at batch=1, FP16 SDPA is itself memory-bandwidth-bound (reads weights).")
    print("TQ3 can match FP16 when KV bandwidth dominates (batch > ~4 at seq≥32K).")
    return results


if __name__ == "__main__":
    cos = _test_triton_attention()
    results = _benchmark_throughput()
