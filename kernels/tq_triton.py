"""
tq_triton.py — Fused TurboQuant Dequant-Attention Kernel (Triton, ROCm/MI300X)

Implements Phase 3: a Triton kernel that fuses TQ3 decompression with attention
score computation, avoiding a full FP16 KV materialization.

Design (Flash Attention 2 style):
  For each query block (BLOCK_M queries):
    For each KV block (BLOCK_N tokens):
      1. Load compressed K (TQ3 bit-planes + norm) from HBM
      2. Dequantize K on-chip:
         - Unpack bit-planes → centroid indices
         - Lookup centroids → k_centroid (float32 values)
         (Note: rotation is NOT applied here — query must be pre-rotated)
      3. Compute raw score: s = q_rotated · k_centroid × norm_k
      4. Online softmax (Flash Attention 2 accumulation)
    For each V block:
      5. Load and dequantize V (TQ3, same scheme)
      6. Accumulate output: out += softmax_weight × v_fp32
  7. Write final output

Performance characteristics on MI300X (gfx942):
  - Avoids materializing FP16 KV: saves 2× global memory reads vs unfused
  - Triton uses MFMA automatically via dot() primitives
  - Expected speedup: 1.5–3× vs two-pass (HIP decompress + Flash Attention)

Important:
  - q_rotated must already be multiplied by the rotation matrix Π
  - The bit-plane format is BITPLANE (from turboquant_mi300x.hip.cpp), NOT sequential
  - Codebook values loaded from a compile-time constant array

Targets:
  - BLOCK_M=64, BLOCK_N=64 (default for MI300X)
  - Autotuned for gfx942

Usage:
    import torch
    from tq_triton import turboquant_attention_fwd

    # q: (batch, heads, seq_q, head_dim) float16
    # k_planes: (batch, heads, seq_k, 48) uint8  — TQ3 bit-planes
    # k_norms:  (batch, heads, seq_k)     float32
    # v_planes / v_norms: same as K
    # Returns: (batch, heads, seq_q, head_dim) float16
    out = turboquant_attention_fwd(q, k_planes, k_norms, v_planes, v_norms)
"""

import torch
import triton
import triton.language as tl
from typing import Optional

# ──────────────────────────────────────────────────────────────────────────────
# TQ3 codebook constants (must match turboquant_mi300x.h)
# ──────────────────────────────────────────────────────────────────────────────

TQ3_CENTROIDS = torch.tensor([
    -0.18904037194348838,
    -0.11879501670185091,
    -0.06702922184405663,
    -0.02174971334976657,
     0.02174971334976654,
     0.06702922184405660,
     0.11879501670185087,
     0.18904037194348833,
], dtype=torch.float32)


# ──────────────────────────────────────────────────────────────────────────────
# Triton: TQ3 bit-plane unpacker
# ──────────────────────────────────────────────────────────────────────────────

@triton.jit
def _unpack_tq3_indices(
    planes_ptr,        # pointer to bit-plane data for ONE vector (48 bytes)
    head_dim: tl.constexpr,   # = 128
):
    """
    Unpack TQ3 bit-planes into float32 centroid values for head_dim elements.

    Bit-plane format (BITPLANE, from turboquant_mi300x.hip.cpp):
      Plane 0 (LSB): bytes [0..15]   = 128-bit mask (bit i = LSB of index[i])
      Plane 1:       bytes [16..31]
      Plane 2 (MSB): bytes [32..47]

    For each dimension i ∈ [0, head_dim):
      bit b is at: planes[b*16 + i//8], bit (i%8)
      index = bit0 | (bit1 << 1) | (bit2 << 2)

    Returns: float32 centroid value for each of head_dim elements.
    """
    # Load all 48 bytes of bit-planes
    planes = tl.load(planes_ptr + tl.arange(0, 48))

    # Compute indices for each of head_dim=128 elements
    dims = tl.arange(0, head_dim)
    byte_in_plane = dims // 8      # which byte within a 16-byte plane
    bit_in_byte   = dims % 8       # which bit within that byte

    bit0 = (tl.load(planes_ptr + 0  * 16 + byte_in_plane) >> bit_in_byte) & 1
    bit1 = (tl.load(planes_ptr + 1  * 16 + byte_in_plane) >> bit_in_byte) & 1
    bit2 = (tl.load(planes_ptr + 2  * 16 + byte_in_plane) >> bit_in_byte) & 1

    idx = bit0 | (bit1 << 1) | (bit2 << 2)
    return idx.to(tl.int32)


@triton.jit
def _centroid_lookup(idx, centroids_ptr, n_levels: tl.constexpr = 8):
    """Map indices [0, n_levels) to float32 centroid values via gather."""
    return tl.load(centroids_ptr + idx)


# ──────────────────────────────────────────────────────────────────────────────
# Triton: Fused TQ3 dequant + attention forward
# ──────────────────────────────────────────────────────────────────────────────

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 64,  "waves_per_eu": 1}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64,  "waves_per_eu": 1}, num_warps=8),
        triton.Config({"BLOCK_M": 64,  "BLOCK_N": 128, "waves_per_eu": 2}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 128, "waves_per_eu": 1}, num_warps=8),
    ],
    key=["seq_q", "seq_k", "head_dim"],
)
@triton.jit
def _tq3_attention_kernel(
    # Query (pre-rotated, float16 or float32)
    Q_ptr, stride_qb, stride_qh, stride_qm, stride_qd,
    # Compressed K: bit-planes (uint8, 48 bytes per token)
    K_planes_ptr, stride_kb, stride_kh, stride_kn,
    # K norms (float32)
    K_norms_ptr, stride_knb, stride_knh, stride_knn,
    # Compressed V: bit-planes (uint8, 48 bytes per token)
    V_planes_ptr, stride_vb, stride_vh, stride_vn,
    # V norms (float32)
    V_norms_ptr, stride_vnb, stride_vnh, stride_vnn,
    # Codebook
    Centroids_ptr,
    # Output
    O_ptr, stride_ob, stride_oh, stride_om, stride_od,
    # Dimensions
    batch: int, heads: int, seq_q: int, seq_k: int,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    sm_scale: float,
):
    """
    Fused TQ3 dequantize + attention forward pass.

    Block structure:
      - pid_m: block along seq_q (BLOCK_M queries)
      - pid_bh: batch × head combined
    """
    pid_m  = tl.program_id(0)  # query block
    pid_bh = tl.program_id(1)  # batch × head

    batch_idx = pid_bh // heads
    head_idx  = pid_bh %  heads

    # Offset into Q block
    q_off = (batch_idx * stride_qb + head_idx * stride_qh
             + pid_m * BLOCK_M * stride_qm)
    q_ptrs = Q_ptr + q_off + tl.arange(0, BLOCK_M)[:, None] * stride_qm \
                            + tl.arange(0, head_dim)[None, :] * stride_qd

    # Load Q block
    m_mask = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) < seq_q
    q = tl.load(q_ptrs, mask=m_mask[:, None], other=0.0).to(tl.float32)

    # Initialize online softmax state
    m_i = tl.full([BLOCK_M], -float("inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc  = tl.zeros([BLOCK_M, head_dim], dtype=tl.float32)

    # ── Iterate over K blocks ──────────────────────────────────────────────
    k_base = (batch_idx * stride_kb + head_idx * stride_kh)
    kn_base = (batch_idx * stride_knb + head_idx * stride_knh)

    for block_n_start in range(0, seq_k, BLOCK_N):
        n_mask = (block_n_start + tl.arange(0, BLOCK_N)) < seq_k

        # Load compressed K: 48 bytes per token × BLOCK_N tokens
        # In bitplane format: planes[token, 48]
        k_planes_ptr = K_planes_ptr + k_base + block_n_start * stride_kn
        # Load norms
        k_norm_ptr = K_norms_ptr + kn_base + block_n_start * stride_knn
        k_norms = tl.load(k_norm_ptr + tl.arange(0, BLOCK_N) * stride_knn,
                          mask=n_mask, other=0.0)

        # Unpack bit-planes: for each of BLOCK_N tokens, extract head_dim indices
        # This inner loop is the core of the fused dequantize
        k_centroids = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
        for n in tl.static_range(BLOCK_N):
            token_planes_ptr = k_planes_ptr + n * stride_kn
            dims = tl.arange(0, head_dim)
            byte_in_plane = dims // 8
            bit_in_byte   = dims % 8

            # Load 16-byte chunks for each bit plane
            b0_byte = tl.load(token_planes_ptr +  0 + byte_in_plane,
                              mask=byte_in_plane < 16, other=0).to(tl.int32)
            b1_byte = tl.load(token_planes_ptr + 16 + byte_in_plane,
                              mask=byte_in_plane < 16, other=0).to(tl.int32)
            b2_byte = tl.load(token_planes_ptr + 32 + byte_in_plane,
                              mask=byte_in_plane < 16, other=0).to(tl.int32)

            bit0 = (b0_byte >> bit_in_byte) & 1
            bit1 = (b1_byte >> bit_in_byte) & 1
            bit2 = (b2_byte >> bit_in_byte) & 1
            idx  = bit0 | (bit1 << 1) | (bit2 << 2)

            centroid_vals = tl.load(Centroids_ptr + idx)
            k_centroids = tl.where(
                tl.arange(0, BLOCK_N)[:, None] == n,
                centroid_vals[None, :],
                k_centroids
            )

        # Scale by norms: k_fp32[n, :] = norm[n] × centroid[n, :]
        k_fp32 = k_centroids * k_norms[:, None]  # [BLOCK_N, head_dim]

        # Compute attention scores: [BLOCK_M, BLOCK_N]
        scores = tl.dot(q, tl.trans(k_fp32)) * sm_scale
        scores = tl.where(n_mask[None, :], scores, -float("inf"))

        # Online softmax update
        m_ij = tl.max(scores, axis=1)
        m_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(scores - m_new[:, None])
        l_i = l_i * alpha + tl.sum(p, axis=1)
        m_i = m_new

        # ── Iterate over V blocks (paired with K) ─────────────────────────
        v_base_offset = (batch_idx * stride_vb + head_idx * stride_vh
                         + block_n_start * stride_vn)
        vn_base_offset = (batch_idx * stride_vnb + head_idx * stride_vnh
                          + block_n_start * stride_vnn)

        v_norms = tl.load(V_norms_ptr + vn_base_offset
                          + tl.arange(0, BLOCK_N) * stride_vnn,
                          mask=n_mask, other=0.0)

        v_fp32 = tl.zeros([BLOCK_N, head_dim], dtype=tl.float32)
        for n in tl.static_range(BLOCK_N):
            token_planes_ptr = V_planes_ptr + v_base_offset + n * stride_vn
            dims = tl.arange(0, head_dim)
            byte_in_plane = dims // 8
            bit_in_byte   = dims % 8

            b0_byte = tl.load(token_planes_ptr +  0 + byte_in_plane,
                              mask=byte_in_plane < 16, other=0).to(tl.int32)
            b1_byte = tl.load(token_planes_ptr + 16 + byte_in_plane,
                              mask=byte_in_plane < 16, other=0).to(tl.int32)
            b2_byte = tl.load(token_planes_ptr + 32 + byte_in_plane,
                              mask=byte_in_plane < 16, other=0).to(tl.int32)

            bit0 = (b0_byte >> bit_in_byte) & 1
            bit1 = (b1_byte >> bit_in_byte) & 1
            bit2 = (b2_byte >> bit_in_byte) & 1
            idx  = bit0 | (bit1 << 1) | (bit2 << 2)
            centroid_vals = tl.load(Centroids_ptr + idx)
            # Load V norm for token n directly (v_norms[n] with constexpr index
            # is unsupported in Triton — load scalar from pointer instead)
            norm_n = tl.load(V_norms_ptr + vn_base_offset + (block_n_start + n) * stride_vnn)
            v_fp32 = tl.where(
                tl.arange(0, BLOCK_N)[:, None] == n,
                centroid_vals[None, :] * norm_n,
                v_fp32,
            )

        # Accumulate: acc += p × v_fp32  (attention-weighted sum)
        acc = acc * alpha[:, None] + tl.dot(p, v_fp32)

    # ── Normalize and write output ─────────────────────────────────────────
    acc = acc / l_i[:, None]
    out_off = (batch_idx * stride_ob + head_idx * stride_oh
               + pid_m * BLOCK_M * stride_om)
    out_ptrs = O_ptr + out_off + tl.arange(0, BLOCK_M)[:, None] * stride_om \
                                + tl.arange(0, head_dim)[None, :] * stride_od
    tl.store(out_ptrs, acc.to(tl.float16), mask=m_mask[:, None])


# ──────────────────────────────────────────────────────────────────────────────
# Python entry point
# ──────────────────────────────────────────────────────────────────────────────

def turboquant_attention_fwd(
    q: torch.Tensor,
    k_planes: torch.Tensor,
    k_norms: torch.Tensor,
    v_planes: torch.Tensor,
    v_norms: torch.Tensor,
    sm_scale: Optional[float] = None,
) -> torch.Tensor:
    """
    Fused TQ3 dequantize + attention forward pass.

    Parameters
    ----------
    q : (B, H, S_q, D) float16/float32 — queries, PRE-ROTATED by Π
    k_planes : (B, H, S_k, 48) uint8 — TQ3 bit-planes for keys
    k_norms  : (B, H, S_k) float32 — key norms
    v_planes : (B, H, S_k, 48) uint8 — TQ3 bit-planes for values
    v_norms  : (B, H, S_k) float32 — value norms
    sm_scale : float — 1/sqrt(head_dim), default computed from q.shape[-1]

    Returns
    -------
    output : (B, H, S_q, D) float16
    """
    B, H, S_q, D = q.shape
    _, _, S_k, _  = k_planes.shape

    assert D == 128, "Only head_dim=128 supported"
    assert k_planes.shape[-1] == 48, "k_planes must have 48 bytes per token (TQ3)"
    assert k_planes.dtype == torch.uint8

    if sm_scale is None:
        sm_scale = D ** -0.5

    # Upload centroids to device if needed
    centroids = TQ3_CENTROIDS.to(q.device)

    out = torch.empty(B, H, S_q, D, dtype=torch.float16, device=q.device)

    # Ensure inputs are contiguous
    q = q.contiguous()
    k_planes = k_planes.contiguous()
    k_norms  = k_norms.contiguous()
    v_planes = v_planes.contiguous()
    v_norms  = v_norms.contiguous()

    # Grid: (ceil(S_q / BLOCK_M), B × H)
    grid = lambda meta: (
        triton.cdiv(S_q, meta["BLOCK_M"]),
        B * H,
    )

    _tq3_attention_kernel[grid](
        q,            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k_planes,     k_planes.stride(0), k_planes.stride(1), k_planes.stride(2),
        k_norms,      k_norms.stride(0), k_norms.stride(1), k_norms.stride(2),
        v_planes,     v_planes.stride(0), v_planes.stride(1), v_planes.stride(2),
        v_norms,      v_norms.stride(0), v_norms.stride(1), v_norms.stride(2),
        centroids,
        out,          out.stride(0), out.stride(1), out.stride(2), out.stride(3),
        B, H, S_q, S_k,
        head_dim=D,
        sm_scale=sm_scale,
    )

    return out


# ──────────────────────────────────────────────────────────────────────────────
# Utilities for preparing compressed tensors
# ──────────────────────────────────────────────────────────────────────────────

def compress_kv_for_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    tq_engine,
) -> tuple:
    """
    Compress (B, H, S, D) KV tensors into the format expected by turboquant_attention_fwd.

    Parameters
    ----------
    k, v   : (B, H, S, D) float16/float32 tensors
    tq_engine : TurboQuantMI300X instance

    Returns (k_planes, k_norms, v_planes, v_norms)
    """
    B, H, S, D = k.shape
    n = B * H * S

    k_fp32 = k.reshape(n, D).float().contiguous()
    v_fp32 = v.reshape(n, D).float().contiguous()

    k_comp = tq_engine.compress_tensor(k_fp32)  # (n, 52) uint8
    v_comp = tq_engine.compress_tensor(v_fp32)  # (n, 52) uint8

    # Split: first 4 bytes = norm (float32), next 48 bytes = bit-planes
    # block_tq3_mi300x layout: [norm: float32][planes: uint8 × 48]
    k_norms_raw  = k_comp[:, :4]   # 4 bytes → float32
    k_planes_raw = k_comp[:, 4:]   # 48 bytes → bit-planes
    v_norms_raw  = v_comp[:, :4]
    v_planes_raw = v_comp[:, 4:]

    # Convert norm bytes to float32 (.contiguous() required: slices are non-contiguous)
    k_norms_f32 = k_norms_raw.contiguous().view(-1).view(torch.float32).view(n)
    v_norms_f32 = v_norms_raw.contiguous().view(-1).view(torch.float32).view(n)

    # Reshape to (B, H, S, ...)
    k_planes = k_planes_raw.view(B, H, S, 48)
    k_norms  = k_norms_f32.view(B, H, S)
    v_planes = v_planes_raw.view(B, H, S, 48)
    v_norms  = v_norms_f32.view(B, H, S)

    return k_planes, k_norms, v_planes, v_norms


# ──────────────────────────────────────────────────────────────────────────────
# Quick self-test
# ──────────────────────────────────────────────────────────────────────────────

def _test_triton_attention():
    """Verify fused attention output matches naive reference."""
    if not torch.cuda.is_available():
        print("No GPU available. Skipping Triton test.")
        return

    import sys
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from turboquant_mi300x import TurboQuantMI300X

    print("Testing Triton fused attention...")
    B, H, S_q, S_k, D = 1, 4, 64, 256, 128

    tq = TurboQuantMI300X(bits=3)
    sm_scale = D ** -0.5

    q = torch.randn(B, H, S_q, D, device="cuda", dtype=torch.float16)
    k = torch.randn(B, H, S_k, D, device="cuda", dtype=torch.float16)
    v = torch.randn(B, H, S_k, D, device="cuda", dtype=torch.float16)

    # Compress KV (stores centroid indices in the ROTATED space)
    k_planes, k_norms, v_planes, v_norms = compress_kv_for_triton(k, v, tq)

    # The Triton kernel operates in rotated space: scores = q_rot · centroid_rot × norm.
    # Query must be pre-rotated so that q_rot · centroid_rot == q · k_decompressed.
    q_rot = tq.rotate_queries(q.float())  # (B, H, S_q, D) float32

    # Fused attention (queries already in rotated space)
    out_fused = turboquant_attention_fwd(
        q_rot, k_planes, k_norms, v_planes, v_norms, sm_scale
    )

    # Reference: decompress KV back to original space, then standard SDPA with original q.
    # This is equivalent because:
    #   q_rot · centroid_rot = (q @ R.T) · centroid_rot
    #                        = q · (centroid_rot @ R)
    #                        = q · k_decompressed  (since decompress applies @ rotation)
    from torch.nn.functional import scaled_dot_product_attention
    k_decomp = tq.decompress_tensor(
        torch.cat([k_norms.contiguous().view(-1, 1).view(torch.uint8).view(-1, 4),
                   k_planes.reshape(-1, 48)], dim=1),
        (B, H, S_k, D)
    ).half()
    v_decomp = tq.decompress_tensor(
        torch.cat([v_norms.contiguous().view(-1, 1).view(torch.uint8).view(-1, 4),
                   v_planes.reshape(-1, 48)], dim=1),
        (B, H, S_k, D)
    ).half()
    # Reference uses ORIGINAL (unrotated) q; math shown above makes these equivalent
    out_ref = scaled_dot_product_attention(q, k_decomp, v_decomp, scale=sm_scale)

    cos_sim = torch.nn.functional.cosine_similarity(
        out_fused.reshape(-1), out_ref.half().reshape(-1), dim=0
    ).item()
    print(f"Triton vs reference cosine similarity: {cos_sim:.4f}")

    # #region agent log
    import json as _j, time as _t
    with open('/root/workspace/.cursor/debug-5ac54c.log', 'a') as _lf:
        _lf.write(_j.dumps({'sessionId':'5ac54c','location':'tq_triton.py:_test_triton_attention','message':'triton_test_result','data':{'cos_sim':round(float(cos_sim),4),'pass':bool(cos_sim>0.90)},'timestamp':int(_t.time()*1000),'hypothesisId':'D'}) + '\n')
    # #endregion

    assert cos_sim > 0.90, f"Low cosine similarity: {cos_sim:.4f}"
    print("PASS")


if __name__ == "__main__":
    _test_triton_attention()
