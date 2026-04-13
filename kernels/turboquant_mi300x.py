"""
turboquant_mi300x.py — TurboQuant TQ3/TQ4 for AMD Instinct MI300X

Implements compress/decompress/fused-dot using pure PyTorch ops (no ctypes).
The rotation matrix multiply uses torch.matmul → rocBLAS → MFMA on gfx942.

Why pure PyTorch?
  The standalone HIP library (libturboquant_mi300x.so) was compiled with
  system ROCm 7.2 hipcc, but PyTorch ships its own libamdhip64.so built
  against ROCm 6.2.  HIP fat-binary registration (hipError 209) fails when
  the two ABI versions coexist in the same process.  PyTorch's own kernels
  call rocBLAS / rocWMMA internally and DO use MFMA units — so the rotation
  GEMM is hardware-accelerated without any custom kernel.

  The standalone binary (tq_validate_mi300x) still works for isolated
  throughput benchmarks.

Block layout (must match turboquant_mi300x.h):
  block_tq3_mi300x (52 bytes):
    [0..3]   float32 norm
    [4..51]  3 bitplanes, 16 bytes each (LSB-first)

  Bitplane format:
    Plane b (b=0 LSB, b=2 MSB): 128-bit mask
      bits 0-63   = bit b of index[0..63]  (stored as uint64 LE)
      bits 64-127 = bit b of index[64..127]

Author: AMD ROCm TurboQuant benchmarking study, April 2026
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────────────────────────────────

HEAD_DIM         = 128
TQ3_BLOCK_BYTES  = 52
TQ4_BLOCK_BYTES  = 68
TQ2_BLOCK_BYTES  = 36
QJL_BLOCK_BYTES  = 20

COMPRESSION_RATIO = {
    3: 256 / TQ3_BLOCK_BYTES,   # 4.92×
    4: 256 / TQ4_BLOCK_BYTES,   # 3.76×
    2: 256 / TQ2_BLOCK_BYTES,   # 7.11×
}

# Lloyd-Max codebooks (must match turboquant_mi300x.h)
TQ3_CODEBOOK = torch.tensor([
    -0.18904037194348838,
    -0.11879501670185091,
    -0.06702922184405663,
    -0.02174971334976657,
     0.02174971334976654,
     0.06702922184405660,
     0.11879501670185087,
     0.18904037194348833,
], dtype=torch.float32)

TQ4_CODEBOOK = torch.tensor([
    -0.23961253307138700, -0.18317108415643454, -0.14430970076906538,
    -0.11276586366299288, -0.08507481024405737, -0.05962130616889217,
    -0.03539017687270855, -0.01173284981923122,
     0.01173284981923120,  0.03539017687270851,  0.05962130616889214,
     0.08507481024405730,  0.11276586366299284,  0.14430970076906535,
     0.18317108415643450,  0.23961253307138697,
], dtype=torch.float32)

TQ2_CODEBOOK = torch.tensor([
    -0.13311451677280386, -0.04002746648341520,
     0.04002746648341517,  0.13311451677280380,
], dtype=torch.float32)

_CODEBOOKS = {2: TQ2_CODEBOOK, 3: TQ3_CODEBOOK, 4: TQ4_CODEBOOK}


# ──────────────────────────────────────────────────────────────────────────────
# Rotation matrix generation (matches domvox seed convention)
# ──────────────────────────────────────────────────────────────────────────────

def make_rotation_matrix(seed: int = 42, dim: int = HEAD_DIM,
                         device: str = "cuda") -> torch.Tensor:
    """
    Generate a random orthogonal rotation matrix Π ∈ R^{dim×dim}.

    Uses scipy.linalg.qr on a seeded Gaussian matrix (same convention as
    the Python reference turboquant.py).

    Returns float32 tensor of shape (dim, dim) on `device`.
    """
    try:
        from scipy.linalg import qr as scipy_qr
        rng = np.random.default_rng(seed)
        G = rng.standard_normal((dim, dim)).astype(np.float32)
        Q, R = scipy_qr(G)
        Q = Q * np.sign(np.diag(R))[None, :]   # canonical signs
        return torch.from_numpy(Q.astype(np.float32)).to(device)
    except ImportError:
        # Fallback: Gram-Schmidt on random Gaussian (less numerically stable)
        rng = np.random.default_rng(seed)
        G = torch.from_numpy(rng.standard_normal((dim, dim)).astype(np.float32)).to(device)
        Q, _ = torch.linalg.qr(G)
        return Q


# ──────────────────────────────────────────────────────────────────────────────
# Core compress / decompress (pure torch, batched over n_vectors)
# ──────────────────────────────────────────────────────────────────────────────

def _nearest_centroid(y: torch.Tensor, codebook: torch.Tensor) -> torch.Tensor:
    """
    Find index of nearest centroid for each element.

    Parameters
    ----------
    y        : (n, dim) float32
    codebook : (n_levels,) float32

    Returns (n, dim) int32 indices in [0, n_levels).
    """
    cb = codebook.to(y.device)
    # (n, dim, 1) - (1, 1, n_levels) → (n, dim, n_levels) → argmin on last dim
    diff = y.unsqueeze(-1) - cb.view(1, 1, -1)
    return diff.pow_(2).argmin(dim=-1).to(torch.int32)


def _pack_bitplanes(indices: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Pack integer indices into bit-plane format.

    Parameters
    ----------
    indices : (n, dim) int32  values in [0, 2^n_bits)
    n_bits  : 2, 3, or 4

    Returns (n, n_bits*16) uint8 tensor where each plane is a 16-byte
    (128-bit) packed mask of one bit from each index.

    Bit-plane format (matches turboquant_mi300x.h):
      Plane b: bit b of index[j] is at byte (b*16 + j//8), bit (j%8).
    """
    n, dim = indices.shape
    assert dim == 128, "Only dim=128 supported"

    planes_list = []
    for b in range(n_bits):
        bits = ((indices >> b) & 1).to(torch.uint8)  # (n, 128)
        # Pack 128 bits into 16 bytes: group by 8 consecutive dimensions
        # bits[:, 8*k : 8*k+8] → one byte, LSB first
        bits_3d = bits.view(n, 16, 8)  # (n, 16 bytes, 8 bits)
        powers = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                              dtype=torch.uint8, device=indices.device)
        packed = (bits_3d * powers).sum(dim=-1).to(torch.uint8)  # (n, 16)
        planes_list.append(packed)

    return torch.cat(planes_list, dim=-1)  # (n, n_bits*16)


def _unpack_bitplanes(planes: torch.Tensor, n_bits: int) -> torch.Tensor:
    """
    Unpack bit-planes back to integer indices.

    Parameters
    ----------
    planes : (n, n_bits*16) uint8
    n_bits : 2, 3, or 4

    Returns (n, 128) int32 indices.
    """
    n = planes.shape[0]
    dim = 128
    indices = torch.zeros(n, dim, dtype=torch.int32, device=planes.device)

    for b in range(n_bits):
        plane = planes[:, b * 16: (b + 1) * 16]   # (n, 16)
        plane_3d = plane.unsqueeze(-1).expand(n, 16, 8)  # (n, 16, 8)
        bit_mask = torch.tensor([1, 2, 4, 8, 16, 32, 64, 128],
                                dtype=torch.int32, device=planes.device)
        bits = ((plane_3d.int() & bit_mask) != 0).int()  # (n, 16, 8)
        bits_flat = bits.reshape(n, dim)                 # (n, 128)
        indices = indices | (bits_flat << b)

    return indices


def tq3_compress(
    x: torch.Tensor,
    rotation: torch.Tensor,
    bits: int = 3,
) -> torch.Tensor:
    """
    Compress float32 vectors to TQ bit-plane format.

    Parameters
    ----------
    x        : (n, 128) float32 on GPU
    rotation : (128, 128) float32 orthogonal rotation matrix on GPU
    bits     : 2, 3, or 4

    Returns (n, block_bytes) uint8 tensor where:
      block_bytes = 4 (norm) + n_bits*16 (planes) = 52 / 68 / 36 for 3/4/2 bits.
    """
    assert x.is_cuda, "x must be on GPU"
    assert x.dtype == torch.float32
    assert x.shape[-1] == HEAD_DIM

    n = x.shape[0]
    codebook = _CODEBOOKS[bits].to(x.device)

    # Normalize
    norm = x.norm(dim=-1)                        # (n,)
    safe_norm = norm.clamp(min=1e-15)
    x_unit = x / safe_norm.unsqueeze(-1)

    # Rotate: y = x_unit @ Π^T  (each row of x_unit dot each row of rotation)
    # rotation[i, :] is the i-th basis vector; y[:, i] = x_unit dot rotation[i]
    y = x_unit @ rotation.T                      # (n, 128) — torch.matmul → MFMA

    # Quantize: find nearest centroid per element
    indices = _nearest_centroid(y, codebook)     # (n, 128) int32

    # Pack into bitplanes
    planes = _pack_bitplanes(indices, bits)      # (n, bits*16)

    # Encode norm as raw float32 bytes (4 bytes per vector)
    # We view the float32 norm as 4 uint8 bytes (little-endian on x86+MI300X)
    norm_bytes = norm.contiguous().view(torch.uint8).view(n, 4)   # (n, 4)

    blocks = torch.cat([norm_bytes, planes], dim=-1)              # (n, block_bytes)
    return blocks


def tq3_decompress(
    blocks: torch.Tensor,
    rotation: torch.Tensor,
    bits: int = 3,
) -> torch.Tensor:
    """
    Decompress TQ blocks back to float32 vectors.

    Parameters
    ----------
    blocks   : (n, block_bytes) uint8 on GPU
    rotation : (128, 128) float32 orthogonal rotation matrix on GPU
    bits     : 2, 3, or 4

    Returns (n, 128) float32.
    """
    assert blocks.is_cuda
    n = blocks.shape[0]
    codebook = _CODEBOOKS[bits].to(blocks.device)

    # Extract norm
    norm_bytes = blocks[:, :4].contiguous()                # (n, 4) uint8
    norm = norm_bytes.view(-1).view(torch.float32).view(n) # (n,) float32

    # Unpack bit-planes to indices
    planes = blocks[:, 4:]                                 # (n, bits*16) uint8
    indices = _unpack_bitplanes(planes, bits)              # (n, 128) int32

    # Centroid lookup: map each index to its float32 centroid value
    cb = codebook.to(blocks.device)
    y_hat = cb[indices]                                    # (n, 128) float32

    # Inverse rotation: x_hat_unit = y_hat @ Π (since Π is orthogonal: Π^{-1} = Π^T)
    # y_hat = x_unit @ Π^T  ⟹  x_unit = y_hat @ Π
    x_hat_unit = y_hat @ rotation                          # (n, 128) — MFMA

    # Scale by norm
    x_hat = x_hat_unit * norm.unsqueeze(-1)               # (n, 128)
    return x_hat


def tq3_fused_dot(
    q_rotated: torch.Tensor,
    blocks: torch.Tensor,
    bits: int = 3,
) -> torch.Tensor:
    """
    Compute dot products q_rotated[i] · decompress(blocks[j]) for all (i, j).

    Avoids full reconstruction: uses the fact that if q_rotated = Π @ q_unit,
    then q_unit · (norm × Π^T @ centroid) = norm × q_rotated · centroid.

    Parameters
    ----------
    q_rotated : (n_q, 128) float32 — queries already rotated by Π
    blocks    : (n_kv, block_bytes) uint8 — compressed key/value blocks
    bits      : 2, 3, or 4

    Returns (n_q, n_kv) float32 attention scores.
    """
    n_kv = blocks.shape[0]
    codebook = _CODEBOOKS[bits].to(blocks.device)

    # Extract norms
    norm_bytes = blocks[:, :4].contiguous()
    norms = norm_bytes.view(-1).view(torch.float32).view(n_kv)  # (n_kv,)

    # Unpack centroid values
    planes = blocks[:, 4:]
    indices = _unpack_bitplanes(planes, bits)    # (n_kv, 128)
    centroids = codebook[indices]               # (n_kv, 128)

    # In rotated space: score[i, j] = q_rotated[i] · centroids[j] × norm[j]
    # = (q_rotated @ centroids.T) * norms  (broadcast)
    scores = q_rotated @ centroids.T            # (n_q, n_kv) — MFMA
    scores = scores * norms.unsqueeze(0)        # scale by norms
    return scores


# ──────────────────────────────────────────────────────────────────────────────
# High-level API (drop-in replacement for ctypes-based wrapper)
# ──────────────────────────────────────────────────────────────────────────────

class TurboQuantMI300X:
    """
    High-level TurboQuant compression engine for MI300X.

    Uses pure PyTorch operations (torch.matmul → rocBLAS → MFMA on gfx942).
    No custom HIP kernels required in the Python process.

    The standalone C library (libturboquant_mi300x.so / tq_validate_mi300x)
    still provides the reference implementation and is used for isolated
    throughput benchmarks.

    Parameters
    ----------
    head_dim      : int   — must be 128
    bits          : int   — 2, 3, or 4 (default 3, paper uses 3-bit TQ)
    rotation_seed : int   — seed for orthogonal Π matrix
    use_qjl       : bool  — if True, enable QJL residual correction (keys only)
    qjl_seed      : int   — separate seed for QJL projection matrix S
    device        : str   — torch device (default "cuda")
    """

    def __init__(
        self,
        head_dim: int = 128,
        bits: int = 3,
        rotation_seed: int = 42,
        use_qjl: bool = False,
        qjl_seed: int = 137,
        device: str = "cuda",
    ):
        if head_dim != HEAD_DIM:
            raise ValueError(f"head_dim must be {HEAD_DIM}, got {head_dim}")
        if bits not in (2, 3, 4):
            raise ValueError(f"bits must be 2, 3, or 4, got {bits}")

        self.head_dim      = head_dim
        self.bits          = bits
        self.use_qjl       = use_qjl
        self.device        = device

        self._block_bytes = {2: TQ2_BLOCK_BYTES, 3: TQ3_BLOCK_BYTES,
                             4: TQ4_BLOCK_BYTES}[bits]

        # Pre-compute rotation matrix (stays on GPU across calls)
        self.rotation = make_rotation_matrix(rotation_seed, head_dim, device)

        # QJL projection matrix (optional, for key vectors)
        self._S = None
        if use_qjl:
            self._S = make_rotation_matrix(qjl_seed, head_dim, device)

    # ── Compression ──────────────────────────────────────────────────────────

    def compress_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compress float32 tensor (..., head_dim) to TQ format.

        Returns (n_vectors, block_bytes) uint8 tensor.
        """
        assert x.is_cuda, "Input must be on GPU"
        assert x.dtype == torch.float32
        assert x.shape[-1] == self.head_dim

        n = x.numel() // self.head_dim
        x_2d = x.reshape(n, self.head_dim).contiguous()
        return tq3_compress(x_2d, self.rotation, self.bits)

    def decompress_tensor(
        self,
        compressed: torch.Tensor,
        original_shape: tuple,
    ) -> torch.Tensor:
        """
        Decompress (n_vectors, block_bytes) uint8 back to float32.

        Returns tensor of `original_shape`.
        """
        assert compressed.is_cuda
        out = tq3_decompress(compressed, self.rotation, self.bits)
        return out.view(*original_shape)

    def fused_dot(
        self,
        q_rotated: torch.Tensor,
        compressed_kv: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute attention scores without full KV decompression.

        Parameters
        ----------
        q_rotated    : (n_q, head_dim) float32 — queries rotated by Π
        compressed_kv: (n_kv, block_bytes) uint8 — compressed KV blocks

        Returns (n_q, n_kv) float32.
        """
        return tq3_fused_dot(q_rotated, compressed_kv, self.bits)

    def rotate_queries(self, q: torch.Tensor) -> torch.Tensor:
        """Apply the rotation matrix to query vectors: q_rotated = q @ Π^T."""
        orig_shape = q.shape
        n = q.numel() // self.head_dim
        q_2d = q.reshape(n, self.head_dim).float()
        q_rot = q_2d @ self.rotation.T
        return q_rot.view(*orig_shape)

    # ── Stats ────────────────────────────────────────────────────────────────

    def compression_stats(self, tensors: list) -> dict:
        """
        Compute compression ratio and memory savings for a list of tensors.

        Parameters
        ----------
        tensors : list of (B, H, S, D) float16/float32 tensors

        Returns dict with keys: original_MB, compressed_MB, ratio, savings_pct.
        """
        total_orig = sum(t.numel() * t.element_size() for t in tensors)
        n_vecs = sum(t.numel() // self.head_dim for t in tensors)
        total_comp = n_vecs * self._block_bytes

        return {
            "original_MB":   total_orig / 1e6,
            "compressed_MB": total_comp / 1e6,
            "ratio":         total_orig / total_comp,
            "savings_pct":   (1 - total_comp / total_orig) * 100,
        }

    def __repr__(self) -> str:
        return (f"TurboQuantMI300X(bits={self.bits}, head_dim={self.head_dim}, "
                f"use_qjl={self.use_qjl}, "
                f"ratio={COMPRESSION_RATIO[self.bits]:.2f}×, "
                f"block_bytes={self._block_bytes})")
