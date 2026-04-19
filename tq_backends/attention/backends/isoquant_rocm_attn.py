from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import os
import torch
import torch.nn.functional as F

_IQ_METHOD = os.environ.get("VLLM_IQ_METHOD", "iso")
_IQ_BITS = int(os.environ.get("VLLM_IQ_BITS", "3"))
_IQ_SEED = int(os.environ.get("VLLM_IQ_SEED", "42"))

IQ_HEAD_DIM = 128
NORM_BYTES = 4
INDEX_BYTES = IQ_HEAD_DIM
COMPRESSED_BYTES = NORM_BYTES + INDEX_BYTES

_IQ_ENGINE: Optional[Any] = None


def _get_iq_engine(device: str = "cuda") -> Any:
    global _IQ_ENGINE
    if _IQ_ENGINE is None:
        from block_quant_rocm import make_quantizer
        _IQ_ENGINE = make_quantizer(
            _IQ_METHOD, bits=_IQ_BITS, head_dim=IQ_HEAD_DIM, seed=_IQ_SEED, device=device
        )
    return _IQ_ENGINE


def _pack(rows: torch.Tensor, engine: Any) -> torch.Tensor:
    comp = engine.compress(rows.float())
    norms = comp["norms"].contiguous().view(torch.uint8).reshape(rows.shape[0], NORM_BYTES)
    idx = comp["indices"].contiguous().view(torch.uint8)
    out = torch.empty(rows.shape[0], COMPRESSED_BYTES, dtype=torch.uint8, device=rows.device)
    out[:, :NORM_BYTES] = norms
    out[:, NORM_BYTES:] = idx
    return out


def _unpack(packed: torch.Tensor, engine: Any, shape: Tuple[int, ...]) -> torch.Tensor:
    n = packed.shape[0]
    norms = packed[:, :NORM_BYTES].contiguous().view(torch.float32).reshape(n)
    idx = packed[:, NORM_BYTES:].contiguous().view(torch.int8)
    return engine.decompress(
        {"norms": norms, "indices": idx, "method": _IQ_METHOD, "bits": _IQ_BITS},
        (n, IQ_HEAD_DIM),
    ).reshape(shape)


def _store_tokens(
    kv_cache: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    slot_mapping: torch.Tensor,
    engine: Any,
) -> None:
    if slot_mapping is None or slot_mapping.numel() == 0:
        return
    valid = slot_mapping >= 0
    if not torch.all(valid):
        k, v, slot_mapping = k[valid], v[valid], slot_mapping[valid]
        if slot_mapping.numel() == 0:
            return

    num_toks, num_kv_heads, head_dim = k.shape
    if head_dim != IQ_HEAD_DIM:
        raise ValueError(f"Expected head_dim={IQ_HEAD_DIM}, got {head_dim}")
    block_size = kv_cache.shape[3]

    k_packed = _pack(k.reshape(-1, head_dim), engine).view(num_toks, num_kv_heads, COMPRESSED_BYTES)
    v_packed = _pack(v.reshape(-1, head_dim), engine).view(num_toks, num_kv_heads, COMPRESSED_BYTES)
    block_idx = slot_mapping // block_size
    block_pos = slot_mapping % block_size
    for i in range(num_toks):
        kv_cache[0, block_idx[i], :, block_pos[i], :] = k_packed[i]
        kv_cache[1, block_idx[i], :, block_pos[i], :] = v_packed[i]


def _gather_sequence(
    kv_cache: torch.Tensor,
    block_table: torch.Tensor,
    seq_len: int,
    engine: Any,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_kv_heads = kv_cache.shape[2]
    block_size = kv_cache.shape[3]
    n_blocks = (seq_len + block_size - 1) // block_size
    k_chunks: List[torch.Tensor] = []
    v_chunks: List[torch.Tensor] = []
    for b in range(n_blocks):
        blk = block_table[b].item()
        n_tok = min(block_size, seq_len - b * block_size)
        k_blk = kv_cache[0, blk, :, :n_tok, :]
        v_blk = kv_cache[1, blk, :, :n_tok, :]
        k_chunks.append(_unpack(k_blk.reshape(num_kv_heads * n_tok, COMPRESSED_BYTES), engine,
                                (num_kv_heads, n_tok, IQ_HEAD_DIM)))
        v_chunks.append(_unpack(v_blk.reshape(num_kv_heads * n_tok, COMPRESSED_BYTES), engine,
                                (num_kv_heads, n_tok, IQ_HEAD_DIM)))
    return (
        torch.cat(k_chunks, dim=1).unsqueeze(0).to(torch.float16),
        torch.cat(v_chunks, dim=1).unsqueeze(0).to(torch.float16),
    )


@dataclass
class IsoQuantROCmAttentionMetadata:
    num_prefills: int = 0
    num_prefill_tokens: int = 0
    num_decode_tokens: int = 0
    slot_mapping: Optional[torch.Tensor] = None
    seq_lens: Optional[List[int]] = None
    seq_lens_tensor: Optional[torch.Tensor] = None
    max_query_len: Optional[int] = None
    max_prefill_seq_len: Optional[int] = None
    query_start_loc: Optional[torch.Tensor] = None
    block_tables: Optional[torch.Tensor] = None
    seq_lens_decode: Optional[List[int]] = None
    max_decode_seq_len: Optional[int] = None
    use_cuda_graph: bool = False
    _cached_prefill_metadata: Optional["IsoQuantROCmAttentionMetadata"] = field(default=None, repr=False)
    _cached_decode_metadata: Optional["IsoQuantROCmAttentionMetadata"] = field(default=None, repr=False)

    @property
    def is_prompt(self) -> bool:
        return self.num_prefill_tokens > 0 and self.num_decode_tokens == 0

    @property
    def is_decode(self) -> bool:
        return self.num_decode_tokens > 0 and self.num_prefill_tokens == 0

    @property
    def is_mixed(self) -> bool:
        return self.num_prefill_tokens > 0 and self.num_decode_tokens > 0


class IsoQuantROCmAttentionImpl:
    def __init__(self, num_heads: int, head_size: int, scale: float, num_kv_heads: int, **_: Any):
        if head_size != IQ_HEAD_DIM:
            raise ValueError(f"IsoQuant backend only supports head_size={IQ_HEAD_DIM}.")
        if num_heads % num_kv_heads != 0:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.gqa_ratio = num_heads // num_kv_heads

    def _expand_gqa(self, k: torch.Tensor, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.gqa_ratio == 1:
            return k, v
        return k.repeat_interleave(self.gqa_ratio, dim=-3), v.repeat_interleave(self.gqa_ratio, dim=-3)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: Optional[torch.Tensor],
        attn_metadata: IsoQuantROCmAttentionMetadata,
        kv_scale: float = 1.0,
        attn_type: str = "decoder",
    ) -> torch.Tensor:
        del kv_scale, attn_type
        n = query.shape[0]
        q = query.view(n, self.num_heads, self.head_size)
        k = key.view(n, self.num_kv_heads, self.head_size)
        v = value.view(n, self.num_kv_heads, self.head_size)
        engine = _get_iq_engine(str(query.device))

        if attn_metadata.is_prompt:
            if kv_cache is not None and attn_metadata.slot_mapping is not None:
                _store_tokens(kv_cache, k, v, attn_metadata.slot_mapping[:attn_metadata.num_prefill_tokens], engine)
            q4, k4, v4 = q.unsqueeze(0).transpose(1, 2), k.unsqueeze(0).transpose(1, 2), v.unsqueeze(0).transpose(1, 2)
            k4, v4 = self._expand_gqa(k4, v4)
            t = q.shape[0]
            mask = torch.ones(t, t, device=q.device, dtype=torch.bool).tril()
            out = F.scaled_dot_product_attention(q4, k4, v4, attn_mask=mask, scale=self.scale, is_causal=False)
            return out.transpose(1, 2).reshape(n, self.num_heads * self.head_size)

        if kv_cache is not None and attn_metadata.slot_mapping is not None:
            _store_tokens(kv_cache, k, v, attn_metadata.slot_mapping, engine)
        if kv_cache is None or attn_metadata.block_tables is None or attn_metadata.seq_lens_decode is None:
            raise ValueError("Decode requires kv_cache, block_tables, and seq_lens_decode.")

        outputs: List[torch.Tensor] = []
        for i in range(attn_metadata.num_decode_tokens):
            k_i, v_i = _gather_sequence(kv_cache, attn_metadata.block_tables[i], int(attn_metadata.seq_lens_decode[i]), engine)
            k_i, v_i = self._expand_gqa(k_i, v_i)
            q_i = q[i:i + 1].unsqueeze(2)
            out_i = F.scaled_dot_product_attention(q_i, k_i.to(q_i.dtype), v_i.to(q_i.dtype), scale=self.scale, is_causal=False)
            outputs.append(out_i.squeeze(2))
        return torch.cat(outputs, dim=0).reshape(n, self.num_heads * self.head_size)


class IsoQuantROCmAttentionBackend:
    @staticmethod
    def get_name() -> str:
        return "ISOQUANT_ROCM"

    @staticmethod
    def get_impl_cls():
        return IsoQuantROCmAttentionImpl

    @staticmethod
    def get_metadata_cls():
        return IsoQuantROCmAttentionMetadata

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> Tuple[int, ...]:
        """Paged KV layout: (2, num_blocks, num_kv_heads, block_size, packed_bytes)."""
        del cache_dtype_str
        if head_size != IQ_HEAD_DIM:
            raise ValueError(f"IsoQuant backend expects head_size={IQ_HEAD_DIM}.")
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, num_kv_heads, block_size, COMPRESSED_BYTES)

    @staticmethod
    def swap_blocks(src: torch.Tensor, dst: torch.Tensor, block_mapping: Dict[int, int]) -> None:
        for src_idx, dst_idx in block_mapping.items():
            dst[dst_idx].copy_(src[src_idx], non_blocking=True)

    @staticmethod
    def copy_blocks(kv_caches: List[torch.Tensor], src_to_dists: Dict[int, List[int]]) -> None:
        for kv_cache in kv_caches:
            for src_idx, dsts in src_to_dists.items():
                for dst_idx in dsts:
                    kv_cache[:, dst_idx].copy_(kv_cache[:, src_idx], non_blocking=True)
