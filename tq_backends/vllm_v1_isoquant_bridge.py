"""
vLLM V1 engine bridge: IsoQuant / PlanarQuant-style packed KV + ROCm SDPA path.

Registers on ``AttentionBackendEnum.CUSTOM`` (same enum slot as TurboQuant).
**Do not** call ``register_turboquant_rocm_backend()`` in the same process if you
need IsoQuant — only one CUSTOM override can be active.

Requires:
  - ``bash scripts/install_isoquant_vllm_backend.sh`` (``iq3`` cache dtype patch)
  - Repo root on ``PYTHONPATH`` for ``tq_backends`` + ``kernels``
"""

from __future__ import annotations

from typing import ClassVar

import torch

from tq_backends.attention.backends.isoquant_rocm_attn import (
    COMPRESSED_BYTES,
    IQ_HEAD_DIM,
    IsoQuantROCmAttentionImpl,
    IsoQuantROCmAttentionMetadata,
)
from vllm.config.cache import CacheDType  # type: ignore[attr-defined]
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
)


def _rocm_meta_to_isoquant(m: RocmAttentionMetadata) -> IsoQuantROCmAttentionMetadata:
    if m.use_cascade:
        raise NotImplementedError(
            "IsoQuant v1 bridge: cascade / prefix attention is not supported yet."
        )
    ql = m.query_start_loc[1:] - m.query_start_loc[:-1]
    ql_cpu = ql.detach().cpu().tolist()
    n_req = int(m.seq_lens.shape[0])
    num_actual = int(m.num_actual_tokens)

    if all(l == 1 for l in ql_cpu):
        nd = num_actual
        rows = m.block_table.shape[0]
        bt = m.block_table[:nd] if rows >= nd else m.block_table
        sl = m.seq_lens[:nd] if m.seq_lens.shape[0] >= nd else m.seq_lens
        return IsoQuantROCmAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=nd,
            slot_mapping=m.slot_mapping,
            block_tables=bt,
            seq_lens_decode=[int(x) for x in sl.detach().cpu().tolist()],
            max_decode_seq_len=int(sl.detach().max().item()) if sl.numel() else 0,
        )

    if all(l > 1 for l in ql_cpu):
        return IsoQuantROCmAttentionMetadata(
            num_prefills=n_req,
            num_prefill_tokens=num_actual,
            num_decode_tokens=0,
            slot_mapping=m.slot_mapping,
            seq_lens_tensor=m.seq_lens,
        )

    if any(l > 1 for l in ql_cpu) and any(l == 1 for l in ql_cpu):
        p_tokens = sum(l for l in ql_cpu if l > 1)
        d_tokens = sum(l for l in ql_cpu if l == 1)
        n_pref_req = sum(1 for l in ql_cpu if l > 1)
        n_dec_req = sum(1 for l in ql_cpu if l == 1)
        dec_lens = [int(m.seq_lens[i].item()) for i, l in enumerate(ql_cpu) if l == 1]
        return IsoQuantROCmAttentionMetadata(
            num_prefills=n_pref_req,
            num_prefill_tokens=p_tokens,
            num_decode_tokens=d_tokens,
            slot_mapping=m.slot_mapping,
            seq_lens_tensor=m.seq_lens,
            block_tables=m.block_table[-n_dec_req:],
            seq_lens_decode=dec_lens,
            max_decode_seq_len=max(dec_lens) if dec_lens else 0,
        )

    return IsoQuantROCmAttentionMetadata(
        num_prefills=n_req,
        num_prefill_tokens=num_actual,
        num_decode_tokens=0,
        slot_mapping=m.slot_mapping,
        seq_lens_tensor=m.seq_lens,
    )


class IsoQuantRocmV1Impl(RocmAttentionImpl):
    """Delegates ``iq3`` KV paths to ``IsoQuantROCmAttentionImpl``; other dtypes use ROCm V1."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._iq_impl = IsoQuantROCmAttentionImpl(
            self.num_heads,
            self.head_size,
            self.scale,
            self.num_kv_heads,
        )

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RocmAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None
        if output_block_scale is not None:
            raise NotImplementedError(
                "IsoQuant v1 bridge: fused block_scale output quantization unsupported"
            )
        if attn_metadata is None:
            return output.fill_(0)

        if self.kv_cache_dtype != "iq3":
            return super().forward(
                layer,
                query,
                key,
                value,
                kv_cache,
                attn_metadata,
                output,
                output_scale,
                output_block_scale,
            )

        if self.attn_type is not AttentionType.DECODER:
            raise NotImplementedError(
                "IsoQuant v1 bridge: only decoder self-attention is supported"
            )

        iq_meta = _rocm_meta_to_isoquant(attn_metadata)
        n = int(attn_metadata.num_actual_tokens)
        flat = self._iq_impl.forward(
            query[:n],
            key[:n] if key is not None else key,
            value[:n] if value is not None else value,
            kv_cache,
            iq_meta,
        )
        output[:n].copy_(flat)
        return output


class IsoQuantRocmV1Backend(RocmAttentionBackend):
    """V1 backend: ``CUSTOM`` slot + paged ``iq3`` cache (packed Iso/Planar-style bytes)."""

    forward_includes_kv_cache_update: ClassVar[bool] = True

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(
        RocmAttentionBackend.supported_kv_cache_dtypes
    ) + ["iq3"]  # type: ignore[list-item]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[IsoQuantRocmV1Impl]:
        return IsoQuantRocmV1Impl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "iq3":
            if head_size != IQ_HEAD_DIM:
                raise ValueError(
                    f"IsoQuant iq3 expects head_size={IQ_HEAD_DIM}, got {head_size}"
                )
            if block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")
            return (2, num_blocks, num_kv_heads, block_size, COMPRESSED_BYTES)
        return RocmAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype == "iq3":
            return True
        return super().supports_kv_cache_dtype(kv_cache_dtype)
