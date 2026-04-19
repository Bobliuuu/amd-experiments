"""
vLLM V1 engine bridge: TurboQuant TQ3 KV + ROCm attention path.

vLLM 0.19+ routes attention through ``vllm.v1.attention.*`` and ``AttentionBackendEnum``.
TurboQuant's original backend is wired for the pre-V1 registry; this module exposes a
V1-compatible backend registered as ``AttentionBackendEnum.CUSTOM`` (``get_name()``
returns ``\"CUSTOM\"`` so ``Attention`` layer bookkeeping matches the enum slot).

Requires:
  - ``bash scripts/install_turboquant_vllm_backend.sh``
  - ``python3 scripts/patch_vllm_cache_dtype_tq3.py`` (or run via install script)
  - ``kernels/`` on ``PYTHONPATH`` for Triton / TurboQuant helpers
"""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm.attention.backends.rocm_flash_attn import (
    TQ3_BLOCK_BYTES,
    TQ3_HEAD_DIM,
    TurboQuantROCmAttentionImpl,
    TurboQuantROCmAttentionMetadata,
)
from vllm.config.cache import CacheDType  # type: ignore[attr-defined]
from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.backends.rocm_attn import (
    RocmAttentionBackend,
    RocmAttentionImpl,
    RocmAttentionMetadata,
)


def _rocm_meta_to_turboquant(m: RocmAttentionMetadata) -> TurboQuantROCmAttentionMetadata:
    if m.use_cascade:
        raise NotImplementedError(
            "TurboQuant v1 bridge: cascade / prefix attention is not supported yet."
        )
    ql = m.query_start_loc[1:] - m.query_start_loc[:-1]
    n_req = int(m.seq_lens.shape[0])
    num_actual = int(m.num_actual_tokens)

    # Uniform decode / prefill: stay on-device to avoid syncing all query lengths to CPU.
    if bool(torch.all(ql == 1).item()):
        nd = num_actual
        rows = m.block_table.shape[0]
        bt = m.block_table[:nd] if rows >= nd else m.block_table
        sl = m.seq_lens[:nd] if m.seq_lens.shape[0] >= nd else m.seq_lens
        return TurboQuantROCmAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=nd,
            slot_mapping=m.slot_mapping,
            block_tables=bt,
            seq_lens_decode=[int(x) for x in sl.detach().cpu().tolist()],
            max_decode_seq_len=int(sl.detach().max().item()) if sl.numel() else 0,
        )

    if bool(torch.all(ql > 1).item()):
        return TurboQuantROCmAttentionMetadata(
            num_prefills=n_req,
            num_prefill_tokens=num_actual,
            num_decode_tokens=0,
            slot_mapping=m.slot_mapping,
            seq_lens_tensor=m.seq_lens,
        )

    ql_cpu = ql.detach().cpu().tolist()
    if any(l > 1 for l in ql_cpu) and any(l == 1 for l in ql_cpu):
        p_tokens = sum(l for l in ql_cpu if l > 1)
        d_tokens = sum(l for l in ql_cpu if l == 1)
        n_pref_req = sum(1 for l in ql_cpu if l > 1)
        n_dec_req = sum(1 for l in ql_cpu if l == 1)
        dec_lens = [int(m.seq_lens[i].item()) for i, l in enumerate(ql_cpu) if l == 1]
        return TurboQuantROCmAttentionMetadata(
            num_prefills=n_pref_req,
            num_prefill_tokens=p_tokens,
            num_decode_tokens=d_tokens,
            slot_mapping=m.slot_mapping,
            seq_lens_tensor=m.seq_lens,
            block_tables=m.block_table[-n_dec_req:],
            seq_lens_decode=dec_lens,
            max_decode_seq_len=max(dec_lens) if dec_lens else 0,
        )

    # Fallback: treat as prefill (e.g. variable query len > 1 but not uniform)
    return TurboQuantROCmAttentionMetadata(
        num_prefills=n_req,
        num_prefill_tokens=num_actual,
        num_decode_tokens=0,
        slot_mapping=m.slot_mapping,
        seq_lens_tensor=m.seq_lens,
    )


class TurboQuantRocmV1Impl(RocmAttentionImpl):
    """Delegates TQ3 KV paths to ``TurboQuantROCmAttentionImpl``; FP16 KV uses ROCm V1."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._tq_impl = TurboQuantROCmAttentionImpl(
            self.num_heads,
            self.head_size,
            self.scale,
            self.num_kv_heads,
            self.alibi_slopes,
            None,
            self.kv_cache_dtype,
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
                "TurboQuant v1 bridge: fused block_scale output quantization unsupported"
            )
        if attn_metadata is None:
            return output.fill_(0)

        if self.kv_cache_dtype != "tq3":
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
                "TurboQuant v1 bridge: only decoder self-attention is supported"
            )

        tq_meta = _rocm_meta_to_turboquant(attn_metadata)
        n = int(attn_metadata.num_actual_tokens)
        flat = self._tq_impl.forward(
            query[:n],
            key[:n] if key is not None else key,
            value[:n] if value is not None else value,
            kv_cache,
            tq_meta,
        )
        output[:n].copy_(flat)
        return output


class TurboQuantRocmV1Backend(RocmAttentionBackend):
    """
    V1 attention backend: ``AttentionBackendEnum.CUSTOM`` slot + ROCm scheduling.

    ``get_name()`` must be ``\"CUSTOM\"`` (see ``Attention`` layer enum indexing).
    """

    forward_includes_kv_cache_update: ClassVar[bool] = True

    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = list(
        RocmAttentionBackend.supported_kv_cache_dtypes
    ) + ["tq3"]  # type: ignore[list-item]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[TurboQuantRocmV1Impl]:
        return TurboQuantRocmV1Impl

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if cache_dtype_str == "tq3":
            if head_size != TQ3_HEAD_DIM:
                raise ValueError(
                    f"TurboQuant TQ3 expects head_size={TQ3_HEAD_DIM}, got {head_size}"
                )
            if block_size % 16 != 0:
                raise ValueError("Block size must be a multiple of 16.")
            return (2, num_blocks, num_kv_heads, block_size, TQ3_BLOCK_BYTES)
        return RocmAttentionBackend.get_kv_cache_shape(
            num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str
        )

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        if kv_cache_dtype == "tq3":
            return True
        return super().supports_kv_cache_dtype(kv_cache_dtype)
