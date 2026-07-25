from __future__ import annotations

import importlib
import os

from cannbench.operators.materialize import materialized_values_to_buffer

from .bound_inputs import bound_indices
from .materialize import (
    _requires_direct_device_inputs,
    materialize_sparse_attention_inputs,
)

_CUDA_DSA_ADAPTER_ENV = "CANNBENCH_CUDA_DSA_ADAPTER"
_DEFAULT_CUDA_DSA_ADAPTER_MODULE = "cannbench_cuda_dsa"


def build_cuda_library_callable(ctx):
    adapter_op = _resolve_cuda_dsa_adapter("sparse_attention")
    payload = materialize_sparse_attention_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    query = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["query"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["query_shape"])
    shared_kv = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["shared_kv"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["shared_kv_shape"])
    indices = bound_indices(ctx, payload["indices_shape"], dtype=ctx.torch.int32)
    if indices is None:
        indices = ctx.backend._tensor(
            ctx.torch,
            payload["indices"],
            device=ctx.device,
            dtype=ctx.torch.int32,
        ).reshape(payload["indices_shape"])
    softmax_scale = payload["query_shape"][-1] ** -0.5

    def operator():
        return adapter_op(
            torch=ctx.torch,
            request=ctx.request,
            case=ctx.case,
            payload=payload,
            device=ctx.device,
            dtype=ctx.dtype,
            query=query,
            shared_kv=shared_kv,
            indices=indices,
            causal=payload["causal"],
            phase=payload["phase"],
            softmax_scale=softmax_scale,
        )

    return operator


def _resolve_cuda_dsa_adapter(op_name: str):
    module_name = os.environ.get(_CUDA_DSA_ADAPTER_ENV) or _DEFAULT_CUDA_DSA_ADAPTER_MODULE
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name != module_name:
            raise
        raise RuntimeError(
            "cuda_library sparse_attention benchmarking requires an external "
            f"adapter. Install {_DEFAULT_CUDA_DSA_ADAPTER_MODULE} or set "
            f"{_CUDA_DSA_ADAPTER_ENV}=<module> with callable {op_name}."
        ) from exc
    op_callable = getattr(module, op_name, None)
    if not callable(op_callable):
        raise RuntimeError(
            f"CUDA sparse_attention adapter {module_name} must expose callable {op_name}"
        )
    return op_callable


def build_vllm_ascend_callable(ctx):
    if (ctx.case.qk_head_dim, ctx.case.value_head_dim) == (576, 512):
        return _build_vllm_sparse_flash_attention_callable(ctx)
    if ctx.case.qk_head_dim != ctx.case.value_head_dim:
        raise RuntimeError(
            "vllm_ascend sparse_attention only supports equal QK/value head "
            "dimensions or the MLA 576/512 layout"
        )
    return _build_vllm_sharedkv_callable(ctx)


def _build_vllm_sparse_flash_attention_callable(ctx):
    attention_op = getattr(
        ctx.backend._ascend_custom_ops(ctx.torch),
        "npu_sparse_flash_attention",
        None,
    )
    if attention_op is None:
        raise RuntimeError(
            "vllm_ascend V3.2 sparse_attention requires "
            "torch.ops._C_ascend.npu_sparse_flash_attention"
        )

    direct_device_inputs = _requires_direct_device_inputs(ctx.case)
    batch = ctx.case.batch
    query_heads = ctx.case.query_heads
    query_tokens = ctx.case.query_tokens
    qk_head_dim = ctx.case.qk_head_dim
    kv_heads = ctx.case.kv_heads
    context_tokens = ctx.case.context_tokens
    value_head_dim = ctx.case.value_head_dim
    selected_tokens = ctx.case.selected_tokens
    rope_head_dim = qk_head_dim - value_head_dim
    block_size = _sfa_page_block_size(context_tokens)
    blocks_per_batch = context_tokens // block_size

    query_shape = (batch * query_tokens, query_heads, value_head_dim)
    query_rope_shape = (batch * query_tokens, query_heads, rope_head_dim)
    key_shape = (
        batch * blocks_per_batch,
        block_size,
        kv_heads,
        value_head_dim,
    )
    key_rope_shape = (
        batch * blocks_per_batch,
        block_size,
        kv_heads,
        rope_head_dim,
    )
    indices_shape = (batch * query_tokens, kv_heads, selected_tokens)

    if direct_device_inputs:
        query = ctx.torch.zeros(query_shape, device=ctx.device, dtype=ctx.dtype)
        query_rope = ctx.torch.zeros(
            query_rope_shape, device=ctx.device, dtype=ctx.dtype
        )
        key = ctx.torch.zeros(key_shape, device=ctx.device, dtype=ctx.dtype)
        key_rope = ctx.torch.zeros(
            key_rope_shape, device=ctx.device, dtype=ctx.dtype
        )
        value = ctx.torch.zeros(key_shape, device=ctx.device, dtype=ctx.dtype)
        sparse_indices = bound_indices(ctx, indices_shape, dtype=ctx.torch.int32)
        if sparse_indices is None:
            sparse_indices = ctx.torch.zeros(
                indices_shape, device=ctx.device, dtype=ctx.torch.int32
            )
    else:
        payload = materialize_sparse_attention_inputs(
            ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
        )
        query_values, query_rope_values = _split_bhtd_values_as_bthd(
            payload["query"],
            batch=batch,
            heads=query_heads,
            tokens=query_tokens,
            part_dims=(value_head_dim, rope_head_dim),
        )
        key_values, key_rope_values = _split_bhtd_values_as_bthd(
            payload["shared_kv"],
            batch=batch,
            heads=kv_heads,
            tokens=context_tokens,
            part_dims=(value_head_dim, rope_head_dim),
        )
        value_values, _ = _split_bhtd_values_as_bthd(
            payload["shared_kv"],
            batch=batch,
            heads=kv_heads,
            tokens=context_tokens,
            part_dims=(value_head_dim, rope_head_dim),
        )
        query = ctx.backend._tensor(
            ctx.torch, query_values, device=ctx.device, dtype=ctx.dtype
        ).reshape(query_shape)
        query_rope = ctx.backend._tensor(
            ctx.torch, query_rope_values, device=ctx.device, dtype=ctx.dtype
        ).reshape(query_rope_shape)
        key = ctx.backend._tensor(
            ctx.torch, key_values, device=ctx.device, dtype=ctx.dtype
        ).reshape(key_shape)
        key_rope = ctx.backend._tensor(
            ctx.torch, key_rope_values, device=ctx.device, dtype=ctx.dtype
        ).reshape(key_rope_shape)
        value = ctx.backend._tensor(
            ctx.torch, value_values, device=ctx.device, dtype=ctx.dtype
        ).reshape(key_shape)
        sparse_indices = bound_indices(ctx, indices_shape, dtype=ctx.torch.int32)
        if sparse_indices is None:
            sparse_indices = ctx.backend._tensor(
                ctx.torch,
                payload["indices"],
                device=ctx.device,
                dtype=ctx.torch.int32,
            ).reshape(indices_shape)
    block_table = ctx.backend._tensor(
        ctx.torch,
        tuple(range(batch * blocks_per_batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(batch, blocks_per_batch)
    actual_seq_lengths_query = ctx.backend._tensor(
        ctx.torch,
        tuple((index + 1) * query_tokens for index in range(batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    actual_seq_lengths_kv = ctx.backend._tensor(
        ctx.torch,
        tuple(context_tokens for _ in range(batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )

    def operator():
        result = attention_op(
            query=query,
            key=key,
            value=value,
            sparse_indices=sparse_indices,
            scale_value=qk_head_dim**-0.5,
            block_table=block_table,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=key_rope,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
        )
        return result[0] if isinstance(result, tuple) else result

    return operator


def _sfa_page_block_size(context_tokens: int) -> int:
    for block_size in (128, 64, 32, 16):
        if context_tokens % block_size == 0:
            return block_size
    raise RuntimeError(
        "vllm_ascend V3.2 sparse_attention requires context_tokens to be "
        "divisible by a supported page block size"
    )


def _split_bhtd_values_as_bthd(
    values,
    *,
    batch: int,
    heads: int,
    tokens: int,
    part_dims: tuple[int, ...],
):
    logical_dim = sum(part_dims)
    parts = tuple([] for _ in part_dims)
    for batch_index in range(batch):
        for token_index in range(tokens):
            for head_index in range(heads):
                offset = (
                    ((batch_index * heads + head_index) * tokens + token_index)
                    * logical_dim
                )
                part_offset = offset
                for part, part_dim in zip(parts, part_dims, strict=True):
                    part.extend(values[part_offset : part_offset + part_dim])
                    part_offset += part_dim
    return tuple(materialized_values_to_buffer(part) for part in parts)


def _build_vllm_sharedkv_callable(ctx):
    metadata_op, attention_op = ctx.backend._custom_op_pair(
        ctx.torch,
        "npu_sparse_attn_sharedkv_metadata",
        "npu_sparse_attn_sharedkv",
    )
    if metadata_op is None or attention_op is None:
        raise RuntimeError(
            "vllm_ascend sparse_attention requires "
            "torch.ops._C_ascend.npu_sparse_attn_sharedkv_metadata and "
            "torch.ops._C_ascend.npu_sparse_attn_sharedkv"
        )

    payload = materialize_sparse_attention_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    batch, query_heads, query_tokens, qk_head_dim = payload["query_shape"]
    _, kv_heads, context_tokens, _ = payload["shared_kv_shape"]
    selected_tokens = payload["indices_shape"][2]
    block_size = 128 if context_tokens % 128 == 0 else context_tokens
    blocks_per_batch = context_tokens // block_size

    query = ctx.backend._tensor(
        ctx.torch,
        ctx.backend._materialized_bhtd_values_as_bthd(
            payload["query"],
            batch=batch,
            heads=query_heads,
            tokens=query_tokens,
            dim=qk_head_dim,
        ),
        device=ctx.device,
        dtype=ctx.dtype,
    )
    query = query.reshape(batch * query_tokens, query_heads, qk_head_dim)
    cmp_kv = ctx.backend._tensor(
        ctx.torch,
        ctx.backend._materialized_kv_values_as_bthd(
            payload["shared_kv"],
            batch=batch,
            kv_heads=kv_heads,
            context_tokens=context_tokens,
            kept_context_tokens=context_tokens,
            logical_dim=qk_head_dim,
            physical_dim=qk_head_dim,
        ),
        device=ctx.device,
        dtype=ctx.dtype,
    )
    cmp_kv = cmp_kv.reshape(
        batch * blocks_per_batch, block_size, kv_heads, qk_head_dim
    )
    cmp_sparse_indices = bound_indices(
        ctx,
        (batch * query_tokens, kv_heads, selected_tokens),
        dtype=ctx.torch.int32,
    )
    if cmp_sparse_indices is None:
        cmp_sparse_indices = ctx.backend._tensor(
            ctx.torch,
            payload["indices"],
            device=ctx.device,
            dtype=ctx.torch.int32,
        ).reshape(batch * query_tokens, kv_heads, selected_tokens)
    cmp_block_table = ctx.backend._tensor(
        ctx.torch,
        tuple(range(batch * blocks_per_batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(batch, blocks_per_batch)
    cu_seqlens_q = ctx.backend._tensor(
        ctx.torch,
        tuple(index * query_tokens for index in range(batch + 1)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    seqused_kv = ctx.backend._tensor(
        ctx.torch,
        tuple(context_tokens for _ in range(batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    softmax_scale = qk_head_dim ** -0.5

    metadata = metadata_op(
        num_heads_q=query_heads,
        num_heads_kv=kv_heads,
        head_dim=qk_head_dim,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=None,
        cu_seqlens_cmp_kv=None,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=batch,
        max_seqlen_q=query_tokens,
        max_seqlen_kv=context_tokens,
        ori_topk=0,
        cmp_topk=selected_tokens,
        cmp_ratio=1,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=0,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=False,
        has_cmp_kv=True,
        device=str(ctx.device),
    )

    def operator():
        return attention_op(
            query,
            ori_kv=None,
            cmp_kv=cmp_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=None,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            sinks=None,
            metadata=metadata,
            softmax_scale=softmax_scale,
            cmp_ratio=1,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=0,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            return_softmax_lse=True,
        )[0]

    return operator
