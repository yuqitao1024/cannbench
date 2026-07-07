from __future__ import annotations

from cannbench.operators.materialize import materialized_values_to_buffer

from .materialize import materialize_lightning_indexer_inputs


def build_cuda_library_callable(ctx):
    adapter_op = ctx.backend._resolve_cuda_dsa_adapter("lightning_indexer")
    payload = materialize_lightning_indexer_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    query = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["query"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["query_shape"])
    keys = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["keys"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["key_shape"])
    weights = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["weights"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["weight_shape"])

    def operator():
        return adapter_op(
            torch=ctx.torch,
            request=ctx.request,
            case=ctx.case,
            payload=payload,
            device=ctx.device,
            dtype=ctx.dtype,
            query=query,
            keys=keys,
            weights=weights,
            top_k=payload["top_k"],
        )

    return operator


def build_vllm_ascend_callable(ctx):
    metadata_op, indexer_op = ctx.backend._custom_op_pair(
        ctx.torch,
        "npu_vllm_quant_lightning_indexer_metadata",
        "npu_vllm_quant_lightning_indexer",
    )
    if metadata_op is not None and indexer_op is not None:
        return _build_vllm_ascend_quant_callable(
            ctx,
            metadata_op=metadata_op,
            indexer_op=indexer_op,
        )

    try:
        import torch_npu
    except ModuleNotFoundError as exc:
        raise RuntimeError("torch_npu is required for vllm_ascend lightning_indexer") from exc
    if not hasattr(torch_npu, "npu_lightning_indexer"):
        raise RuntimeError(
            "vllm_ascend lightning_indexer requires torch_npu.npu_lightning_indexer"
        )

    payload = materialize_lightning_indexer_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    query_shape = payload["query_shape"]
    key_shape = payload["key_shape"]
    batch, query_tokens, index_heads, index_dim = query_shape
    context_tokens = key_shape[1]

    query = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["query"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(batch * query_tokens, index_heads, index_dim)
    keys = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["keys"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(batch, context_tokens, 1, index_dim)
    weights = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["weights"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["weight_shape"])
    actual_seq_lengths_query = ctx.backend._tensor(
        ctx.torch,
        tuple((index + 1) * query_tokens for index in range(batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    actual_seq_lengths_key = ctx.backend._tensor(
        ctx.torch,
        tuple((index + 1) * context_tokens for index in range(batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )

    def operator():
        result = torch_npu.npu_lightning_indexer(
            query=query,
            key=keys,
            weights=weights,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_key=actual_seq_lengths_key,
            block_table=None,
            layout_query="TND",
            layout_key="BSND",
            sparse_count=payload["top_k"],
            sparse_mode=3,
        )
        return result[0] if isinstance(result, tuple) else result

    return operator


def _build_vllm_ascend_quant_callable(ctx, *, metadata_op, indexer_op):
    payload = materialize_lightning_indexer_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    query_shape = payload["query_shape"]
    key_shape = payload["key_shape"]
    batch, query_tokens, index_heads, index_dim = query_shape
    context_tokens = key_shape[1]
    block_size = 128 if context_tokens % 128 == 0 else context_tokens
    blocks_per_batch = context_tokens // block_size
    quant_dtype = getattr(ctx.torch, "float8_e4m3fn", getattr(ctx.torch, "int8", ctx.dtype))
    scale_dtype = getattr(ctx.torch, "float32", ctx.dtype)

    query = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["query"]),
        device=ctx.device,
        dtype=quant_dtype,
    ).reshape(batch * query_tokens, index_heads, index_dim)
    keys = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["keys"]),
        device=ctx.device,
        dtype=quant_dtype,
    ).reshape(batch * blocks_per_batch, block_size, 1, index_dim)
    weights = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["weights"]),
        device=ctx.device,
        dtype=scale_dtype,
    ).reshape(batch * query_tokens, index_heads)
    query_dequant_scale = ctx.backend._tensor(
        ctx.torch,
        tuple(1.0 for _ in range(batch * query_tokens * index_heads)),
        device=ctx.device,
        dtype=scale_dtype,
    ).reshape(batch * query_tokens, index_heads)
    key_dequant_scale = ctx.backend._tensor(
        ctx.torch,
        tuple(1.0 for _ in range(batch * blocks_per_batch * block_size)),
        device=ctx.device,
        dtype=scale_dtype,
    ).reshape(batch * blocks_per_batch, block_size, 1)
    actual_seq_lengths_query = ctx.backend._tensor(
        ctx.torch,
        tuple((index + 1) * query_tokens for index in range(batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    actual_seq_lengths_key = ctx.backend._tensor(
        ctx.torch,
        tuple((index + 1) * context_tokens for index in range(batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    block_table = ctx.backend._tensor(
        ctx.torch,
        tuple(range(batch * blocks_per_batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(batch, blocks_per_batch)
    common_kwargs = {
        "actual_seq_lengths_query": actual_seq_lengths_query,
        "actual_seq_lengths_key": actual_seq_lengths_key,
        "num_heads_q": index_heads,
        "num_heads_k": 1,
        "head_dim": index_dim,
        "query_quant_mode": 0,
        "key_quant_mode": 0,
        "batch_size": batch,
        "max_seqlen_q": query_tokens,
        "max_seqlen_k": context_tokens,
        "layout_query": "TND",
        "layout_key": "PA_BSND",
        "sparse_count": payload["top_k"],
        "sparse_mode": 3,
        "pre_tokens": (1 << 63) - 1,
        "next_tokens": (1 << 63) - 1,
        "cmp_ratio": 4,
    }
    metadata = metadata_op(**common_kwargs, device=str(ctx.device))
    metadata_only_kwargs = {
        "num_heads_q",
        "num_heads_k",
        "head_dim",
        "batch_size",
        "max_seqlen_q",
        "max_seqlen_k",
    }
    indexer_kwargs = {
        key: value
        for key, value in common_kwargs.items()
        if key not in metadata_only_kwargs
    }

    def operator():
        result = indexer_op(
            query=query,
            key=keys,
            weights=weights,
            query_dequant_scale=query_dequant_scale,
            key_dequant_scale=key_dequant_scale,
            block_table=block_table,
            metadata=metadata,
            return_value=False,
            **indexer_kwargs,
        )
        return result[0] if isinstance(result, tuple) else result

    return operator
