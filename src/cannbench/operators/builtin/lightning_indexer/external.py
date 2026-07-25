from __future__ import annotations

import importlib
import os

from cannbench.operators.materialize import materialized_values_to_buffer

from .materialize import materialize_lightning_indexer_inputs

_CUDA_DSA_ADAPTER_ENV = "CANNBENCH_CUDA_DSA_ADAPTER"
_DEFAULT_CUDA_DSA_ADAPTER_MODULE = "cannbench_cuda_dsa"


def build_cuda_library_callable(ctx):
    adapter_op = _resolve_cuda_dsa_adapter("lightning_indexer")
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
            score_scale=payload["score_scale"],
            tie_policy=payload["tie_policy"],
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
            "cuda_library lightning_indexer benchmarking requires an external "
            f"adapter. Install {_DEFAULT_CUDA_DSA_ADAPTER_MODULE} or set "
            f"{_CUDA_DSA_ADAPTER_ENV}=<module> with callable {op_name}."
        ) from exc
    op_callable = getattr(module, op_name, None)
    if not callable(op_callable):
        raise RuntimeError(
            f"CUDA lightning_indexer adapter {module_name} must expose callable {op_name}"
        )
    return op_callable


def build_vllm_ascend_callable(ctx):
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
    query_values = _pack_padded_rows(
        payload["query"],
        row_width=index_heads * index_dim,
        padded_rows=query_tokens,
        row_counts=payload["query_lens"],
    )
    key_values = _pack_padded_rows(
        payload["keys"],
        row_width=index_dim,
        padded_rows=context_tokens,
        row_counts=payload["context_lens"],
    )
    weight_values = _pack_padded_rows(
        payload["weights"],
        row_width=index_heads,
        padded_rows=query_tokens,
        row_counts=payload["query_lens"],
    )
    total_query_tokens = payload["cu_seqlens_q"][-1]
    total_context_tokens = payload["cu_seqlens_kv"][-1]

    query = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(query_values),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(total_query_tokens, index_heads, index_dim)
    keys = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(key_values),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(total_context_tokens, 1, index_dim)
    weights = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(weight_values),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(total_query_tokens, index_heads)
    actual_seq_lengths_query = ctx.backend._tensor(
        ctx.torch,
        payload["cu_seqlens_q"][1:],
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    actual_seq_lengths_key = ctx.backend._tensor(
        ctx.torch,
        payload["cu_seqlens_kv"][1:],
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
            layout_key="TND",
            sparse_count=payload["top_k"],
            sparse_mode=3,
        )
        indices = result[0] if isinstance(result, tuple) else result
        indices = indices.reshape(total_query_tokens, payload["top_k"])
        if total_query_tokens == batch * query_tokens:
            return indices.reshape(batch, query_tokens, payload["top_k"])
        padded = ctx.torch.full(
            (batch, query_tokens, payload["top_k"]),
            -1,
            device=ctx.device,
            dtype=ctx.torch.int32,
        )
        offset = 0
        for batch_index, query_len in enumerate(payload["query_lens"]):
            padded[batch_index, :query_len] = indices[offset : offset + query_len]
            offset += query_len
        return padded

    return operator


def _pack_padded_rows(
    values,
    *,
    row_width: int,
    padded_rows: int,
    row_counts: tuple[int, ...],
):
    packed = []
    batch_stride = padded_rows * row_width
    for batch_index, row_count in enumerate(row_counts):
        start = batch_index * batch_stride
        packed.extend(values[start : start + row_count * row_width])
    return tuple(packed)
