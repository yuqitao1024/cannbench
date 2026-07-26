from __future__ import annotations

import importlib
import os

from cannbench.operators.materialize import materialized_values_to_buffer

from .materialize import materialize_lightning_indexer_inputs

_CUDA_DSA_ADAPTER_ENV = "CANNBENCH_CUDA_DSA_ADAPTER"
_DEFAULT_CUDA_DSA_ADAPTER_MODULE = "cannbench_cuda_dsa"
_CUDA_NVTX_RANGE = "cannbench_lightning_indexer_dynamic"


def build_cuda_library_callable(ctx):
    adapter_op = _resolve_cuda_dsa_adapter("lightning_indexer")
    adapter_prepare = _resolve_cuda_dsa_preparer("lightning_indexer")
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

    adapter_kwargs = {
        "torch": ctx.torch,
        "request": ctx.request,
        "case": ctx.case,
        "payload": payload,
        "device": ctx.device,
        "dtype": ctx.dtype,
        "query": query,
        "keys": keys,
        "weights": weights,
        "top_k": payload["top_k"],
        "score_scale": payload["score_scale"],
        "tie_policy": payload["tie_policy"],
    }
    if adapter_prepare is not None:
        dynamic_operator = adapter_prepare(**adapter_kwargs)
        return _with_cuda_nvtx_range(
            ctx.torch, _CUDA_NVTX_RANGE, dynamic_operator
        )

    def operator():
        return adapter_op(**adapter_kwargs)

    return _with_cuda_nvtx_range(ctx.torch, _CUDA_NVTX_RANGE, operator)


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


def _resolve_cuda_dsa_preparer(op_name: str):
    module_name = (
        os.environ.get(_CUDA_DSA_ADAPTER_ENV)
        or _DEFAULT_CUDA_DSA_ADAPTER_MODULE
    )
    module = importlib.import_module(module_name)
    prepare = getattr(module, f"prepare_{op_name}", None)
    return prepare if callable(prepare) else None


def _with_cuda_nvtx_range(torch, range_name: str, operator):
    nvtx = getattr(getattr(torch, "cuda", None), "nvtx", None)
    if nvtx is None:
        return operator

    def profiled_operator():
        nvtx.range_push(range_name)
        try:
            return operator()
        finally:
            nvtx.range_pop()

    return profiled_operator


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
    key_values = _pack_padded_rows(
        payload["keys"],
        row_width=index_dim,
        padded_rows=context_tokens,
        row_counts=payload["context_lens"],
    )
    total_query_tokens = payload["cu_seqlens_q"][-1]
    total_context_tokens = payload["cu_seqlens_kv"][-1]

    query = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["query"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(query_shape)
    keys = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(key_values),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(total_context_tokens, 1, index_dim)
    weights = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["weights"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["weight_shape"])
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
        lowered_query = _lower_vllm_indexer_query(
            ctx.torch,
            query,
            query_lens=payload["query_lens"],
            padded_query_tokens=query_tokens,
            index_heads=index_heads,
            index_dim=index_dim,
        )
        lowered_weights = _lower_vllm_indexer_weights(
            ctx.torch,
            weights,
            query_lens=payload["query_lens"],
            padded_query_tokens=query_tokens,
            index_heads=index_heads,
        )
        result = torch_npu.npu_lightning_indexer(
            query=lowered_query,
            key=keys,
            weights=lowered_weights,
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


def _lower_vllm_indexer_query(
    torch,
    query,
    *,
    query_lens,
    padded_query_tokens: int,
    index_heads: int,
    index_dim: int,
):
    total_query_tokens = sum(query_lens)
    if total_query_tokens == len(query_lens) * padded_query_tokens:
        return query.reshape(total_query_tokens, index_heads, index_dim)
    rows = tuple(
        query[batch_index, :query_len].reshape(
            query_len, index_heads, index_dim
        )
        for batch_index, query_len in enumerate(query_lens)
    )
    return torch.cat(rows, dim=0)


def _lower_vllm_indexer_weights(
    torch,
    weights,
    *,
    query_lens,
    padded_query_tokens: int,
    index_heads: int,
):
    total_query_tokens = sum(query_lens)
    if total_query_tokens == len(query_lens) * padded_query_tokens:
        return weights.reshape(total_query_tokens, index_heads)
    rows = tuple(
        weights[batch_index, :query_len].reshape(query_len, index_heads)
        for batch_index, query_len in enumerate(query_lens)
    )
    return torch.cat(rows, dim=0)


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
