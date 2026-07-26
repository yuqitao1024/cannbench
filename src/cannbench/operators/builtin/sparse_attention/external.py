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
_CUDA_NVTX_RANGE = "cannbench_sparse_attention_dynamic"


def build_cuda_library_callable(ctx):
    adapter_op = _resolve_cuda_dsa_adapter("sparse_attention")
    adapter_prepare = _resolve_cuda_dsa_preparer("sparse_attention")
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
    softmax_scale = payload["softmax_scale"]
    topk_lengths = None
    if any(
        length != payload["selected_tokens"]
        for length in payload["topk_lengths"]
    ):
        topk_length_values = payload["topk_lengths"]
        topk_length_shape = (
            payload["query_shape"][0], payload["query_shape"][2]
        )
        if payload["phase"] == "decode":
            batch, query_tokens = topk_length_shape
            rows = tuple(
                topk_length_values[index * query_tokens : (index + 1) * query_tokens]
                for index in range(batch)
            )
            if any(len(set(row)) != 1 for row in rows):
                raise RuntimeError(
                    "FlashMLA decode requires one topk_length per request"
                )
            topk_length_values = tuple(row[0] for row in rows)
            topk_length_shape = (batch,)
        topk_lengths = ctx.backend._tensor(
            ctx.torch,
            topk_length_values,
            device=ctx.device,
            dtype=ctx.torch.int32,
        ).reshape(topk_length_shape)

    adapter_kwargs = {
        "torch": ctx.torch,
        "request": ctx.request,
        "case": ctx.case,
        "payload": payload,
        "device": ctx.device,
        "dtype": ctx.dtype,
        "query": query,
        "shared_kv": shared_kv,
        "indices": indices,
        "causal": payload["causal"],
        "phase": payload["phase"],
        "softmax_scale": softmax_scale,
        "topk_lengths": topk_lengths,
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
        try:
            import torch_npu
        except ModuleNotFoundError:
            torch_npu = None
        attention_op = getattr(
            torch_npu, "npu_sparse_flash_attention", None
        )
    if attention_op is None:
        raise RuntimeError(
            "vllm_ascend V3.2 sparse_attention requires "
            "torch.ops._C_ascend.npu_sparse_flash_attention or "
            "torch_npu.npu_sparse_flash_attention"
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
    query_lens = tuple(
        getattr(ctx.case, "resolved_query_lens", (query_tokens,) * batch)
    )
    context_lens = tuple(
        getattr(ctx.case, "resolved_context_lens", (context_tokens,) * batch)
    )
    cu_seqlens_q = tuple(
        getattr(
            ctx.case,
            "cu_seqlens_q",
            tuple(index * query_tokens for index in range(batch + 1)),
        )
    )
    total_query_tokens = cu_seqlens_q[-1]
    rope_head_dim = qk_head_dim - value_head_dim
    total_context_tokens = sum(context_lens)

    canonical_query_shape = (
        batch,
        query_heads,
        query_tokens,
        qk_head_dim,
    )
    key_shape = (
        total_context_tokens,
        kv_heads,
        value_head_dim,
    )
    key_rope_shape = (
        total_context_tokens,
        kv_heads,
        rope_head_dim,
    )
    canonical_indices_shape = (batch, query_tokens, selected_tokens)
    softmax_scale = float(
        getattr(ctx.case, "softmax_scale", None) or qk_head_dim**-0.5
    )

    if direct_device_inputs:
        canonical_query = ctx.torch.zeros(
            canonical_query_shape, device=ctx.device, dtype=ctx.dtype
        )
        key = ctx.torch.zeros(key_shape, device=ctx.device, dtype=ctx.dtype)
        key_rope = ctx.torch.zeros(
            key_rope_shape, device=ctx.device, dtype=ctx.dtype
        )
        value = ctx.torch.zeros(key_shape, device=ctx.device, dtype=ctx.dtype)
        canonical_indices = ctx.bound_inputs.get("indices")
        if canonical_indices is None:
            canonical_indices = ctx.torch.zeros(
                canonical_indices_shape,
                device=ctx.device,
                dtype=ctx.torch.int32,
            )
    else:
        payload = materialize_sparse_attention_inputs(
            ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
        )
        key_values, key_rope_values = _split_bhtd_values_as_bthd(
            payload["shared_kv"],
            batch=batch,
            heads=kv_heads,
            tokens=context_tokens,
            part_dims=(value_head_dim, rope_head_dim),
            token_lens=context_lens,
        )
        value_values, _ = _split_bhtd_values_as_bthd(
            payload["shared_kv"],
            batch=batch,
            heads=kv_heads,
            tokens=context_tokens,
            part_dims=(value_head_dim, rope_head_dim),
            token_lens=context_lens,
        )
        canonical_query = ctx.backend._tensor(
            ctx.torch,
            materialized_values_to_buffer(payload["query"]),
            device=ctx.device,
            dtype=ctx.dtype,
        ).reshape(canonical_query_shape)
        key = ctx.backend._tensor(
            ctx.torch, key_values, device=ctx.device, dtype=ctx.dtype
        ).reshape(key_shape)
        key_rope = ctx.backend._tensor(
            ctx.torch, key_rope_values, device=ctx.device, dtype=ctx.dtype
        ).reshape(key_rope_shape)
        value = ctx.backend._tensor(
            ctx.torch, value_values, device=ctx.device, dtype=ctx.dtype
        ).reshape(key_shape)
        canonical_indices = ctx.bound_inputs.get("indices")
        if canonical_indices is None:
            canonical_indices = ctx.backend._tensor(
                ctx.torch,
                payload["indices"],
                device=ctx.device,
                dtype=ctx.torch.int32,
            ).reshape(canonical_indices_shape)
    actual_seq_lengths_query = ctx.backend._tensor(
        ctx.torch,
        cu_seqlens_q[1:],
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    actual_seq_lengths_kv = ctx.backend._tensor(
        ctx.torch,
        _cumulative_lengths(context_lens),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )

    def operator():
        query, query_rope = _lower_vllm_sparse_attention_query(
            ctx.torch,
            canonical_query,
            query_lens=query_lens,
            padded_query_tokens=query_tokens,
            query_heads=query_heads,
            value_head_dim=value_head_dim,
            rope_head_dim=rope_head_dim,
        )
        sparse_indices = _lower_vllm_sparse_attention_indices(
            ctx.torch,
            canonical_indices,
            query_lens=query_lens,
            padded_query_tokens=query_tokens,
            kv_heads=kv_heads,
            selected_tokens=selected_tokens,
        )
        result = attention_op(
            query=query,
            key=key,
            value=value,
            sparse_indices=sparse_indices,
            scale_value=softmax_scale,
            block_table=None,
            actual_seq_lengths_query=actual_seq_lengths_query,
            actual_seq_lengths_kv=actual_seq_lengths_kv,
            query_rope=query_rope,
            key_rope=key_rope,
            sparse_block_size=1,
            layout_query="TND",
            layout_kv="TND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=True,
        )
        return _normalize_ascend_sfa_result(
            ctx.torch,
            result,
            batch=batch,
            query_tokens=query_tokens,
            query_heads=query_heads,
            value_head_dim=value_head_dim,
            query_lens=query_lens,
        )

    return operator


def _cumulative_lengths(lengths):
    total = 0
    cumulative = []
    for length in lengths:
        total += int(length)
        cumulative.append(total)
    return tuple(cumulative)


def _split_bhtd_values_as_bthd(
    values,
    *,
    batch: int,
    heads: int,
    tokens: int,
    part_dims: tuple[int, ...],
    token_lens: tuple[int, ...] | None = None,
):
    logical_dim = sum(part_dims)
    parts = tuple([] for _ in part_dims)
    if token_lens is None:
        token_lens = (tokens,) * batch
    for batch_index in range(batch):
        for token_index in range(token_lens[batch_index]):
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


def _lower_vllm_sparse_attention_query(
    torch,
    query,
    *,
    query_lens,
    padded_query_tokens: int,
    query_heads: int,
    value_head_dim: int,
    rope_head_dim: int,
):
    packed = query.permute(0, 2, 1, 3)
    total_query_tokens = sum(query_lens)
    if total_query_tokens == len(query_lens) * padded_query_tokens:
        packed = packed.reshape(
            total_query_tokens,
            query_heads,
            value_head_dim + rope_head_dim,
        )
    else:
        rows = tuple(
            packed[batch_index, :query_len].reshape(
                query_len,
                query_heads,
                value_head_dim + rope_head_dim,
            )
            for batch_index, query_len in enumerate(query_lens)
        )
        packed = torch.cat(rows, dim=0)
    return (
        packed[..., :value_head_dim].contiguous(),
        packed[..., value_head_dim:].contiguous(),
    )


def _lower_vllm_sparse_attention_indices(
    torch,
    indices,
    *,
    query_lens,
    padded_query_tokens: int,
    kv_heads: int,
    selected_tokens: int,
):
    if getattr(indices, "dtype", torch.int32) != torch.int32:
        indices = indices.to(dtype=torch.int32)
    total_query_tokens = sum(query_lens)
    if total_query_tokens == len(query_lens) * padded_query_tokens:
        return indices.reshape(total_query_tokens, kv_heads, selected_tokens)
    rows = tuple(
        indices[batch_index, :query_len].reshape(
            query_len, kv_heads, selected_tokens
        )
        for batch_index, query_len in enumerate(query_lens)
    )
    return torch.cat(rows, dim=0)


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
    softmax_scale = payload["softmax_scale"]

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
        result = attention_op(
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
        )
        return _normalize_ascend_sfa_result(
            ctx.torch,
            result,
            batch=batch,
            query_tokens=query_tokens,
            query_heads=query_heads,
            value_head_dim=qk_head_dim,
        )

    return operator


def _normalize_ascend_sfa_result(
    torch,
    result,
    *,
    batch: int,
    query_tokens: int,
    query_heads: int,
    value_head_dim: int,
    query_lens: tuple[int, ...] | None = None,
):
    if not isinstance(result, tuple) or len(result) < 2:
        raise RuntimeError(
            "vllm_ascend sparse_attention must return output and LSE statistics"
        )
    if len(result) == 2:
        output, lse = result
        return _reshape_or_pad_tnd_result(
            torch,
            output,
            lse,
            batch=batch,
            query_tokens=query_tokens,
            query_heads=query_heads,
            value_head_dim=value_head_dim,
            query_lens=query_lens,
        )
    output, softmax_max, softmax_sum = result[:3]
    lse = softmax_max + torch.log(softmax_sum)
    return _reshape_or_pad_tnd_result(
        torch,
        output,
        lse,
        batch=batch,
        query_tokens=query_tokens,
        query_heads=query_heads,
        value_head_dim=value_head_dim,
        query_lens=query_lens,
    )


def _reshape_or_pad_tnd_result(
    torch,
    output,
    lse,
    *,
    batch: int,
    query_tokens: int,
    query_heads: int,
    value_head_dim: int,
    query_lens: tuple[int, ...] | None,
):
    if query_lens is None:
        query_lens = (query_tokens,) * batch
    total_query_tokens = sum(query_lens)
    output = output.reshape(total_query_tokens, query_heads, value_head_dim)
    lse = lse.reshape(total_query_tokens, query_heads)
    if total_query_tokens == batch * query_tokens:
        return (
            output.reshape(batch, query_tokens, query_heads, value_head_dim),
            lse.reshape(batch, query_tokens, query_heads),
        )
    padded_output = torch.zeros(
        (batch, query_tokens, query_heads, value_head_dim),
        device=output.device,
        dtype=output.dtype,
    )
    padded_lse = torch.full(
        (batch, query_tokens, query_heads),
        float("-inf"),
        device=lse.device,
        dtype=lse.dtype,
    )
    offset = 0
    for batch_index, query_len in enumerate(query_lens):
        padded_output[batch_index, :query_len] = output[offset : offset + query_len]
        padded_lse[batch_index, :query_len] = lse[offset : offset + query_len]
        offset += query_len
    return padded_output, padded_lse
