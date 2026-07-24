from __future__ import annotations

import importlib
import os

from cannbench.operators.materialize import materialized_values_to_buffer

from .materialize import materialize_sparse_attention_inputs

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
    keys = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["keys"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["key_shape"])
    values = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["value_shape"])
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
            keys=keys,
            values=values,
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
    batch, query_heads, query_tokens, head_dim = payload["query_shape"]
    _, kv_heads, context_tokens, _ = payload["key_shape"]
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
            dim=head_dim,
        ),
        device=ctx.device,
        dtype=ctx.dtype,
    )
    query = query.reshape(batch * query_tokens, query_heads, head_dim)
    cmp_kv = ctx.backend._tensor(
        ctx.torch,
        ctx.backend._materialized_kv_values_as_bthd(
            payload["keys"],
            batch=batch,
            kv_heads=kv_heads,
            context_tokens=context_tokens,
            kept_context_tokens=context_tokens,
            logical_dim=head_dim,
            physical_dim=head_dim,
        ),
        device=ctx.device,
        dtype=ctx.dtype,
    )
    cmp_kv = cmp_kv.reshape(batch * blocks_per_batch, block_size, kv_heads, head_dim)
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
    softmax_scale = head_dim ** -0.5

    metadata = metadata_op(
        num_heads_q=query_heads,
        num_heads_kv=kv_heads,
        head_dim=head_dim,
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
