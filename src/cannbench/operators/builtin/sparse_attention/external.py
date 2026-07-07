from __future__ import annotations

from cannbench.operators.materialize import materialized_values_to_buffer

from .materialize import materialize_sparse_attention_inputs


def build_cuda_library_callable(ctx):
    adapter_op = ctx.backend._resolve_cuda_dsa_adapter("sparse_attention")
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


def build_vllm_ascend_callable(ctx):
    metadata_op, attention_op = ctx.backend._custom_op_pair(
        ctx.torch,
        "npu_kv_quant_sparse_attn_sharedkv_metadata",
        "npu_kv_quant_sparse_attn_sharedkv",
    )
    if metadata_op is not None and attention_op is not None:
        return _build_vllm_ascend_quant_callable(
            ctx,
            metadata_op=metadata_op,
            attention_op=attention_op,
        )

    metadata_op, attention_op = ctx.backend._custom_op_pair(
        ctx.torch,
        "npu_sparse_attn_sharedkv_metadata",
        "npu_sparse_attn_sharedkv",
    )
    if metadata_op is None or attention_op is None:
        raise RuntimeError(
            "vllm_ascend sparse_attention requires "
            "torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv_metadata and "
            "torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv, or "
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


def _build_vllm_ascend_quant_callable(ctx, *, metadata_op, attention_op):
    payload = materialize_sparse_attention_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    batch, query_heads, query_tokens, head_dim = payload["query_shape"]
    _, kv_heads, context_tokens, _ = payload["key_shape"]
    selected_tokens = payload["indices_shape"][2]
    a5_physical_layout = (
        head_dim == 512
        and kv_heads == 1
        and query_heads in {64, 128}
        and selected_tokens in {512, 1024}
    )
    cmp_ratio = 4 if a5_physical_layout else 1
    ori_context_tokens = context_tokens
    cmp_context_tokens = context_tokens // cmp_ratio
    ori_block_size = 128 if ori_context_tokens % 128 == 0 else ori_context_tokens
    cmp_block_size = 128 if cmp_context_tokens % 128 == 0 else cmp_context_tokens
    ori_blocks_per_batch = ori_context_tokens // ori_block_size
    cmp_blocks_per_batch = cmp_context_tokens // cmp_block_size
    kv_head_dim = 640 if a5_physical_layout else head_dim
    query_dtype = getattr(ctx.torch, "bfloat16", ctx.dtype)
    kv_dtype = getattr(ctx.torch, "float8_e4m3fn", ctx.dtype)
    scale_dtype = getattr(ctx.torch, "float32", ctx.dtype)

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
        dtype=query_dtype,
    )
    query = query.reshape(batch * query_tokens, query_heads, head_dim)
    ori_kv = None
    ori_block_table = None
    cu_seqlens_ori_kv = None
    if a5_physical_layout:
        ori_kv = ctx.backend._tensor(
            ctx.torch,
            ctx.backend._materialized_kv_values_as_bthd(
                payload["values"],
                batch=batch,
                kv_heads=kv_heads,
                context_tokens=context_tokens,
                kept_context_tokens=ori_context_tokens,
                logical_dim=head_dim,
                physical_dim=kv_head_dim,
            ),
            device=ctx.device,
            dtype=kv_dtype,
        )
        ori_kv = ori_kv.reshape(
            batch * ori_blocks_per_batch,
            ori_block_size,
            kv_heads,
            kv_head_dim,
        )
        ori_block_table = ctx.backend._tensor(
            ctx.torch,
            tuple(range(batch * ori_blocks_per_batch)),
            device=ctx.device,
            dtype=ctx.torch.int32,
        ).reshape(batch, ori_blocks_per_batch)
        cu_seqlens_ori_kv = ctx.backend._tensor(
            ctx.torch,
            tuple(index * ori_context_tokens for index in range(batch + 1)),
            device=ctx.device,
            dtype=ctx.torch.int32,
        )
    cmp_kv = ctx.backend._tensor(
        ctx.torch,
        ctx.backend._materialized_kv_values_as_bthd(
            payload["keys"],
            batch=batch,
            kv_heads=kv_heads,
            context_tokens=context_tokens,
            kept_context_tokens=cmp_context_tokens,
            logical_dim=head_dim,
            physical_dim=kv_head_dim,
        ),
        device=ctx.device,
        dtype=kv_dtype,
    )
    cmp_kv = cmp_kv.reshape(
        batch * cmp_blocks_per_batch, cmp_block_size, kv_heads, kv_head_dim
    )
    cmp_indices = payload["indices"]
    if a5_physical_layout:
        cmp_indices = tuple(index % cmp_context_tokens for index in cmp_indices)
    cmp_sparse_indices = ctx.backend._tensor(
        ctx.torch,
        cmp_indices,
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(batch * query_tokens, kv_heads, selected_tokens)
    cmp_block_table = ctx.backend._tensor(
        ctx.torch,
        tuple(range(batch * cmp_blocks_per_batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(batch, cmp_blocks_per_batch)
    cu_seqlens_q = ctx.backend._tensor(
        ctx.torch,
        tuple(index * query_tokens for index in range(batch + 1)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    cu_seqlens_cmp_kv = ctx.backend._tensor(
        ctx.torch,
        tuple(index * cmp_context_tokens for index in range(batch + 1)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    seqused_kv = ctx.backend._tensor(
        ctx.torch,
        tuple(context_tokens for _ in range(batch)),
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    sinks = None
    if a5_physical_layout:
        sinks = ctx.backend._tensor(
            ctx.torch,
            tuple(0.0 for _ in range(query_heads)),
            device=ctx.device,
            dtype=scale_dtype,
        ).reshape((query_heads,))
    softmax_scale = head_dim ** -0.5

    metadata = metadata_op(
        num_heads_q=query_heads,
        num_heads_kv=kv_heads,
        head_dim=head_dim,
        kv_quant_mode=1,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_ori_kv=cu_seqlens_ori_kv,
        cu_seqlens_cmp_kv=cu_seqlens_cmp_kv,
        seqused_q=None,
        seqused_kv=seqused_kv,
        batch_size=batch,
        max_seqlen_q=query_tokens,
        max_seqlen_kv=context_tokens,
        ori_topk=0,
        cmp_topk=selected_tokens,
        tile_size=64,
        rope_head_dim=64,
        cmp_ratio=cmp_ratio,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=127,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_ND",
        has_ori_kv=a5_physical_layout,
        has_cmp_kv=True,
        device=str(ctx.device),
    )

    def operator():
        return attention_op(
            query,
            kv_quant_mode=1,
            ori_kv=ori_kv,
            cmp_kv=cmp_kv,
            ori_sparse_indices=None,
            cmp_sparse_indices=cmp_sparse_indices,
            ori_block_table=ori_block_table,
            cmp_block_table=cmp_block_table,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_ori_kv=None,
            cu_seqlens_cmp_kv=None,
            seqused_q=None,
            seqused_kv=seqused_kv,
            sinks=sinks,
            metadata=metadata,
            tile_size=64,
            rope_head_dim=64,
            softmax_scale=softmax_scale,
            cmp_ratio=cmp_ratio,
            ori_mask_mode=4,
            cmp_mask_mode=3,
            ori_win_left=127,
            ori_win_right=0,
            layout_q="TND",
            layout_kv="PA_ND",
            return_softmax_lse=True,
        )[0]

    return operator
