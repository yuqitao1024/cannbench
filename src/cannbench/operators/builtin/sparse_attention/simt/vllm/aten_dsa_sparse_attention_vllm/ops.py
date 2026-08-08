from __future__ import annotations

from importlib import import_module


def sparse_flash_attention_forward(**kwargs):
    """Call the vLLM-Ascend sparse-flash-attention ABI selected by CANN."""
    attention_op = _resolve_attention_op()
    return attention_op(**kwargs)


def sparse_attention_forward(
    query,
    shared_kv,
    indices,
    *,
    value_head_dim: int,
    phase: str,
    family: str,
    causal: bool,
):
    """BHTD test wrapper for the copied V3.2 decode implementation."""
    torch = import_module("torch")
    if phase != "decode" or not causal:
        raise RuntimeError(
            "aten_dsa_sparse_attention_vllm only supports causal decode"
        )
    if family != "family_hd576" or query.shape[-1] != 576:
        raise RuntimeError(
            "aten_dsa_sparse_attention_vllm requires the MLA 576/512 family"
        )
    if value_head_dim != 512:
        raise RuntimeError(
            "aten_dsa_sparse_attention_vllm requires value_head_dim=512"
        )

    batch, query_heads, query_tokens, qk_head_dim = query.shape
    _, kv_heads, context_tokens, _ = shared_kv.shape
    selected_tokens = indices.shape[-1]
    packed_query = query.permute(0, 2, 1, 3).reshape(
        batch * query_tokens, query_heads, qk_head_dim
    )
    packed_kv = shared_kv.permute(0, 2, 1, 3).reshape(
        batch * context_tokens, kv_heads, qk_head_dim
    )
    query_nope = packed_query[..., :value_head_dim].contiguous()
    query_rope = packed_query[..., value_head_dim:].contiguous()
    key = packed_kv[..., :value_head_dim].contiguous()
    key_rope = packed_kv[..., value_head_dim:].contiguous()
    value = key.clone()
    sparse_indices = indices.to(dtype=torch.int32).reshape(
        batch * query_tokens, kv_heads, selected_tokens
    )
    actual_seq_lengths_query = torch.tensor(
        tuple((index + 1) * query_tokens for index in range(batch)),
        dtype=torch.int32,
        device=query.device,
    )
    actual_seq_lengths_kv = torch.tensor(
        tuple((index + 1) * context_tokens for index in range(batch)),
        dtype=torch.int32,
        device=query.device,
    )
    result = sparse_flash_attention_forward(
        query=query_nope,
        key=key,
        value=value,
        sparse_indices=sparse_indices,
        scale_value=qk_head_dim**-0.5,
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
    if not isinstance(result, tuple) or len(result) != 3:
        raise RuntimeError(
            "vLLM-Ascend arch35 sparse attention must return output, "
            "softmax_max, and softmax_sum"
        )
    output, softmax_max, softmax_sum = result
    lse = torch.log(softmax_sum) + softmax_max
    return (
        output.reshape(batch, query_tokens, query_heads, value_head_dim),
        lse.permute(1, 0, 2).reshape(batch, query_tokens, query_heads),
    )


def _resolve_attention_op():
    import_module("torch")
    torch_npu = import_module("torch_npu")
    attention_op = getattr(torch_npu, "npu_sparse_flash_attention", None)
    if attention_op is None:
        raise RuntimeError(
            "aten_dsa_sparse_attention_vllm requires the three-output "
            "torch_npu.npu_sparse_flash_attention binding"
        )
    return attention_op
