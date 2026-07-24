from __future__ import annotations

import math

from cannbench.core.profile import ProfileKernelSelection
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_sparse_attention_inputs
from .cases import (
    get_sparse_attention_case,
    get_sparse_attention_dataset,
)
from cannbench.operators.plugin import OperatorPlugin, ProfileKernelSelectionContext
from cannbench.operators.spec import OperatorSpec
from .external import build_cuda_library_callable, build_vllm_ascend_callable


def _build_torch_callable(ctx):
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
        dtype=ctx.torch.long,
    ).reshape(payload["indices_shape"])
    def operator():
        batch, query_heads, query_tokens, head_dim = query.shape
        context_tokens = keys.shape[2]
        selected_tokens = indices.shape[2]
        expanded_keys = keys
        expanded_values = values
        if expanded_keys.shape[1] != query_heads:
            repeats = query_heads // expanded_keys.shape[1]
            expanded_keys = expanded_keys.repeat_interleave(repeats, dim=1)
            expanded_values = expanded_values.repeat_interleave(repeats, dim=1)

        gather_index = indices[:, None, :, :, None].expand(
            batch, query_heads, query_tokens, selected_tokens, head_dim
        )
        key_source = expanded_keys[:, :, None, :, :].expand(
            batch, query_heads, query_tokens, context_tokens, head_dim
        )
        value_source = expanded_values[:, :, None, :, :].expand(
            batch, query_heads, query_tokens, context_tokens, head_dim
        )
        selected_keys = ctx.torch.gather(key_source, 3, gather_index)
        selected_values = ctx.torch.gather(value_source, 3, gather_index)
        scores = (query.unsqueeze(3) * selected_keys).sum(dim=-1) / math.sqrt(head_dim)
        if payload["causal"] and payload["phase"] == "prefill":
            positions = ctx.torch.arange(query_tokens, device=query.device).reshape(
                1, 1, query_tokens, 1
            )
            scores = scores.masked_fill(indices[:, None, :, :] > positions, float("-inf"))
        probabilities = ctx.torch.softmax(scores.float(), dim=-1).to(dtype=query.dtype)
        return (probabilities.unsqueeze(-1) * selected_values).sum(dim=-2)

    return operator


def _simt_module_name(version: str | None) -> str | None:
    if (version or "v1") == "v1":
        return "aten_dsa_sparse_attention"
    return None


def _select_simt_family(payload: dict[str, object]) -> str:
    if payload["head_dim"] == 512 and payload["kv_heads"] == 1:
        return "family_hd512"
    if payload["head_dim"] == 576 and payload["kv_heads"] == 1:
        return "family_hd576"
    if payload["head_dim"] == 256 and payload["kv_heads"] == 1:
        return "family_hd256"
    if (
        payload["head_dim"] == 128
        and payload["kv_heads"] == 1
        and payload["query_heads"] == 128
    ):
        return "family_hd128"
    return "fallback"


def _build_simt_callable(ctx):
    if ctx.implementation_module is None:
        raise RuntimeError("sparse_attention SIMT implementation module is not loaded")
    payload = materialize_sparse_attention_inputs(
        ctx.case,
        dtype=ctx.request.dtype,
        seed=ctx.request.seed,
    )
    family = _select_simt_family(payload)
    if family == "fallback":
        raise RuntimeError(
            "sparse_attention SIMT custom op does not support this shape family"
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
        dtype=ctx.torch.long,
    ).reshape(payload["indices_shape"])
    return lambda: ctx.implementation_module.ops.sparse_attention_forward(
        query,
        keys,
        values,
        indices,
        phase=str(payload["phase"]),
        family=family,
        causal=bool(payload["causal"]),
    )


def _build_profile_kernel_selection(ctx: ProfileKernelSelectionContext):
    if ctx.implementation == "simt":
        return ProfileKernelSelection(
            kernel_name_patterns=("sparse_attention", "aten_dsa_sparse_attention")
        )
    return ProfileKernelSelection(kernel_name_patterns=("sparse", "attention"))


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="sparse_attention",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="sparse_attention",
        runner_name="sparse_attention",
    ),
    get_dataset=get_sparse_attention_dataset,
    get_case=get_sparse_attention_case,
    materialize_inputs=materialize_sparse_attention_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=13,
    build_cuda_library_callable=build_cuda_library_callable,
    build_vllm_ascend_callable=build_vllm_ascend_callable,
    build_simt_callable=_build_simt_callable,
    simt_module_name=_simt_module_name,
    build_profile_kernel_selection=_build_profile_kernel_selection,
)
