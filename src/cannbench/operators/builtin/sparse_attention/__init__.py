from __future__ import annotations

import math

from cannbench.core.profile import ProfileKernelSelection
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import (
    _requires_direct_device_inputs,
    materialize_sparse_attention_inputs,
)
from .cases import (
    get_sparse_attention_case,
    get_sparse_attention_dataset,
)
from .bound_inputs import bound_indices
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
    shared_kv = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["shared_kv"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["shared_kv_shape"])
    indices = bound_indices(ctx, payload["indices_shape"], dtype=ctx.torch.long)
    if indices is None:
        indices = ctx.backend._tensor(
            ctx.torch,
            payload["indices"],
            device=ctx.device,
            dtype=ctx.torch.long,
        ).reshape(payload["indices_shape"])
    topk_lengths = ctx.backend._tensor(
        ctx.torch,
        payload["topk_lengths"],
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(payload["query_shape"][0], payload["query_shape"][2])
    def operator():
        batch, query_heads, query_tokens, qk_head_dim = query.shape
        value_head_dim = int(payload["value_head_dim"])
        values = shared_kv[:, :, :, :value_head_dim]
        context_tokens = shared_kv.shape[2]
        selected_tokens = indices.shape[2]
        expanded_keys = shared_kv
        expanded_values = values
        if expanded_keys.shape[1] != query_heads:
            repeats = query_heads // expanded_keys.shape[1]
            expanded_keys = expanded_keys.repeat_interleave(repeats, dim=1)
            expanded_values = expanded_values.repeat_interleave(repeats, dim=1)

        invalid_indices = (indices < 0) | (indices >= context_tokens)
        topk_positions = ctx.torch.arange(
            selected_tokens, device=query.device
        ).reshape(1, 1, selected_tokens)
        invalid_indices = invalid_indices | (
            topk_positions >= topk_lengths.unsqueeze(-1)
        )
        safe_indices = indices.clamp(min=0, max=context_tokens - 1)
        key_gather_index = safe_indices[:, None, :, :, None].expand(
            batch, query_heads, query_tokens, selected_tokens, qk_head_dim
        )
        value_gather_index = safe_indices[:, None, :, :, None].expand(
            batch, query_heads, query_tokens, selected_tokens, value_head_dim
        )
        key_source = expanded_keys[:, :, None, :, :].expand(
            batch, query_heads, query_tokens, context_tokens, qk_head_dim
        )
        value_source = expanded_values[:, :, None, :, :].expand(
            batch, query_heads, query_tokens, context_tokens, value_head_dim
        )
        selected_keys = ctx.torch.gather(key_source, 3, key_gather_index)
        selected_values = ctx.torch.gather(value_source, 3, value_gather_index)
        scores = (query.unsqueeze(3) * selected_keys).sum(dim=-1)
        scores = scores * payload["softmax_scale"]
        invalid_mask = invalid_indices[:, None, :, :]
        if payload["causal"]:
            positions = (
                ctx.torch.arange(query_tokens, device=query.device)
                + (context_tokens - query_tokens)
            ).reshape(1, 1, query_tokens, 1)
            invalid_mask = invalid_mask | (indices[:, None, :, :] > positions)
        scores = scores.masked_fill(invalid_mask, float("-inf"))
        scores_float = scores.float()
        probabilities = ctx.torch.softmax(scores_float, dim=-1)
        lse = ctx.torch.logsumexp(scores_float, dim=-1)
        all_invalid = invalid_mask.all(dim=-1)
        probabilities = probabilities.masked_fill(all_invalid.unsqueeze(-1), 0.0)
        lse = lse.masked_fill(all_invalid, float("-inf"))
        output = (
            probabilities.to(dtype=query.dtype).unsqueeze(-1) * selected_values
        ).sum(dim=-2)
        return output.permute(0, 2, 1, 3), lse.permute(0, 2, 1)

    return operator


def _simt_module_name(version: str | None) -> str | None:
    if (version or "v1") == "v1":
        return "aten_dsa_sparse_attention"
    return None


def _select_simt_family(payload: dict[str, object]) -> str:
    dimensions = (payload["qk_head_dim"], payload["value_head_dim"])
    if dimensions == (512, 512) and payload["kv_heads"] == 1:
        return "family_hd512"
    if dimensions == (576, 512) and payload["kv_heads"] == 1:
        return "family_hd576"
    if dimensions == (256, 256) and payload["kv_heads"] == 1:
        return "family_hd256"
    if (
        dimensions == (128, 128)
        and payload["kv_heads"] == 1
        and payload["query_heads"] == 128
    ):
        return "family_hd128"
    return "fallback"


def _build_simt_callable(ctx):
    if ctx.implementation_module is None:
        raise RuntimeError("sparse_attention SIMT implementation module is not loaded")
    direct_device_inputs = _requires_direct_device_inputs(ctx.case)
    if direct_device_inputs:
        payload = {
            "query_shape": (
                ctx.case.batch,
                ctx.case.query_heads,
                ctx.case.query_tokens,
                ctx.case.qk_head_dim,
            ),
            "shared_kv_shape": (
                ctx.case.batch,
                ctx.case.kv_heads,
                ctx.case.context_tokens,
                ctx.case.qk_head_dim,
            ),
            "indices_shape": (
                ctx.case.batch,
                ctx.case.query_tokens,
                ctx.case.selected_tokens,
            ),
            "query_heads": ctx.case.query_heads,
            "kv_heads": ctx.case.kv_heads,
            "selected_tokens": ctx.case.selected_tokens,
            "qk_head_dim": ctx.case.qk_head_dim,
            "value_head_dim": ctx.case.value_head_dim,
            "causal": ctx.case.causal,
            "phase": ctx.case.phase,
            "softmax_scale": ctx.case.softmax_scale,
            "topk_lengths": ctx.case.resolved_topk_lengths,
        }
    else:
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
    if any(
        length != payload["selected_tokens"]
        for length in payload["topk_lengths"]
    ):
        raise RuntimeError(
            "sparse_attention SIMT custom op requires full topk_lengths"
        )
    expected_scale = payload["qk_head_dim"] ** -0.5
    if not math.isclose(payload["softmax_scale"], expected_scale):
        raise RuntimeError(
            "sparse_attention SIMT custom op requires qk_head_dim**-0.5 "
            "softmax_scale"
        )

    if direct_device_inputs:
        query = ctx.torch.zeros(
            payload["query_shape"], device=ctx.device, dtype=ctx.dtype
        )
        shared_kv = ctx.torch.zeros(
            payload["shared_kv_shape"], device=ctx.device, dtype=ctx.dtype
        )
        indices = bound_indices(ctx, payload["indices_shape"])
        if indices is None:
            indices = ctx.torch.zeros(
                payload["indices_shape"], device=ctx.device, dtype=ctx.torch.long
            )
    else:
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
        indices = bound_indices(ctx, payload["indices_shape"])
        if indices is None:
            indices = ctx.backend._tensor(
                ctx.torch,
                payload["indices"],
                device=ctx.device,
                dtype=ctx.torch.long,
            ).reshape(payload["indices_shape"])
    return lambda: ctx.implementation_module.ops.sparse_attention_forward(
        query,
        shared_kv,
        indices,
        value_head_dim=int(payload["value_head_dim"]),
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
