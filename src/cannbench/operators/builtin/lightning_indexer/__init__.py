from __future__ import annotations

from .cases import (
    get_lightning_indexer_case,
    get_lightning_indexer_dataset,
)
from cannbench.core.profile import ProfileKernelSelection
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_lightning_indexer_inputs, valid_context_lengths
from cannbench.operators.plugin import OperatorPlugin, ProfileKernelSelectionContext
from cannbench.operators.spec import OperatorSpec
from .external import build_cuda_library_callable, build_vllm_ascend_callable

_MAX_HOST_MATERIALIZED_ELEMENTS = 64 * 1024 * 1024


def _build_torch_callable(ctx):
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
    valid_lengths = ctx.backend._tensor(
        ctx.torch,
        payload["valid_context_lengths"],
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(payload["query_shape"][:2])
    query_lens = ctx.backend._tensor(
        ctx.torch,
        payload["query_lens"],
        device=ctx.device,
        dtype=ctx.torch.int32,
    )
    def operator():
        index_scores = ctx.torch.einsum("bqhd,bcd->bqhc", query, keys)
        index_scores = ctx.torch.relu(index_scores)
        index_scores = index_scores * weights.unsqueeze(-1)
        index_scores = index_scores.sum(dim=2)
        if payload["score_scale"] != 1.0:
            index_scores = index_scores * payload["score_scale"]
        context_positions = ctx.torch.arange(
            payload["key_shape"][1], device=query.device
        ).reshape(1, 1, -1)
        index_scores = index_scores.masked_fill(
            context_positions >= valid_lengths.unsqueeze(-1),
            float("-inf"),
        )
        result = ctx.torch.topk(
            index_scores,
            payload["top_k"],
            dim=-1,
            largest=True,
            sorted=True,
        ).indices
        valid_queries = ctx.torch.arange(
            payload["query_shape"][1], device=query.device
        ).reshape(1, -1) < query_lens.reshape(-1, 1)
        return result.masked_fill(~valid_queries.unsqueeze(-1), -1)

    return operator


def _simt_module_name(version: str | None) -> str | None:
    if (version or "v1") == "v1":
        return "aten_dsa_lightning_indexer"
    return None


def _select_simt_family(payload: dict[str, object]) -> str:
    if payload["index_heads"] == 64 and payload["index_dim"] == 128:
        return "family_64x128"
    if payload["index_heads"] == 32 and payload["index_dim"] == 128:
        return "family_32x128"
    if payload["index_heads"] == 4 and payload["index_dim"] == 64:
        return "family_4x64"
    return "fallback"


def _build_simt_callable(ctx):
    if ctx.implementation_module is None:
        raise RuntimeError("lightning_indexer SIMT implementation module is not loaded")
    if _requires_direct_device_inputs(ctx.case):
        payload = {
            "query_shape": (
                ctx.case.batch,
                ctx.case.query_tokens,
                ctx.case.index_heads,
                ctx.case.index_dim,
            ),
            "key_shape": (
                ctx.case.batch,
                ctx.case.context_tokens,
                ctx.case.index_dim,
            ),
            "weight_shape": (
                ctx.case.batch,
                ctx.case.query_tokens,
                ctx.case.index_heads,
            ),
            "index_heads": ctx.case.index_heads,
            "index_dim": ctx.case.index_dim,
            "top_k": ctx.case.top_k,
            "phase": ctx.case.phase,
            "causal": ctx.case.causal,
            "score_scale": ctx.case.score_scale,
            "tie_policy": ctx.case.tie_policy,
            "query_lens": ctx.case.resolved_query_lens,
            "context_lens": ctx.case.resolved_context_lens,
            "query_start_positions": ctx.case.resolved_query_start_positions,
            "cu_seqlens_q": ctx.case.cu_seqlens_q,
            "cu_seqlens_kv": ctx.case.cu_seqlens_kv,
            "page_block_size": ctx.case.resolved_page_block_size,
            "block_tables": ctx.case.block_tables,
            "valid_context_lengths": valid_context_lengths(ctx.case),
        }
        query = ctx.torch.zeros(
            payload["query_shape"], device=ctx.device, dtype=ctx.dtype
        )
        keys = ctx.torch.zeros(
            payload["key_shape"], device=ctx.device, dtype=ctx.dtype
        )
        weights = ctx.torch.zeros(
            payload["weight_shape"], device=ctx.device, dtype=ctx.dtype
        )
    else:
        payload = materialize_lightning_indexer_inputs(
            ctx.case,
            dtype=ctx.request.dtype,
            seed=ctx.request.seed,
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
    native_valid_context_lengths = ctx.backend._tensor(
        ctx.torch,
        payload["valid_context_lengths"],
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(payload["query_shape"][:2])
    family = _select_simt_family(payload)
    ragged = (
        payload["query_lens"] != (payload["query_shape"][1],) * ctx.case.batch
        or payload["context_lens"]
        != (payload["key_shape"][1],) * ctx.case.batch
    )
    if ragged:
        return _build_ragged_simt_operator(
            ctx,
            payload,
            query=query,
            keys=keys,
            weights=weights,
            valid_context_lengths=native_valid_context_lengths,
            family=family,
        )
    return lambda: ctx.implementation_module.ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        valid_context_lengths=native_valid_context_lengths,
        top_k=int(payload["top_k"]),
        phase=str(payload["phase"]),
        family=family,
    )


def _build_ragged_simt_operator(
    ctx,
    payload,
    *,
    query,
    keys,
    weights,
    valid_context_lengths,
    family: str,
):
    batch, padded_query_tokens = payload["query_shape"][:2]

    def operator():
        output = ctx.torch.full(
            (batch, padded_query_tokens, payload["top_k"]),
            -1,
            device=ctx.device,
            dtype=ctx.torch.int32,
        )
        for batch_index, (query_len, context_len) in enumerate(
            zip(payload["query_lens"], payload["context_lens"], strict=True)
        ):
            if query_len == 0:
                continue
            output[batch_index : batch_index + 1, :query_len] = (
                ctx.implementation_module.ops.lightning_indexer_forward(
                    query[batch_index : batch_index + 1, :query_len],
                    keys[batch_index : batch_index + 1, :context_len],
                    weights[batch_index : batch_index + 1, :query_len],
                    valid_context_lengths=valid_context_lengths[
                        batch_index : batch_index + 1, :query_len
                    ],
                    top_k=int(payload["top_k"]),
                    phase=str(payload["phase"]),
                    family=family,
                )
            )
        return output

    return operator


def _requires_direct_device_inputs(case) -> bool:
    sizes = (
        case.batch * case.query_tokens * case.index_heads * case.index_dim,
        case.batch * case.context_tokens * case.index_dim,
        case.batch * case.query_tokens * case.index_heads,
    )
    return max(sizes) > _MAX_HOST_MATERIALIZED_ELEMENTS


def _build_profile_kernel_selection(ctx: ProfileKernelSelectionContext):
    if ctx.implementation == "simt":
        return ProfileKernelSelection(
            kernel_name_patterns=("lightning_indexer", "aten_dsa_lightning_indexer")
        )
    if ctx.backend == "nvidia" and ctx.implementation == "cuda_library":
        return ProfileKernelSelection(
            kernel_name_patterns=("mqa_logits", "topk"),
            launch_count=2,
        )
    return ProfileKernelSelection(kernel_name_patterns=("lightning", "indexer"))


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="lightning_indexer",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="lightning_indexer",
        runner_name="lightning_indexer",
    ),
    get_dataset=get_lightning_indexer_dataset,
    get_case=get_lightning_indexer_case,
    materialize_inputs=materialize_lightning_indexer_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=12,
    build_cuda_library_callable=build_cuda_library_callable,
    build_vllm_ascend_callable=build_vllm_ascend_callable,
    build_simt_callable=_build_simt_callable,
    simt_module_name=_simt_module_name,
    build_profile_kernel_selection=_build_profile_kernel_selection,
)
