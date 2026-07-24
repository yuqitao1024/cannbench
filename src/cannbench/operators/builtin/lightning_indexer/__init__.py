from __future__ import annotations

from .cases import (
    get_lightning_indexer_case,
    get_lightning_indexer_dataset,
)
from cannbench.core.profile import ProfileKernelSelection
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_lightning_indexer_inputs
from cannbench.operators.plugin import OperatorPlugin, ProfileKernelSelectionContext
from cannbench.operators.spec import OperatorSpec
from .external import build_cuda_library_callable, build_vllm_ascend_callable


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
    def operator():
        index_scores = ctx.torch.einsum("bqhd,bcd->bqhc", query, keys)
        index_scores = ctx.torch.relu(index_scores)
        index_scores = index_scores * weights.unsqueeze(-1)
        index_scores = index_scores.sum(dim=2)
        return ctx.torch.topk(
            index_scores,
            payload["top_k"],
            dim=-1,
            largest=True,
            sorted=True,
        ).indices

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
    payload = materialize_lightning_indexer_inputs(
        ctx.case,
        dtype=ctx.request.dtype,
        seed=ctx.request.seed,
    )
    family = _select_simt_family(payload)
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
    return lambda: ctx.implementation_module.ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        top_k=int(payload["top_k"]),
        phase=str(payload["phase"]),
        family=family,
    )


def _build_profile_kernel_selection(ctx: ProfileKernelSelectionContext):
    if ctx.implementation == "simt":
        return ProfileKernelSelection(
            kernel_name_patterns=("lightning_indexer", "aten_dsa_lightning_indexer")
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
