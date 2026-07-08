from __future__ import annotations

from cannbench.core.profile import ProfileKernelSelection
from cannbench.operators.materialize import materialized_values_to_buffer
from cannbench.operators.plugin import OperatorPlugin, ProfileKernelSelectionContext
from cannbench.operators.spec import OperatorSpec

from .cases import get_index_add_case, get_index_add_dataset
from .materialize import materialize_index_add_inputs


def _build_torch_callable(ctx):
    payload = materialize_index_add_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
    input_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["input_shape"])
    index_tensor = ctx.backend._tensor(
        ctx.torch,
        payload["indices"],
        device=ctx.device,
        dtype=ctx.torch.long,
    ).reshape(payload["index_shape"])
    src_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["src"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["src_shape"])
    return lambda: ctx.torch.index_add(
        input_tensor,
        payload["dim"],
        index_tensor,
        src_tensor,
    )


def _simt_module_name(version: str | None) -> str | None:
    if (version or "v1") == "v1":
        return "aten_index_add"
    return None


def _build_simt_callable(ctx):
    if ctx.implementation_module is None:
        raise RuntimeError("index_add SIMT implementation module is not loaded")
    payload = materialize_index_add_inputs(
        ctx.case,
        dtype=ctx.request.dtype,
        seed=ctx.request.seed,
    )
    input_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["input_shape"])
    index_tensor = ctx.backend._tensor(
        ctx.torch,
        payload["indices"],
        device=ctx.device,
        dtype=ctx.torch.int32,
    ).reshape(payload["index_shape"])
    src_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["src"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["src_shape"])
    return lambda: ctx.implementation_module.ops.index_add_forward(
        input_tensor,
        int(payload["dim"]),
        index_tensor,
        src_tensor,
    )


def _build_profile_kernel_selection(ctx: ProfileKernelSelectionContext):
    if ctx.implementation == "simt":
        return ProfileKernelSelection(
            kernel_name_patterns=("index_add", "aten_index_add")
        )
    if ctx.backend == "nvidia":
        return ProfileKernelSelection(
            kernel_name_patterns=(
                "indexadd",
                "inplaceindexadd",
                "indexfuncsmallindex",
                "indexfunclargeindex",
            )
        )
    return ProfileKernelSelection(kernel_name_patterns=("indexadd", "inplaceindexadd"))


def _profile_launch_count(ctx: ProfileKernelSelectionContext) -> int | None:
    if (
        ctx.backend == "ascend"
        and ctx.implementation == "simt"
        and (ctx.implementation_version or "v1") == "v1"
        and ctx.dtype in {"float16", "float32"}
        and ctx.iterations is not None
    ):
        return ctx.iterations * 2
    if (
        ctx.backend == "ascend"
        and ctx.implementation != "simt"
        and ctx.iterations is not None
    ):
        return ctx.iterations * 10
    return None


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="index_add",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="index_add",
        runner_name="index_add",
    ),
    get_dataset=get_index_add_dataset,
    get_case=get_index_add_case,
    materialize_inputs=materialize_index_add_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=4,
    build_simt_callable=_build_simt_callable,
    simt_module_name=_simt_module_name,
    build_profile_kernel_selection=_build_profile_kernel_selection,
    profile_launch_count=_profile_launch_count,
)
