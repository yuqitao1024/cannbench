from __future__ import annotations

from cannbench.core.profile import ProfileKernelSelection
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_softmax_inputs
from .cases import get_softmax_case, get_softmax_dataset
from cannbench.operators.plugin import OperatorPlugin, ProfileKernelSelectionContext
from cannbench.operators.spec import OperatorSpec


def _build_torch_callable(ctx):
    payload = materialize_softmax_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
    tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["shape"])
    return lambda: ctx.torch.softmax(tensor, dim=payload["dim"])


def _simt_module_name(version: str | None) -> str | None:
    return {
        "v1": "aten_softmax",
        "v2": "aten_softmax_v2",
        "v3": "aten_softmax_v3",
    }.get(version or "v1")


def _build_simt_callable(ctx):
    if ctx.implementation_module is None:
        raise RuntimeError("softmax SIMT implementation module is not loaded")
    payload = materialize_softmax_inputs(ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed)
    tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["shape"])
    return lambda: ctx.implementation_module.ops.spatial_softmax_forward(
        tensor,
        int(payload["dim"]),
    )


def _build_profile_kernel_selection(ctx: ProfileKernelSelectionContext):
    if (
        ctx.backend == "ascend"
        and ctx.implementation == "simt"
        and ctx.implementation_version == "v3"
    ):
        return ProfileKernelSelection(
            kernel_name_patterns=("softmax", "aten_softmax_v3"),
            aggregate_across_files=True,
        )
    return ProfileKernelSelection(kernel_name_patterns=("softmax",))


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="softmax",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="softmax",
        runner_name="softmax",
    ),
    get_dataset=get_softmax_dataset,
    get_case=get_softmax_case,
    materialize_inputs=materialize_softmax_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=0,
    build_simt_callable=_build_simt_callable,
    simt_module_name=_simt_module_name,
    build_profile_kernel_selection=_build_profile_kernel_selection,
)
