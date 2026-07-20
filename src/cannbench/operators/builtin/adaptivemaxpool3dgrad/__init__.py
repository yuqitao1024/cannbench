from __future__ import annotations

from cannbench.core.profile import ProfileKernelSelection
from cannbench.operators.materialize import materialized_values_to_buffer
from cannbench.operators.plugin import OperatorPlugin, ProfileKernelSelectionContext
from cannbench.operators.spec import OperatorSpec

from .cases import (
    get_adaptivemaxpool3dgrad_case,
    get_adaptivemaxpool3dgrad_dataset,
)
from .materialize import materialize_adaptivemaxpool3dgrad_inputs


def _indices_for_backend(ctx, indices):
    if getattr(ctx.device, "type", None) == "npu":
        return indices.to(ctx.torch.int32)
    return indices


def _build_torch_callable(ctx):
    payload = materialize_adaptivemaxpool3dgrad_inputs(
        ctx.case,
        dtype=ctx.request.dtype,
        seed=ctx.request.seed,
    )
    input_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["input_values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["input_shape"])
    grad_output = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["grad_output_values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["output_shape"])
    indices = ctx.backend._tensor(
        ctx.torch,
        payload["indices"],
        device=ctx.device,
        dtype=ctx.torch.long,
    )
    indices = indices.reshape(payload["output_shape"])
    indices = _indices_for_backend(ctx, indices)
    return lambda: ctx.torch.ops.aten.adaptive_max_pool3d_backward(
        grad_output,
        input_tensor,
        indices,
    )


def _simt_module_name(version: str | None) -> str | None:
    resolved_version = version or "v1"
    if resolved_version == "v1":
        return "aten_adaptive_max_pool3d_grad"
    if resolved_version == "v2":
        return "aten_adaptive_max_pool3d_grad_v2"
    if resolved_version == "v3":
        return "aten_adaptive_max_pool3d_grad_v3"
    if resolved_version == "v4":
        return "aten_adaptive_max_pool3d_grad_v4"
    if resolved_version == "v5":
        return "aten_adaptive_max_pool3d_grad_v5"
    return None


def _build_simt_callable(ctx):
    if ctx.implementation_module is None:
        raise RuntimeError("adaptivemaxpool3dgrad SIMT implementation module is not loaded")
    payload = materialize_adaptivemaxpool3dgrad_inputs(
        ctx.case,
        dtype=ctx.request.dtype,
        seed=ctx.request.seed,
    )
    input_tensor = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["input_values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["input_shape"])
    grad_output = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["grad_output_values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["output_shape"])
    indices = ctx.backend._tensor(
        ctx.torch,
        payload["indices"],
        device=ctx.device,
        dtype=ctx.torch.long,
    ).reshape(payload["output_shape"])
    indices = _indices_for_backend(ctx, indices)
    return lambda: ctx.implementation_module.ops.adaptive_max_pool3d_backward(
        grad_output,
        input_tensor,
        indices,
    )


def _build_profile_kernel_selection(ctx: ProfileKernelSelectionContext):
    if ctx.implementation == "simt":
        if ctx.implementation_version == "v5":
            return ProfileKernelSelection(
                kernel_name_patterns=(
                    "adaptive_max_pool3d_grad_v5_zero_kernel",
                    "adaptive_max_pool3d_grad_v5_scatter_kernel",
                ),
                aggregate_across_files=True,
            )
        return ProfileKernelSelection(
            kernel_name_patterns=(
                "aten_adaptive_max_pool3d_grad",
                "aten_adaptive_max_pool3d_grad_v2",
                "aten_adaptive_max_pool3d_grad_v3",
                "aten_adaptive_max_pool3d_grad_v4",
                "adaptive_max_pool3d_grad",
            )
        )
    if ctx.backend == "ascend":
        return ProfileKernelSelection(
            kernel_name_patterns=(
                "AdaptiveMaxPool3DGrad",
                "AdaptiveMaxPool3DGradD",
                "adaptive_max_pool3d_backward",
                "adaptivemaxpool3dgrad",
            )
        )
    return ProfileKernelSelection(
        kernel_name_patterns=(
            "adaptive_max_pool3d_backward",
            "adaptivemaxgradinput",
            "atomicadaptivemaxgradinput",
            "adaptivemaxpool3dgrad",
        ),
        launch_count=4,
    )


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="adaptivemaxpool3dgrad",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="adaptivemaxpool3dgrad",
        runner_name="adaptivemaxpool3dgrad",
    ),
    get_dataset=get_adaptivemaxpool3dgrad_dataset,
    get_case=get_adaptivemaxpool3dgrad_case,
    materialize_inputs=materialize_adaptivemaxpool3dgrad_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=20,
    build_simt_callable=_build_simt_callable,
    simt_module_name=_simt_module_name,
    build_profile_kernel_selection=_build_profile_kernel_selection,
    payload_summary_key_order=("input_shape", "output_size"),
)
