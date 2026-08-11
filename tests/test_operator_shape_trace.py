from dataclasses import replace

import pytest

from cannbench.operators.shape_trace import (
    DeviceExecutionTrace,
    DeviceKernelTrace,
    ShapeAxis,
    ShapeStage,
    ShapeTensor,
    ShapeTrace,
    ShapeTraceKey,
    shape_trace_to_payload,
)


def _valid_trace() -> ShapeTrace:
    q = ShapeTensor(
        id="q",
        label="Q",
        axes=(
            ShapeAxis("H", 128, "query heads", "preserved"),
            ShapeAxis("Dqk", 576, "QK feature", "contracted"),
        ),
    )
    k = ShapeTensor(
        id="k",
        label="K^T",
        axes=(
            ShapeAxis("Dqk", 576, "QK feature", "contracted"),
            ShapeAxis("S", 2048, "selected tokens", "produced"),
        ),
    )
    scores = ShapeTensor(
        id="scores",
        label="scores",
        axes=(
            ShapeAxis("H", 128, "query heads", "preserved"),
            ShapeAxis("S", 2048, "selected tokens", "produced"),
        ),
    )
    return ShapeTrace(
        schema_version=1,
        operator="dsa_decode",
        dataset="realistic",
        case_id="case",
        phase="decode",
        group="deepseek-v32",
        symbols=(
            ShapeAxis("H", 128, "query heads", "preserved"),
            ShapeAxis("Dqk", 576, "QK feature", "contracted"),
            ShapeAxis("S", 2048, "selected tokens", "produced"),
        ),
        stages=(
            ShapeStage(
                id="qk",
                component="sparse_attention",
                title="QK",
                operation="matmul",
                formula="[H,Dqk] x [Dqk,S] -> [H,S]",
                scope="one query row",
                tensors=(q, k, scores),
                input_ids=("q", "k"),
                output_ids=("scores",),
                contracted_axes=("Dqk",),
                insight="Dqk contracts.",
            ),
        ),
        device_execution=DeviceExecutionTrace(
            status="unavailable",
            implementation="simt",
            version=None,
            message="No device trace.",
            kernels=(),
        ),
    )


def _valid_kernel() -> DeviceKernelTrace:
    return DeviceKernelTrace(
        id="kernel",
        title="Kernel",
        summary="Runs one tile.",
        task_count=2,
        used_core_count=1,
        task_formula="B x Q",
        task_axes=(ShapeAxis("B", 1, "batch", "preserved"),),
        tile_tensors=(
            ShapeTensor(
                id="tile",
                label="Tile",
                axes=(ShapeAxis("T", 64, "tile rows", "produced"),),
            ),
        ),
        steps=("Load tile.",),
    )


def test_shape_trace_serializes_nested_tuples_to_json_payload():
    payload = shape_trace_to_payload(_valid_trace())
    assert payload["schema_version"] == 1
    assert payload["stages"][0]["tensors"][0]["axes"][1]["value"] == 576


def test_shape_trace_rejects_non_positive_axis_values():
    with pytest.raises(ValueError, match="axis value must be positive"):
        ShapeAxis("S", 0, "selected tokens", "produced")


def test_shape_trace_rejects_missing_stage_tensor_references():
    trace = _valid_trace()
    with pytest.raises(ValueError, match="unknown tensor id"):
        replace(trace.stages[0], input_ids=("missing",))


def test_shape_stage_rejects_duplicate_tensor_ids():
    trace = _valid_trace()
    duplicate_q = replace(trace.stages[0].tensors[0], label="Duplicate Q")
    with pytest.raises(ValueError, match="duplicate tensor id.*q"):
        replace(trace.stages[0], tensors=(*trace.stages[0].tensors, duplicate_q))


@pytest.mark.parametrize(
    ("field_name", "references", "duplicate_id"),
    (
        ("input_ids", ("q", "q"), "q"),
        ("output_ids", ("scores", "scores"), "scores"),
    ),
)
def test_shape_stage_rejects_duplicate_tensor_references(
    field_name: str,
    references: tuple[str, ...],
    duplicate_id: str,
):
    stage = _valid_trace().stages[0]
    with pytest.raises(
        ValueError,
        match=rf"{field_name} contains duplicate tensor id: {duplicate_id}",
    ):
        replace(stage, **{field_name: references})


def test_shape_stage_allows_the_same_tensor_as_input_and_output():
    stage = _valid_trace().stages[0]
    replaced = replace(stage, input_ids=("q",), output_ids=("q",))
    assert replaced.input_ids == replaced.output_ids == ("q",)


def test_shape_trace_key_keeps_phase_group_and_identity():
    key = ShapeTraceKey("dsa_prefill", "realistic", "case", "prefill", "deepseek-v32")
    assert shape_trace_to_payload(key) == {
        "operator": "dsa_prefill",
        "dataset": "realistic",
        "case_id": "case",
        "phase": "prefill",
        "group": "deepseek-v32",
    }


def test_shape_trace_normalizes_every_collection_to_tuples():
    trace = _valid_trace()
    tensor = replace(
        trace.stages[0].tensors[0],
        axes=list(trace.stages[0].tensors[0].axes),
    )
    stage = replace(
        trace.stages[0],
        tensors=[tensor, *trace.stages[0].tensors[1:]],
        input_ids=list(trace.stages[0].input_ids),
        output_ids=list(trace.stages[0].output_ids),
        contracted_axes=list(trace.stages[0].contracted_axes),
    )
    kernel = replace(
        _valid_kernel(),
        task_axes=list(_valid_kernel().task_axes),
        tile_tensors=list(_valid_kernel().tile_tensors),
        steps=list(_valid_kernel().steps),
    )
    device = DeviceExecutionTrace(
        status="available",
        implementation="simt",
        version="v1",
        message=None,
        kernels=[kernel],
    )
    normalized_trace = replace(
        trace,
        symbols=list(trace.symbols),
        stages=[stage],
        device_execution=device,
    )

    assert isinstance(tensor.axes, tuple)
    assert isinstance(stage.tensors, tuple)
    assert isinstance(stage.input_ids, tuple)
    assert isinstance(stage.output_ids, tuple)
    assert isinstance(stage.contracted_axes, tuple)
    assert isinstance(kernel.task_axes, tuple)
    assert isinstance(kernel.tile_tensors, tuple)
    assert isinstance(kernel.steps, tuple)
    assert isinstance(device.kernels, tuple)
    assert isinstance(normalized_trace.symbols, tuple)
    assert isinstance(normalized_trace.stages, tuple)


def test_shape_trace_rejects_boolean_schema_version():
    with pytest.raises(ValueError, match="schema_version must be 1"):
        replace(_valid_trace(), schema_version=True)


@pytest.mark.parametrize(
    "build",
    (
        lambda: ShapeAxis(" ", 1, "axis", "preserved"),
        lambda: ShapeTensor("tensor", " ", (ShapeAxis("D", 1, "dim", "produced"),)),
        lambda: replace(_valid_trace().stages[0], component=""),
        lambda: replace(_valid_kernel(), summary=" "),
        lambda: ShapeTraceKey("operator", "dataset", "", "decode", "group"),
        lambda: replace(_valid_trace(), group=" "),
    ),
)
def test_shape_trace_contract_rejects_empty_string_fields(build):
    with pytest.raises(ValueError, match="must not be empty"):
        build()


def test_shape_tensor_rejects_duplicate_axis_symbols():
    axis = ShapeAxis("D", 64, "dimension", "contracted")
    with pytest.raises(ValueError, match="tensor axis symbols must be unique"):
        ShapeTensor("tensor", "Tensor", (axis, axis))


@pytest.mark.parametrize("contracted_axes", (("missing",), ("Dqk", "Dqk")))
def test_shape_stage_rejects_invalid_contracted_axis_references(contracted_axes):
    with pytest.raises(ValueError, match="contracted_axes"):
        replace(_valid_trace().stages[0], contracted_axes=contracted_axes)


@pytest.mark.parametrize(
    ("task_count", "used_core_count"),
    ((0, 0), (1, 0), (1, 2)),
)
def test_device_kernel_rejects_invalid_counts(task_count, used_core_count):
    with pytest.raises(ValueError, match="task/core counts"):
        replace(
            _valid_kernel(),
            task_count=task_count,
            used_core_count=used_core_count,
        )


def test_device_kernel_rejects_duplicate_axes_tensors_and_empty_steps():
    kernel = _valid_kernel()
    with pytest.raises(ValueError, match="task axis symbols must be unique"):
        replace(kernel, task_axes=(*kernel.task_axes, kernel.task_axes[0]))
    with pytest.raises(ValueError, match="tile tensor ids must be unique"):
        replace(kernel, tile_tensors=(*kernel.tile_tensors, kernel.tile_tensors[0]))
    with pytest.raises(ValueError, match="steps must contain non-empty strings"):
        replace(kernel, steps=("",))


@pytest.mark.parametrize(
    "device",
    (
        DeviceExecutionTrace("unavailable", "simt", None, "Unavailable.", ()),
        DeviceExecutionTrace("unavailable", "simt", "v2", "Unavailable.", ()),
    ),
)
def test_device_execution_accepts_unavailable_version_states(device):
    assert device.kernels == ()


@pytest.mark.parametrize(
    "build",
    (
        lambda: DeviceExecutionTrace("available", "simt", None, None, (_valid_kernel(),)),
        lambda: DeviceExecutionTrace("available", "simt", "v1", "unexpected", (_valid_kernel(),)),
        lambda: DeviceExecutionTrace("available", "simt", "v1", None, ()),
        lambda: DeviceExecutionTrace("unavailable", "simt", None, None, ()),
        lambda: DeviceExecutionTrace("unavailable", "simt", None, "Unavailable.", (_valid_kernel(),)),
    ),
)
def test_device_execution_rejects_inconsistent_status_fields(build):
    with pytest.raises(ValueError, match="device execution"):
        build()


def test_shape_trace_rejects_unknown_stage_axis_and_duplicate_symbols():
    trace = _valid_trace()
    unknown_axis = ShapeAxis("X", 1, "unknown", "produced")
    unknown_tensor = ShapeTensor("x", "X", (unknown_axis,))
    stage = ShapeStage(
        id="x-stage",
        component="component",
        title="X",
        operation="identity",
        formula="X -> X",
        scope="one row",
        tensors=(unknown_tensor,),
        input_ids=("x",),
        output_ids=("x",),
        contracted_axes=(),
        insight="Identity.",
    )
    with pytest.raises(ValueError, match="unknown shape trace axis"):
        replace(trace, stages=(stage,))
    with pytest.raises(ValueError, match="symbol ids must be unique"):
        replace(trace, symbols=(*trace.symbols, trace.symbols[0]))


def test_device_execution_rejects_duplicate_kernel_ids():
    kernel = _valid_kernel()
    with pytest.raises(ValueError, match="kernel ids must be unique"):
        DeviceExecutionTrace(
            "available", "simt", "v1", None, (kernel, kernel)
        )
