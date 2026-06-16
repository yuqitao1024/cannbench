from __future__ import annotations

from cannbench.operators.spec import OperatorSpec

_OPERATOR_SPECS = {
    "softmax": OperatorSpec(
        name="softmax",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="softmax",
        runner_name="softmax",
    ),
    "embedding": OperatorSpec(
        name="embedding",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="embedding",
        runner_name="embedding",
    ),
    "gather": OperatorSpec(
        name="gather",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="gather",
        runner_name="gather",
    ),
    "index_select": OperatorSpec(
        name="index_select",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="index_select",
        runner_name="index_select",
    ),
    "take_along_dim": OperatorSpec(
        name="take_along_dim",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="take_along_dim",
        runner_name="take_along_dim",
    ),
}


def get_operator_spec(name: str) -> OperatorSpec:
    try:
        return _OPERATOR_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported operator: {name}") from exc


def list_operator_names() -> tuple[str, ...]:
    return tuple(_OPERATOR_SPECS)
