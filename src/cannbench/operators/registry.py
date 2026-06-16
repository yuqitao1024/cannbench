from __future__ import annotations

from cannbench.operators.spec import OperatorSpec

_OPERATOR_SPECS = {
    "softmax": OperatorSpec(
        name="softmax",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="softmax",
        runner_name="softmax",
    ),
}


def get_operator_spec(name: str) -> OperatorSpec:
    try:
        return _OPERATOR_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported operator: {name}") from exc


def list_operator_names() -> tuple[str, ...]:
    return tuple(_OPERATOR_SPECS)

