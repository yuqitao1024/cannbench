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
    "index_add": OperatorSpec(
        name="index_add",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="index_add",
        runner_name="index_add",
    ),
    "index_put": OperatorSpec(
        name="index_put",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="index_put",
        runner_name="index_put",
    ),
    "take_along_dim": OperatorSpec(
        name="take_along_dim",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="take_along_dim",
        runner_name="take_along_dim",
    ),
    "masked_select": OperatorSpec(
        name="masked_select",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="masked_select",
        runner_name="masked_select",
    ),
    "cross_entropy": OperatorSpec(
        name="cross_entropy",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="cross_entropy",
        runner_name="cross_entropy",
    ),
    "scatter_add": OperatorSpec(
        name="scatter_add",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="scatter_add",
        runner_name="scatter_add",
    ),
    "scatter": OperatorSpec(
        name="scatter",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="scatter",
        runner_name="scatter",
    ),
    "topk": OperatorSpec(
        name="topk",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="topk",
        runner_name="topk",
    ),
    "lightning_indexer": OperatorSpec(
        name="lightning_indexer",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="lightning_indexer",
        runner_name="lightning_indexer",
    ),
    "sparse_attention": OperatorSpec(
        name="sparse_attention",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="sparse_attention",
        runner_name="sparse_attention",
    ),
}


def get_operator_spec(name: str) -> OperatorSpec:
    try:
        return _OPERATOR_SPECS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported operator: {name}") from exc


def list_operator_names() -> tuple[str, ...]:
    return tuple(_OPERATOR_SPECS)
