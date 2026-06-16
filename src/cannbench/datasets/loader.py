from __future__ import annotations

from dataclasses import dataclass

from cannbench.datasets.gather import get_gather_case, get_gather_dataset
from cannbench.datasets.index_select import (
    get_index_select_case,
    get_index_select_dataset,
)
from cannbench.datasets.take_along_dim import (
    get_take_along_dim_case,
    get_take_along_dim_dataset,
)
from cannbench.datasets.masked_select import (
    get_masked_select_case,
    get_masked_select_dataset,
)
from cannbench.datasets.cross_entropy import (
    get_cross_entropy_case,
    get_cross_entropy_dataset,
)
from cannbench.datasets.scatter_add import (
    get_scatter_add_case,
    get_scatter_add_dataset,
)
from cannbench.datasets.embedding import get_embedding_case, get_embedding_dataset
from cannbench.datasets.softmax import get_softmax_case, get_softmax_dataset


@dataclass(frozen=True)
class OperatorDataset:
    name: str
    dataset_namespace: str

    def get(self, split: str):
        if self.dataset_namespace == "softmax":
            return get_softmax_dataset(split)
        if self.dataset_namespace == "embedding":
            return get_embedding_dataset(split)
        if self.dataset_namespace == "gather":
            return get_gather_dataset(split)
        if self.dataset_namespace == "index_select":
            return get_index_select_dataset(split)
        if self.dataset_namespace == "take_along_dim":
            return get_take_along_dim_dataset(split)
        if self.dataset_namespace == "masked_select":
            return get_masked_select_dataset(split)
        if self.dataset_namespace == "cross_entropy":
            return get_cross_entropy_dataset(split)
        if self.dataset_namespace == "scatter_add":
            return get_scatter_add_dataset(split)
        raise ValueError(f"Unknown operator dataset namespace: {self.dataset_namespace}")


def get_operator_dataset(name: str) -> OperatorDataset:
    if name == "softmax":
        return OperatorDataset(name="softmax", dataset_namespace="softmax")
    if name == "embedding":
        return OperatorDataset(name="embedding", dataset_namespace="embedding")
    if name == "gather":
        return OperatorDataset(name="gather", dataset_namespace="gather")
    if name == "index_select":
        return OperatorDataset(name="index_select", dataset_namespace="index_select")
    if name == "take_along_dim":
        return OperatorDataset(
            name="take_along_dim", dataset_namespace="take_along_dim"
        )
    if name == "masked_select":
        return OperatorDataset(name="masked_select", dataset_namespace="masked_select")
    if name == "cross_entropy":
        return OperatorDataset(name="cross_entropy", dataset_namespace="cross_entropy")
    if name == "scatter_add":
        return OperatorDataset(name="scatter_add", dataset_namespace="scatter_add")
    raise ValueError(f"Unsupported operator: {name}")


def get_operator_case(op_name: str, dataset_name: str, case_id: str):
    if op_name == "softmax":
        return get_softmax_case(dataset_name, case_id)
    if op_name == "embedding":
        return get_embedding_case(dataset_name, case_id)
    if op_name == "gather":
        return get_gather_case(dataset_name, case_id)
    if op_name == "index_select":
        return get_index_select_case(dataset_name, case_id)
    if op_name == "take_along_dim":
        return get_take_along_dim_case(dataset_name, case_id)
    if op_name == "masked_select":
        return get_masked_select_case(dataset_name, case_id)
    if op_name == "cross_entropy":
        return get_cross_entropy_case(dataset_name, case_id)
    if op_name == "scatter_add":
        return get_scatter_add_case(dataset_name, case_id)
    raise ValueError(f"Unsupported operator: {op_name}")
