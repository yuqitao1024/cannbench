from __future__ import annotations

from dataclasses import dataclass

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
        raise ValueError(f"Unknown operator dataset namespace: {self.dataset_namespace}")


def get_operator_dataset(name: str) -> OperatorDataset:
    if name == "softmax":
        return OperatorDataset(name="softmax", dataset_namespace="softmax")
    if name == "embedding":
        return OperatorDataset(name="embedding", dataset_namespace="embedding")
    raise ValueError(f"Unsupported operator: {name}")


def get_operator_case(op_name: str, dataset_name: str, case_id: str):
    if op_name == "softmax":
        return get_softmax_case(dataset_name, case_id)
    if op_name == "embedding":
        return get_embedding_case(dataset_name, case_id)
    raise ValueError(f"Unsupported operator: {op_name}")
