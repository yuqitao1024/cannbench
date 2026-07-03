from __future__ import annotations

from dataclasses import dataclass

from cannbench.datasets.gather import get_gather_case, get_gather_dataset
from cannbench.datasets.index_select import (
    get_index_select_case,
    get_index_select_dataset,
)
from cannbench.datasets.index_add import (
    get_index_add_case,
    get_index_add_dataset,
)
from cannbench.datasets.index_put import (
    get_index_put_case,
    get_index_put_dataset,
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
from cannbench.datasets.scatter import (
    get_scatter_case,
    get_scatter_dataset,
)
from cannbench.datasets.lightning_indexer import (
    get_lightning_indexer_case,
    get_lightning_indexer_dataset,
)
from cannbench.datasets.sparse_attention import (
    get_sparse_attention_case,
    get_sparse_attention_dataset,
)
from cannbench.datasets.embedding import get_embedding_case, get_embedding_dataset
from cannbench.datasets.softmax import get_softmax_case, get_softmax_dataset
from cannbench.datasets.topk import get_topk_case, get_topk_dataset


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
        if self.dataset_namespace == "index_add":
            return get_index_add_dataset(split)
        if self.dataset_namespace == "index_put":
            return get_index_put_dataset(split)
        if self.dataset_namespace == "take_along_dim":
            return get_take_along_dim_dataset(split)
        if self.dataset_namespace == "masked_select":
            return get_masked_select_dataset(split)
        if self.dataset_namespace == "cross_entropy":
            return get_cross_entropy_dataset(split)
        if self.dataset_namespace == "scatter_add":
            return get_scatter_add_dataset(split)
        if self.dataset_namespace == "scatter":
            return get_scatter_dataset(split)
        if self.dataset_namespace == "topk":
            return get_topk_dataset(split)
        if self.dataset_namespace == "lightning_indexer":
            return get_lightning_indexer_dataset(split)
        if self.dataset_namespace == "sparse_attention":
            return get_sparse_attention_dataset(split)
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
    if name == "index_add":
        return OperatorDataset(name="index_add", dataset_namespace="index_add")
    if name == "index_put":
        return OperatorDataset(name="index_put", dataset_namespace="index_put")
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
    if name == "scatter":
        return OperatorDataset(name="scatter", dataset_namespace="scatter")
    if name == "topk":
        return OperatorDataset(name="topk", dataset_namespace="topk")
    if name == "lightning_indexer":
        return OperatorDataset(
            name="lightning_indexer", dataset_namespace="lightning_indexer"
        )
    if name == "sparse_attention":
        return OperatorDataset(
            name="sparse_attention", dataset_namespace="sparse_attention"
        )
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
    if op_name == "index_add":
        return get_index_add_case(dataset_name, case_id)
    if op_name == "index_put":
        return get_index_put_case(dataset_name, case_id)
    if op_name == "take_along_dim":
        return get_take_along_dim_case(dataset_name, case_id)
    if op_name == "masked_select":
        return get_masked_select_case(dataset_name, case_id)
    if op_name == "cross_entropy":
        return get_cross_entropy_case(dataset_name, case_id)
    if op_name == "scatter_add":
        return get_scatter_add_case(dataset_name, case_id)
    if op_name == "scatter":
        return get_scatter_case(dataset_name, case_id)
    if op_name == "topk":
        return get_topk_case(dataset_name, case_id)
    if op_name == "lightning_indexer":
        return get_lightning_indexer_case(dataset_name, case_id)
    if op_name == "sparse_attention":
        return get_sparse_attention_case(dataset_name, case_id)
    raise ValueError(f"Unsupported operator: {op_name}")
