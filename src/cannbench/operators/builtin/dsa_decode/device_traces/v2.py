from __future__ import annotations

from cannbench.operators.builtin.lightning_indexer.cases import LightningIndexerCase
from cannbench.operators.builtin.sparse_attention.cases import SparseAttentionCase
from cannbench.operators.shape_trace import (
    AxisRole,
    DeviceExecutionTrace,
    DeviceKernelTrace,
    ShapeAxis,
    ShapeTensor,
)


def _axis(symbol: str, value: int, meaning: str, role: AxisRole) -> ShapeAxis:
    return ShapeAxis(symbol=symbol, value=value, meaning=meaning, role=role)


def build_v2_device_trace(
    indexer: LightningIndexerCase,
    sparse: SparseAttentionCase,
) -> DeviceExecutionTrace:
    query_atom_count = (indexer.query_tokens + 1) // 2
    base_tasks = indexer.batch * query_atom_count
    context_shards = max(
        value for value in (16, 8, 4, 2, 1) if base_tasks * value <= 32
    )
    index_tasks = base_tasks * context_shards
    context_per_shard = indexer.context_tokens // context_shards
    context_tile = 32
    head_tile = 64
    selected_tile = 64
    qk_tile = 128
    head_groups = sparse.query_heads // head_tile
    attention_tasks = sparse.batch * sparse.query_tokens * head_groups

    index_kernel = DeviceKernelTrace(
        id="lightning-indexer-v2",
        title="Lightning Indexer v2",
        summary="Query-pair atoms split the context across adaptive shards.",
        task_count=index_tasks,
        used_core_count=min(index_tasks, 32),
        task_formula=(
            f"B={indexer.batch} x query_atoms={query_atom_count} "
            f"x context_shards={context_shards} "
            f"({indexer.batch} x {query_atom_count} x {context_shards})"
        ),
        task_axes=(
            _axis("B", indexer.batch, "batch requests", "preserved"),
            _axis(
                "Qa",
                query_atom_count,
                "two-query atoms per request",
                "preserved",
            ),
            _axis(
                "Cs",
                context_shards,
                "context shards per query atom",
                "produced",
            ),
        ),
        tile_tensors=(
            ShapeTensor(
                id="index-query-atom",
                label="Query atom",
                axes=(
                    _axis("Qta", 2, "query rows per atom", "preserved"),
                    _axis("Hi", indexer.index_heads, "index heads", "preserved"),
                    _axis("Di", indexer.index_dim, "index dimension", "contracted"),
                ),
            ),
            ShapeTensor(
                id="index-key-tile",
                label="Key tile",
                axes=(
                    _axis(
                        "Ct",
                        context_tile,
                        "context positions per tile",
                        "preserved",
                    ),
                    _axis("Di", indexer.index_dim, "index dimension", "contracted"),
                ),
            ),
            ShapeTensor(
                id="index-score-tile",
                label="Score tile",
                axes=(
                    _axis("Qta", 2, "query rows per atom", "preserved"),
                    _axis("Hi", indexer.index_heads, "index heads", "reduced"),
                    _axis("Ct", context_tile, "context positions per tile", "produced"),
                ),
                logical_only=True,
            ),
        ),
        steps=(
            f"Assign {context_per_shard} context positions to each shard.",
            f"Process context tiles of {context_tile} positions.",
            "Reduce index heads and maintain shard-local TopK candidates.",
        ),
    )
    attention_kernel = DeviceKernelTrace(
        id="sparse-attention-v2",
        title="Sparse Attention v2",
        summary="Head64, P=1 direct output, no Combine",
        task_count=attention_tasks,
        used_core_count=min(attention_tasks, 32),
        task_formula=(
            f"B={sparse.batch} x Q={sparse.query_tokens} "
            f"x head_groups={head_groups}"
        ),
        task_axes=(
            _axis("B", sparse.batch, "batch requests", "preserved"),
            _axis("Q", sparse.query_tokens, "query tokens", "preserved"),
            _axis("Hg", head_groups, "groups of 64 query heads", "produced"),
        ),
        tile_tensors=(
            ShapeTensor(
                id="attention-query-tile",
                label="Q tile",
                axes=(
                    _axis("Ht", head_tile, "query heads per tile", "preserved"),
                    _axis(
                        "Dqkt",
                        qk_tile,
                        "QK features per compute tile",
                        "contracted",
                    ),
                ),
            ),
            ShapeTensor(
                id="attention-selected-key-tile",
                label="Selected K transposed",
                axes=(
                    _axis(
                        "Dqkt",
                        qk_tile,
                        "QK features per compute tile",
                        "contracted",
                    ),
                    _axis("St", selected_tile, "selected tokens per tile", "produced"),
                ),
            ),
            ShapeTensor(
                id="attention-score-tile",
                label="Score tile",
                axes=(
                    _axis("Ht", head_tile, "query heads per tile", "preserved"),
                    _axis("St", selected_tile, "selected tokens per tile", "produced"),
                ),
                logical_only=True,
            ),
        ),
        steps=(
            "Load one Head64 query group.",
            f"Stream selected tokens in tiles of {selected_tile}.",
            f"Contract QK features in tiles of {qk_tile}.",
            "Accumulate QK, online softmax, and PV directly to output.",
        ),
    )
    return DeviceExecutionTrace(
        status="available",
        implementation="simt",
        version="v2",
        message=None,
        kernels=(index_kernel, attention_kernel),
    )


__all__ = ["build_v2_device_trace"]
