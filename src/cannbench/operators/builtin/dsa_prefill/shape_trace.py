from __future__ import annotations

from cannbench.operators.builtin.lightning_indexer.cases import (
    LightningIndexerCase,
    get_lightning_indexer_case,
)
from cannbench.operators.builtin.sparse_attention.cases import (
    SparseAttentionCase,
    get_sparse_attention_case,
)
from cannbench.operators.shape_trace import (
    DeviceExecutionTrace,
    ShapeAxis,
    ShapeStage,
    ShapeTensor,
    ShapeTrace,
    ShapeTraceKey,
)

from .cases import OPERATOR_NAME, get_dsa_prefill_case, get_dsa_prefill_dataset

STAGE_SPECS = (
    (
        "index-inputs",
        "lightning_indexer",
        "Indexer projection",
        "inputs",
        "Q_idx and K_idx use the retrieval feature space",
        (),
        ("index_query", "index_key"),
        (),
    ),
    (
        "index-matmul",
        "lightning_indexer",
        "Per-head dot product",
        "matmul",
        "[Hi,Di] x [Di,C] -> [Hi,C]",
        ("index_query", "index_key_t"),
        ("head_scores",),
        ("Di",),
    ),
    (
        "index-reduce",
        "lightning_indexer",
        "ReLU, weight and head reduction",
        "reduction",
        "[Hi,C] -> [C]",
        ("head_scores", "weights"),
        ("index_scores",),
        ("Hi",),
    ),
    (
        "topk",
        "lightning_indexer",
        "Select sparse context positions",
        "topk",
        "TopK([C], k=S) -> indices [S]",
        ("index_scores",),
        ("indices",),
        ("C",),
    ),
    (
        "gather",
        "sparse_attention",
        "Gather shared KV",
        "gather",
        "shared_kv [C,Dqk] + indices [S] -> K [S,Dqk], V [S,Dv]",
        ("shared_kv", "indices"),
        ("selected_k", "selected_v"),
        ("C",),
    ),
    (
        "qk",
        "sparse_attention",
        "Sparse attention score matrix",
        "matmul",
        "[H,Dqk] x [Dqk,S] -> [H,S]",
        ("query", "selected_k_t"),
        ("scores",),
        ("Dqk",),
    ),
    (
        "softmax",
        "sparse_attention",
        "Normalize selected-token scores",
        "softmax",
        "scores [H,S] -> P [H,S] + LSE [H]",
        ("scores",),
        ("probabilities", "lse"),
        ("S",),
    ),
    (
        "pv-output",
        "sparse_attention",
        "Weighted value accumulation",
        "matmul",
        "[H,S] x [S,Dv] -> output [H,Dv]",
        ("probabilities", "selected_v"),
        ("output",),
        ("S",),
    ),
)

_GROUP = "deepseek-v32"
_SOURCE_MODEL = "DeepSeek-V3.2"
_COMPONENT_DATASET = "realistic_prefill"
_DEVICE_UNAVAILABLE_MESSAGE = (
    "Prefill is not optimized yet. This view intentionally shows only "
    "the algorithm-level matrix flow."
)

_AGGREGATE_AXES = {
    "index_query_all": ("R", "Hi", "Di"),
    "head_scores_all": ("R", "Hi", "C"),
    "index_scores_all": ("R", "C"),
    "indices_all": ("R", "S"),
    "query_all": ("R", "H", "Dqk"),
    "selected_k_all": ("R", "S", "Dqk"),
    "selected_v_all": ("R", "S", "Dv"),
    "scores_all": ("R", "H", "S"),
    "output_all": ("R", "H", "Dv"),
    "lse_all": ("R", "H"),
}

_AGGREGATE_LABELS = {
    "index_query_all": "All index queries",
    "head_scores_all": "All per-head scores",
    "index_scores_all": "All index scores",
    "indices_all": "All selected indices",
    "query_all": "All attention queries",
    "selected_k_all": "All selected K",
    "selected_v_all": "All selected V",
    "scores_all": "All attention scores / probabilities",
    "output_all": "All outputs",
    "lse_all": "All log-sum-exp values",
}

_LOGICAL_AGGREGATES = {
    "head_scores_all",
    "index_scores_all",
    "selected_k_all",
    "selected_v_all",
    "scores_all",
}

_STAGE_AGGREGATES = {
    "index-inputs": ("index_query_all",),
    "index-matmul": ("head_scores_all",),
    "index-reduce": ("index_scores_all",),
    "topk": ("indices_all",),
    "gather": ("selected_k_all", "selected_v_all"),
    "qk": ("query_all", "scores_all"),
    "softmax": ("scores_all", "lse_all"),
    "pv-output": ("output_all",),
}


def _tensor(
    symbols: dict[str, ShapeAxis],
    tensor_id: str,
    label: str,
    axes: tuple[str, ...],
    *,
    logical_only: bool = False,
) -> ShapeTensor:
    return ShapeTensor(
        id=tensor_id,
        label=label,
        axes=tuple(symbols[symbol] for symbol in axes),
        logical_only=logical_only,
    )


def _build_symbols(
    indexer: LightningIndexerCase,
    sparse: SparseAttentionCase,
) -> tuple[ShapeAxis, ...]:
    return (
        ShapeAxis("B", indexer.batch, "batch requests", "preserved"),
        ShapeAxis("Q", indexer.query_tokens, "query tokens per request", "preserved"),
        ShapeAxis(
            "R",
            indexer.batch * indexer.query_tokens,
            "flattened query rows",
            "produced",
        ),
        ShapeAxis("Hi", indexer.index_heads, "index heads", "reduced"),
        ShapeAxis("Di", indexer.index_dim, "index feature dimension", "contracted"),
        ShapeAxis("C", indexer.context_tokens, "context tokens", "reduced"),
        ShapeAxis("H", sparse.query_heads, "query attention heads", "preserved"),
        ShapeAxis("Hkv", sparse.kv_heads, "shared KV heads", "preserved"),
        ShapeAxis("S", sparse.selected_tokens, "selected context tokens", "contracted"),
        ShapeAxis("Dqk", sparse.qk_head_dim, "QK feature dimension", "contracted"),
        ShapeAxis("Dv", sparse.value_head_dim, "value dimension", "produced"),
    )


def _build_row_tensors(symbols: dict[str, ShapeAxis]) -> dict[str, ShapeTensor]:
    return {
        "index_query": _tensor(symbols, "index_query", "Index query", ("Hi", "Di")),
        "index_key": _tensor(symbols, "index_key", "Index key", ("C", "Di")),
        "index_key_t": _tensor(
            symbols,
            "index_key_t",
            "Index key transposed",
            ("Di", "C"),
        ),
        "head_scores": _tensor(
            symbols,
            "head_scores",
            "Per-head scores",
            ("Hi", "C"),
            logical_only=True,
        ),
        "weights": _tensor(symbols, "weights", "Head weights", ("Hi",)),
        "index_scores": _tensor(
            symbols,
            "index_scores",
            "Index scores",
            ("C",),
            logical_only=True,
        ),
        "indices": _tensor(symbols, "indices", "Selected indices", ("S",)),
        "shared_kv": _tensor(symbols, "shared_kv", "Shared KV", ("C", "Dqk")),
        "selected_k": _tensor(
            symbols,
            "selected_k",
            "Selected K",
            ("S", "Dqk"),
            logical_only=True,
        ),
        "selected_v": _tensor(
            symbols,
            "selected_v",
            "Selected V",
            ("S", "Dv"),
            logical_only=True,
        ),
        "query": _tensor(symbols, "query", "Query", ("H", "Dqk")),
        "selected_k_t": _tensor(
            symbols,
            "selected_k_t",
            "Selected K transposed",
            ("Dqk", "S"),
            logical_only=True,
        ),
        "scores": _tensor(
            symbols,
            "scores",
            "Attention scores",
            ("H", "S"),
            logical_only=True,
        ),
        "probabilities": _tensor(
            symbols,
            "probabilities",
            "Probabilities",
            ("H", "S"),
            logical_only=True,
        ),
        "lse": _tensor(symbols, "lse", "Log-sum-exp", ("H",)),
        "output": _tensor(symbols, "output", "Output", ("H", "Dv")),
    }


def _build_aggregate_tensors(
    symbols: dict[str, ShapeAxis],
    indexer: LightningIndexerCase,
    sparse: SparseAttentionCase,
) -> dict[str, ShapeTensor]:
    r = indexer.batch * indexer.query_tokens
    aggregate_shapes = {
        "index_query_all": (r, indexer.index_heads, indexer.index_dim),
        "head_scores_all": (r, indexer.index_heads, indexer.context_tokens),
        "index_scores_all": (r, indexer.context_tokens),
        "indices_all": (r, indexer.top_k),
        "query_all": (r, sparse.query_heads, sparse.qk_head_dim),
        "selected_k_all": (r, sparse.selected_tokens, sparse.qk_head_dim),
        "selected_v_all": (r, sparse.selected_tokens, sparse.value_head_dim),
        "scores_all": (r, sparse.query_heads, sparse.selected_tokens),
        "output_all": (r, sparse.query_heads, sparse.value_head_dim),
        "lse_all": (r, sparse.query_heads),
    }

    tensors = {}
    for tensor_id, shape in aggregate_shapes.items():
        axis_symbols = _AGGREGATE_AXES[tensor_id]
        axes = tuple(symbols[symbol] for symbol in axis_symbols)
        if tuple(axis.value for axis in axes) != shape:
            raise ValueError(f"Aggregate shape mismatch for {tensor_id}")
        tensors[tensor_id] = ShapeTensor(
            id=tensor_id,
            label=_AGGREGATE_LABELS[tensor_id],
            axes=axes,
            logical_only=tensor_id in _LOGICAL_AGGREGATES,
        )
    return tensors


def _build_stages(
    row_tensors: dict[str, ShapeTensor],
    aggregate_tensors: dict[str, ShapeTensor],
    causal_insight: str,
) -> tuple[ShapeStage, ...]:
    stages = []
    for (
        stage_id,
        component,
        title,
        operation,
        formula,
        input_ids,
        output_ids,
        contracted_axes,
    ) in STAGE_SPECS:
        row_tensor_ids = tuple(dict.fromkeys((*input_ids, *output_ids)))
        stages.append(
            ShapeStage(
                id=stage_id,
                component=component,
                title=title,
                operation=operation,
                formula=formula,
                scope="one flattened query row; aggregate shapes cover all R rows",
                tensors=(
                    *(row_tensors[tensor_id] for tensor_id in row_tensor_ids),
                    *(
                        aggregate_tensors[tensor_id]
                        for tensor_id in _STAGE_AGGREGATES[stage_id]
                    ),
                ),
                input_ids=input_ids,
                output_ids=output_ids,
                contracted_axes=contracted_axes,
                insight=causal_insight if stage_id == "topk" else formula,
            )
        )
    return tuple(stages)


def _build_causal_insight(
    indexer: LightningIndexerCase,
    sparse: SparseAttentionCase,
) -> str:
    start = indexer.resolved_query_start_positions[0]
    if sparse.resolved_query_start_positions != (start,):
        raise ValueError("Prefill component query positions do not match")
    q_end = indexer.query_tokens - 1
    first_valid = min(indexer.context_tokens, start + 1)
    last_valid = min(indexer.context_tokens, start + indexer.query_tokens)
    return (
        f"Canonical causal prefill: q=0..{q_end}; position(q)={start}+q; "
        f"valid_length(q)=min({indexer.context_tokens},{start}+q+1); "
        f"valid length range {first_valid}..{last_valid}. "
        "Indexer masks c >= valid_length(q); "
        "Attention masks index > position(q)."
    )


def list_dsa_prefill_shape_trace_cases() -> tuple[ShapeTraceKey, ...]:
    dataset = "realistic"
    return tuple(
        ShapeTraceKey(
            operator=OPERATOR_NAME,
            dataset=dataset,
            case_id=case.case_id,
            phase=case.phase,
            group=_GROUP,
        )
        for case in get_dsa_prefill_dataset(dataset).cases
        if case.source_model == _SOURCE_MODEL
    )


def build_dsa_prefill_shape_trace(dataset: str, case_id: str) -> ShapeTrace:
    workflow_case = get_dsa_prefill_case(dataset, case_id)
    if dataset != "realistic" or workflow_case.source_model != _SOURCE_MODEL:
        raise ValueError(f"Shape trace unavailable for DSA prefill case: {case_id}")

    indexer = get_lightning_indexer_case(_COMPONENT_DATASET, case_id)
    sparse = get_sparse_attention_case(_COMPONENT_DATASET, case_id)

    from . import _validate_component_pair

    _validate_component_pair(sparse, indexer)
    symbol_values = _build_symbols(indexer, sparse)
    symbols = {axis.symbol: axis for axis in symbol_values}
    return ShapeTrace(
        schema_version=1,
        operator=OPERATOR_NAME,
        dataset=dataset,
        case_id=case_id,
        phase=workflow_case.phase,
        group=_GROUP,
        symbols=symbol_values,
        stages=_build_stages(
            _build_row_tensors(symbols),
            _build_aggregate_tensors(symbols, indexer, sparse),
            _build_causal_insight(indexer, sparse),
        ),
        device_execution=DeviceExecutionTrace(
            status="unavailable",
            implementation="simt",
            version=None,
            message=_DEVICE_UNAVAILABLE_MESSAGE,
            kernels=(),
        ),
    )


__all__ = [
    "STAGE_SPECS",
    "build_dsa_prefill_shape_trace",
    "list_dsa_prefill_shape_trace_cases",
]
