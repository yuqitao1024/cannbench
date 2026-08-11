from __future__ import annotations

import re
from importlib.resources import files

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

from .cases import (
    COMPONENT_OPERATORS,
    OPERATOR_NAME,
    get_dsa_decode_case,
    get_dsa_decode_dataset,
)
from .device_traces import DEVICE_TRACE_BUILDERS

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

_VERSION_RE = re.compile(r"^v([1-9][0-9]*)$")
_GROUP = "deepseek-v32"
_SOURCE_MODEL = "DeepSeek-V3.2"


def latest_numeric_common_version(version_sets: tuple[set[str], ...]) -> str | None:
    if not version_sets:
        return None
    common = set.intersection(*(set(values) for values in version_sets))
    numeric = [
        (int(match.group(1)), version)
        for version in common
        if (match := _VERSION_RE.fullmatch(version)) is not None
    ]
    return max(numeric)[1] if numeric else None


def _component_simt_versions(op: str) -> set[str]:
    root = files(f"cannbench.operators.builtin.{op}").joinpath("simt")
    return {child.name for child in root.iterdir() if child.is_dir()}


def latest_common_simt_version(component_ops: tuple[str, ...]) -> str | None:
    return latest_numeric_common_version(
        tuple(_component_simt_versions(op) for op in component_ops)
    )


def _axis(symbols: dict[str, ShapeAxis], symbol: str) -> ShapeAxis:
    return symbols[symbol]


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
        axes=tuple(_axis(symbols, symbol) for symbol in axes),
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


def _build_tensors(symbols: dict[str, ShapeAxis]) -> dict[str, ShapeTensor]:
    return {
        "index_query": _tensor(symbols, "index_query", "Index query", ("Hi", "Di")),
        "index_key": _tensor(symbols, "index_key", "Index key", ("C", "Hi", "Di")),
        "index_key_t": _tensor(
            symbols,
            "index_key_t",
            "Index key transposed",
            ("Hi", "Di", "C"),
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


def _build_stages(tensors: dict[str, ShapeTensor]) -> tuple[ShapeStage, ...]:
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
        tensor_ids = tuple(dict.fromkeys((*input_ids, *output_ids)))
        stages.append(
            ShapeStage(
                id=stage_id,
                component=component,
                title=title,
                operation=operation,
                formula=formula,
                scope="one flattened query row",
                tensors=tuple(tensors[tensor_id] for tensor_id in tensor_ids),
                input_ids=input_ids,
                output_ids=output_ids,
                contracted_axes=contracted_axes,
                insight=formula,
            )
        )
    return tuple(stages)


def _build_device_execution(
    indexer: LightningIndexerCase,
    sparse: SparseAttentionCase,
) -> DeviceExecutionTrace:
    selected = latest_common_simt_version(COMPONENT_OPERATORS)
    if selected is None:
        return DeviceExecutionTrace(
            status="unavailable",
            implementation="simt",
            version=None,
            message="No common SIMT version is available.",
            kernels=(),
        )
    builder = DEVICE_TRACE_BUILDERS.get(selected)
    if builder is None:
        return DeviceExecutionTrace(
            status="unavailable",
            implementation="simt",
            version=selected,
            message=f"Device trace unavailable for {selected}.",
            kernels=(),
        )
    return builder(indexer, sparse)


def list_dsa_decode_shape_trace_cases() -> tuple[ShapeTraceKey, ...]:
    dataset = "realistic"
    return tuple(
        ShapeTraceKey(
            operator=OPERATOR_NAME,
            dataset=dataset,
            case_id=case.case_id,
            phase=case.phase,
            group=_GROUP,
        )
        for case in get_dsa_decode_dataset(dataset).cases
        if case.source_model == _SOURCE_MODEL
    )


def build_dsa_decode_shape_trace(dataset: str, case_id: str) -> ShapeTrace:
    workflow_case = get_dsa_decode_case(dataset, case_id)
    if dataset != "realistic" or workflow_case.source_model != _SOURCE_MODEL:
        raise ValueError(f"Shape trace unavailable for DSA decode case: {case_id}")

    component_dataset = "realistic_decode"
    indexer = get_lightning_indexer_case(component_dataset, case_id)
    sparse = get_sparse_attention_case(component_dataset, case_id)

    from . import _validate_component_pair

    _validate_component_pair(sparse, indexer)
    symbol_values = _build_symbols(indexer, sparse)
    symbols = {axis.symbol: axis for axis in symbol_values}
    tensors = _build_tensors(symbols)
    return ShapeTrace(
        schema_version=1,
        operator=OPERATOR_NAME,
        dataset=dataset,
        case_id=case_id,
        phase=workflow_case.phase,
        group=_GROUP,
        symbols=symbol_values,
        stages=_build_stages(tensors),
        device_execution=_build_device_execution(indexer, sparse),
    )


__all__ = [
    "STAGE_SPECS",
    "build_dsa_decode_shape_trace",
    "latest_common_simt_version",
    "latest_numeric_common_version",
    "list_dsa_decode_shape_trace_cases",
]
