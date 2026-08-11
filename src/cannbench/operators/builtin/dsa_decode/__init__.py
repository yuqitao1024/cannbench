from __future__ import annotations

from cannbench.operators.builtin.lightning_indexer.cases import (
    LightningIndexerCase,
    get_lightning_indexer_case,
)
from cannbench.operators.builtin.sparse_attention.cases import (
    SparseAttentionCase,
    get_sparse_attention_case,
)
from cannbench.operators.plugin import (
    OperatorPlugin,
    OperatorWorkflow,
    OperatorWorkflowStep,
)
from cannbench.operators.spec import OperatorSpec

from .cases import (
    COMPONENT_OPERATORS,
    OPERATOR_NAME,
    PHASE,
    DsaDecodeCase,
    DsaDecodeDataset,
    get_dsa_decode_case,
    get_dsa_decode_dataset,
)
from .materialize import materialize_dsa_decode_inputs
from .shape_trace import (
    build_dsa_decode_shape_trace,
    list_dsa_decode_shape_trace_cases,
)

DsaFusedWorkflow = OperatorWorkflow
DsaWorkflowStep = OperatorWorkflowStep


def build_dsa_decode_workflow(
    *,
    dataset: str,
    case_id: str,
    dtype: str,
    seed: int,
) -> DsaFusedWorkflow:
    workflow_case = get_dsa_decode_case(dataset, case_id)
    component_dataset = _component_dataset(dataset)
    sparse_case = get_sparse_attention_case(component_dataset, case_id)
    try:
        indexer_case = get_lightning_indexer_case(component_dataset, case_id)
    except ValueError as exc:
        raise ValueError(
            f"No matching lightning_indexer case for sparse_attention case: {case_id}"
        ) from exc

    _validate_component_pair(sparse_case, indexer_case)
    if sparse_case.phase != workflow_case.phase:
        raise ValueError(
            "DSA decode workflow manifest phase mismatch: "
            f"workflow is {workflow_case.phase}, sparse_attention is {sparse_case.phase}"
        )

    return OperatorWorkflow(
        workflow=workflow_case.workflow,
        phase=sparse_case.phase,
        dataset=dataset,
        case_id=case_id,
        steps=(
            OperatorWorkflowStep(
                contract="dsa_index_select",
                op="lightning_indexer",
                dataset=component_dataset,
                case_id=case_id,
                consumes=(),
                produces=("indices",),
                prepared=_build_prepared_operator_input(
                    op="lightning_indexer",
                    dtype=dtype,
                    dataset=component_dataset,
                    case_id=case_id,
                    seed=seed,
                ),
            ),
            OperatorWorkflowStep(
                contract="sparse_mla_decode",
                op="sparse_attention",
                dataset=component_dataset,
                case_id=case_id,
                consumes=("indices",),
                produces=("out", "lse"),
                prepared=_build_prepared_operator_input(
                    op="sparse_attention",
                    dtype=dtype,
                    dataset=component_dataset,
                    case_id=case_id,
                    seed=seed,
                ),
            ),
        ),
    )


def list_dsa_decode_workflows(
    dataset: str,
    *,
    dtype: str = "float16",
    seed: int = 0,
) -> tuple[DsaFusedWorkflow, ...]:
    return tuple(
        build_dsa_decode_workflow(
            dataset=dataset,
            case_id=workflow_case.case_id,
            dtype=dtype,
            seed=seed,
        )
        for workflow_case in get_dsa_decode_dataset(dataset).cases
    )


def _build_unsupported_direct_callable(ctx):
    def _raise_direct_run_error():
        raise RuntimeError(
            f"{ctx.case.payload.get('workflow', 'DSA decode')} direct callable "
            "is not implemented; run it through the bench workflow expansion path"
        )

    return _raise_direct_run_error


def _validate_component_pair(
    sparse_case: SparseAttentionCase, indexer_case: LightningIndexerCase
) -> None:
    indexer_phase = _phase_from_indexer_case(indexer_case)
    if sparse_case.phase != indexer_phase:
        raise ValueError(
            "DSA decode component phase mismatch: "
            f"sparse_attention is {sparse_case.phase}, "
            f"lightning_indexer is {indexer_phase}"
        )
    if sparse_case.phase != PHASE:
        raise ValueError(f"DSA decode requires {PHASE} component cases")
    if sparse_case.batch != indexer_case.batch:
        raise ValueError("DSA decode component batch mismatch")
    if sparse_case.query_tokens != indexer_case.query_tokens:
        raise ValueError("DSA decode component query_tokens mismatch")
    if sparse_case.context_tokens != indexer_case.context_tokens:
        raise ValueError("DSA decode component context_tokens mismatch")
    if sparse_case.selected_tokens != indexer_case.top_k:
        raise ValueError("DSA decode component top_k mismatch")
    for name, sparse_value, indexer_value in (
        (
            "query_lens",
            sparse_case.resolved_query_lens,
            indexer_case.resolved_query_lens,
        ),
        (
            "context_lens",
            sparse_case.resolved_context_lens,
            indexer_case.resolved_context_lens,
        ),
        (
            "query_start_positions",
            sparse_case.resolved_query_start_positions,
            indexer_case.resolved_query_start_positions,
        ),
        (
            "page_block_size",
            sparse_case.resolved_page_block_size,
            indexer_case.resolved_page_block_size,
        ),
        ("block_tables", sparse_case.block_tables, indexer_case.block_tables),
        ("shape_scope", sparse_case.shape_scope, indexer_case.shape_scope),
        ("tp_size", sparse_case.tp_size, indexer_case.tp_size),
        ("dp_size", sparse_case.dp_size, indexer_case.dp_size),
        ("cp_size", sparse_case.cp_size, indexer_case.cp_size),
        ("kv_shard", sparse_case.kv_shard, indexer_case.kv_shard),
    ):
        if sparse_value != indexer_value:
            raise ValueError(f"DSA decode component {name} mismatch")


def _phase_from_indexer_case(case: LightningIndexerCase) -> str:
    if case.family.startswith("decode_") or "_decode_" in case.family:
        return "decode"
    if case.family.startswith("prefill_") or "_prefill_" in case.family:
        return "prefill"
    raise ValueError(f"Unable to infer DSA phase from indexer family: {case.family}")


def _component_dataset(dataset: str) -> str:
    if dataset == "realistic":
        return "realistic_decode"
    return dataset


def _build_prepared_operator_input(**kwargs):
    from cannbench.core.prepared_input import build_prepared_operator_input

    return build_prepared_operator_input(**kwargs)


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name=OPERATOR_NAME,
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace=OPERATOR_NAME,
        runner_name=OPERATOR_NAME,
    ),
    get_dataset=get_dsa_decode_dataset,
    get_case=get_dsa_decode_case,
    materialize_inputs=materialize_dsa_decode_inputs,
    build_torch_callable=_build_unsupported_direct_callable,
    sort_order=30,
    build_workflow=build_dsa_decode_workflow,
    list_workflows=list_dsa_decode_workflows,
    component_operator_names=COMPONENT_OPERATORS,
    build_shape_trace=build_dsa_decode_shape_trace,
    list_shape_trace_cases=list_dsa_decode_shape_trace_cases,
)

__all__ = [
    "DsaDecodeCase",
    "DsaDecodeDataset",
    "DsaFusedWorkflow",
    "DsaWorkflowStep",
    "PLUGIN",
    "build_dsa_decode_shape_trace",
    "build_dsa_decode_workflow",
    "get_dsa_decode_case",
    "get_dsa_decode_dataset",
    "list_dsa_decode_shape_trace_cases",
    "list_dsa_decode_workflows",
]
