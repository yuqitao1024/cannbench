from __future__ import annotations

from cannbench.operators.builtin._dsa_fused import (
    DsaFusedWorkflow,
    DsaWorkflowStep,
    build_dsa_fused_workflow,
    create_dsa_fused_plugin,
    list_dsa_fused_workflows,
)

from .cases import (
    OPERATOR_NAME,
    DsaPrefillCase,
    DsaPrefillDataset,
    get_dsa_prefill_case,
    get_dsa_prefill_dataset,
)
from .materialize import materialize_dsa_prefill_inputs


def build_dsa_prefill_workflow(
    *,
    dataset: str,
    case_id: str,
    dtype: str,
    seed: int,
) -> DsaFusedWorkflow:
    return build_dsa_fused_workflow(
        __package__,
        operator_name=OPERATOR_NAME,
        dataset=dataset,
        case_id=case_id,
        dtype=dtype,
        seed=seed,
    )


def list_dsa_prefill_workflows(
    dataset: str,
    *,
    dtype: str = "float16",
    seed: int = 0,
) -> tuple[DsaFusedWorkflow, ...]:
    return list_dsa_fused_workflows(
        __package__,
        operator_name=OPERATOR_NAME,
        dataset=dataset,
        dtype=dtype,
        seed=seed,
    )


PLUGIN = create_dsa_fused_plugin(
    operator_name=OPERATOR_NAME,
    get_dataset=get_dsa_prefill_dataset,
    get_case=get_dsa_prefill_case,
    materialize_inputs=materialize_dsa_prefill_inputs,
    build_workflow=build_dsa_prefill_workflow,
    list_workflows=list_dsa_prefill_workflows,
    sort_order=31,
)

__all__ = [
    "DsaFusedWorkflow",
    "DsaPrefillCase",
    "DsaPrefillDataset",
    "DsaWorkflowStep",
    "PLUGIN",
    "build_dsa_prefill_workflow",
    "get_dsa_prefill_case",
    "get_dsa_prefill_dataset",
    "list_dsa_prefill_workflows",
]
