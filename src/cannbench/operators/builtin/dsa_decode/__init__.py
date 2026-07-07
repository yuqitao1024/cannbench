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
    DsaDecodeCase,
    DsaDecodeDataset,
    get_dsa_decode_case,
    get_dsa_decode_dataset,
)
from .materialize import materialize_dsa_decode_inputs


def build_dsa_decode_workflow(
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


def list_dsa_decode_workflows(
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
    get_dataset=get_dsa_decode_dataset,
    get_case=get_dsa_decode_case,
    materialize_inputs=materialize_dsa_decode_inputs,
    build_workflow=build_dsa_decode_workflow,
    list_workflows=list_dsa_decode_workflows,
    sort_order=30,
)

__all__ = [
    "DsaDecodeCase",
    "DsaDecodeDataset",
    "DsaFusedWorkflow",
    "DsaWorkflowStep",
    "PLUGIN",
    "build_dsa_decode_workflow",
    "get_dsa_decode_case",
    "get_dsa_decode_dataset",
    "list_dsa_decode_workflows",
]
