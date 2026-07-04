from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING

from cannbench.datasets.lightning_indexer import (
    LightningIndexerCase,
    get_lightning_indexer_case,
)
from cannbench.datasets.sparse_attention import (
    SparseAttentionCase,
    get_sparse_attention_case,
)

DSA_PHASES = {"decode", "prefill"}
DSA_WORKFLOWS = {"dsa_decode", "dsa_prefill"}

if TYPE_CHECKING:
    from cannbench.core.prepared_input import PreparedOperatorInput


@dataclass(frozen=True)
class DsaInferenceWorkflowCase:
    case_id: str
    workflow: str
    phase: str
    family: str

    def __post_init__(self) -> None:
        if self.workflow not in DSA_WORKFLOWS:
            raise ValueError(f"unsupported DSA workflow: {self.workflow}")
        _validate_phase(self.phase)
        if self.phase != _workflow_phase(self.workflow):
            raise ValueError("DSA workflow case phase mismatch")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")


@dataclass(frozen=True)
class DsaInferenceWorkflowDataset:
    name: str
    cases: tuple[DsaInferenceWorkflowCase, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cases", tuple(self.cases))


@dataclass(frozen=True)
class DsaWorkflowStep:
    contract: str
    op: str
    dataset: str
    case_id: str
    consumes: tuple[str, ...]
    produces: tuple[str, ...]
    prepared: PreparedOperatorInput


@dataclass(frozen=True)
class DsaInferenceWorkflow:
    workflow: str
    phase: str
    dataset: str
    case_id: str
    steps: tuple[DsaWorkflowStep, ...]


def build_dsa_inference_workflow(
    *,
    dataset: str,
    case_id: str,
    dtype: str,
    seed: int,
) -> DsaInferenceWorkflow:
    workflow_case = get_dsa_inference_workflow_case(dataset, case_id)
    sparse_case = get_sparse_attention_case(dataset, case_id)
    try:
        indexer_case = get_lightning_indexer_case(dataset, case_id)
    except ValueError as exc:
        raise ValueError(
            f"No matching lightning_indexer case for sparse_attention case: {case_id}"
        ) from exc

    _validate_component_pair(sparse_case, indexer_case)
    if sparse_case.phase != workflow_case.phase:
        raise ValueError(
            "DSA workflow manifest phase mismatch: "
            f"workflow is {workflow_case.phase}, sparse_attention is {sparse_case.phase}"
        )
    sparse_contract = (
        "sparse_mla_decode" if sparse_case.phase == "decode" else "sparse_mla_prefill"
    )
    return DsaInferenceWorkflow(
        workflow=workflow_case.workflow,
        phase=sparse_case.phase,
        dataset=dataset,
        case_id=case_id,
        steps=(
            DsaWorkflowStep(
                contract="dsa_index_select",
                op="lightning_indexer",
                dataset=dataset,
                case_id=case_id,
                consumes=(),
                produces=("indices",),
                prepared=_build_prepared_operator_input(
                    op="lightning_indexer",
                    dtype=dtype,
                    dataset=dataset,
                    case_id=case_id,
                    seed=seed,
                ),
            ),
            DsaWorkflowStep(
                contract=sparse_contract,
                op="sparse_attention",
                dataset=dataset,
                case_id=case_id,
                consumes=("indices",),
                produces=("out", "lse"),
                prepared=_build_prepared_operator_input(
                    op="sparse_attention",
                    dtype=dtype,
                    dataset=dataset,
                    case_id=case_id,
                    seed=seed,
                ),
            ),
        ),
    )


def list_dsa_inference_workflows(
    dataset: str,
    *,
    phase: str | None = None,
    dtype: str = "float16",
    seed: int = 0,
) -> tuple[DsaInferenceWorkflow, ...]:
    if phase is not None:
        _validate_phase(phase)

    workflows: list[DsaInferenceWorkflow] = []
    for workflow_case in get_dsa_inference_workflow_dataset(dataset).cases:
        if phase is not None and workflow_case.phase != phase:
            continue
        workflows.append(
            build_dsa_inference_workflow(
                dataset=dataset,
                case_id=workflow_case.case_id,
                dtype=dtype,
                seed=seed,
            )
        )
    return tuple(workflows)


@lru_cache(maxsize=None)
def get_dsa_inference_workflow_dataset(name: str) -> DsaInferenceWorkflowDataset:
    resource = files("cannbench.datasets.data.dsa_inference_workflow").joinpath(
        f"{name}.json"
    )
    if not resource.is_file():
        raise ValueError(f"Unknown DSA inference workflow dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(DsaInferenceWorkflowCase(**item) for item in payload["cases"])
    return DsaInferenceWorkflowDataset(name=payload["name"], cases=cases)


def get_dsa_inference_workflow_case(
    dataset_name: str, case_id: str
) -> DsaInferenceWorkflowCase:
    dataset = get_dsa_inference_workflow_dataset(dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown DSA inference workflow case: {case_id}")


def _validate_component_pair(
    sparse_case: SparseAttentionCase, indexer_case: LightningIndexerCase
) -> None:
    indexer_phase = _phase_from_indexer_case(indexer_case)
    if sparse_case.phase != indexer_phase:
        raise ValueError(
            "DSA component phase mismatch: "
            f"sparse_attention is {sparse_case.phase}, "
            f"lightning_indexer is {indexer_phase}"
        )
    if sparse_case.batch != indexer_case.batch:
        raise ValueError("DSA component batch mismatch")
    if sparse_case.query_tokens != indexer_case.query_tokens:
        raise ValueError("DSA component query_tokens mismatch")
    if sparse_case.context_tokens != indexer_case.context_tokens:
        raise ValueError("DSA component context_tokens mismatch")
    if sparse_case.selected_tokens != indexer_case.top_k:
        raise ValueError("DSA component top_k mismatch")


def _phase_from_indexer_case(case: LightningIndexerCase) -> str:
    if case.family.startswith("decode_") or "_decode_" in case.family:
        return "decode"
    if case.family.startswith("prefill_") or "_prefill_" in case.family:
        return "prefill"
    raise ValueError(f"Unable to infer DSA phase from indexer family: {case.family}")


def _validate_phase(phase: str) -> None:
    if phase not in DSA_PHASES:
        raise ValueError("phase must be decode or prefill")


def _workflow_phase(workflow: str) -> str:
    if workflow == "dsa_decode":
        return "decode"
    if workflow == "dsa_prefill":
        return "prefill"
    raise ValueError(f"unsupported DSA workflow: {workflow}")


def _build_prepared_operator_input(**kwargs) -> PreparedOperatorInput:
    from cannbench.core.prepared_input import build_prepared_operator_input

    return build_prepared_operator_input(**kwargs)
