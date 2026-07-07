from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import TYPE_CHECKING, Callable

from cannbench.operators.builtin.lightning_indexer.cases import (
    LightningIndexerCase,
    get_lightning_indexer_case,
)
from cannbench.operators.builtin.sparse_attention.cases import (
    SparseAttentionCase,
    get_sparse_attention_case,
)
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec

DSA_PHASES = {"decode", "prefill"}
DSA_FUSED_OPERATORS = ("dsa_decode", "dsa_prefill")

if TYPE_CHECKING:
    from cannbench.core.prepared_input import PreparedOperatorInput


@dataclass(frozen=True)
class DsaFusedOperatorCase:
    case_id: str
    workflow: str
    phase: str
    family: str
    source_kind: str = "fused_workflow"
    source_project: str = "cannbench"
    source_model: str = "DeepSeek-A5-compatible"
    source_file: str = "serving_buckets/deepseek_a5_dsa.json"
    source_op: str = ""

    def __post_init__(self) -> None:
        if self.workflow not in DSA_FUSED_OPERATORS:
            raise ValueError(f"unsupported DSA fused operator: {self.workflow}")
        _validate_phase(self.phase)
        if self.phase != _operator_phase(self.workflow):
            raise ValueError("DSA fused operator case phase mismatch")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if not self.source_op:
            object.__setattr__(self, "source_op", self.workflow)

    @property
    def payload(self) -> dict[str, object]:
        return {
            "workflow": self.workflow,
            "phase": self.phase,
            "component_ops": ("lightning_indexer", "sparse_attention"),
        }


@dataclass(frozen=True)
class DsaFusedOperatorDataset:
    name: str
    cases: tuple[DsaFusedOperatorCase, ...]

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
class DsaFusedWorkflow:
    workflow: str
    phase: str
    dataset: str
    case_id: str
    steps: tuple[DsaWorkflowStep, ...]


def get_dsa_fused_dataset(
    package: str,
    *,
    operator_name: str,
    name: str,
) -> DsaFusedOperatorDataset:
    return _get_dsa_fused_dataset(package, operator_name, name)


@lru_cache(maxsize=None)
def _get_dsa_fused_dataset(
    package: str,
    operator_name: str,
    name: str,
) -> DsaFusedOperatorDataset:
    resource = files(package).joinpath("data", f"{name}.json")
    if not resource.is_file():
        raise ValueError(f"Unknown {operator_name} dataset: {name}")

    payload = json.loads(resource.read_text())
    cases = tuple(DsaFusedOperatorCase(**item) for item in payload["cases"])
    phase = _operator_phase(operator_name)
    for case in cases:
        if case.workflow != operator_name or case.phase != phase:
            raise ValueError(
                f"{operator_name} dataset {name} contains non-{phase} case {case.case_id}"
            )
    return DsaFusedOperatorDataset(name=payload["name"], cases=cases)


def get_dsa_fused_case(
    package: str,
    *,
    operator_name: str,
    dataset_name: str,
    case_id: str,
) -> DsaFusedOperatorCase:
    dataset = get_dsa_fused_dataset(package, operator_name=operator_name, name=dataset_name)
    for case in dataset.cases:
        if case.case_id == case_id:
            return case
    raise ValueError(f"Unknown DSA {_operator_phase(operator_name)} case: {case_id}")


def build_dsa_fused_workflow(
    package: str,
    *,
    operator_name: str,
    dataset: str,
    case_id: str,
    dtype: str,
    seed: int,
) -> DsaFusedWorkflow:
    workflow_case = get_dsa_fused_case(
        package,
        operator_name=operator_name,
        dataset_name=dataset,
        case_id=case_id,
    )
    component_dataset = _component_dataset(dataset, workflow_case.phase)
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
            "DSA workflow manifest phase mismatch: "
            f"workflow is {workflow_case.phase}, sparse_attention is {sparse_case.phase}"
        )
    sparse_contract = (
        "sparse_mla_decode" if sparse_case.phase == "decode" else "sparse_mla_prefill"
    )
    return DsaFusedWorkflow(
        workflow=workflow_case.workflow,
        phase=sparse_case.phase,
        dataset=dataset,
        case_id=case_id,
        steps=(
            DsaWorkflowStep(
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
            DsaWorkflowStep(
                contract=sparse_contract,
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


def list_dsa_fused_workflows(
    package: str,
    *,
    operator_name: str,
    dataset: str,
    dtype: str = "float16",
    seed: int = 0,
) -> tuple[DsaFusedWorkflow, ...]:
    workflows: list[DsaFusedWorkflow] = []
    for workflow_case in get_dsa_fused_dataset(
        package, operator_name=operator_name, name=dataset
    ).cases:
        workflows.append(
            build_dsa_fused_workflow(
                package,
                operator_name=operator_name,
                dataset=dataset,
                case_id=workflow_case.case_id,
                dtype=dtype,
                seed=seed,
            )
        )
    return tuple(workflows)


def materialize_dsa_fused_inputs(
    case: DsaFusedOperatorCase, *, dtype: str, seed: int
) -> dict[str, object]:
    return {
        "workflow": case.workflow,
        "phase": case.phase,
        "case_id": case.case_id,
        "dtype": dtype,
        "seed": seed,
        "component_ops": ("lightning_indexer", "sparse_attention"),
    }


def create_dsa_fused_plugin(
    *,
    operator_name: str,
    get_dataset: Callable[[str], DsaFusedOperatorDataset],
    get_case: Callable[[str, str], DsaFusedOperatorCase],
    materialize_inputs: Callable[..., dict[str, object]],
    build_workflow: Callable[..., DsaFusedWorkflow],
    list_workflows: Callable[..., tuple[DsaFusedWorkflow, ...]],
    sort_order: int,
) -> OperatorPlugin:
    return OperatorPlugin(
        spec=OperatorSpec(
            name=operator_name,
            supported_dtypes=("float32", "float16", "bfloat16"),
            dataset_namespace=operator_name,
            runner_name=operator_name,
        ),
        get_dataset=get_dataset,
        get_case=get_case,
        materialize_inputs=materialize_inputs,
        build_torch_callable=_build_unsupported_direct_callable,
        sort_order=sort_order,
        build_workflow=build_workflow,
        list_workflows=list_workflows,
        component_operator_names=("lightning_indexer", "sparse_attention"),
    )


def _build_unsupported_direct_callable(ctx):
    def _raise_direct_run_error():
        raise RuntimeError(
            f"{ctx.case.payload.get('workflow', 'DSA fused operator')} direct callable "
            "is not implemented; run it through the bench workflow expansion path"
        )

    return _raise_direct_run_error


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


def _operator_phase(operator_name: str) -> str:
    if operator_name == "dsa_decode":
        return "decode"
    if operator_name == "dsa_prefill":
        return "prefill"
    raise ValueError(f"unsupported DSA fused operator: {operator_name}")


def _component_dataset(dataset: str, phase: str) -> str:
    if dataset == "realistic":
        return f"realistic_{phase}"
    return dataset


def _build_prepared_operator_input(**kwargs) -> PreparedOperatorInput:
    from cannbench.core.prepared_input import build_prepared_operator_input

    return build_prepared_operator_input(**kwargs)
