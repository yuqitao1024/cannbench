from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from cannbench.core.profile import ProfileKernelSelection
from cannbench.core.result import OperatorCase
from cannbench.operators.spec import OperatorSpec


@dataclass(frozen=True)
class TorchOperatorContext:
    backend: Any
    torch: Any
    request: Any
    case: Any
    device: Any
    dtype: Any
    implementation_module: Any | None = None


@dataclass(frozen=True)
class ProfileKernelSelectionContext:
    backend: str
    implementation: str | None
    implementation_version: str | None


@dataclass(frozen=True)
class OperatorPlugin:
    spec: OperatorSpec
    get_dataset: Callable[[str], Any]
    get_case: Callable[[str, str], Any]
    materialize_inputs: Callable[..., dict[str, object]]
    build_torch_callable: Callable[[TorchOperatorContext], Callable[[], Any]]
    sort_order: int = 1000
    build_cuda_library_callable: Callable[[TorchOperatorContext], Callable[[], Any]] | None = None
    build_vllm_ascend_callable: Callable[[TorchOperatorContext], Callable[[], Any]] | None = None
    build_simt_callable: Callable[[TorchOperatorContext], Callable[[], Any]] | None = None
    simt_module_name: Callable[[str | None], str | None] | None = None
    build_profile_kernel_selection: (
        Callable[[ProfileKernelSelectionContext], ProfileKernelSelection] | None
    ) = None
    build_workflow: Callable[..., Any] | None = None
    list_workflows: Callable[..., tuple[Any, ...]] | None = None
    component_operator_names: tuple[str, ...] = ()

    def build_result_case(self, case: Any) -> OperatorCase:
        return OperatorCase(
            case_id=case.case_id,
            family=case.family,
            source_kind=case.source_kind,
            source_project=case.source_project,
            source_model=case.source_model,
            source_file=case.source_file,
            source_op=case.source_op,
            payload=case.payload,
        )

    def profile_kernel_selection(
        self,
        *,
        backend: str,
        implementation: str | None,
        implementation_version: str | None,
    ) -> ProfileKernelSelection:
        if self.build_profile_kernel_selection is not None:
            return self.build_profile_kernel_selection(
                ProfileKernelSelectionContext(
                    backend=backend,
                    implementation=implementation,
                    implementation_version=implementation_version,
                )
            )
        return ProfileKernelSelection(kernel_name_patterns=(self.spec.name,))
