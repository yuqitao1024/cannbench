from dataclasses import dataclass, field

from cannbench.core.prepared_input import OperatorInputBinding
from cannbench.datasets import get_operator_case

SUPPORTED_DTYPES = {"float32", "float16", "bfloat16"}
SUPPORTED_IMPLEMENTATIONS = {
    "cann_ops_library",
    "simt",
    "cuda_library",
    "vllm_ascend",
}


@dataclass(frozen=True)
class OperatorBenchmarkRequest:
    backend: str
    op: str
    dtype: str
    dataset: str
    case_id: str
    implementation: str | None = None
    seed: int = 0
    implementation_version: str | None = None
    aic_metrics: str = "BasicInfo"
    input_bindings: dict[str, OperatorInputBinding] = field(default_factory=dict)
    case_payload: dict[str, object] = field(init=False)
    dimensions: tuple[int, ...] | None = field(init=False, default=None)
    dim: int | None = field(init=False, default=None)
    family: str = field(init=False)
    source_kind: str = field(init=False)
    source_project: str = field(init=False)
    source_model: str = field(init=False)
    source_file: str = field(init=False)
    source_op: str = field(init=False)

    def __post_init__(self) -> None:
        if self.implementation is not None:
            implementation = self.implementation.strip()
            if implementation not in SUPPORTED_IMPLEMENTATIONS:
                raise ValueError(f"Unsupported implementation: {self.implementation}")
            object.__setattr__(self, "implementation", implementation)
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.implementation_version is not None:
            version = self.implementation_version.strip()
            if not version:
                raise ValueError("implementation_version must not be empty")
            object.__setattr__(self, "implementation_version", version)
        aic_metrics = self.aic_metrics.strip()
        if not aic_metrics:
            raise ValueError("aic_metrics must not be empty")
        if self.backend != "ascend" and aic_metrics != "BasicInfo":
            raise ValueError("aic_metrics is only supported for the ascend backend")
        object.__setattr__(self, "aic_metrics", aic_metrics)

        case = get_operator_case(self.op, self.dataset, self.case_id)
        object.__setattr__(self, "case_payload", case.payload)
        object.__setattr__(self, "family", case.family)
        object.__setattr__(self, "source_kind", case.source_kind)
        object.__setattr__(self, "source_project", case.source_project)
        object.__setattr__(self, "source_model", case.source_model)
        object.__setattr__(self, "source_file", case.source_file)
        object.__setattr__(self, "source_op", case.source_op)
        if "dimensions" in case.payload and "dim" in case.payload:
            object.__setattr__(self, "dimensions", tuple(case.payload["dimensions"]))
            object.__setattr__(self, "dim", int(case.payload["dim"]))
