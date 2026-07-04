from dataclasses import dataclass, field

from cannbench.datasets import get_operator_case
from cannbench.core.output import SUPPORTED_OUTPUT_FORMATS
from cannbench.core.result import SUPPORTED_SOFTMAX_DIMS

SUPPORTED_DTYPES = {"float32", "float16", "bfloat16"}
SUPPORTED_IMPLEMENTATIONS = {
    "cann_ops_library",
    "simt",
    "cuda_library",
    "vllm_ascend",
}
SUPPORTED_DATASETS = {
    "smoke",
    "realistic",
    "realistic_decode",
    "realistic_prefill",
    "stress",
}


@dataclass(frozen=True)
class OperatorBenchmarkRequest:
    backend: str
    op: str
    dtype: str
    dataset: str
    case_id: str
    warmup: int
    iterations: int
    implementation: str | None = None
    seed: int = 0
    use_simt_op: bool = False
    deploy_simt_op: bool = False
    implementation_version: str | None = None
    output_formats: tuple[str, ...] = field(
        default_factory=lambda: ("json", "csv", "md")
    )
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
        if self.dataset not in SUPPORTED_DATASETS:
            raise ValueError(f"Unknown operator dataset: {self.dataset}")
        if not self.case_id.strip():
            raise ValueError("case_id must not be empty")
        if self.output_formats and any(
            format_name not in SUPPORTED_OUTPUT_FORMATS
            for format_name in self.output_formats
        ):
            unsupported = sorted(
                {
                    format_name
                    for format_name in self.output_formats
                    if format_name not in SUPPORTED_OUTPUT_FORMATS
                }
            )
            raise ValueError(f"unsupported output format: {', '.join(unsupported)}")
        if self.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if self.iterations <= 0:
            raise ValueError("iterations must be > 0")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        if self.implementation_version is not None:
            version = self.implementation_version.strip()
            if not version:
                raise ValueError("implementation_version must not be empty")
            object.__setattr__(self, "implementation_version", version)

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
