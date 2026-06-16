from dataclasses import dataclass, field

from cannbench.datasets import get_operator_case
from cannbench.core.output import SUPPORTED_OUTPUT_FORMATS
from cannbench.core.result import SUPPORTED_SOFTMAX_DIMS

SUPPORTED_DTYPES = {"float32", "float16", "bfloat16"}


@dataclass(frozen=True)
class OperatorBenchmarkRequest:
    backend: str
    op: str
    dtype: str
    dataset: str
    case_id: str
    warmup: int
    iterations: int
    seed: int = 0
    output_formats: tuple[str, ...] = field(
        default_factory=lambda: ("json", "csv", "md")
    )
    case_payload: dict[str, object] = field(init=False)
    dimensions: tuple[int, ...] = field(init=False)
    dim: int = field(init=False)
    family: str = field(init=False)
    source_kind: str = field(init=False)
    source_project: str = field(init=False)
    source_model: str = field(init=False)
    source_file: str = field(init=False)
    source_op: str = field(init=False)

    def __post_init__(self) -> None:
        if self.dtype not in SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {self.dtype}")
        if self.dataset not in {"smoke", "realistic", "stress"}:
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
