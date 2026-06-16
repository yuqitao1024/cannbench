from dataclasses import dataclass

SUPPORTED_SOFTMAX_DIMS = {-1, -2, 0, 1}


@dataclass(frozen=True)
class BenchmarkMetrics:
    iterations: int
    warmup: int
    latency_ms_avg: float
    latency_ms_p50: float
    latency_ms_p95: float
    latency_ms_p99: float
    throughput_ops_per_sec: float


@dataclass(frozen=True)
class SoftmaxShape:
    dimensions: tuple[int, ...]
    dim: int
    case_id: str
    family: str
    source_kind: str
    source_project: str
    source_model: str
    source_file: str
    source_op: str

    def __post_init__(self) -> None:
        dimensions = tuple(int(value) for value in self.dimensions)
        if not dimensions:
            raise ValueError("dimensions must be non-empty")
        if any(value <= 0 for value in dimensions):
            raise ValueError("dimensions must be positive integers")
        if self.dim not in SUPPORTED_SOFTMAX_DIMS:
            raise ValueError("dim must address an axis in dimensions")
        object.__setattr__(self, "dimensions", dimensions)

    @property
    def rows(self) -> int:
        return self.dimensions[0]

    @property
    def cols(self) -> int:
        return self.dimensions[1] if len(self.dimensions) > 1 else 1

    def to_json_dict(self) -> dict[str, object]:
        return {
            "dimensions": list(self.dimensions),
            "dim": self.dim,
            "case_id": self.case_id,
            "family": self.family,
            "source_kind": self.source_kind,
            "source_project": self.source_project,
            "source_model": self.source_model,
            "source_file": self.source_file,
            "source_op": self.source_op,
        }


@dataclass(frozen=True)
class OperatorBenchmarkResult:
    backend: str
    device_name: str
    op: str
    dtype: str
    shape: SoftmaxShape
    metrics: BenchmarkMetrics

    def to_json_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "device_name": self.device_name,
            "op": self.op,
            "dtype": self.dtype,
            "shape": self.shape.to_json_dict(),
            "metrics": {
                "iterations": self.metrics.iterations,
                "warmup": self.metrics.warmup,
                "latency_ms_avg": self.metrics.latency_ms_avg,
                "latency_ms_p50": self.metrics.latency_ms_p50,
                "latency_ms_p95": self.metrics.latency_ms_p95,
                "latency_ms_p99": self.metrics.latency_ms_p99,
                "throughput_ops_per_sec": self.metrics.throughput_ops_per_sec,
            },
        }
