from dataclasses import dataclass

from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.result import OperatorBenchmarkResult


@dataclass(frozen=True)
class OperatorBackend:
    name: str
    device_type: str

    def validate_request(self, request: OperatorBenchmarkRequest) -> None:
        if request.warmup < 0:
            raise ValueError("warmup must be >= 0")
        if request.iterations <= 0:
            raise ValueError("iterations must be > 0")

    def run_operator(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        raise NotImplementedError

    def run_softmax(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        return self.run_operator(request)
