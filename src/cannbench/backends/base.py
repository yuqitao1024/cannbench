from dataclasses import dataclass

from cannbench.core.config import OperatorBenchmarkRequest, WorkflowBenchmarkRequest
from cannbench.core.operator_output import CapturedOperatorOutput
from cannbench.core.profile import LocalDeviceProfileResult
from cannbench.core.result import OperatorBenchmarkResult, WorkflowBenchmarkResult


@dataclass(frozen=True)
class OperatorBackend:
    name: str
    device_type: str

    def validate_request(self, request: OperatorBenchmarkRequest) -> None:
        del request

    def run_operator(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        raise NotImplementedError

    def run_workflow(
        self, request: WorkflowBenchmarkRequest
    ) -> WorkflowBenchmarkResult:
        raise NotImplementedError

    def capture_operator_output(
        self, request: OperatorBenchmarkRequest
    ) -> CapturedOperatorOutput:
        raise NotImplementedError

    def profile_operator_device_time(
        self, request: OperatorBenchmarkRequest
    ) -> LocalDeviceProfileResult:
        raise NotImplementedError
