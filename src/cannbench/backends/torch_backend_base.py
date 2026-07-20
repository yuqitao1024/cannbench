from __future__ import annotations

from cannbench.backends.base import OperatorBackend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.operator_output import CapturedOperatorOutput
from cannbench.core.profile import LocalDeviceProfileResult
from cannbench.core.result import OperatorBenchmarkResult
from cannbench.datasets import get_operator_case
from cannbench.operators import (
    TorchOperatorContext,
    get_operator_plugin,
    get_operator_spec,
)


class TorchOperatorBackend(OperatorBackend):
    def __init__(self, *, name: str, device_type: str) -> None:
        super().__init__(name=name, device_type=device_type)

    def _torch_module(self):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise RuntimeError(f"PyTorch is required for the {self.name} backend") from exc
        return torch

    def _device_module(self, torch):
        return getattr(torch, self.device_type)

    def _is_available(self, torch) -> bool:
        return self._device_module(torch).is_available()

    def _synchronize(self, torch) -> None:
        self._device_module(torch).synchronize()

    def _device_name(self, torch, device) -> str:
        return self._device_module(torch).get_device_name(device)

    def _device(self, torch):
        return torch.device(self.device_type)

    def _tensor(self, torch, values, *, device, dtype):
        return torch.tensor(values, device=device, dtype=dtype)

    def _operator_callable(self, torch, request, case, *, device, dtype):
        plugin = get_operator_plugin(request.op)
        return plugin.build_torch_callable(
            TorchOperatorContext(
                backend=self,
                torch=torch,
                request=request,
                case=case,
                device=device,
                dtype=dtype,
            )
        )

    def _captured_output_from_tensor(
        self,
        *,
        torch,
        request: OperatorBenchmarkRequest,
        device,
        output,
    ) -> CapturedOperatorOutput:
        if hasattr(output, "detach"):
            output = output.detach()
        if hasattr(output, "cpu"):
            output = output.cpu()
        if hasattr(output, "to"):
            output = output.to(dtype=torch.float32)
        shape = tuple(int(value) for value in getattr(output, "shape", ()))
        if hasattr(output, "flatten"):
            flat = output.flatten()
        else:
            flat = output.reshape(-1)
        values = tuple(float(value) for value in flat.tolist())
        return CapturedOperatorOutput(
            backend=self.name,
            device_name=self._device_name(torch, device),
            op=request.op,
            dtype=request.dtype,
            dataset=request.dataset,
            case_id=request.case_id,
            seed=request.seed,
            shape=shape,
            values=values,
        )

    def capture_operator_output(
        self, request: OperatorBenchmarkRequest
    ) -> CapturedOperatorOutput:
        self.validate_request(request)
        self._before_run_operator(request)
        spec = get_operator_spec(request.op)
        torch = self._torch_module()
        if not self._is_available(torch):
            raise RuntimeError(self._availability_error())
        if request.dtype not in spec.supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for {request.op}: {request.dtype}")

        device = self._device(torch)
        dtype = getattr(torch, request.dtype)
        case = get_operator_case(request.op, request.dataset, request.case_id)
        output = self._operator_callable(
            torch,
            request,
            case,
            device=device,
            dtype=dtype,
        )()
        self._synchronize(torch)
        return self._captured_output_from_tensor(
            torch=torch,
            request=request,
            device=device,
            output=output,
        )

    def _availability_error(self) -> str:
        return f"{self.device_type.upper()} is required for the {self.name} backend"

    def _before_run_operator(self, request: OperatorBenchmarkRequest) -> None:
        del request

    def run_operator(self, request: OperatorBenchmarkRequest) -> OperatorBenchmarkResult:
        self.validate_request(request)
        self._before_run_operator(request)
        spec = get_operator_spec(request.op)
        torch = self._torch_module()
        if not self._is_available(torch):
            raise RuntimeError(self._availability_error())

        device = self._device(torch)
        dtype = getattr(torch, request.dtype)
        if request.dtype not in spec.supported_dtypes:
            raise RuntimeError(f"Unsupported dtype for {request.op}: {request.dtype}")
        case = get_operator_case(request.op, request.dataset, request.case_id)
        operator = self._operator_callable(
            torch,
            request,
            case,
            device=device,
            dtype=dtype,
        )

        output = operator()
        self._synchronize(torch)
        del output
        return OperatorBenchmarkResult(
            backend=self.name,
            device_name=self._device_name(torch, device),
            op=request.op,
            dtype=request.dtype,
            case=get_operator_plugin(request.op).build_result_case(case),
        )

    def profile_operator_device_time(
        self, request: OperatorBenchmarkRequest
    ) -> LocalDeviceProfileResult:
        raise NotImplementedError
