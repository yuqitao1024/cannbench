from __future__ import annotations

from cannbench.backends.base import OperatorBackend
from cannbench.core.config import OperatorBenchmarkRequest, WorkflowBenchmarkRequest
from cannbench.core.operator_output import CapturedOperatorOutput
from cannbench.core.profile import LocalDeviceProfileResult
from cannbench.core.result import OperatorBenchmarkResult, WorkflowBenchmarkResult
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

    def _request_for_input_binding(self, request, binding):
        return OperatorBenchmarkRequest(
            backend=request.backend,
            op=binding.op,
            dtype=binding.dtype,
            dataset=binding.dataset,
            case_id=binding.case_id,
            implementation=request.implementation,
            seed=binding.seed,
            implementation_version=request.implementation_version,
            aic_metrics=request.aic_metrics,
        )

    def _request_for_workflow_step(self, request, step):
        prepared = step.prepared
        return OperatorBenchmarkRequest(
            backend=request.backend,
            op=prepared.op,
            dtype=prepared.dtype,
            dataset=prepared.dataset,
            case_id=prepared.case.case_id,
            implementation=request.implementation,
            seed=prepared.seed,
            implementation_version=request.implementation_version,
            aic_metrics=request.aic_metrics,
        )

    def _resolve_input_bindings(self, torch, request, *, device):
        bound_inputs = {}
        for name, binding in request.input_bindings.items():
            producer_request = self._request_for_input_binding(request, binding)
            self._before_run_operator(producer_request)
            producer_case = get_operator_case(
                producer_request.op,
                producer_request.dataset,
                producer_request.case_id,
            )
            producer = self._operator_callable(
                torch,
                producer_request,
                producer_case,
                device=device,
                dtype=getattr(torch, producer_request.dtype),
            )
            output = producer()
            if isinstance(output, (tuple, list)):
                try:
                    output = output[binding.output_index]
                except IndexError as exc:
                    raise RuntimeError(
                        f"bound input {name!r} requested output index "
                        f"{binding.output_index}, but {binding.op} returned "
                        f"{len(output)} outputs"
                    ) from exc
            elif binding.output_index != 0:
                raise RuntimeError(
                    f"bound input {name!r} requested output index "
                    f"{binding.output_index}, but {binding.op} returned one output"
                )
            bound_inputs[name] = output
        return bound_inputs

    def _operator_context(
        self,
        torch,
        request,
        case,
        *,
        device,
        dtype,
        implementation_module=None,
        bound_inputs=None,
    ):
        return TorchOperatorContext(
            backend=self,
            torch=torch,
            request=request,
            case=case,
            device=device,
            dtype=dtype,
            implementation_module=implementation_module,
            bound_inputs=(
                self._resolve_input_bindings(torch, request, device=device)
                if bound_inputs is None
                else dict(bound_inputs)
            ),
        )

    def _operator_callable(
        self, torch, request, case, *, device, dtype, bound_inputs=None
    ):
        plugin = get_operator_plugin(request.op)
        return plugin.build_torch_callable(
            self._operator_context(
                torch,
                request,
                case,
                device=device,
                dtype=dtype,
                bound_inputs=bound_inputs,
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

    def run_workflow(
        self, request: WorkflowBenchmarkRequest
    ) -> WorkflowBenchmarkResult:
        torch = self._torch_module()
        if not self._is_available(torch):
            raise RuntimeError(self._availability_error())

        device = self._device(torch)
        outputs: dict[str, object] = {}
        step_results: list[OperatorBenchmarkResult] = []
        for step in request.prepared.steps:
            step_request = self._request_for_workflow_step(request, step)
            self.validate_request(step_request)
            self._before_run_operator(step_request)
            spec = get_operator_spec(step_request.op)
            if step_request.dtype not in spec.supported_dtypes:
                raise RuntimeError(
                    f"Unsupported dtype for {step_request.op}: {step_request.dtype}"
                )
            case = get_operator_case(
                step_request.op,
                step_request.dataset,
                step_request.case_id,
            )
            bound_inputs = {name: outputs[name] for name in step.consumes}
            operator = self._operator_callable(
                torch,
                step_request,
                case,
                device=device,
                dtype=getattr(torch, step_request.dtype),
                bound_inputs=bound_inputs,
            )
            step_output = operator()
            if isinstance(step_output, (tuple, list)):
                if len(step_output) != len(step.produces):
                    raise RuntimeError(
                        f"workflow step {step.contract} returned {len(step_output)} "
                        f"outputs but declares {len(step.produces)}"
                    )
                produced_values = tuple(step_output)
            elif len(step.produces) == 1:
                produced_values = (step_output,)
            else:
                raise RuntimeError(
                    f"workflow step {step.contract} returned one output but declares "
                    f"{len(step.produces)}"
                )
            outputs.update(zip(step.produces, produced_values, strict=True))
            step_results.append(
                OperatorBenchmarkResult(
                    backend=self.name,
                    device_name=self._device_name(torch, device),
                    op=step_request.op,
                    dtype=step_request.dtype,
                    case=get_operator_plugin(step_request.op).build_result_case(case),
                )
            )

        self._synchronize(torch)
        return WorkflowBenchmarkResult(
            backend=self.name,
            device_name=self._device_name(torch, device),
            workflow=request.prepared.workflow,
            phase=request.prepared.phase,
            dataset=request.prepared.dataset,
            case_id=request.prepared.case_id,
            steps=tuple(step_results),
        )

    def profile_operator_device_time(
        self, request: OperatorBenchmarkRequest
    ) -> LocalDeviceProfileResult:
        raise NotImplementedError
