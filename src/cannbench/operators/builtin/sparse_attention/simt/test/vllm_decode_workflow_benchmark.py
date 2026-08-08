from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from cannbench.backends.pytorch_backend import AscendBackend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.datasets import get_operator_case
from cannbench.operators.builtin.dsa_decode import build_dsa_decode_workflow


DATASET = "realistic"
CASE_ID = "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048"


def summarize(samples_us: list[float]) -> dict[str, object]:
    return {
        "samples_us": samples_us,
        "median_us": statistics.median(samples_us),
        "min_us": min(samples_us),
        "max_us": max(samples_us),
    }


def _time_operator(torch, operator, *, warmups: int, iters: int) -> list[float]:
    for _ in range(warmups):
        operator()
    torch.npu.synchronize()

    samples = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        operator()
        end.record()
        end.synchronize()
        samples.append(float(start.elapsed_time(end)) * 1000.0)
    return samples


def run_benchmark(*, warmups: int, iters: int, seed: int) -> dict[str, object]:
    import torch
    import torch_npu  # noqa: F401

    workflow = build_dsa_decode_workflow(
        dataset=DATASET,
        case_id=CASE_ID,
        dtype="bfloat16",
        seed=seed,
    )
    backend = AscendBackend()
    components = []
    component_samples = []
    for step in workflow.steps:
        prepared = step.prepared
        request = OperatorBenchmarkRequest(
            backend="ascend",
            op=prepared.op,
            dtype=prepared.dtype,
            dataset=prepared.dataset,
            case_id=prepared.case.case_id,
            implementation="simt",
            implementation_version="vllm",
            seed=prepared.seed,
            input_bindings=prepared.input_bindings,
        )
        backend._before_run_operator(request)
        case = get_operator_case(request.op, request.dataset, request.case_id)
        operator = backend._operator_callable(
            torch,
            request,
            case,
            device=backend._device(torch),
            dtype=getattr(torch, request.dtype),
        )
        samples_us = _time_operator(
            torch, operator, warmups=warmups, iters=iters
        )
        component_samples.append(samples_us)
        components.append({"op": request.op, **summarize(samples_us)})

    workflow_samples = [sum(values) for values in zip(*component_samples)]
    return {
        "workflow": workflow.workflow,
        "case_id": workflow.case_id,
        "device": torch.npu.get_device_name(0),
        "warmups": warmups,
        "iters": iters,
        "seed": seed,
        "components": components,
        "workflow_timing": summarize(workflow_samples),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.iters <= 0:
        parser.error("warmups must be nonnegative and iters must be positive")

    payload = run_benchmark(
        warmups=args.warmups, iters=args.iters, seed=args.seed
    )
    rendered = json.dumps(payload, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
