import argparse
from pathlib import Path

from cannbench.backends import get_backend
from cannbench.core.batch import expand_prepared_input_plans
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.cuda_events import write_cuda_event_profile_csv
from cannbench.core.layout import build_run_layout
from cannbench.core.publish import publish_run_artifacts
from cannbench.serve import serve_cannbench
from cannbench.core.operator_output import (
    compare_operator_outputs,
    read_operator_output,
    write_operator_output,
    write_output_comparison,
)
from cannbench.core.profile import read_device_profile, write_device_profile_summary
from cannbench.core.prepared_input import (
    build_prepared_operator_input,
    read_prepared_operator_input,
    write_prepared_operator_input,
)
from cannbench.core.remote import collect_remote_artifacts
from cannbench.core.report import write_local_report
from cannbench.core.output import write_benchmark_outputs
from cannbench.operators import list_operator_names


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("warmup must be >= 0")
    return parsed


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("iterations must be > 0")
    return parsed


def _non_negative_float(value: str) -> float:
    parsed = float(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def _benchmark_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=["nvidia", "ascend"], required=True)
    parser.add_argument("--implementation", choices=["cann_ops_library", "simt"])
    parser.add_argument("--op", choices=list_operator_names())
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--dataset", choices=["smoke", "realistic", "stress"])
    parser.add_argument("--case-id")
    parser.add_argument("--prepared-input", type=Path)
    parser.add_argument("--prepared-dir", type=Path)
    parser.add_argument("--deploy-custom-op", action="store_true", default=False)
    parser.add_argument("--warmup", type=_non_negative_int, default=10)
    parser.add_argument("--iterations", type=_positive_int, default=1)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--run-name")


def _resolve_deploy_custom_op(backend: str, implementation: str | None, deploy_custom_op: bool) -> bool:
    if backend != "ascend":
        return deploy_custom_op
    if implementation == "simt":
        return True
    if implementation == "cann_ops_library":
        return False
    return deploy_custom_op


def _prepared_input_path(output_dir: Path, op: str, dataset: str, case_id: str, dtype: str, seed: int) -> Path:
    layout = build_run_layout(output_dir, "_prepared")
    return layout.root / op / dataset / f"{case_id}-{dtype}-seed{seed}.json"


def _build_request_from_args(args: argparse.Namespace) -> OperatorBenchmarkRequest:
    if args.prepared_input is not None:
        prepared = read_prepared_operator_input(args.prepared_input)
        return OperatorBenchmarkRequest(
            backend=args.backend,
            op=prepared.op,
            dtype=prepared.dtype,
            dataset=prepared.dataset,
            case_id=prepared.case.case_id,
            warmup=args.warmup,
            iterations=args.iterations,
            seed=prepared.seed,
            deploy_custom_op=_resolve_deploy_custom_op(
                args.backend, getattr(args, "implementation", None), args.deploy_custom_op
            ),
        )
    if not args.op:
        raise ValueError("--op is required")
    if not args.dataset or not args.case_id:
        raise ValueError("--dataset and --case-id are required for single-case execution")
    return OperatorBenchmarkRequest(
        backend=args.backend,
        op=args.op,
        dtype=args.dtype,
        dataset=args.dataset,
        case_id=args.case_id,
        warmup=args.warmup,
        iterations=args.iterations,
        seed=getattr(args, "seed", 0),
        deploy_custom_op=_resolve_deploy_custom_op(
            args.backend, getattr(args, "implementation", None), args.deploy_custom_op
        ),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cannbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    bench = subparsers.add_parser("bench")
    _benchmark_args(bench)

    operator = subparsers.add_parser("operator")
    _benchmark_args(operator)

    prepare = subparsers.add_parser("prepare")
    prepare.add_argument("--op", choices=list_operator_names(), required=True)
    prepare.add_argument("--dtype", default="float16")
    prepare.add_argument("--dataset", choices=["smoke", "realistic", "stress"], default="realistic")
    prepare.add_argument("--case-id", required=True)
    prepare.add_argument("--seed", type=_non_negative_int, default=0)
    prepare.add_argument("--output", type=Path, required=True)

    capture = subparsers.add_parser("capture-output")
    capture.add_argument("--backend", choices=["nvidia", "ascend"], required=True)
    capture.add_argument("--op", choices=list_operator_names())
    capture.add_argument("--dtype", default="float16")
    capture.add_argument("--dataset", choices=["smoke", "realistic", "stress"], default="realistic")
    capture.add_argument("--case-id")
    capture.add_argument("--prepared-input", type=Path)
    capture.add_argument("--seed", type=_non_negative_int, default=0)
    capture.add_argument("--deploy-custom-op", action="store_true", default=False)
    capture.add_argument("--output", type=Path, required=True)

    compare = subparsers.add_parser("compare-output")
    compare.add_argument("--left", type=Path, required=True)
    compare.add_argument("--right", type=Path, required=True)
    compare.add_argument("--rtol", type=_non_negative_float, default=0.001)
    compare.add_argument("--atol", type=_non_negative_float, default=0.001)
    compare.add_argument("--output", type=Path, required=True)

    collect = subparsers.add_parser("collect")
    collect.add_argument("--endpoint", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--run-id")
    collect.add_argument("--run-name")
    collect.add_argument("--capture-output", action="store_true", default=False)
    collect.add_argument("--profile-device-time", action="store_true", default=False)
    collect.add_argument("--summarize-profile", action="store_true", default=False)
    collect.add_argument("--implementation", choices=["cann_ops_library", "simt"])
    collect.add_argument("--op", choices=list_operator_names())
    collect.add_argument("--dtype", default="float16")
    collect.add_argument("--dataset", choices=["smoke", "realistic", "stress"])
    collect.add_argument("--case-id")
    collect.add_argument("--seed", type=_non_negative_int, default=0)
    collect.add_argument("--prepared-input", type=Path)
    collect.add_argument("--prepared-dir", type=Path)
    collect.add_argument("--warmup", type=_non_negative_int, default=10)
    collect.add_argument("--iterations", type=_positive_int, default=1)
    collect.add_argument("--deploy-custom-op", action="store_true", default=False)

    publish = subparsers.add_parser("publish")
    publish.add_argument("--source", type=Path, required=True)
    publish.add_argument("--dest", type=Path, required=True)

    serve = subparsers.add_parser("serve")
    serve.add_argument("--frontend-dir", type=Path, default=Path("web/dist"))
    serve.add_argument("--published-dir", type=Path, default=Path("published"))
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--enable-gpu-upload", action="store_true", default=False)

    report = subparsers.add_parser("report")
    report.add_argument("--nvidia", type=Path, required=True)
    report.add_argument("--ascend", type=Path, required=True)
    report.add_argument("--accuracy", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    summarize_profile = subparsers.add_parser("summarize-profile")
    summarize_profile.add_argument("--backend", choices=["nvidia", "ascend"], required=True)
    summarize_profile.add_argument("--profile-dir", type=Path, required=True)
    summarize_profile.add_argument("--output", type=Path, required=True)

    cuda_event = subparsers.add_parser("cuda-event-profile")
    cuda_event.add_argument("--backend", choices=["nvidia"], required=True)
    cuda_event.add_argument("--prepared-input", type=Path, required=True)
    cuda_event.add_argument("--warmup", type=_non_negative_int, default=10)
    cuda_event.add_argument("--iterations", type=_positive_int, default=1)
    cuda_event.add_argument("--profile-dir", type=Path, required=True)
    cuda_event.add_argument("--output-dir", type=Path, required=True)
    cuda_event.add_argument("--run-name", default="benchmark")

    return parser


def _is_batch_mode(args: argparse.Namespace) -> bool:
    if getattr(args, "prepared_dir", None) is not None:
        return True
    return getattr(args, "prepared_input", None) is None and getattr(args, "case_id", None) is None


def _validate_benchmark_selection(
    args: argparse.Namespace,
    *,
    allow_batch: bool,
) -> None:
    prepared_input = getattr(args, "prepared_input", None)
    prepared_dir = getattr(args, "prepared_dir", None)
    case_id = getattr(args, "case_id", None)
    op = getattr(args, "op", None)

    if prepared_input is not None and prepared_dir is not None:
        raise ValueError("--prepared-input and --prepared-dir are mutually exclusive")
    if (prepared_input is not None or prepared_dir is not None) and case_id is not None:
        raise ValueError("--case-id cannot be used with --prepared-input or --prepared-dir")
    if prepared_dir is not None and not op:
        raise ValueError("--op is required with --prepared-dir")
    if prepared_input is None and not op:
        raise ValueError("--op is required unless --prepared-input is set")
    if not allow_batch and prepared_dir is not None:
        raise ValueError("--prepared-dir is only supported for bench and collect")
    if allow_batch and _is_batch_mode(args) and not getattr(args, "run_name", None):
        raise ValueError("--run-name is required for batch execution")


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"bench", "operator"}:
        try:
            _validate_benchmark_selection(args, allow_batch=args.command == "bench")
            if args.command == "bench" and _is_batch_mode(args):
                expand_prepared_input_plans(
                    op=args.op,
                    dtype=args.dtype,
                    dataset=args.dataset,
                    case_id=args.case_id,
                    prepared_input=args.prepared_input,
                    prepared_dir=args.prepared_dir,
                )
                parser.error("batch bench execution is not implemented yet")
            request = _build_request_from_args(args)
            backend = get_backend(args.backend)
            result = backend.run_operator(request)
            write_benchmark_outputs(
                args.output_dir, args.run_name or "operator-benchmark", result, request.output_formats
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "prepare":
        prepared = build_prepared_operator_input(
            op=args.op,
            dtype=args.dtype,
            dataset=args.dataset,
            case_id=args.case_id,
            seed=args.seed,
        )
        write_prepared_operator_input(args.output, prepared)
    elif args.command == "capture-output":
        try:
            if args.prepared_input is not None:
                prepared = read_prepared_operator_input(args.prepared_input)
                request = OperatorBenchmarkRequest(
                    backend=args.backend,
                    op=prepared.op,
                    dtype=prepared.dtype,
                    dataset=prepared.dataset,
                    case_id=prepared.case.case_id,
                    warmup=0,
                    iterations=1,
                    seed=prepared.seed,
                    deploy_custom_op=args.deploy_custom_op,
                )
            else:
                if not args.op or not args.case_id:
                    parser.error("--op and --case-id are required unless --prepared-input is set")
                request = OperatorBenchmarkRequest(
                    backend=args.backend,
                    op=args.op,
                    dtype=args.dtype,
                    dataset=args.dataset,
                    case_id=args.case_id,
                    warmup=0,
                    iterations=1,
                    seed=args.seed,
                    deploy_custom_op=args.deploy_custom_op,
                )
            backend = get_backend(args.backend)
            output = backend.capture_operator_output(request)
            write_operator_output(args.output, output)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "compare-output":
        try:
            left = read_operator_output(args.left)
            right = read_operator_output(args.right)
            result = compare_operator_outputs(
                left,
                right,
                rtol=args.rtol,
                atol=args.atol,
            )
            write_output_comparison(args.output, result)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "collect":
        try:
            _validate_benchmark_selection(args, allow_batch=True)
            if _is_batch_mode(args):
                expand_prepared_input_plans(
                    op=args.op,
                    dtype=args.dtype,
                    dataset=args.dataset,
                    case_id=args.case_id,
                    seed=args.seed,
                    prepared_input=args.prepared_input,
                    prepared_dir=args.prepared_dir,
                )
                parser.error("batch collect execution is not implemented yet")
            prepared_input = args.prepared_input
            if prepared_input is None:
                if not args.dataset or not args.case_id:
                    parser.error("--dataset and --case-id are required for single-case execution")
                prepared = build_prepared_operator_input(
                    op=args.op,
                    dtype=args.dtype,
                    dataset=args.dataset,
                    case_id=args.case_id,
                    seed=args.seed,
                )
                prepared_input = _prepared_input_path(
                    args.output_dir, args.op, args.dataset, args.case_id, args.dtype, args.seed
                )
                write_prepared_operator_input(prepared_input, prepared)
            collect_remote_artifacts(
                endpoint_path=args.endpoint,
                prepared_input=prepared_input,
                output_dir=args.output_dir,
                run_id=args.run_id,
                capture_output=args.capture_output,
                profile_device_time=args.profile_device_time,
                summarize_profile=args.summarize_profile,
                warmup=args.warmup,
                iterations=args.iterations,
                deploy_custom_op=_resolve_deploy_custom_op(
                    "ascend", args.implementation, args.deploy_custom_op
                ),
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "report":
        try:
            write_local_report(
                output_path=args.output,
                nvidia_dir=args.nvidia,
                ascend_dir=args.ascend,
                accuracy_path=args.accuracy,
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "publish":
        try:
            publish_run_artifacts(args.source, args.dest)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "serve":
        try:
            serve_cannbench(
                frontend_dir=args.frontend_dir,
                published_dir=args.published_dir,
                host=args.host,
                port=args.port,
                enable_gpu_upload=args.enable_gpu_upload,
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "summarize-profile":
        try:
            summary = read_device_profile(args.profile_dir, backend=args.backend)
            write_device_profile_summary(args.output, summary)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))
    elif args.command == "cuda-event-profile":
        try:
            prepared = read_prepared_operator_input(args.prepared_input)
            request = OperatorBenchmarkRequest(
                backend=args.backend,
                op=prepared.op,
                dtype=prepared.dtype,
                dataset=prepared.dataset,
                case_id=prepared.case.case_id,
                warmup=args.warmup,
                iterations=args.iterations,
                seed=prepared.seed,
            )
            backend = get_backend(args.backend)
            event_result = backend.profile_operator_with_cuda_events(request)
            write_cuda_event_profile_csv(args.profile_dir, event_result)
            write_benchmark_outputs(
                args.output_dir,
                args.run_name,
                event_result.benchmark_result,
                request.output_formats,
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    return 0
