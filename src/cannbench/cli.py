import argparse
from pathlib import Path

from cannbench.backends import get_backend
from cannbench.core.config import OperatorBenchmarkRequest
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cannbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    operator = subparsers.add_parser("operator")
    operator.add_argument("--backend", choices=["nvidia", "ascend"], required=True)
    operator.add_argument("--op", choices=list_operator_names())
    operator.add_argument("--dtype", default="float16")
    operator.add_argument("--dataset", choices=["smoke", "realistic", "stress"], default="realistic")
    operator.add_argument("--case-id")
    operator.add_argument("--prepared-input", type=Path)
    operator.add_argument("--deploy-custom-op", action="store_true", default=False)
    operator.add_argument("--warmup", type=_non_negative_int, default=10)
    operator.add_argument("--iterations", type=_positive_int, default=50)
    operator.add_argument("--output-dir", type=Path, default=Path("results"))
    operator.add_argument("--run-name", default="operator-benchmark")

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
    collect.add_argument("--prepared-input", type=Path, required=True)
    collect.add_argument("--output-dir", type=Path, required=True)
    collect.add_argument("--run-id")
    collect.add_argument("--capture-output", action="store_true", default=False)
    collect.add_argument("--profile-device-time", action="store_true", default=False)
    collect.add_argument("--summarize-profile", action="store_true", default=False)
    collect.add_argument("--warmup", type=_non_negative_int, default=10)
    collect.add_argument("--iterations", type=_positive_int, default=50)
    collect.add_argument("--deploy-custom-op", action="store_true", default=False)

    report = subparsers.add_parser("report")
    report.add_argument("--nvidia", type=Path, required=True)
    report.add_argument("--ascend", type=Path, required=True)
    report.add_argument("--accuracy", type=Path, required=True)
    report.add_argument("--output", type=Path, required=True)

    summarize_profile = subparsers.add_parser("summarize-profile")
    summarize_profile.add_argument("--backend", choices=["nvidia", "ascend"], required=True)
    summarize_profile.add_argument("--profile-dir", type=Path, required=True)
    summarize_profile.add_argument("--output", type=Path, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "operator":
        try:
            if args.prepared_input is not None:
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
                    warmup=args.warmup,
                    iterations=args.iterations,
                    deploy_custom_op=args.deploy_custom_op,
                )
            backend = get_backend(args.backend)
            result = backend.run_operator(request)
            write_benchmark_outputs(
                args.output_dir, args.run_name, result, request.output_formats
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
            collect_remote_artifacts(
                endpoint_path=args.endpoint,
                prepared_input=args.prepared_input,
                output_dir=args.output_dir,
                run_id=args.run_id,
                capture_output=args.capture_output,
                profile_device_time=args.profile_device_time,
                summarize_profile=args.summarize_profile,
                warmup=args.warmup,
                iterations=args.iterations,
                deploy_custom_op=args.deploy_custom_op,
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
    elif args.command == "summarize-profile":
        try:
            summary = read_device_profile(args.profile_dir, backend=args.backend)
            write_device_profile_summary(args.output, summary)
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    return 0
