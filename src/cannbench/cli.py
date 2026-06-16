import argparse
from pathlib import Path

from cannbench.backends import get_backend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.prepared_input import (
    build_prepared_operator_input,
    read_prepared_operator_input,
    write_prepared_operator_input,
)
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cannbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    operator = subparsers.add_parser("operator")
    operator.add_argument("--backend", choices=["nvidia"], required=True)
    operator.add_argument("--op", choices=list_operator_names())
    operator.add_argument("--dtype", default="float16")
    operator.add_argument("--dataset", choices=["smoke", "realistic", "stress"], default="realistic")
    operator.add_argument("--case-id")
    operator.add_argument("--prepared-input", type=Path)
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

    return 0
