import argparse
from pathlib import Path

from cannbench.backends import get_backend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.output import write_benchmark_outputs


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
    operator.add_argument("--op", choices=["softmax"], required=True)
    operator.add_argument("--dtype", default="float16")
    operator.add_argument("--dataset", choices=["smoke", "realistic", "stress"], default="realistic")
    operator.add_argument("--case-id", required=True)
    operator.add_argument("--warmup", type=_non_negative_int, default=10)
    operator.add_argument("--iterations", type=_positive_int, default=50)
    operator.add_argument("--output-dir", type=Path, default=Path("results"))
    operator.add_argument("--run-name", default="operator-benchmark")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "operator":
        try:
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
            result = backend.run_softmax(request)
            write_benchmark_outputs(
                args.output_dir, args.run_name, result, request.output_formats
            )
        except (RuntimeError, ValueError) as exc:
            parser.error(str(exc))

    return 0
