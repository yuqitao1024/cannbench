import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cannbench")
    subparsers = parser.add_subparsers(dest="command", required=True)

    operator = subparsers.add_parser("operator")
    operator.add_argument("--backend", choices=["nvidia", "ascend"], required=True)
    operator.add_argument("--op", choices=["softmax"], required=True)
    operator.add_argument("--m", type=int, required=True)
    operator.add_argument("--k", type=int, required=True)
    operator.add_argument("--n", type=int, required=True)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    parser.parse_args(argv)
    return 0
