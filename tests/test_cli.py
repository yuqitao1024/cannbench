import runpy

import pytest

from cannbench.cli import build_parser


def test_build_parser_exposes_operator_subcommand():
    parser = build_parser()
    args = parser.parse_args(
        [
            "operator",
            "--backend",
            "nvidia",
            "--op",
            "softmax",
            "--m",
            "256",
            "--k",
            "256",
            "--n",
            "256",
        ]
    )

    assert args.command == "operator"
    assert args.backend == "nvidia"
    assert args.op == "softmax"
    assert args.m == 256
    assert args.k == 256
    assert args.n == 256


def test_python_m_cannbench_exits_with_main_return_code(monkeypatch):
    monkeypatch.setattr("cannbench.cli.main", lambda: 7)

    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("cannbench", run_name="__main__")

    assert excinfo.value.code == 7
