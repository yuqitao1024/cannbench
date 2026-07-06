#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if SRC_ROOT.is_dir() and str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))
if SRC_ROOT.is_dir():
    pythonpath_parts = [part for part in os.environ.get("PYTHONPATH", "").split(os.pathsep) if part]
    if str(SRC_ROOT) not in pythonpath_parts:
        os.environ["PYTHONPATH"] = os.pathsep.join((str(SRC_ROOT), *pythonpath_parts))


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


class Checker:
    def __init__(self) -> None:
        self.results: list[CheckResult] = []

    def ok(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name=name, status="OK", detail=detail))

    def fail(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name=name, status="FAIL", detail=detail))

    def warn(self, name: str, detail: str) -> None:
        self.results.append(CheckResult(name=name, status="WARN", detail=detail))

    def print(self) -> None:
        width = max((len(result.name) for result in self.results), default=0)
        for result in self.results:
            print(f"[{result.status}] {result.name:<{width}} {result.detail}")

    def exit_code(self) -> int:
        return 1 if any(result.status == "FAIL" for result in self.results) else 0


def _run_command(command: list[str], timeout: int = 15) -> tuple[int, str]:
    try:
        result = subprocess.run(
            command,
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except FileNotFoundError:
        return 127, ""
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    output = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part.strip())
    return result.returncode, output


def _check_python_imports(checker: Checker) -> None:
    try:
        import cannbench  # noqa: F401
    except Exception as exc:
        checker.fail("cannbench import", f"{type(exc).__name__}: {exc}")
    else:
        checker.ok("cannbench import", str(SRC_ROOT if SRC_ROOT.is_dir() else "installed package"))


def _check_workflow_cases(checker: Checker, dtype: str, seed: int) -> None:
    try:
        from cannbench.datasets.dsa_workflow import list_dsa_inference_workflows
    except Exception as exc:
        checker.fail("DSA workflow loader", f"{type(exc).__name__}: {exc}")
        return

    for workflow, dataset in (
        ("dsa_decode", "realistic_decode"),
        ("dsa_prefill", "realistic_prefill"),
    ):
        try:
            workflows = list_dsa_inference_workflows(dataset, dtype=dtype, seed=seed)
        except Exception as exc:
            checker.fail(f"workflow {workflow}/{dataset}", f"{type(exc).__name__}: {exc}")
            continue
        component_count = sum(len(item.steps) for item in workflows)
        checker.ok(
            f"workflow {workflow}/{dataset}",
            f"{len(workflows)} cases, {component_count} component runs",
        )


def _check_nvidia_tools(checker: Checker, *, skip_ncu: bool) -> None:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        checker.fail("nvidia-smi", "not found in PATH")
    else:
        code, output = _run_command([nvidia_smi, "-L"])
        if code == 0 and output:
            checker.ok("nvidia-smi", output.splitlines()[0])
        else:
            checker.fail("nvidia-smi", output or f"exit {code}")

    nvcc = shutil.which("nvcc")
    if not nvcc:
        checker.warn("nvcc", "not found in PATH; required only if CUDA extensions must be built")
    else:
        code, output = _run_command([nvcc, "--version"])
        checker.ok("nvcc", output.splitlines()[-1] if output else f"exit {code}")

    if skip_ncu:
        checker.warn("ncu", "skipped by --skip-ncu")
        return
    ncu = shutil.which("ncu")
    if not ncu:
        checker.fail("ncu", "not found in PATH; CannBench GPU profiling uses Nsight Compute CLI")
    else:
        code, output = _run_command([ncu, "--version"])
        if code == 0:
            checker.ok("ncu", output.splitlines()[0] if output else "available")
        else:
            checker.fail("ncu", output or f"exit {code}")


def _check_torch_cuda(checker: Checker) -> None:
    try:
        import torch
    except Exception as exc:
        checker.fail("torch import", f"{type(exc).__name__}: {exc}")
        return

    checker.ok("torch import", getattr(torch, "__version__", "unknown version"))
    cuda_version = getattr(getattr(torch, "version", object()), "cuda", None)
    if cuda_version:
        checker.ok("torch CUDA build", str(cuda_version))
    else:
        checker.fail("torch CUDA build", "torch.version.cuda is empty")

    try:
        available = bool(torch.cuda.is_available())
    except Exception as exc:
        checker.fail("torch.cuda.is_available", f"{type(exc).__name__}: {exc}")
        return
    if not available:
        checker.fail("torch CUDA runtime", "torch.cuda.is_available() is False")
        return
    try:
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        capability = ".".join(str(part) for part in torch.cuda.get_device_capability(0))
    except Exception as exc:
        checker.fail("torch CUDA device", f"{type(exc).__name__}: {exc}")
        return
    checker.ok("torch CUDA device", f"{device_count} visible, device0={device_name}, cc={capability}")


def _check_optional_import(checker: Checker, module_name: str) -> None:
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError:
        checker.warn(module_name, "not importable")
    except Exception as exc:
        checker.fail(module_name, f"{type(exc).__name__}: {exc}")
    else:
        version = getattr(module, "__version__", None)
        checker.ok(module_name, f"importable{f' version={version}' if version else ''}")


def _check_adapter(checker: Checker, module_name: str) -> None:
    os.environ["CANNBENCH_CUDA_DSA_ADAPTER"] = module_name
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        checker.fail(f"CUDA DSA adapter module {module_name}", f"{type(exc).__name__}: {exc}")
        return
    checker.ok(
        f"CUDA DSA adapter module {module_name}",
        getattr(module, "__file__", "imported"),
    )
    for callable_name in ("lightning_indexer", "sparse_attention"):
        candidate = getattr(module, callable_name, None)
        if callable(candidate):
            checker.ok(f"adapter callable {callable_name}", "available")
        else:
            checker.fail(f"adapter callable {callable_name}", "missing or not callable")


def _check_bench_command_import(checker: Checker) -> None:
    code, output = _run_command([sys.executable, "-m", "cannbench", "--help"])
    if code == 0 and "bench" in output:
        checker.ok("cannbench CLI", f"{sys.executable} -m cannbench")
    else:
        checker.fail("cannbench CLI", output or f"exit {code}")


def _print_next_commands(dtype: str, warmup: int, iterations: int) -> None:
    print("\nSuggested smoke commands after all required checks pass:")
    print(
        "CANNBENCH_CUDA_DSA_ADAPTER=${CANNBENCH_CUDA_DSA_ADAPTER:-cannbench_cuda_dsa} "
        f"PYTHONPATH=src python3 -m cannbench bench --backend nvidia "
        f"--implementation cuda_library --workflow dsa_decode --dataset realistic_decode "
        f"--dtype {dtype} --warmup {warmup} --iterations {iterations} --output-dir runs"
    )
    print(
        "CANNBENCH_CUDA_DSA_ADAPTER=${CANNBENCH_CUDA_DSA_ADAPTER:-cannbench_cuda_dsa} "
        f"PYTHONPATH=src python3 -m cannbench bench --backend nvidia "
        f"--implementation cuda_library --workflow dsa_prefill --dataset realistic_prefill "
        f"--dtype {dtype} --warmup {warmup} --iterations {iterations} --output-dir runs"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Check whether a GPU host has the dependencies needed for CannBench DSA CUDA workflow tests."
    )
    parser.add_argument(
        "--adapter-module",
        help=(
            "CUDA DSA adapter module exposing lightning_indexer and sparse_attention. "
            "Defaults to CANNBENCH_CUDA_DSA_ADAPTER or cannbench_cuda_dsa for full checks."
        ),
    )
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--skip-ncu", action="store_true", help="Skip Nsight Compute CLI check.")
    parser.add_argument(
        "--check-cases-only",
        action="store_true",
        help="Only validate CannBench imports, DSA workflow datasets, and adapter import.",
    )
    parser.add_argument(
        "--optional-module",
        action="append",
        default=[],
        help="Additional optional CUDA library module to import, e.g. flash_mla or deep_gemm.",
    )
    args = parser.parse_args(argv)

    checker = Checker()
    _check_python_imports(checker)
    _check_workflow_cases(checker, dtype=args.dtype, seed=args.seed)
    _check_bench_command_import(checker)
    adapter_module = (
        args.adapter_module
        or os.environ.get("CANNBENCH_CUDA_DSA_ADAPTER")
        or "cannbench_cuda_dsa"
    )
    adapter_explicit = bool(args.adapter_module or os.environ.get("CANNBENCH_CUDA_DSA_ADAPTER"))
    if args.check_cases_only and not adapter_explicit:
        checker.warn("CUDA DSA adapter", "skipped by --check-cases-only")
    else:
        _check_adapter(checker, adapter_module)
    for module_name in args.optional_module:
        _check_optional_import(checker, module_name)
    if args.check_cases_only:
        checker.warn("GPU checks", "GPU checks skipped by --check-cases-only")
    else:
        _check_nvidia_tools(checker, skip_ncu=args.skip_ncu)
        _check_torch_cuda(checker)

    checker.print()
    _print_next_commands(dtype=args.dtype, warmup=args.warmup, iterations=args.iterations)
    return checker.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
