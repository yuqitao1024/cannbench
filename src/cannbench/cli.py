import argparse
import json
import shutil
from pathlib import Path

from cannbench.backends import get_backend
from cannbench.core.batch import (
    BatchFailureRecord,
    PreparedInputPlan,
    BatchResultRecord,
    BatchSummaryMetadata,
    expand_prepared_input_plans,
    write_batch_failures_json,
    write_batch_summary_csv,
    write_batch_summary_json,
)
from cannbench.core.benchmark_records import (
    build_benchmark_record,
    write_benchmark_records_json,
)
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.layout import build_run_layout
from cannbench.core.publish import publish_run_artifacts
from cannbench.serve import serve_cannbench
from cannbench.core.operator_output import (
    compare_operator_outputs,
    read_operator_output,
    write_operator_output,
    write_output_comparison,
)
from cannbench.core.profile import (
    ProfileArtifacts,
    read_device_profile,
    write_device_profile_summary,
    write_profile_artifacts,
)
from cannbench.core.prepared_input import (
    PreparedOperatorInput,
    build_prepared_operator_input,
    read_prepared_operator_input,
    write_prepared_operator_input,
)
from cannbench.core.remote import collect_remote_artifacts, read_remote_endpoint
from cannbench.core.report import write_local_report
from cannbench.core.output import (
    build_benchmark_artifact_stem,
    write_benchmark_outputs,
)
from cannbench.core.execution import (
    BenchCaseExecutionResult,
    LocalBenchExecutor,
    RemoteBenchExecutor,
)
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
    parser.add_argument("--seed", type=_non_negative_int, default=0)
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


def _prepared_manifest_path(base_dir: Path, op: str, dataset: str, case_id: str, dtype: str, seed: int) -> Path:
    return base_dir / op / dataset / f"{case_id}-{dtype}-seed{seed}.json"


def _build_request_from_prepared(
    prepared: PreparedOperatorInput,
    args: argparse.Namespace,
) -> OperatorBenchmarkRequest:
    if args.op and prepared.op != args.op:
        raise ValueError(
            f"prepared input operator mismatch: expected {args.op}, got {prepared.op}"
        )
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


def _build_request_from_args(args: argparse.Namespace) -> OperatorBenchmarkRequest:
    if args.prepared_input is not None:
        return _build_request_from_prepared(read_prepared_operator_input(args.prepared_input), args)
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
    bench.add_argument("--endpoint", type=Path)
    bench.add_argument("--run-id")
    bench.add_argument("--capture-output", action="store_true", default=False)

    internal_run = subparsers.add_parser("internal-run")
    _benchmark_args(internal_run)

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
    dataset = getattr(args, "dataset", None)
    op = getattr(args, "op", None)

    if prepared_input is not None and prepared_dir is not None:
        raise ValueError("--prepared-input and --prepared-dir are mutually exclusive")
    if (prepared_input is not None or prepared_dir is not None) and dataset is not None:
        raise ValueError("--dataset cannot be used with --prepared-input or --prepared-dir")
    if (prepared_input is not None or prepared_dir is not None) and case_id is not None:
        raise ValueError("--case-id cannot be used with --prepared-input or --prepared-dir")
    if prepared_dir is not None and not op:
        raise ValueError("--op is required with --prepared-dir")
    if prepared_input is None and not op:
        raise ValueError("--op is required unless --prepared-input is set")
    if not allow_batch and prepared_dir is not None:
        raise ValueError("--prepared-dir is only supported for bench")
    if allow_batch and _is_batch_mode(args) and not getattr(args, "run_name", None):
        raise ValueError("--run-name is required for batch execution")


def _relative_artifact_path(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    return path.relative_to(root).as_posix()


def _prepared_reference_for_plan(
    args: argparse.Namespace,
    prepared_dir: Path,
    layout_root: Path,
    plan,
) -> tuple[Path, str]:
    prepared_path = _prepared_manifest_path(
        prepared_dir,
        plan.op,
        plan.dataset,
        plan.case_id,
        plan.dtype,
        plan.seed,
    )
    if plan.source_path is not None:
        prepared_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(plan.source_path, prepared_path)
    else:
        write_prepared_operator_input(prepared_path, plan.prepared)
    return prepared_path, _relative_artifact_path(layout_root, prepared_path) or prepared_path.name


def _single_case_plan_from_args(args: argparse.Namespace) -> PreparedInputPlan:
    if args.prepared_input is not None:
        plans = expand_prepared_input_plans(
            op=args.op,
            dtype=args.dtype,
            dataset=args.dataset,
            case_id=args.case_id,
            seed=args.seed,
            prepared_input=args.prepared_input,
        )
        if len(plans) != 1:
            raise ValueError("single-case bench requires exactly one prepared input plan")
        return plans[0]

    request = _build_request_from_args(args)
    prepared = build_prepared_operator_input(
        op=request.op,
        dtype=request.dtype,
        dataset=request.dataset,
        case_id=request.case_id,
        seed=request.seed,
    )
    return PreparedInputPlan(
        op=prepared.op,
        dataset=prepared.dataset,
        case_id=prepared.case.case_id,
        dtype=prepared.dtype,
        seed=prepared.seed,
        prepared=prepared,
    )


def _validate_unique_batch_artifact_stems(plans: list) -> None:
    seen: dict[str, str] = {}
    for plan in plans:
        stem = build_benchmark_artifact_stem(
            op=plan.op,
            dataset=plan.dataset,
            case_id=plan.case_id,
            dtype=plan.dtype,
            seed=plan.seed,
        )
        plan_ref = (
            plan.source_path.as_posix()
            if plan.source_path is not None
            else f"{plan.op}/{plan.dataset}/{plan.case_id}"
        )
        previous_ref = seen.get(stem)
        if previous_ref is not None:
            raise ValueError(
                "duplicate batch artifact stem "
                f"{stem!r} from prepared inputs {previous_ref!r} and {plan_ref!r}"
            )
        seen[stem] = plan_ref


def _write_perf_artifacts(
    perf_dir: Path,
    artifact_stem: str,
    artifacts: tuple[tuple[str, bytes], ...],
) -> Path | None:
    if not artifacts:
        return None
    perf_dir.mkdir(parents=True, exist_ok=True)
    preferred_result_path: Path | None = None
    for relative_name, content in artifacts:
        source_name = Path(relative_name).name
        if source_name.startswith("benchmark."):
            dest_path = perf_dir / f"{artifact_stem}{Path(source_name).suffix}"
        else:
            dest_path = perf_dir / f"{artifact_stem}-{source_name}"
        dest_path.write_bytes(content)
        if dest_path.suffix == ".json":
            preferred_result_path = dest_path
        elif preferred_result_path is None:
            preferred_result_path = dest_path
    return preferred_result_path


def _finalize_profile_artifacts(
    layout,
    artifact_stem: str,
    profile: ProfileArtifacts,
) -> Path | None:
    if layout.profile_dir is not None:
        profile_dir = layout.profile_dir / artifact_stem
        write_profile_artifacts(profile_dir, profile.profile_artifacts)
        write_device_profile_summary(
            profile_dir / "profile-summary.json",
            profile.profile_summary,
        )
    return _write_perf_artifacts(layout.perf_dir, artifact_stem, profile.perf_artifacts)


def _finalize_remote_execution_artifacts(
    *,
    layout,
    artifact_stem: str,
    execution_result,
    capture_output: bool,
) -> Path | None:
    if capture_output:
        output_dest = layout.output_dir / artifact_stem
        write_profile_artifacts(output_dest, execution_result.artifacts.output_artifacts)
    if execution_result.artifacts.profile is None:
        return None
    return _finalize_profile_artifacts(
        layout,
        artifact_stem,
        execution_result.artifacts.profile,
    )


def _finalize_local_execution_artifacts(
    *,
    layout,
    artifact_stem: str,
    execution_result: BenchCaseExecutionResult,
) -> Path | None:
    result_path = execution_result.result_path
    if execution_result.artifacts.profile is None:
        return result_path
    profiled_result_path = _finalize_profile_artifacts(
        layout,
        artifact_stem,
        execution_result.artifacts.profile,
    )
    return profiled_result_path or result_path


def _build_benchmark_record_entry(
    *,
    run_id: str,
    backend: str,
    implementation: str | None,
    prepared,
    execution_result: BenchCaseExecutionResult,
) -> dict[str, object]:
    if execution_result.artifacts.profile is None:
        raise RuntimeError("missing profile summary for profiled bench case")
    return build_benchmark_record(
        run_id=run_id,
        backend=backend,
        implementation=implementation,
        prepared=prepared,
        device_name=execution_result.artifacts.profile.device_name,
        profile_summary=execution_result.artifacts.profile.profile_summary,
    )


def _prepare_batch_run_layout(output_dir: Path, run_name: str):
    layout = build_run_layout(output_dir, run_name)
    if layout.root.exists() and any(layout.root.iterdir()):
        raise ValueError(
            f"batch run directory already exists and is not empty: {layout.root}"
        )
    layout.root.mkdir(parents=True, exist_ok=True)
    return layout


def _write_bench_metadata(
    *,
    layout,
    backend: str,
    run_name: str,
    implementation: str | None,
    summary_rows: list[BatchResultRecord],
    failure_rows: list[BatchFailureRecord],
    benchmark_records: list[dict[str, object]],
) -> Path:
    metadata = BatchSummaryMetadata(
        backend=backend,
        run_name=run_name,
        implementation=implementation,
        total_cases=len(summary_rows),
        success_count=sum(1 for row in summary_rows if row.status == "ok"),
        failure_count=len(failure_rows),
    )
    write_batch_summary_json(layout.meta_dir / "summary.json", summary_rows, metadata)
    write_batch_summary_csv(layout.meta_dir / "summary.csv", summary_rows)
    if benchmark_records:
        write_benchmark_records_json(layout.meta_dir / "benchmark-records.json", benchmark_records)
    return write_batch_failures_json(layout.meta_dir / "failures.json", failure_rows, metadata)


def _build_success_row(
    *,
    plan,
    prepared_reference: str,
    layout_root: Path,
    result_path: Path | None,
) -> BatchResultRecord:
    return BatchResultRecord(
        op=plan.op,
        dataset=plan.dataset,
        case_id=plan.case_id,
        dtype=plan.dtype,
        seed=plan.seed,
        status="ok",
        prepared_input=prepared_reference,
        result_path=_relative_artifact_path(layout_root, result_path),
    )


def _build_failure_row(
    *,
    plan,
    prepared_reference: str,
) -> BatchResultRecord:
    return BatchResultRecord(
        op=plan.op,
        dataset=plan.dataset,
        case_id=plan.case_id,
        dtype=plan.dtype,
        seed=plan.seed,
        status="failed",
        prepared_input=prepared_reference,
        result_path=None,
    )


def _build_failure_record(
    *,
    plan,
    prepared_reference: str,
    error: Exception,
) -> BatchFailureRecord:
    return BatchFailureRecord(
        op=plan.op,
        dataset=plan.dataset,
        case_id=plan.case_id,
        dtype=plan.dtype,
        seed=plan.seed,
        prepared_input=prepared_reference,
        error=str(error),
    )


def _run_local_bench_with_plans(
    args: argparse.Namespace,
    *,
    plans: list,
    run_name: str,
) -> None:
    _validate_unique_batch_artifact_stems(plans)
    layout = _prepare_batch_run_layout(args.output_dir, run_name)
    backend = get_backend(args.backend)
    executor = LocalBenchExecutor(backend, write_benchmark_outputs)
    summary_rows: list[BatchResultRecord] = []
    benchmark_records: list[dict[str, object]] = []
    failure_rows: list[BatchFailureRecord] = []

    for plan in plans:
        _, prepared_reference = _prepared_reference_for_plan(
            args, layout.prepared_dir, layout.root, plan
        )
        request = _build_request_from_prepared(plan.prepared, args)
        artifact_stem = build_benchmark_artifact_stem(
            op=plan.op,
            dataset=plan.dataset,
            case_id=plan.case_id,
            dtype=plan.dtype,
            seed=plan.seed,
        )
        try:
            execution_result = executor.execute_case(
                request,
                output_dir=layout.perf_dir,
                run_name=artifact_stem,
            )
            result_path = _finalize_local_execution_artifacts(
                layout=layout,
                artifact_stem=artifact_stem,
                execution_result=execution_result,
            )
            if execution_result.artifacts.profile is not None:
                benchmark_records.append(
                    _build_benchmark_record_entry(
                        run_id=f"{run_name}/{artifact_stem}",
                        backend=args.backend,
                        implementation=getattr(args, "implementation", None),
                        prepared=plan.prepared,
                        execution_result=execution_result,
                    )
                )
            summary_rows.append(
                _build_success_row(
                    plan=plan,
                    prepared_reference=prepared_reference,
                    layout_root=layout.root,
                    result_path=result_path,
                )
            )
        except Exception as exc:
            summary_rows.append(
                _build_failure_row(
                    plan=plan,
                    prepared_reference=prepared_reference,
                )
            )
            failure_rows.append(
                _build_failure_record(
                    plan=plan,
                    prepared_reference=prepared_reference,
                    error=exc,
                )
            )

    failures_path = _write_bench_metadata(
        layout=layout,
        backend=args.backend,
        run_name=run_name,
        implementation=getattr(args, "implementation", None),
        summary_rows=summary_rows,
        failure_rows=failure_rows,
        benchmark_records=benchmark_records,
    )
    if failure_rows:
        label = "single" if len(plans) == 1 else "batch"
        raise RuntimeError(
            f"{label} bench completed with {len(failure_rows)} failures; see {failures_path}"
        )


def _run_remote_bench_with_plans(
    args: argparse.Namespace,
    *,
    plans: list,
    run_name: str,
    endpoint,
    parent_run_id: str,
) -> None:
    _validate_unique_batch_artifact_stems(plans)
    layout = _prepare_batch_run_layout(args.output_dir, run_name)
    executor = RemoteBenchExecutor(
        collect_remote_artifacts,
        endpoint,
        endpoint_path=args.endpoint,
    )
    summary_rows: list[BatchResultRecord] = []
    benchmark_records: list[dict[str, object]] = []
    failure_rows: list[BatchFailureRecord] = []

    for plan in plans:
        prepared_path, prepared_reference = _prepared_reference_for_plan(
            args, layout.prepared_dir, layout.root, plan
        )
        artifact_stem = build_benchmark_artifact_stem(
            op=plan.op,
            dataset=plan.dataset,
            case_id=plan.case_id,
            dtype=plan.dtype,
            seed=plan.seed,
        )
        remote_run_id = f"{parent_run_id}/{artifact_stem}" if len(plans) > 1 else parent_run_id
        try:
            execution_result = executor.execute_case(
                prepared_input=prepared_path,
                layout_root=layout.root,
                artifact_stem=artifact_stem,
                run_id=remote_run_id,
                capture_output=args.capture_output,
                warmup=args.warmup,
                iterations=args.iterations,
                deploy_custom_op=_resolve_deploy_custom_op(
                    endpoint.backend, args.implementation, args.deploy_custom_op
                ),
            )
            result_path = _finalize_remote_execution_artifacts(
                layout=layout,
                artifact_stem=artifact_stem,
                execution_result=execution_result,
                capture_output=args.capture_output,
            )
            execution_result = BenchCaseExecutionResult(
                artifacts=execution_result.artifacts,
                result_path=result_path,
            )
            if execution_result.result_path is None:
                raise RuntimeError(
                    f"missing perf artifacts for profiled remote bench case {artifact_stem}"
                )
            if execution_result.artifacts.profile is None:
                raise RuntimeError(
                    f"missing profile summary for profiled remote bench case {artifact_stem}"
                )
            benchmark_records.append(
                _build_benchmark_record_entry(
                    run_id=remote_run_id,
                    backend=endpoint.backend,
                    implementation=args.implementation,
                    prepared=plan.prepared,
                    execution_result=execution_result,
                )
            )
            summary_rows.append(
                _build_success_row(
                    plan=plan,
                    prepared_reference=prepared_reference,
                    layout_root=layout.root,
                    result_path=execution_result.result_path,
                )
            )
        except Exception as exc:
            summary_rows.append(
                _build_failure_row(
                    plan=plan,
                    prepared_reference=prepared_reference,
                )
            )
            failure_rows.append(
                _build_failure_record(
                    plan=plan,
                    prepared_reference=prepared_reference,
                    error=exc,
                )
            )

    failures_path = _write_bench_metadata(
        layout=layout,
        backend=endpoint.backend,
        run_name=run_name,
        implementation=getattr(args, "implementation", None),
        summary_rows=summary_rows,
        failure_rows=failure_rows,
        benchmark_records=benchmark_records,
    )
    if failure_rows:
        label = "single" if len(plans) == 1 else "batch"
        raise RuntimeError(
            f"{label} bench completed with {len(failure_rows)} failures; see {failures_path}"
        )


def _run_batch_local_bench(args: argparse.Namespace) -> None:
    plans = expand_prepared_input_plans(
        op=args.op,
        dtype=args.dtype,
        dataset=args.dataset,
        case_id=args.case_id,
        prepared_input=args.prepared_input,
        prepared_dir=args.prepared_dir,
    )
    _run_local_bench_with_plans(args, plans=plans, run_name=args.run_name)


def _run_batch_remote_bench(args: argparse.Namespace) -> None:
    plans = expand_prepared_input_plans(
        op=args.op,
        dtype=args.dtype,
        dataset=args.dataset,
        case_id=args.case_id,
        seed=args.seed,
        prepared_input=args.prepared_input,
        prepared_dir=args.prepared_dir,
    )
    endpoint = read_remote_endpoint(args.endpoint)
    _run_remote_bench_with_plans(
        args,
        plans=plans,
        run_name=args.run_name,
        endpoint=endpoint,
        parent_run_id=args.run_id or args.run_name,
    )


def _run_single_bench(args: argparse.Namespace) -> None:
    run_name = args.run_name or "operator-benchmark"
    if args.endpoint is None:
        _run_local_bench_with_plans(
            args,
            plans=[_single_case_plan_from_args(args)],
            run_name=run_name,
        )
        return

    run_name = args.run_name or args.run_id
    endpoint = read_remote_endpoint(args.endpoint)
    if not run_name:
        raise ValueError("single-case remote bench requires --run-name or --run-id")
    _run_remote_bench_with_plans(
        args,
        plans=[_single_case_plan_from_args(args)],
        run_name=run_name,
        endpoint=endpoint,
        parent_run_id=args.run_id or run_name,
    )


def _run_bench_command(args: argparse.Namespace) -> None:
    if _is_batch_mode(args):
        if args.endpoint is None:
            _run_batch_local_bench(args)
        else:
            _run_batch_remote_bench(args)
        return
    _run_single_bench(args)


def _run_internal_command(args: argparse.Namespace) -> None:
    request = _build_request_from_args(args)
    backend = get_backend(args.backend)
    result = backend.run_operator(request)
    write_benchmark_outputs(
        args.output_dir, args.run_name or "internal-run-benchmark", result, request.output_formats
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command in {"bench", "internal-run"}:
        try:
            _validate_benchmark_selection(args, allow_batch=args.command == "bench")
            if args.command == "bench":
                _run_bench_command(args)
                return 0
            _run_internal_command(args)
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
    return 0
