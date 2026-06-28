from __future__ import annotations

from pathlib import Path

from cannbench.core.benchmark_records import (
    build_benchmark_record,
    build_collect_benchmark_record,
    read_profile_summary,
    write_benchmark_records_json,
)
from cannbench.core.prepared_input import build_prepared_operator_input
from cannbench.core.profile import DeviceProfileSummary, write_device_profile_summary


def test_build_collect_benchmark_record_for_ascend_simt():
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="realistic",
        case_id="t5_attention",
        seed=7,
    )
    profile_summary = DeviceProfileSummary(
        backend="ascend",
        sample_count=1,
        latency_ms_avg=1.0,
        latency_ms_p50=1.0,
        latency_ms_p95=1.1,
        latency_ms_p99=1.2,
        source_files=("op_summary.csv",),
    )

    record = build_collect_benchmark_record(
        run_id="opbench-ascend-950pr-simt-v1-softmax-realistic-float16",
        backend="ascend",
        implementation="simt",
        prepared=prepared,
        perf_payload={"device_name": "Ascend950PR_9589"},
        profile_summary=profile_summary,
    )

    assert record["implementation"] == "simt"
    assert record["implementation_version"] == "v1"
    assert record["device_class"] == "950PR"
    assert record["shape"] == [4, 8, 1024, 1024]
    assert record["family"] == "attention"
    assert record["source_kind"] == "real_model"
    assert record["source_project"] == "TritonBench"
    assert record["source_model"] == "T5Small"
    assert record["source_file"] == "hf_train/T5Small_train.json"
    assert record["source_op"] == "aten._softmax.default"
    assert record["diff_ref"] == "softmax/simt/v1"


def test_build_collect_benchmark_record_for_ascend_cann_ops():
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="realistic",
        case_id="t5_attention",
        seed=7,
    )
    profile_summary = DeviceProfileSummary(
        backend="ascend",
        sample_count=2,
        latency_ms_avg=0.9,
        latency_ms_p50=0.9,
        latency_ms_p95=1.0,
        latency_ms_p99=1.1,
        source_files=("op_summary.csv",),
    )

    record = build_collect_benchmark_record(
        run_id="opbench-ascend-950pr-cann-cannops-softmax-realistic-float16",
        backend="ascend",
        implementation="cann_ops_library",
        prepared=prepared,
        perf_payload={"device_name": "Ascend950PR_9589"},
        profile_summary=profile_summary,
    )

    assert record["implementation"] == "cann_ops_library"
    assert record["implementation_version"] == "cannops"
    assert record["device_class"] == "950PR"


def test_build_benchmark_record_for_nvidia_ncu():
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="realistic",
        case_id="t5_attention",
        seed=7,
    )
    profile_summary = DeviceProfileSummary(
        backend="nvidia",
        sample_count=2,
        latency_ms_avg=0.2,
        latency_ms_p50=0.2,
        latency_ms_p95=0.3,
        latency_ms_p99=0.4,
        source_files=("ncu.csv",),
    )

    record = build_benchmark_record(
        run_id="softmax-realistic-ncu",
        backend="nvidia",
        implementation=None,
        prepared=prepared,
        device_name="NVIDIA H800 PCIe",
        profile_summary=profile_summary,
    )

    assert record["implementation"] == "ncu"
    assert record["implementation_version"] == "ncu"
    assert record["device_class"] == "H800"
    assert record["shape"] == [4, 8, 1024, 1024]
    assert record["family"] == "attention"
    assert record["source_kind"] == "real_model"
    assert record["source_project"] == "TritonBench"
    assert record["source_model"] == "T5Small"
    assert record["source_file"] == "hf_train/T5Small_train.json"
    assert record["source_op"] == "aten._softmax.default"
    assert record["diff_ref"] is None


def test_read_profile_summary_and_write_benchmark_records_json(tmp_path: Path):
    summary_path = tmp_path / "profile-summary.json"
    payload_path = tmp_path / "benchmark-records.json"
    write_device_profile_summary(
        summary_path,
        DeviceProfileSummary(
            backend="nvidia",
            sample_count=2,
            latency_ms_avg=0.2,
            latency_ms_p50=0.2,
            latency_ms_p95=0.3,
            latency_ms_p99=0.4,
            source_files=("ncu.csv",),
        ),
    )

    summary = read_profile_summary(summary_path)
    result = write_benchmark_records_json(payload_path, [{"schema_version": 1, "records": "ok"}])

    assert summary.backend == "nvidia"
    assert summary.source_files == ("ncu.csv",)
    assert result == payload_path
    assert '"records": "ok"' in payload_path.read_text()
