from __future__ import annotations

from pathlib import Path

from cannbench.core.benchmark_records import (
    build_benchmark_record,
    build_collect_benchmark_record,
    build_workflow_benchmark_record,
    read_profile_summary,
    write_benchmark_records_json,
)
from cannbench.core.prepared_input import (
    build_prepared_operator_input,
    prepare_workflow_input,
)
from cannbench.core.profile import DeviceProfileSummary, write_device_profile_summary
from cannbench.operators.builtin.dsa_decode import build_dsa_decode_workflow


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
        latency_ms=1.0,
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
    assert record["metrics"] == {"latency_ms": 1.0}


def test_build_collect_benchmark_record_for_ascend_simt_v2():
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="realistic",
        case_id="t5_attention",
        seed=7,
    )
    profile_summary = DeviceProfileSummary(
        backend="ascend",
        latency_ms=1.0,
        source_files=("op_summary.csv",),
    )

    record = build_collect_benchmark_record(
        run_id="opbench-ascend-950pr-simt-v2-softmax-realistic-float16",
        backend="ascend",
        implementation="simt",
        implementation_version="v2",
        prepared=prepared,
        perf_payload={"device_name": "Ascend950PR_9589"},
        profile_summary=profile_summary,
    )

    assert record["implementation"] == "simt"
    assert record["implementation_version"] == "v2"
    assert record["diff_ref"] == "softmax/simt/v2"


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
        latency_ms=0.9,
        source_files=("op_summary.csv",),
    )

    record = build_collect_benchmark_record(
        run_id="opbench-ascend-950pr-cannops-softmax-realistic-float16",
        backend="ascend",
        implementation="cann_ops_library",
        prepared=prepared,
        perf_payload={"device_name": "Ascend950PR_9589"},
        profile_summary=profile_summary,
    )

    assert record["implementation"] == "cann_ops_library"
    assert record["implementation_version"] == "cannops"
    assert record["device_class"] == "950PR"
    assert record["metrics"] == {"latency_ms": 0.9}


def test_build_benchmark_record_for_vllm_ascend_lightning_indexer_shape():
    prepared = build_prepared_operator_input(
        op="lightning_indexer",
        dtype="bfloat16",
        dataset="smoke",
        case_id="vllm_ascend_a5_decode_b1_ctx512_top512",
        seed=0,
    )
    profile_summary = DeviceProfileSummary(
        backend="ascend",
        latency_ms=0.005,
        source_files=("OpBasicInfo.csv",),
    )

    record = build_benchmark_record(
        run_id="dsa/lightning-indexer",
        backend="ascend",
        implementation="vllm_ascend",
        prepared=prepared,
        device_name="Ascend950PR_9599",
        profile_summary=profile_summary,
    )

    assert record["implementation"] == "vllm_ascend"
    assert record["implementation_version"] == "vllm-ascend"
    assert record["shape"] == [1, 1, 64, 128]


def test_build_benchmark_record_for_vllm_ascend_sparse_attention_shape():
    prepared = build_prepared_operator_input(
        op="sparse_attention",
        dtype="bfloat16",
        dataset="smoke",
        case_id="vllm_ascend_a5_prefill_b1_q512_ctx512_top512",
        seed=0,
    )
    profile_summary = DeviceProfileSummary(
        backend="ascend",
        latency_ms=0.014,
        source_files=("OpBasicInfo.csv",),
    )

    record = build_benchmark_record(
        run_id="dsa/sparse-attention",
        backend="ascend",
        implementation="vllm_ascend",
        prepared=prepared,
        device_name="Ascend950PR_9599",
        profile_summary=profile_summary,
    )

    assert record["implementation"] == "vllm_ascend"
    assert record["implementation_version"] == "vllm-ascend"
    assert record["shape"] == [512, 64, 512]


def test_build_benchmark_record_for_v32_sparse_attention_uses_qk_dimension():
    prepared = build_prepared_operator_input(
        op="sparse_attention",
        dtype="bfloat16",
        dataset="realistic_decode",
        case_id="deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
        seed=0,
    )
    profile_summary = DeviceProfileSummary(
        backend="ascend",
        latency_ms=0.014,
        source_files=("OpBasicInfo.csv",),
    )

    record = build_benchmark_record(
        run_id="dsa/sparse-attention-v32",
        backend="ascend",
        implementation="simt",
        prepared=prepared,
        device_name="Ascend950PR_9599",
        profile_summary=profile_summary,
    )

    assert record["shape"] == [2, 128, 576]


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
        latency_ms=0.2,
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

    assert record["implementation"] == "cuda-pytorch"
    assert record["implementation_version"] == "cuda-pytorch"
    assert record["device_class"] == "H800"
    assert record["shape"] == [4, 8, 1024, 1024]
    assert record["family"] == "attention"
    assert record["source_kind"] == "real_model"
    assert record["source_project"] == "TritonBench"
    assert record["source_model"] == "T5Small"
    assert record["source_file"] == "hf_train/T5Small_train.json"
    assert record["source_op"] == "aten._softmax.default"
    assert record["diff_ref"] is None
    assert record["metrics"] == {"latency_ms": 0.2}


def test_read_profile_summary_and_write_benchmark_records_json(tmp_path: Path):
    summary_path = tmp_path / "profile-summary.json"
    payload_path = tmp_path / "benchmark-records.json"
    write_device_profile_summary(
        summary_path,
        DeviceProfileSummary(
            backend="nvidia",
            latency_ms=0.2,
            source_files=("ncu.csv",),
        ),
    )

    summary = read_profile_summary(summary_path)
    result = write_benchmark_records_json(payload_path, [{"schema_version": 1, "records": "ok"}])

    assert summary.backend == "nvidia"
    assert summary.source_files == ("ncu.csv",)
    assert result == payload_path
    assert '"records": "ok"' in payload_path.read_text()


def test_build_workflow_benchmark_record_preserves_published_contract():
    prepared = prepare_workflow_input(
        build_dsa_decode_workflow(
            dataset="stress",
            case_id="vllm_ascend_a5_decode_b1_ctx512_top512",
            dtype="bfloat16",
            seed=0,
        )
    )

    record = build_workflow_benchmark_record(
        run_id=(
            "dsa-decode-profiled/"
            "dsa_decode-stress-vllm_ascend_a5_decode_b1_ctx512_top512-"
            "bfloat16-seed0"
        ),
        backend="nvidia",
        implementation="cuda_library",
        prepared=prepared,
        device_name="NVIDIA H800 PCIe",
        profile_summary=DeviceProfileSummary(
            backend="nvidia",
            latency_ms=0.01,
            source_files=("ncu.csv",),
        ),
    )

    assert record["operator"] == "dsa_decode"
    assert record["dataset"] == "stress"
    assert record["family"] == "decode_workflow"
    assert record["shape"] == [1, 64, 512]
    assert record["implementation"] == "cuda_library"
    assert record["implementation_version"] == "cuda-library"
    assert record["metrics"] == {"latency_ms": 0.01}
    assert record["diff_ref"] is None
