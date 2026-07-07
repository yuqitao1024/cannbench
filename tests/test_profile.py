import json

import pytest

from cannbench.core.profile import (
    ProfileKernelSelection,
    read_device_profile,
    write_device_profile_summary,
)


def test_read_ascend_msprof_csv_duration_summary(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "op_summary.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "softmax,1000\n"
        "softmax,2000\n"
    )

    summary = read_device_profile(profile_dir, backend="ascend")

    assert summary.backend == "ascend"
    assert summary.sample_count == 2
    assert summary.latency_ms_avg == 1.5
    assert summary.latency_ms_p50 == 1.5
    assert summary.source_files == ("op_summary.csv",)


def test_read_nvidia_ncu_csv_duration_metric_summary(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "ncu.csv").write_text(
        "Kernel Name,Metric Name,Metric Unit,Metric Value\n"
        "softmax,gpu__time_duration.avg,usecond,1000\n"
        "softmax,gpu__time_duration.avg,usecond,2000\n"
    )

    summary = read_device_profile(profile_dir, backend="nvidia")

    assert summary.backend == "nvidia"
    assert summary.sample_count == 2
    assert summary.latency_ms_avg == 1.5
    assert summary.latency_ms_p95 == 1.95
    assert summary.source_files == ("ncu.csv",)


def test_read_nvidia_ncu_wide_csv_uses_avg_duration_and_unit_row(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "ncu.csv").write_text(
        '"ID","Kernel Name","gpu__dram_cycles_active.avg","gpu__time_duration.avg"\n'
        '"","","cycle","usecond"\n'
        '"1","softmax_kernel","55295584","60.125"\n'
    )

    summary = read_device_profile(profile_dir, backend="nvidia")

    assert summary.backend == "nvidia"
    assert summary.sample_count == 1
    assert summary.latency_ms_avg == 0.060125
    assert summary.source_files == ("ncu.csv",)


def test_read_device_profile_filters_to_expected_kernel_name(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "StatelessRandomNormalV3,400\n"
        "SoftmaxV2_float16_high_precision_1000,58\n"
    )

    summary = read_device_profile(
        profile_dir,
        backend="ascend",
        expected_kernel_name_patterns=("softmax",),
    )

    assert summary.sample_count == 1
    assert summary.latency_ms_avg == 0.058
    assert summary.source_files == ("OpBasicInfo.csv",)


def test_read_device_profile_can_sum_multiple_matching_kernels(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "lightning_indexer_kernel,400\n"
        "sparse_attention_kernel,600\n"
        "unrelated_kernel,999\n"
    )

    summary = read_device_profile(
        profile_dir,
        backend="ascend",
        kernel_selection=ProfileKernelSelection(
            kernel_name_patterns=("lightning_indexer", "sparse_attention"),
        ),
    )

    assert summary.sample_count == 1
    assert summary.latency_ms_avg == 1.0


def test_read_device_profile_rejects_unexpected_kernel_name(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "OpBasicInfo.csv").write_text(
        "Op Name,Task Duration(us)\n"
        "StatelessRandomNormalV3,400\n"
    )

    with pytest.raises(ValueError, match="expected profiler kernel"):
        read_device_profile(
            profile_dir,
            backend="ascend",
            expected_kernel_name_patterns=("softmax",),
        )


def test_write_device_profile_summary_json(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "op_summary.csv").write_text("Name,Duration(ms)\nsoftmax,1.25\n")
    summary = read_device_profile(profile_dir, backend="ascend")

    path = write_device_profile_summary(tmp_path / "profile-summary.json", summary)

    payload = json.loads(path.read_text())
    assert payload["backend"] == "ascend"
    assert payload["latency_ms_avg"] == 1.25
