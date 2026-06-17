import json

from cannbench.core.profile import read_device_profile, write_device_profile_summary


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
        "softmax,gpu__time_duration.sum,nsecond,1000000\n"
        "softmax,gpu__time_duration.sum,nsecond,2000000\n"
    )

    summary = read_device_profile(profile_dir, backend="nvidia")

    assert summary.backend == "nvidia"
    assert summary.sample_count == 2
    assert summary.latency_ms_avg == 1.5
    assert summary.latency_ms_p95 == 1.95
    assert summary.source_files == ("ncu.csv",)


def test_write_device_profile_summary_json(tmp_path):
    profile_dir = tmp_path / "profile"
    profile_dir.mkdir()
    (profile_dir / "op_summary.csv").write_text("Name,Duration(ms)\nsoftmax,1.25\n")
    summary = read_device_profile(profile_dir, backend="ascend")

    path = write_device_profile_summary(tmp_path / "profile-summary.json", summary)

    payload = json.loads(path.read_text())
    assert payload["backend"] == "ascend"
    assert payload["latency_ms_avg"] == 1.25
