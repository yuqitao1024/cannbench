import json

from cannbench.core.report import write_local_report


def _write_perf(path, *, backend, avg):
    perf_dir = path / "perf"
    perf_dir.mkdir(parents=True)
    (perf_dir / "benchmark.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "device_name": f"Fake {backend}",
                "op": "softmax",
                "dtype": "float16",
                "case": {
                    "case_id": "tiny_logits",
                    "family": "lm_logits",
                    "source_kind": "synthetic_smoke",
                    "source_project": "cannbench",
                    "source_model": "smoke_fixture",
                    "source_file": "built-in",
                    "source_op": "softmax",
                    "payload": {"dimensions": [32, 128], "dim": -1},
                },
                "metrics": {
                    "iterations": 5,
                    "warmup": 3,
                    "latency_ms_avg": avg,
                    "latency_ms_p50": avg,
                    "latency_ms_p95": avg,
                    "latency_ms_p99": avg,
                    "throughput_ops_per_sec": 1000.0 / avg,
                },
            }
        )
        + "\n"
    )
    (path / "profile").mkdir()
    (path / "profile-summary.json").write_text(
        json.dumps(
            {
                "backend": backend,
                "sample_count": 2,
                "latency_ms_avg": avg + 0.1,
                "latency_ms_p50": avg + 0.1,
                "latency_ms_p95": avg + 0.2,
                "latency_ms_p99": avg + 0.3,
                "source_files": ["profile.csv"],
            }
        )
        + "\n"
    )


def test_write_local_report_summarizes_perf_accuracy_and_profile_paths(tmp_path):
    nvidia_dir = tmp_path / "nvidia"
    ascend_dir = tmp_path / "ascend"
    _write_perf(nvidia_dir, backend="nvidia", avg=1.2)
    _write_perf(ascend_dir, backend="ascend", avg=1.5)
    accuracy_path = tmp_path / "accuracy.json"
    accuracy_path.write_text(
        json.dumps(
            {
                "passed": True,
                "left_backend": "nvidia",
                "right_backend": "ascend",
                "op": "softmax",
                "case_id": "tiny_logits",
                "numel": 4096,
                "mismatch_count": 0,
                "max_abs_error": 0.001,
                "max_rel_error": 0.002,
                "mean_abs_error": 0.0001,
                "rmse": 0.0002,
                "rtol": 0.001,
                "atol": 0.001,
            }
        )
        + "\n"
    )

    report_path = write_local_report(
        output_path=tmp_path / "report.md",
        nvidia_dir=nvidia_dir,
        ascend_dir=ascend_dir,
        accuracy_path=accuracy_path,
    )

    report = report_path.read_text()
    assert "# CannBench Local Comparison Report" in report
    assert "| nvidia | Fake nvidia | softmax | tiny_logits | float16 | 1.2 |" in report
    assert "| ascend | Fake ascend | softmax | tiny_logits | float16 | 1.5 |" in report
    assert "| passed | True |" in report
    assert f"| nvidia_profile | {nvidia_dir / 'profile'} |" in report
    assert "| nvidia_device_latency_ms_avg | 1.3 |" in report
    assert "| ascend_device_latency_ms_avg | 1.6 |" in report
