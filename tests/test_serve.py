from pathlib import Path

from cannbench.serve import build_simt_operator_diff, validate_gpu_benchmark_upload


def _valid_gpu_upload():
    return {
        "records": [
            {
                "schema_version": 1,
                "run_id": "softmax-h800",
                "operator": "softmax",
                "dataset": "realistic",
                "case_id": "t5_attention",
                "shape": [4, 8, 1024, 1024],
                "dtype": "float16",
                "backend": "nvidia",
                "device_class": "H800",
                "implementation": "ncu",
                "implementation_version": "ncu",
                "metrics": {
                    "latency_ms_avg": 1.0,
                    "latency_ms_p50": 1.0,
                    "latency_ms_p95": 1.1,
                    "sample_count": 1,
                },
                "accuracy": {
                    "passed": True,
                    "max_abs_error": 0.0,
                    "max_rel_error": 0.0,
                },
                "diff_ref": None,
            }
        ]
    }


def test_validate_gpu_benchmark_upload_accepts_minimal_gpu_record():
    result = validate_gpu_benchmark_upload(_valid_gpu_upload())

    assert result.ok is True
    assert result.accepted_count == 1
    assert result.errors == ()


def test_validate_gpu_benchmark_upload_rejects_sensitive_fields():
    payload = _valid_gpu_upload()
    payload["records"][0]["env"] = {"CUDA_VISIBLE_DEVICES": "0"}

    result = validate_gpu_benchmark_upload(payload)

    assert result.ok is False
    assert "sensitive field rejected at payload.records[0].env" in result.errors


def test_validate_gpu_benchmark_upload_rejects_non_gpu_backend():
    payload = _valid_gpu_upload()
    payload["records"][0]["backend"] = "ascend"

    result = validate_gpu_benchmark_upload(payload)

    assert result.ok is False
    assert "records[0].backend must be nvidia or gpu" in result.errors


def test_validate_gpu_benchmark_upload_rejects_code_snippet_in_allowed_field():
    payload = _valid_gpu_upload()
    payload["records"][0]["implementation_version"] = "def kernel(x):\n    return x\n"

    result = validate_gpu_benchmark_upload(payload)

    assert result.ok is False
    assert "code-like content rejected at payload.records[0].implementation_version" in result.errors


def test_build_simt_operator_diff_uses_real_version_directories(tmp_path: Path):
    datasets_root = tmp_path / "datasets"
    base_root = datasets_root / "softmax" / "custom_ops" / "ascend" / "dynamic-ubuf"
    compare_root = datasets_root / "softmax" / "custom_ops" / "ascend" / "tiled-v2"
    base_root.mkdir(parents=True)
    compare_root.mkdir(parents=True)

    shared_relative = Path("aten_softmax/csrc/simt/spatial_softmax.asc")
    (base_root / shared_relative).parent.mkdir(parents=True, exist_ok=True)
    (compare_root / shared_relative).parent.mkdir(parents=True, exist_ok=True)
    (base_root / shared_relative).write_text("alpha\nbeta\n", encoding="utf-8")
    (compare_root / shared_relative).write_text("alpha\ngamma\n", encoding="utf-8")

    diff = build_simt_operator_diff(
        "softmax",
        "dynamic-ubuf",
        "tiled-v2",
        datasets_root=datasets_root,
    )

    assert diff.operator == "softmax"
    assert diff.base_version == "dynamic-ubuf"
    assert diff.compare_version == "tiled-v2"
    assert diff.patch.startswith(
        "diff --git a/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc "
        "b/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc"
    )
    assert "src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc" in diff.patch
    assert "-beta" in diff.patch
    assert "+gamma" in diff.patch
