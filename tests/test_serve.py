from pathlib import Path

from cannbench.serve import (
    build_simt_operator_diff,
    CannBenchRequestHandler,
    list_simt_operator_versions,
    validate_gpu_benchmark_upload,
)


def _valid_gpu_upload():
    return {
        "records": [
            {
                "schema_version": 1,
                "run_id": "softmax-h800",
                "operator": "softmax",
                "dataset": "realistic",
                "case_id": "t5_attention",
                "family": "attention",
                "shape": [4, 8, 1024, 1024],
                "dtype": "float16",
                "backend": "nvidia",
                "device_class": "H800",
                "implementation": "cuda-pytorch",
                "implementation_version": "cuda-pytorch",
                "source_kind": "real_model",
                "source_project": "TritonBench",
                "source_model": "T5Small",
                "source_file": "hf_train/T5Small_train.json",
                "source_op": "aten._softmax.default",
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


def test_validate_gpu_benchmark_upload_rejects_non_cuda_pytorch_implementation():
    payload = _valid_gpu_upload()
    payload["records"][0]["implementation"] = "cuda_event"

    result = validate_gpu_benchmark_upload(payload)

    assert result.ok is False
    assert "records[0].implementation must be cuda-pytorch" in result.errors


def test_list_simt_operator_versions_returns_sorted_directory_names(tmp_path: Path):
    datasets_root = tmp_path / "datasets"
    simt_root = datasets_root / "softmax" / "simt"
    (simt_root / "v2").mkdir(parents=True)
    (simt_root / "v1").mkdir(parents=True)
    (simt_root / "test").mkdir(parents=True)
    (simt_root / "tests").mkdir(parents=True)
    (simt_root / "scripts").mkdir(parents=True)
    (simt_root / "__pycache__").mkdir(parents=True)
    (simt_root / ".tmp").mkdir(parents=True)

    versions = list_simt_operator_versions("softmax", datasets_root=datasets_root)

    assert versions == ("v1", "v2")


def test_build_simt_operator_diff_uses_real_version_directories(tmp_path: Path):
    datasets_root = tmp_path / "datasets"
    base_root = datasets_root / "softmax" / "simt" / "v1"
    compare_root = datasets_root / "softmax" / "simt" / "v2"
    base_root.mkdir(parents=True)
    compare_root.mkdir(parents=True)

    shared_relative = Path("aten_softmax/csrc/simt/spatial_softmax.asc")
    (base_root / shared_relative).parent.mkdir(parents=True, exist_ok=True)
    (compare_root / shared_relative).parent.mkdir(parents=True, exist_ok=True)
    (base_root / shared_relative).write_text("alpha\nbeta\n", encoding="utf-8")
    (compare_root / shared_relative).write_text("alpha\ngamma\n", encoding="utf-8")

    diff = build_simt_operator_diff(
        "softmax",
        "v1",
        "v2",
        datasets_root=datasets_root,
    )

    assert diff.operator == "softmax"
    assert diff.base_version == "v1"
    assert diff.compare_version == "v2"
    assert diff.patch.startswith(
        "diff --git a/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/spatial_softmax.asc "
        "b/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/spatial_softmax.asc"
    )
    assert "src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/spatial_softmax.asc" in diff.patch
    assert "-beta" in diff.patch
    assert "+gamma" in diff.patch


def test_build_simt_operator_diff_normalizes_version_project_directory_names(tmp_path: Path):
    datasets_root = tmp_path / "datasets"
    base_root = datasets_root / "softmax" / "simt" / "v1"
    compare_root = datasets_root / "softmax" / "simt" / "v2"

    base_file = base_root / "aten_softmax" / "csrc" / "simt" / "spatial_softmax.asc"
    compare_file = compare_root / "aten_softmax_v2" / "csrc" / "simt" / "spatial_softmax.asc"
    base_file.parent.mkdir(parents=True)
    compare_file.parent.mkdir(parents=True)
    base_file.write_text("alpha\nbeta\n", encoding="utf-8")
    compare_file.write_text("alpha\ngamma\n", encoding="utf-8")

    diff = build_simt_operator_diff(
        "softmax",
        "v1",
        "v2",
        datasets_root=datasets_root,
    )

    expected_path = "src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/spatial_softmax.asc"
    assert f"diff --git a/{expected_path} b/{expected_path}" in diff.patch
    assert "aten_softmax" not in diff.patch
    assert "-beta" in diff.patch
    assert "+gamma" in diff.patch


def test_request_handler_returns_runtime_config(tmp_path: Path):
    class _Headers(dict):
        def get(self, key, default=None):
            return super().get(key, default)

    class _FakeHandler:
        def __init__(self) -> None:
            self.path = "/api/config"
            self.headers = _Headers()
            self.wfile = type("Writer", (), {"write": lambda self, data: setattr(self, "data", data)})()
            self._frontend_dir = tmp_path
            self._published_dir = tmp_path
            self._enable_gpu_upload = True
            self.status = None
            self.sent_headers: list[tuple[str, str]] = []

        def send_response(self, status):
            self.status = status

        def send_header(self, key, value):
            self.sent_headers.append((key, value))

        def end_headers(self):
            return None

        def _handle_config(self):
            return CannBenchRequestHandler._handle_config(self)  # type: ignore[misc]

    handler = _FakeHandler()

    CannBenchRequestHandler.do_GET(handler)  # type: ignore[misc]

    assert handler.status == 200
    assert b'"gpu_upload_enabled": true' in handler.wfile.data
