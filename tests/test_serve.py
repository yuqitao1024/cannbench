from http import HTTPStatus
from io import BytesIO
from pathlib import Path

import cannbench.serve as serve
import pytest
from cannbench.operators.shape_trace import ShapeTraceKey
from cannbench.serve import (
    build_shape_trace_payload,
    build_simt_operator_diff,
    CannBenchRequestHandler,
    list_shape_trace_payloads,
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
                    "latency_ms": 1.0,
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


def test_shape_trace_index_uses_plugin_hooks(monkeypatch):
    key = ShapeTraceKey("example", "realistic", "case", "decode", "group")
    plugin = type(
        "Plugin", (), {"list_shape_trace_cases": staticmethod(lambda: (key,))}
    )()
    monkeypatch.setattr(serve, "list_operator_plugins", lambda: (plugin,))

    assert list_shape_trace_payloads() == [
        {
            "operator": "example",
            "dataset": "realistic",
            "case_id": "case",
            "phase": "decode",
            "group": "group",
        }
    ]


def test_shape_trace_payload_rejects_plugin_without_hook(monkeypatch):
    plugin = type("Plugin", (), {"build_shape_trace": None})()
    monkeypatch.setattr(serve, "get_operator_plugin", lambda name: plugin)

    with pytest.raises(LookupError, match="shape trace is not available"):
        build_shape_trace_payload("example", "realistic", "case")


def test_shape_trace_payload_calls_generic_plugin_hook(monkeypatch):
    trace = object()
    plugin = type(
        "Plugin",
        (),
        {"build_shape_trace": staticmethod(lambda dataset, case: trace)},
    )()
    monkeypatch.setattr(serve, "get_operator_plugin", lambda name: plugin)
    monkeypatch.setattr(
        serve, "shape_trace_to_payload", lambda value: {"case_id": "case"}
    )

    assert build_shape_trace_payload("example", "realistic", "case")["case_id"] == "case"


class _FakeGetHandler(CannBenchRequestHandler):
    def __init__(self, path: str, tmp_path: Path) -> None:
        self.path = path
        self.headers = {}
        self.wfile = BytesIO()
        self._frontend_dir = tmp_path
        self._published_dir = tmp_path
        self._enable_gpu_upload = False
        self.status = None
        self.error_message = None
        self.sent_headers: list[tuple[str, str]] = []

    def send_response(self, status, message=None):
        self.status = status

    def send_header(self, key, value):
        self.sent_headers.append((key, value))

    def end_headers(self):
        return None

    def send_error(self, code, message=None, explain=None):
        self.status = code
        self.error_message = message

    def send_head(self):
        self.status = HTTPStatus.NOT_FOUND
        return None


def _get(path: str, tmp_path: Path) -> _FakeGetHandler:
    handler = _FakeGetHandler(path, tmp_path)
    handler.do_GET()
    return handler


def test_request_handler_returns_shape_trace_index(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        serve,
        "list_shape_trace_payloads",
        lambda: [{"operator": "example", "case_id": "case"}],
    )

    handler = _get("/api/shape-traces", tmp_path)

    assert handler.status == HTTPStatus.OK
    assert b'"traces": [{"operator": "example", "case_id": "case"}]' in (
        handler.wfile.getvalue()
    )


def test_request_handler_returns_shape_trace_detail(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(
        serve,
        "build_shape_trace_payload",
        lambda operator, dataset, case_id: {
            "operator": operator,
            "dataset": dataset,
            "case_id": case_id,
        },
    )

    handler = _get(
        "/api/shape-trace?operator=example&dataset=realistic&case=case", tmp_path
    )

    assert handler.status == HTTPStatus.OK
    assert b'"operator": "example"' in handler.wfile.getvalue()


def test_request_handler_rejects_missing_shape_trace_parameters(tmp_path: Path):
    handler = _get("/api/shape-trace?operator=example", tmp_path)

    assert handler.status == HTTPStatus.BAD_REQUEST


@pytest.mark.parametrize("field", ["operator", "dataset", "case"])
def test_request_handler_rejects_invalid_shape_trace_components(
    field: str, tmp_path: Path
):
    params = {"operator": "example", "dataset": "realistic", "case": "case"}
    params[field] = "../unsafe"
    query = "&".join(f"{key}={value}" for key, value in params.items())

    handler = _get(f"/api/shape-trace?{query}", tmp_path)

    assert handler.status == HTTPStatus.BAD_REQUEST


def test_request_handler_returns_not_found_for_unknown_shape_trace(
    tmp_path: Path, monkeypatch
):
    def _raise_unknown(operator, dataset, case_id):
        raise ValueError(f"Unsupported operator: {operator}")

    monkeypatch.setattr(serve, "build_shape_trace_payload", _raise_unknown)

    handler = _get(
        "/api/shape-trace?operator=unknown&dataset=x&case=y", tmp_path
    )

    assert handler.status == HTTPStatus.NOT_FOUND
    assert handler.error_message == "Unsupported operator: unknown"


def test_request_handler_returns_not_found_for_unavailable_shape_trace(
    tmp_path: Path, monkeypatch
):
    def _raise_unavailable(operator, dataset, case_id):
        raise LookupError(f"shape trace is not available for operator: {operator}")

    monkeypatch.setattr(serve, "build_shape_trace_payload", _raise_unavailable)

    handler = _get(
        "/api/shape-trace?operator=example&dataset=x&case=y", tmp_path
    )

    assert handler.status == HTTPStatus.NOT_FOUND
    assert handler.error_message == "shape trace is not available for operator: example"


def test_request_handler_rewrites_shape_explorer_to_spa(tmp_path: Path):
    handler = _get("/shape-explorer", tmp_path)

    assert handler.path == "/index.html"


def test_validate_gpu_benchmark_upload_accepts_minimal_gpu_record():
    result = validate_gpu_benchmark_upload(_valid_gpu_upload())

    assert result.ok is True
    assert result.accepted_count == 1
    assert result.errors == ()


def test_validate_gpu_benchmark_upload_accepts_cuda_library_record():
    payload = _valid_gpu_upload()
    payload["records"][0]["run_id"] = "opbench-nvidia-h800-cuda-library-dsa_decode-realistic-bfloat16"
    payload["records"][0]["operator"] = "dsa_decode"
    payload["records"][0]["dtype"] = "bfloat16"
    payload["records"][0]["implementation"] = "cuda_library"
    payload["records"][0]["implementation_version"] = "cuda-library"

    result = validate_gpu_benchmark_upload(payload)

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
    assert "records[0].implementation must be cuda-pytorch or cuda_library" in result.errors


def test_list_simt_operator_versions_returns_sorted_directory_names(
    tmp_path: Path, monkeypatch
):
    builtin_root = tmp_path / "operators" / "builtin"
    simt_root = builtin_root / "softmax" / "simt"
    (simt_root / "v2").mkdir(parents=True)
    (simt_root / "v1").mkdir(parents=True)
    (simt_root / "test").mkdir(parents=True)
    (simt_root / "tests").mkdir(parents=True)
    (simt_root / "scripts").mkdir(parents=True)
    (simt_root / "__pycache__").mkdir(parents=True)
    (simt_root / ".tmp").mkdir(parents=True)

    monkeypatch.setattr(serve, "_operators_builtin_root", lambda: builtin_root)

    versions = list_simt_operator_versions("softmax")

    assert versions == ("v1", "v2")


def test_build_simt_operator_diff_uses_real_version_directories(
    tmp_path: Path, monkeypatch
):
    builtin_root = tmp_path / "operators" / "builtin"
    base_root = builtin_root / "softmax" / "simt" / "v1"
    compare_root = builtin_root / "softmax" / "simt" / "v2"
    base_root.mkdir(parents=True)
    compare_root.mkdir(parents=True)

    shared_relative = Path("aten_softmax/csrc/simt/spatial_softmax.asc")
    (base_root / shared_relative).parent.mkdir(parents=True, exist_ok=True)
    (compare_root / shared_relative).parent.mkdir(parents=True, exist_ok=True)
    (base_root / shared_relative).write_text("alpha\nbeta\n", encoding="utf-8")
    (compare_root / shared_relative).write_text("alpha\ngamma\n", encoding="utf-8")
    monkeypatch.setattr(serve, "_operators_builtin_root", lambda: builtin_root)

    diff = build_simt_operator_diff("softmax", "v1", "v2")

    assert diff.operator == "softmax"
    assert diff.base_version == "v1"
    assert diff.compare_version == "v2"
    expected_path = (
        "src/cannbench/operators/builtin/softmax/simt/softmax/csrc/simt/"
        "spatial_softmax.asc"
    )
    assert diff.patch.startswith(
        f"diff --git a/{expected_path} b/{expected_path}"
    )
    assert "-beta" in diff.patch
    assert "+gamma" in diff.patch


def test_build_simt_operator_diff_normalizes_version_project_directory_names(
    tmp_path: Path, monkeypatch
):
    builtin_root = tmp_path / "operators" / "builtin"
    base_root = builtin_root / "softmax" / "simt" / "v1"
    compare_root = builtin_root / "softmax" / "simt" / "v2"

    base_file = base_root / "aten_softmax" / "csrc" / "simt" / "spatial_softmax.asc"
    compare_file = compare_root / "aten_softmax_v2" / "csrc" / "simt" / "spatial_softmax.asc"
    base_file.parent.mkdir(parents=True)
    compare_file.parent.mkdir(parents=True)
    base_file.write_text("alpha\nbeta\n", encoding="utf-8")
    compare_file.write_text("alpha\ngamma\n", encoding="utf-8")
    monkeypatch.setattr(serve, "_operators_builtin_root", lambda: builtin_root)

    diff = build_simt_operator_diff("softmax", "v1", "v2")

    expected_path = "src/cannbench/operators/builtin/softmax/simt/softmax/csrc/simt/spatial_softmax.asc"
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
