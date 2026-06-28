from __future__ import annotations

import difflib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit


SENSITIVE_FIELDS = {
    "hostname",
    "ip",
    "username",
    "env",
    "command",
    "workdir",
    "path",
    "log",
    "stdout",
    "stderr",
    "source_code",
    "diff",
    "profile_raw",
}

RECORD_FIELDS = {
    "schema_version",
    "run_id",
    "operator",
    "dataset",
    "case_id",
    "family",
    "shape",
    "dtype",
    "backend",
    "device_class",
    "implementation",
    "implementation_version",
    "source_kind",
    "source_project",
    "source_model",
    "source_file",
    "source_op",
    "metrics",
    "accuracy",
    "diff_ref",
}

METRIC_FIELDS = {"latency_ms_avg", "latency_ms_p50", "latency_ms_p95", "sample_count"}
ACCURACY_FIELDS = {"passed", "max_abs_error", "max_rel_error"}
SAFE_COMPONENT_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")
CODE_LIKE_PATTERNS = (
    re.compile(r"```"),
    re.compile(r"diff --git ", re.IGNORECASE),
    re.compile(r'#include\s*[<"]', re.IGNORECASE),
    re.compile(r"\bTORCH_LIBRARY\b"),
    re.compile(r"\b__global__\b"),
    re.compile(r"\b__device__\b"),
    re.compile(r"\b__host__\b"),
    re.compile(r"\b__aicore__\b"),
    re.compile(r"(^|\n)\s*(def|class|import|from)\s+[A-Za-z_]", re.MULTILINE),
    re.compile(r"(^|\n)\s*function\s+[A-Za-z_][A-Za-z0-9_]*\s*\(", re.MULTILINE),
    re.compile(r"=>\s*[{(A-Za-z_]"),
    re.compile(r"(^|\n)\s*(const|let|var)\s+[A-Za-z_][A-Za-z0-9_]*\s*=", re.MULTILINE),
    re.compile(r"(^|\n)\s*template\s*<", re.MULTILINE),
)


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    accepted_count: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class SimtDiffResult:
    operator: str
    base_version: str
    compare_version: str
    patch: str


@dataclass(frozen=True)
class ServeConfig:
    frontend_dir: Path
    published_dir: Path
    host: str = "127.0.0.1"
    port: int = 8000
    enable_gpu_upload: bool = False


def _datasets_root() -> Path:
    return Path(__file__).resolve().parent / "datasets" / "data"


def _validate_component(value: str, field_name: str) -> str:
    if not SAFE_COMPONENT_RE.fullmatch(value):
        raise ValueError(f"{field_name} must match {SAFE_COMPONENT_RE.pattern}")
    return value


def _resolve_simt_operator_dir(operator: str, version: str, datasets_root: Path | None = None) -> Path:
    safe_operator = _validate_component(operator, "operator")
    safe_version = _validate_component(version, "version")
    root = (datasets_root or _datasets_root()).resolve()
    target = (root / safe_operator / "custom_ops" / "ascend" / safe_version).resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("resolved diff path escapes datasets root") from exc
    if not target.is_dir():
        raise FileNotFoundError(f"SIMT operator directory not found: {target}")
    return target


def list_simt_operator_versions(
    operator: str,
    datasets_root: Path | None = None,
) -> tuple[str, ...]:
    safe_operator = _validate_component(operator, "operator")
    root = (datasets_root or _datasets_root()).resolve()
    target = (root / safe_operator / "custom_ops" / "ascend").resolve()
    try:
        target.relative_to(root)
    except ValueError as exc:
        raise ValueError("resolved versions path escapes datasets root") from exc
    if not target.is_dir():
        return ()
    return tuple(
        sorted(
            path.name
            for path in target.iterdir()
            if path.is_dir() and path.name != "__pycache__" and not path.name.startswith(".")
        )
    )


def _read_text_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def _iter_version_files(version_dir: Path) -> dict[Path, Path]:
    files: dict[Path, Path] = {}
    for path in sorted(version_dir.rglob("*")):
        if not path.is_file():
            continue
        if "__pycache__" in path.parts:
            continue
        files[path.relative_to(version_dir)] = path
    return files


def build_simt_operator_diff(
    operator: str,
    base_version: str,
    compare_version: str,
    datasets_root: Path | None = None,
) -> SimtDiffResult:
    base_dir = _resolve_simt_operator_dir(operator, base_version, datasets_root)
    compare_dir = _resolve_simt_operator_dir(operator, compare_version, datasets_root)
    base_files = _iter_version_files(base_dir)
    compare_files = _iter_version_files(compare_dir)
    patch_chunks: list[str] = []

    logical_root = Path("src") / "cannbench" / "datasets" / "data" / operator / "custom_ops" / "ascend"
    for relative_path in sorted(set(base_files) | set(compare_files)):
        base_path = base_files.get(relative_path)
        compare_path = compare_files.get(relative_path)
        base_lines = _read_text_lines(base_path) if base_path else []
        compare_lines = _read_text_lines(compare_path) if compare_path else []
        if base_lines == compare_lines:
            continue
        logical_path = logical_root / relative_path
        patch_lines = list(
            difflib.unified_diff(
                base_lines,
                compare_lines,
                fromfile=f"a/{logical_path.as_posix()}",
                tofile=f"b/{logical_path.as_posix()}",
                lineterm="",
            )
        )
        if patch_lines:
            patch_chunks.append(
                "\n".join(
                    [
                        f"diff --git a/{logical_path.as_posix()} b/{logical_path.as_posix()}",
                        *patch_lines,
                    ]
                )
            )

    patch = ("\n".join(patch_chunks) + "\n") if patch_chunks else ""
    return SimtDiffResult(
        operator=operator,
        base_version=base_version,
        compare_version=compare_version,
        patch=patch,
    )


def _is_object(value: Any) -> bool:
    return isinstance(value, dict)


def _check_sensitive_fields(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_sensitive_fields(item, f"{path}[{index}]", errors)
        return
    if not _is_object(value):
        return
    for key, child in value.items():
        if str(key).lower() in SENSITIVE_FIELDS:
            errors.append(f"sensitive field rejected at {path}.{key}")
        _check_sensitive_fields(child, f"{path}.{key}", errors)


def _is_code_like_string(value: str) -> bool:
    if len(value) > 1600:
        return True
    if not re.search(r"[\n\r`#;{}<>()=]", value):
        return False
    return any(pattern.search(value) for pattern in CODE_LIKE_PATTERNS)


def _check_code_like_content(value: Any, path: str, errors: list[str]) -> None:
    if isinstance(value, str):
        if _is_code_like_string(value):
            errors.append(f"code-like content rejected at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _check_code_like_content(item, f"{path}[{index}]", errors)
        return
    if not _is_object(value):
        return
    for key, child in value.items():
        _check_code_like_content(child, f"{path}.{key}", errors)


def _reject_unknown_fields(value: dict[str, Any], allowed: set[str], path: str, errors: list[str]) -> None:
    for key in value:
        if key not in allowed:
            errors.append(f"{path}.{key} is not allowed")


def _require_string(value: Any, path: str, errors: list[str]) -> str | None:
    if not isinstance(value, str) or not value or len(value) > 160:
        errors.append(f"{path} must be a non-empty string up to 160 characters")
        return None
    return value


def _require_number(value: Any, path: str, errors: list[str]) -> float | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        errors.append(f"{path} must be a non-negative finite number")
        return None
    if value < 0:
        errors.append(f"{path} must be a non-negative finite number")
        return None
    return float(value)


def _validate_record(record: Any, index: int, errors: list[str]) -> None:
    path = f"records[{index}]"
    if not _is_object(record):
        errors.append(f"{path} must be an object")
        return

    _reject_unknown_fields(record, RECORD_FIELDS, path, errors)
    if record.get("schema_version") != 1:
        errors.append(f"{path}.schema_version must be 1")

    for key in [
        "run_id",
        "operator",
        "dataset",
        "case_id",
        "family",
        "dtype",
        "device_class",
        "implementation",
        "implementation_version",
        "source_kind",
        "source_project",
        "source_model",
        "source_file",
        "source_op",
    ]:
        _require_string(record.get(key), f"{path}.{key}", errors)

    if record.get("backend") not in {"nvidia", "gpu"}:
        errors.append(f"{path}.backend must be nvidia or gpu")

    if record.get("implementation") != "ncu":
        errors.append(f"{path}.implementation must be ncu")

    shape = record.get("shape")
    if not isinstance(shape, list) or not shape or len(shape) > 8:
        errors.append(f"{path}.shape must be a non-empty numeric array up to 8 dimensions")
    else:
        for dim_index, dimension in enumerate(shape):
            if not isinstance(dimension, int) or dimension <= 0:
                errors.append(f"{path}.shape[{dim_index}] must be a positive integer")

    metrics = record.get("metrics")
    if not _is_object(metrics):
        errors.append(f"{path}.metrics must be an object")
    else:
        _reject_unknown_fields(metrics, METRIC_FIELDS, f"{path}.metrics", errors)
        for key in METRIC_FIELDS:
            _require_number(metrics.get(key), f"{path}.metrics.{key}", errors)

    accuracy = record.get("accuracy")
    if not _is_object(accuracy):
        errors.append(f"{path}.accuracy must be an object")
    else:
        _reject_unknown_fields(accuracy, ACCURACY_FIELDS, f"{path}.accuracy", errors)
        if not isinstance(accuracy.get("passed"), bool):
            errors.append(f"{path}.accuracy.passed must be boolean")
        _require_number(accuracy.get("max_abs_error"), f"{path}.accuracy.max_abs_error", errors)
        _require_number(accuracy.get("max_rel_error"), f"{path}.accuracy.max_rel_error", errors)

    if record.get("diff_ref") is not None:
        errors.append(f"{path}.diff_ref must be null for uploaded GPU records")


def validate_gpu_benchmark_upload(payload: Any) -> ValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    _check_sensitive_fields(payload, "payload", errors)
    _check_code_like_content(payload, "payload", errors)

    if not _is_object(payload):
        return ValidationResult(False, 0, ("payload must be an object",), ())
    _reject_unknown_fields(payload, {"records"}, "payload", errors)

    records = payload.get("records")
    if not isinstance(records, list):
        errors.append("payload.records must be an array")
        return ValidationResult(False, 0, tuple(errors), tuple(warnings))
    if len(records) == 0 or len(records) > 10000:
        errors.append("payload.records must contain 1 to 10000 records")

    for index, record in enumerate(records):
        _validate_record(record, index, errors)

    return ValidationResult(
        ok=not errors,
        accepted_count=len(records) if not errors else 0,
        errors=tuple(errors),
        warnings=tuple(warnings),
    )


class CannBenchRequestHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args: Any, frontend_dir: Path, published_dir: Path, enable_gpu_upload: bool, **kwargs: Any):
        self._frontend_dir = frontend_dir
        self._published_dir = published_dir
        self._enable_gpu_upload = enable_gpu_upload
        super().__init__(*args, directory=str(frontend_dir), **kwargs)

    def translate_path(self, path: str) -> str:
        if path.startswith("/published/"):
            relative = path.removeprefix("/published/").lstrip("/")
            base = self._published_dir.resolve()
            target = (base / relative).resolve()
            try:
                target.relative_to(base)
            except ValueError:
                return str(base / "__not_found__")
            return str(target)
        return super().translate_path(path)

    def do_GET(self) -> None:
        parsed = urlsplit(self.path)
        if parsed.path == "/api/simt-versions":
            self._handle_simt_versions(parsed.query)
            return
        if parsed.path == "/api/simt-diff":
            self._handle_simt_diff(parsed.query)
            return
        if parsed.path == "/":
            self.path = "/index.html"
        else:
            self.path = parsed.path
        super().do_GET()

    def do_POST(self) -> None:
        if self.path != "/api/gpu-results":
            self.send_error(HTTPStatus.NOT_FOUND)
            return
        if not self._enable_gpu_upload:
            self.send_error(HTTPStatus.FORBIDDEN, "GPU upload is disabled")
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        if content_length <= 0 or content_length > 2 * 1024 * 1024:
            self.send_error(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "payload too large")
            return

        raw = self.rfile.read(content_length)
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self.send_error(HTTPStatus.BAD_REQUEST, "invalid JSON")
            return

        result = validate_gpu_benchmark_upload(payload)
        if not result.ok:
            self.send_error(HTTPStatus.BAD_REQUEST, "; ".join(result.errors))
            return

        uploads_dir = self._published_dir / "uploads"
        uploads_dir.mkdir(parents=True, exist_ok=True)
        output_path = uploads_dir / f"gpu-results-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
        output_path.write_text(json.dumps(payload, indent=2) + "\n")

        response = json.dumps({"ok": True, "path": str(output_path.relative_to(self._published_dir))}).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _handle_simt_versions(self, query: str) -> None:
        params = parse_qs(query, keep_blank_values=False)
        operator = params.get("operator", [None])[0]
        if not operator:
            self.send_error(HTTPStatus.BAD_REQUEST, "operator is required")
            return

        try:
            versions = list_simt_operator_versions(operator)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return

        response = json.dumps({"operator": operator, "versions": versions}).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)

    def _handle_simt_diff(self, query: str) -> None:
        params = parse_qs(query, keep_blank_values=False)
        operator = params.get("operator", [None])[0]
        base_version = params.get("base_version", [None])[0]
        compare_version = params.get("compare_version", [None])[0]
        if not operator or not base_version or not compare_version:
            self.send_error(
                HTTPStatus.BAD_REQUEST,
                "operator, base_version, and compare_version are required",
            )
            return

        try:
            diff = build_simt_operator_diff(operator, base_version, compare_version)
        except ValueError as exc:
            self.send_error(HTTPStatus.BAD_REQUEST, str(exc))
            return
        except FileNotFoundError as exc:
            self.send_error(HTTPStatus.NOT_FOUND, str(exc))
            return

        response = json.dumps(
            {
                "operator": diff.operator,
                "base_version": diff.base_version,
                "compare_version": diff.compare_version,
                "patch": diff.patch,
            }
        ).encode()
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(response)))
        self.end_headers()
        self.wfile.write(response)


def serve_cannbench(
    *,
    frontend_dir: Path,
    published_dir: Path,
    host: str = "127.0.0.1",
    port: int = 8000,
    enable_gpu_upload: bool = False,
) -> None:
    handler = partial(
        CannBenchRequestHandler,
        frontend_dir=frontend_dir,
        published_dir=published_dir,
        enable_gpu_upload=enable_gpu_upload,
    )
    server = ThreadingHTTPServer((host, port), handler)
    try:
        server.serve_forever()
    finally:
        server.server_close()
