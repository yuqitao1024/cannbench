from __future__ import annotations

import json
import math
import sys
from array import array
from pathlib import Path


_MANIFEST_COMPATIBILITY_KEYS = (
    "schema_version",
    "case_id",
    "phase",
    "seed",
    "canonical_input",
    "validation_query_rows",
)


def validation_query_rows(query_tokens: int, *, phase: str) -> tuple[int, ...]:
    if phase == "decode":
        return tuple(range(query_tokens))
    if phase != "prefill":
        raise ValueError(f"unsupported DSA phase: {phase}")
    if query_tokens <= 4:
        return tuple(range(query_tokens))
    step = (query_tokens - 1) // 3
    return (0, step, step * 2, query_tokens - 1)


def compare_topk_rows(reference, candidate, *, row_width: int) -> dict[str, float | int]:
    if row_width <= 0:
        raise ValueError("row_width must be positive")
    if len(reference) != len(candidate) or len(reference) % row_width != 0:
        raise ValueError("Top-K artifacts have incompatible lengths")
    recalls: list[float] = []
    jaccards: list[float] = []
    exact_rows = 0
    for start in range(0, len(reference), row_width):
        reference_set = {value for value in reference[start : start + row_width] if value >= 0}
        candidate_set = {value for value in candidate[start : start + row_width] if value >= 0}
        intersection = reference_set & candidate_set
        union = reference_set | candidate_set
        recalls.append(len(intersection) / len(reference_set) if reference_set else 1.0)
        jaccards.append(len(intersection) / len(union) if union else 1.0)
        exact_rows += reference_set == candidate_set
    return {
        "rows": len(recalls),
        "mean_recall": sum(recalls) / len(recalls) if recalls else 1.0,
        "min_recall": min(recalls, default=1.0),
        "mean_jaccard": sum(jaccards) / len(jaccards) if jaccards else 1.0,
        "min_jaccard": min(jaccards, default=1.0),
        "exact_set_rows": exact_rows,
    }


def compare_numeric_values(
    reference,
    candidate,
    *,
    atol: float,
    rtol: float,
) -> dict[str, float | int]:
    if len(reference) != len(candidate):
        raise ValueError("numeric artifacts have incompatible lengths")
    mismatches = 0
    max_abs_error = 0.0
    max_rel_error = 0.0
    for expected, actual in zip(reference, candidate, strict=True):
        if math.isnan(expected) or math.isnan(actual):
            mismatch = not (math.isnan(expected) and math.isnan(actual))
            abs_error = math.inf if mismatch else 0.0
            rel_error = abs_error
        elif math.isinf(expected) or math.isinf(actual):
            mismatch = expected != actual
            abs_error = math.inf if mismatch else 0.0
            rel_error = abs_error
        else:
            abs_error = abs(actual - expected)
            rel_error = abs_error / max(abs(expected), 1e-12)
            mismatch = abs_error > atol + rtol * abs(expected)
        mismatches += mismatch
        max_abs_error = max(max_abs_error, abs_error)
        max_rel_error = max(max_rel_error, rel_error)
    return {
        "numel": len(reference),
        "mismatch_count": mismatches,
        "max_abs_error": max_abs_error,
        "max_rel_error": max_rel_error,
        "atol": atol,
        "rtol": rtol,
    }


def ensure_compatible_manifests(reference: dict, candidate: dict) -> None:
    for key in _MANIFEST_COMPATIBILITY_KEYS:
        if reference.get(key) != candidate.get(key):
            raise ValueError(f"conformance manifest {key} mismatch")


def write_array(path: Path, typecode: str, values) -> None:
    payload = array(typecode, values)
    if sys.byteorder != "little":
        payload.byteswap()
    path.write_bytes(payload.tobytes())


def read_array(path: Path, typecode: str):
    payload = array(typecode)
    payload.frombytes(path.read_bytes())
    if sys.byteorder != "little":
        payload.byteswap()
    return payload


def write_manifest(directory: Path, manifest: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_manifest(directory: Path) -> dict:
    return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))


def compare_artifacts(
    reference_directory: Path,
    candidate_directory: Path,
    *,
    atol: float,
    rtol: float,
) -> dict:
    reference = load_manifest(reference_directory)
    candidate = load_manifest(candidate_directory)
    ensure_compatible_manifests(reference, candidate)
    top_k = int(reference["canonical_input"]["top_k"])
    result = {
        "reference_backend": reference["backend"],
        "candidate_backend": candidate["backend"],
        "case_id": reference["case_id"],
        "indexer": compare_topk_rows(
            read_array(reference_directory / "indexer_indices.i32", "i"),
            read_array(candidate_directory / "indexer_indices.i32", "i"),
            row_width=top_k,
        ),
        "attention": _compare_output_pair(
            reference_directory,
            candidate_directory,
            prefix="attention",
            reference_manifest=reference,
            candidate_manifest=candidate,
            atol=atol,
            rtol=rtol,
        ),
        "workflow": _compare_output_pair(
            reference_directory,
            candidate_directory,
            prefix="workflow",
            reference_manifest=reference,
            candidate_manifest=candidate,
            atol=atol,
            rtol=rtol,
        ),
    }
    return result


def conformance_passed(result: dict, *, min_indexer_recall: float) -> bool:
    indexer = result["indexer"]
    if (
        indexer["mean_recall"] < min_indexer_recall
        or indexer["min_recall"] < min_indexer_recall
    ):
        return False
    for section in ("attention", "workflow"):
        for output_name in ("output", "lse"):
            metrics = result[section][output_name]
            if metrics.get("mismatch_count") != 0:
                return False
    return True


def _compare_output_pair(
    reference_directory: Path,
    candidate_directory: Path,
    *,
    prefix: str,
    reference_manifest: dict,
    candidate_manifest: dict,
    atol: float,
    rtol: float,
) -> dict:
    result = {
        "output": compare_numeric_values(
            read_array(reference_directory / f"{prefix}_output.f32", "f"),
            read_array(candidate_directory / f"{prefix}_output.f32", "f"),
            atol=atol,
            rtol=rtol,
        )
    }
    reference_lse = reference_manifest["outputs"][prefix]["lse"]
    candidate_lse = candidate_manifest["outputs"][prefix]["lse"]
    if reference_lse == "available" and candidate_lse == "available":
        result["lse"] = compare_numeric_values(
            read_array(reference_directory / f"{prefix}_lse.f32", "f"),
            read_array(candidate_directory / f"{prefix}_lse.f32", "f"),
            atol=atol,
            rtol=rtol,
        )
    else:
        result["lse"] = {
            "status": "unavailable",
            "reference": reference_lse,
            "candidate": candidate_lse,
        }
    return result
