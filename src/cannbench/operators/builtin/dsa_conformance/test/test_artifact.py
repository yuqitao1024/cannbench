from array import array

import pytest

from cannbench.operators.builtin.dsa_conformance.artifact import (
    compare_numeric_values,
    compare_topk_rows,
    conformance_passed,
    ensure_compatible_manifests,
    validation_query_rows,
)


def test_prefill_validation_rows_span_full_query_range():
    assert validation_query_rows(4096, phase="prefill") == (0, 1365, 2730, 4095)


def test_decode_validation_rows_include_every_query():
    assert validation_query_rows(2, phase="decode") == (0, 1)


def test_topk_comparison_reports_recall_and_jaccard_per_row():
    reference = array("i", [1, 2, 3, 4, 10, 11, -1, -1])
    candidate = array("i", [1, 2, 8, 9, 10, 12, -1, -1])

    metrics = compare_topk_rows(reference, candidate, row_width=4)

    assert metrics["rows"] == 2
    assert metrics["mean_recall"] == pytest.approx(0.5)
    assert metrics["min_recall"] == pytest.approx(0.5)
    assert metrics["mean_jaccard"] == pytest.approx((2 / 6 + 1 / 3) / 2)


def test_numeric_comparison_uses_combined_absolute_and_relative_tolerance():
    metrics = compare_numeric_values(
        array("f", [1.0, 10.0, -2.0]),
        array("f", [1.04, 10.4, -2.2]),
        atol=0.05,
        rtol=0.05,
    )

    assert metrics["mismatch_count"] == 1
    assert metrics["max_abs_error"] == pytest.approx(0.4)
    assert metrics["numel"] == 3


def test_manifest_compatibility_rejects_different_canonical_inputs():
    reference = {"schema_version": 1, "case_id": "v32", "seed": 7}
    candidate = {"schema_version": 1, "case_id": "v32", "seed": 8}

    with pytest.raises(ValueError, match="seed"):
        ensure_compatible_manifests(reference, candidate)


def test_conformance_gate_requires_recall_and_all_numeric_outputs():
    result = {
        "indexer": {"mean_recall": 0.99, "min_recall": 0.98},
        "attention": {
            "output": {"mismatch_count": 0},
            "lse": {"mismatch_count": 0},
        },
        "workflow": {
            "output": {"mismatch_count": 1},
            "lse": {"mismatch_count": 0},
        },
    }

    assert not conformance_passed(result, min_indexer_recall=0.95)
    result["workflow"]["output"]["mismatch_count"] = 0
    assert conformance_passed(result, min_indexer_recall=0.95)
