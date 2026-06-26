from __future__ import annotations

import json
from pathlib import Path

import pytest

from cannbench.core.batch import (
    DEFAULT_DATASET_SPLITS,
    BatchFailureRecord,
    BatchResultRecord,
    expand_prepared_input_plans,
    write_batch_failures_json,
    write_batch_summary_csv,
    write_batch_summary_json,
)
from cannbench.core.prepared_input import (
    build_prepared_operator_input,
    write_prepared_operator_input,
)
from cannbench.datasets import get_operator_dataset


def test_expand_direct_selection_defaults_to_all_splits_and_float16():
    dataset = get_operator_dataset("softmax")
    expected = [
        (split, case.case_id, "float16")
        for split in DEFAULT_DATASET_SPLITS
        for case in dataset.get(split).cases
    ]

    plans = expand_prepared_input_plans(op="softmax")

    assert [(plan.dataset, plan.case_id, plan.dtype) for plan in plans] == expected


def test_expand_direct_selection_with_case_id_returns_single_item():
    plans = expand_prepared_input_plans(
        op="softmax",
        dataset="smoke",
        case_id="tiny_logits",
        dtype="float16",
    )

    assert len(plans) == 1
    assert plans[0].dataset == "smoke"
    assert plans[0].case_id == "tiny_logits"


def test_expand_single_prepared_input_preserves_single_item(tmp_path):
    prepared_path = tmp_path / "prepared.json"
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )
    write_prepared_operator_input(prepared_path, prepared)

    plans = expand_prepared_input_plans(op="softmax", prepared_input=prepared_path)

    assert len(plans) == 1
    assert plans[0].source_path == prepared_path
    assert plans[0].seed == 7


def test_expand_prepared_dir_sorts_json_files(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    alpha = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=1,
    )
    beta = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="stress",
        case_id="wide_vocab_lm_logits",
        seed=2,
    )
    write_prepared_operator_input(prepared_dir / "b.json", beta)
    write_prepared_operator_input(prepared_dir / "a.json", alpha)

    plans = expand_prepared_input_plans(op="softmax", prepared_dir=prepared_dir)

    assert [plan.source_path.name for plan in plans] == ["a.json", "b.json"]


def test_expand_prepared_dir_rejects_empty_directory(tmp_path):
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir()

    with pytest.raises(
        ValueError,
        match="prepared input directory does not contain any json manifests",
    ):
        expand_prepared_input_plans(op="softmax", prepared_dir=prepared_dir)


def test_expand_prepared_input_rejects_operator_mismatch(tmp_path):
    prepared_path = tmp_path / "prepared.json"
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )
    write_prepared_operator_input(prepared_path, prepared)

    with pytest.raises(ValueError, match="prepared input operator mismatch"):
        expand_prepared_input_plans(op="embedding", prepared_input=prepared_path)


def test_write_batch_summary_and_failures(tmp_path):
    summary_path = tmp_path / "summary.json"
    csv_path = tmp_path / "summary.csv"
    failures_path = tmp_path / "failures.json"

    rows = [
        BatchResultRecord(
            op="softmax",
            dataset="smoke",
            case_id="tiny_logits",
            dtype="float16",
            seed=7,
            status="ok",
            prepared_input="prepared/tiny.json",
        )
    ]
    failures = [
        BatchFailureRecord(
            op="softmax",
            dataset="stress",
            case_id="wide_vocab_logits",
            dtype="float16",
            seed=7,
            prepared_input="prepared/wide.json",
            error="boom",
        )
    ]

    write_batch_summary_json(summary_path, rows)
    write_batch_summary_csv(csv_path, rows)
    write_batch_failures_json(failures_path, failures)

    assert json.loads(summary_path.read_text())["results"][0]["case_id"] == "tiny_logits"
    assert "tiny_logits" in csv_path.read_text()
    assert json.loads(failures_path.read_text())["failures"][0]["error"] == "boom"
