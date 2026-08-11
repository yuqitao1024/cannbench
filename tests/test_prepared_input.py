import json

import pytest

from cannbench.core.prepared_input import (
    OperatorInputBinding,
    PreparedOperatorInput,
    PreparedWorkflowInput,
    PreparedWorkflowStep,
    build_prepared_operator_input,
    read_prepared_operator_input,
    read_prepared_workflow_input,
    prepare_workflow_input,
    write_prepared_operator_input,
    write_prepared_workflow_input,
)
from cannbench.operators.builtin.dsa_decode import build_dsa_decode_workflow


def test_build_prepared_operator_input_resolves_softmax_case_metadata():
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )

    assert prepared.op == "softmax"
    assert prepared.dtype == "float16"
    assert prepared.dataset == "smoke"
    assert prepared.seed == 7
    assert prepared.case.case_id == "tiny_logits"
    assert prepared.case.payload == {
        "dimensions": (32, 128),
        "dim": -1,
    }


def test_prepared_operator_input_json_roundtrip(tmp_path):
    prepared = PreparedOperatorInput(
        op="softmax",
        dtype="float16",
        dataset="realistic",
        seed=11,
        case=build_prepared_operator_input(
            op="softmax",
            dtype="float16",
            dataset="realistic",
            case_id="t5_attention",
            seed=11,
        ).case,
        input_bindings={
            "indices": OperatorInputBinding(
                op="lightning_indexer",
                dtype="bfloat16",
                dataset="realistic_decode",
                case_id="deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
                seed=11,
            )
        },
    )

    path = tmp_path / "prepared-softmax.json"
    write_prepared_operator_input(path, prepared)

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["dataset"] == "realistic"
    assert payload["case"]["case_id"] == "t5_attention"
    assert payload["input_bindings"]["indices"] == {
        "op": "lightning_indexer",
        "dtype": "bfloat16",
        "dataset": "realistic_decode",
        "case_id": "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
        "seed": 11,
        "output_index": 0,
    }

    loaded = read_prepared_operator_input(path)

    assert loaded == prepared


def test_prepared_operator_input_loads_legacy_manifest_without_bindings(tmp_path):
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=0,
    )
    payload = prepared.to_json_dict()
    payload.pop("input_bindings")
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))

    loaded = read_prepared_operator_input(path)

    assert loaded.input_bindings == {}


def test_prepared_workflow_input_json_roundtrip_preserves_order_and_bindings(tmp_path):
    workflow = build_dsa_decode_workflow(
        dataset="realistic",
        case_id="deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
        dtype="bfloat16",
        seed=7,
    )
    prepared = prepare_workflow_input(workflow)
    path = tmp_path / "prepared-workflow.json"

    write_prepared_workflow_input(path, prepared)
    loaded = read_prepared_workflow_input(path)

    assert loaded == prepared
    assert [step.contract for step in loaded.steps] == [
        "dsa_index_select",
        "sparse_mla_decode",
    ]
    assert [step.prepared.op for step in loaded.steps] == [
        "lightning_indexer",
        "sparse_attention",
    ]
    assert loaded.steps[0].produces == ("indices",)
    assert loaded.steps[1].consumes == ("indices",)
    assert all(not step.prepared.input_bindings for step in loaded.steps)


def test_prepared_workflow_input_rejects_empty_steps():
    with pytest.raises(ValueError, match="at least one step"):
        PreparedWorkflowInput(
            workflow="test_workflow",
            phase="decode",
            dataset="smoke",
            case_id="case",
            steps=(),
        )


def test_prepared_workflow_input_rejects_missing_consumed_output():
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=0,
    )

    with pytest.raises(ValueError, match="consumes unknown output 'indices'"):
        PreparedWorkflowInput(
            workflow="test_workflow",
            phase="decode",
            dataset="smoke",
            case_id="tiny_logits",
            steps=(
                PreparedWorkflowStep(
                    contract="consumer",
                    consumes=("indices",),
                    produces=("output",),
                    prepared=prepared,
                ),
            ),
        )


def test_prepared_workflow_input_rejects_duplicate_produced_output():
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=0,
    )

    with pytest.raises(ValueError, match="produces duplicate output 'shared'"):
        PreparedWorkflowInput(
            workflow="test_workflow",
            phase="decode",
            dataset="smoke",
            case_id="tiny_logits",
            steps=(
                PreparedWorkflowStep(
                    contract="producer-one",
                    consumes=(),
                    produces=("shared",),
                    prepared=prepared,
                ),
                PreparedWorkflowStep(
                    contract="producer-two",
                    consumes=("shared",),
                    produces=("shared",),
                    prepared=prepared,
                ),
            ),
        )


def test_prepared_workflow_input_rejects_component_case_mismatch():
    prepared = build_prepared_operator_input(
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=0,
    )

    with pytest.raises(ValueError, match="component case_id mismatch"):
        PreparedWorkflowInput(
            workflow="test_workflow",
            phase="decode",
            dataset="smoke",
            case_id="different-case",
            steps=(
                PreparedWorkflowStep(
                    contract="producer",
                    consumes=(),
                    produces=("output",),
                    prepared=prepared,
                ),
            ),
        )
