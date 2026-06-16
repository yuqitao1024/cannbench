import json

from cannbench.core.prepared_input import (
    PreparedOperatorInput,
    build_prepared_operator_input,
    read_prepared_operator_input,
    write_prepared_operator_input,
)


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
    )

    path = tmp_path / "prepared-softmax.json"
    write_prepared_operator_input(path, prepared)

    payload = json.loads(path.read_text())
    assert payload["schema_version"] == 1
    assert payload["dataset"] == "realistic"
    assert payload["case"]["case_id"] == "t5_attention"

    loaded = read_prepared_operator_input(path)

    assert loaded == prepared
