import json

from cannbench.core.prepared_input import (
    OperatorInputBinding,
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
