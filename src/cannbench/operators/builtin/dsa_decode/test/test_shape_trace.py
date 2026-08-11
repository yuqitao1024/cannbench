from cannbench.operators.builtin.dsa_decode.shape_trace import (
    build_dsa_decode_shape_trace,
    latest_common_simt_version,
    latest_numeric_common_version,
    list_dsa_decode_shape_trace_cases,
)

CASE_ID = "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048"


def test_decode_trace_derives_canonical_symbols_and_stage_order():
    trace = build_dsa_decode_shape_trace("realistic", CASE_ID)
    symbols = {axis.symbol: axis.value for axis in trace.symbols}
    assert symbols == {
        "B": 2,
        "Q": 2,
        "R": 4,
        "Hi": 64,
        "Di": 128,
        "C": 32768,
        "H": 128,
        "Hkv": 1,
        "S": 2048,
        "Dqk": 576,
        "Dv": 512,
    }
    assert [stage.id for stage in trace.stages] == [
        "index-inputs",
        "index-matmul",
        "index-reduce",
        "topk",
        "gather",
        "qk",
        "softmax",
        "pv-output",
    ]
    assert trace.stages[1].formula == "[Hi,Di] x [Di,C] -> [Hi,C]"
    assert trace.stages[5].contracted_axes == ("Dqk",)
    assert trace.stages[7].contracted_axes == ("S",)


def test_decode_trace_lists_only_v32_realistic_case():
    keys = list_dsa_decode_shape_trace_cases()
    assert [(key.dataset, key.case_id, key.phase, key.group) for key in keys] == [
        ("realistic", CASE_ID, "decode", "deepseek-v32"),
    ]


def test_numeric_version_selection_uses_intersection_and_numeric_order():
    assert latest_numeric_common_version(
        (
            {"v1", "v2", "v10", "test"},
            {"v2", "v10", "v11"},
        )
    ) == "v10"
    assert latest_numeric_common_version(({"v1"}, {"v2"})) is None


def test_latest_common_simt_version_discovers_component_directories():
    assert latest_common_simt_version(
        ("lightning_indexer", "sparse_attention")
    ) == "v2"
