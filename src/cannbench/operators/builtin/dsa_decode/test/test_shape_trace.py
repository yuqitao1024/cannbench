from unittest.mock import Mock

from cannbench.operators.builtin.dsa_decode import shape_trace as shape_trace_module
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

    tensors = {
        tensor.id: tensor for stage in trace.stages for tensor in stage.tensors
    }
    assert tuple(axis.symbol for axis in tensors["index_key"].axes) == ("C", "Di")
    assert tuple(axis.symbol for axis in tensors["index_key_t"].axes) == ("Di", "C")
    assert all(
        tensors[tensor_id].logical_only
        for tensor_id in (
            "head_scores",
            "index_scores",
            "selected_k",
            "selected_v",
            "scores",
            "probabilities",
        )
    )


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


def test_device_trace_does_not_fallback_from_unregistered_latest_version(monkeypatch):
    discovered_ops = []

    def discover_versions(op):
        discovered_ops.append(op)
        return {"v2", "v99", "test"}

    v2_builder = Mock(wraps=shape_trace_module.DEVICE_TRACE_BUILDERS["v2"])
    monkeypatch.setattr(
        shape_trace_module, "_component_simt_versions", discover_versions
    )
    monkeypatch.setitem(shape_trace_module.DEVICE_TRACE_BUILDERS, "v2", v2_builder)

    device = build_dsa_decode_shape_trace("realistic", CASE_ID).device_execution

    assert discovered_ops == ["lightning_indexer", "sparse_attention"]
    assert device.status == "unavailable"
    assert device.version == "v99"
    assert device.message == "Device trace unavailable for v99."
    assert device.kernels == ()
    v2_builder.assert_not_called()


def test_device_trace_reports_when_components_have_no_common_version(monkeypatch):
    versions = {
        "lightning_indexer": {"v2", "v3"},
        "sparse_attention": {"v1", "v4"},
    }
    v2_builder = Mock(wraps=shape_trace_module.DEVICE_TRACE_BUILDERS["v2"])
    monkeypatch.setattr(
        shape_trace_module, "_component_simt_versions", versions.__getitem__
    )
    monkeypatch.setitem(shape_trace_module.DEVICE_TRACE_BUILDERS, "v2", v2_builder)

    device = build_dsa_decode_shape_trace("realistic", CASE_ID).device_execution

    assert device.status == "unavailable"
    assert device.version is None
    assert device.message == "No common SIMT version is available."
    assert device.kernels == ()
    v2_builder.assert_not_called()
