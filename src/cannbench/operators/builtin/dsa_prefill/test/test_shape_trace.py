from cannbench.operators.builtin.dsa_prefill.shape_trace import (
    build_dsa_prefill_shape_trace,
    list_dsa_prefill_shape_trace_cases,
)
from cannbench.operators.builtin.dsa_decode.shape_trace import (
    build_dsa_decode_shape_trace,
)

CASE_ID = "deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048"


def test_prefill_trace_derives_aggregate_matrix_shapes():
    trace = build_dsa_prefill_shape_trace("realistic", CASE_ID)
    symbols = {axis.symbol: axis.value for axis in trace.symbols}
    assert symbols["R"] == 4096
    assert symbols["C"] == 32768
    tensors = {
        tensor.id: tuple(axis.value for axis in tensor.axes)
        for stage in trace.stages
        for tensor in stage.tensors
    }
    assert tensors["head_scores_all"] == (4096, 64, 32768)
    assert tensors["indices_all"] == (4096, 2048)
    assert tensors["scores_all"] == (4096, 128, 2048)
    assert tensors["output_all"] == (4096, 128, 512)


def test_prefill_trace_has_no_decode_device_layout():
    trace = build_dsa_prefill_shape_trace("realistic", CASE_ID)
    assert trace.device_execution.status == "unavailable"
    assert trace.device_execution.version is None
    assert trace.device_execution.kernels == ()
    assert trace.device_execution.message == (
        "Prefill is not optimized yet. This view intentionally shows only "
        "the algorithm-level matrix flow."
    )


def test_prefill_trace_lists_only_v32_realistic_case():
    keys = list_dsa_prefill_shape_trace_cases()
    assert [(key.dataset, key.case_id, key.phase, key.group) for key in keys] == [
        ("realistic", CASE_ID, "prefill", "deepseek-v32"),
    ]


def test_prefill_topk_insight_describes_canonical_causal_rules_only_for_prefill():
    prefill = build_dsa_prefill_shape_trace("realistic", CASE_ID)
    topk_insight = next(stage.insight for stage in prefill.stages if stage.id == "topk")
    assert "q=0..4095" in topk_insight
    assert "position(q)=28672+q" in topk_insight
    assert "valid_length(q)=min(32768,28672+q+1)" in topk_insight
    assert "28673..32768" in topk_insight
    assert "Indexer masks c >= valid_length(q)" in topk_insight
    assert "Attention masks index > position(q)" in topk_insight

    decode = build_dsa_decode_shape_trace(
        "realistic", "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048"
    )
    decode_insights = " ".join(stage.insight for stage in decode.stages)
    assert "position(q)=28672+q" not in decode_insights
    assert "valid_length(q)" not in decode_insights
