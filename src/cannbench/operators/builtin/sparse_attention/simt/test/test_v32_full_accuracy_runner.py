import inspect
from pathlib import Path

import pytest

from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)
import head64_reduced_accuracy as reduced_accuracy
import v32_full_accuracy as full_accuracy
from v32_full_accuracy import (
    deterministic_pattern,
    resize_prefill_case,
    validation_query_rows,
)
from v32_prefill_benchmark import summarize_samples


def test_prefill_validation_rows_cover_sequence_boundaries():
    assert validation_query_rows(4096, phase="prefill") == (0, 1365, 2730, 4095)


def test_decode_validation_rows_cover_every_query():
    assert validation_query_rows(2, phase="decode") == (0, 1)


def test_validation_rows_reject_unknown_phase():
    with pytest.raises(ValueError, match="phase"):
        validation_query_rows(2, phase="unknown")


def test_deterministic_values_repeat_every_257_elements():
    pattern = deterministic_pattern(seed=11, nonnegative=False)

    assert len(pattern) == 257
    assert pattern[0] == ((11 * 29) % 257 - 128.0) / 128.0
    assert pattern[1] == ((17 + 11 * 29) % 257 - 128.0) / 128.0


def test_resize_prefill_case_preserves_right_aligned_causal_metadata():
    case = get_sparse_attention_case(
        "realistic_prefill",
        "deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048",
    )

    resized = resize_prefill_case(case, 256)

    assert resized.query_tokens == 256
    assert resized.resolved_query_lens == (256,)
    assert resized.resolved_query_start_positions == (32768 - 256,)


def test_resize_prefill_case_rejects_non_prefill_case():
    case = get_sparse_attention_case(
        "realistic_decode",
        "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
    )

    with pytest.raises(ValueError, match="prefill"):
        resize_prefill_case(case, 1)


def _row_categories(row, *, context: int, causal_limit: int):
    return {
        "negative": sum(index < 0 for index in row),
        "out_of_range": sum(index >= context for index in row),
        "valid_past": sum(0 <= index < causal_limit for index in row),
        "valid_future": sum(causal_limit <= index < context for index in row),
    }


def test_reduced_prefill_has_q4_causal_rows_with_future_and_invalid_indices():
    case = next(
        case
        for case in reduced_accuracy.CASES
        if case["name"] == "causal_q4_c256_s64"
    )

    assert case == {
        "name": "causal_q4_c256_s64",
        "query_tokens": 4,
        "context": 256,
        "selected": 64,
        "mode": "mixed",
        "causal": True,
    }
    rows = reduced_accuracy.index_rows(case)
    for query_index in (0, 1, 3):
        causal_limit = case["context"] - case["query_tokens"] + query_index + 1
        row = rows[query_index]
        categories = _row_categories(
            row,
            context=case["context"],
            causal_limit=causal_limit,
        )
        assert categories["negative"] > 0
        assert categories["out_of_range"] > 0
        assert categories["valid_past"] > 0
        if query_index < case["query_tokens"] - 1:
            assert categories["valid_future"] > 0
        else:
            assert categories["valid_future"] == 0
            assert case["context"] - 1 in row


def test_full_causal_boundary_injection_is_explicit_and_covers_sampled_rows():
    signature = inspect.signature(full_accuracy._build_indices)
    assert signature.parameters["inject_causal_boundaries"].default is False
    builder_source = inspect.getsource(full_accuracy._build_indices)
    assert "values = causal_boundary_values(" in builder_source
    assert "indices[batch_index, query_index, : len(values)]" in builder_source

    for query_index in validation_query_rows(4, phase="prefill"):
        causal_limit = 8 - 4 + query_index + 1
        row = full_accuracy.causal_boundary_values(
            context=8, causal_limit=causal_limit
        )
        categories = _row_categories(
            row,
            context=8,
            causal_limit=causal_limit,
        )
        assert categories["negative"] > 0
        assert categories["out_of_range"] > 0
        assert categories["valid_past"] > 0
        if query_index < 3:
            assert categories["valid_future"] > 0
        else:
            assert categories["valid_future"] == 0
            assert 7 in row

    accuracy_source = inspect.getsource(full_accuracy.run_case)
    benchmark_source = Path(__file__).with_name("v32_prefill_benchmark.py").read_text()
    assert "inject_causal_boundaries=not runtime_only" in accuracy_source
    assert "inject_causal_boundaries" not in benchmark_source


def test_chunked_reference_masks_invalid_and_causal_future_before_softmax():
    source = inspect.getsource(full_accuracy._chunked_reference_metrics)

    safe_gather = source.index("safe_indices = row_indices.clamp")
    gather = source.index("kv.index_select(0, safe_indices)")
    causal_mask = source.index("valid_indices &= row_indices < causal_limit")
    mask_scores = source.index("scores = scores.masked_fill(")
    assert "~valid_indices[None, :]" in source[mask_scores:]
    softmax = source.index("torch.softmax(scores, dim=-1)")
    logsumexp = source.index("torch.logsumexp(scores, dim=-1)")
    assert safe_gather < gather
    assert causal_mask < mask_scores < softmax
    assert mask_scores < logsumexp


def test_prefill_benchmark_summary_uses_median_and_preserves_samples():
    result = summarize_samples([4.0, 2.0, 3.0])

    assert result == {
        "samples_ms": [4.0, 2.0, 3.0],
        "median_ms": 3.0,
        "min_ms": 2.0,
        "max_ms": 4.0,
    }


def test_prefill_benchmark_uses_realistic_materialization_and_timed_contract():
    source = (
        Path(__file__).with_name("v32_prefill_benchmark.py")
        .read_text(encoding="utf-8")
    )

    assert "deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048" in source
    assert "CANNBENCH_SPARSE_ATTENTION_HEAD_TILE" not in source
    assert "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS" not in source
    assert source.index("_fill_deterministic(") < source.index("for _ in range(warmups)")
    assert "torch.npu.synchronize()" in source
    assert "output.shape != expected_output_shape" in source
    assert "lse.shape != expected_lse_shape" in source
    assert "output.dtype != torch.bfloat16" in source
    assert "lse.dtype != torch.float32" in source


def test_prefill_benchmark_synchronizes_materialization_before_warmups():
    source = (
        Path(__file__).with_name("v32_prefill_benchmark.py")
        .read_text(encoding="utf-8")
    )

    materialized = source.index("query, shared_kv, indices = _materialize_inputs")
    prewarmup_sync = source.index("torch.npu.synchronize()", materialized)
    warmups = source.index("for _ in range(warmups)")
    assert materialized < prewarmup_sync < warmups


def test_prefill_benchmark_freezes_public_wrapper_layout_and_dtypes():
    benchmark_source = (
        Path(__file__).with_name("v32_prefill_benchmark.py")
        .read_text(encoding="utf-8")
    )
    wrapper_source = (
        Path(__file__).parents[1]
        / "v1/aten_dsa_sparse_attention/ops.py"
    ).read_text(encoding="utf-8")

    assert "output, lse = result\n    return output.permute(0, 2, 1, 3), lse.permute(0, 2, 1)" in wrapper_source
    assert (
        "expected_output_shape = (\n"
        "        case.batch,\n"
        "        case.query_tokens,\n"
        "        case.query_heads,\n"
        "        case.value_head_dim,\n"
        "    )"
    ) in benchmark_source
    assert "expected_lse_shape = (case.batch, case.query_tokens, case.query_heads)" in benchmark_source
    assert "if output.dtype != torch.bfloat16:" in benchmark_source
    assert "if lse.dtype != torch.float32:" in benchmark_source
