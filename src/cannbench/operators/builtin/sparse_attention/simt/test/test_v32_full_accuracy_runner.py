import pytest

from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)
from v32_full_accuracy import (
    deterministic_pattern,
    resize_prefill_case,
    validation_query_rows,
)


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
