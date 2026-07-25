from dataclasses import replace
import math

import pytest

from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)
from cannbench.operators.builtin.sparse_attention.materialize import (
    materialize_sparse_attention_inputs,
)
from cannbench.operators.builtin.sparse_attention.simt.v1.aten_dsa_sparse_attention import (
    ops,
)


def test_v32_case_exposes_softmax_and_topk_contract():
    case = get_sparse_attention_case(
        "realistic_decode",
        "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
    )

    assert case.softmax_scale == pytest.approx(576**-0.5)
    assert case.resolved_topk_lengths == (2048,) * 4
    assert case.payload["softmax_scale"] == pytest.approx(576**-0.5)
    assert case.payload["topk_lengths"] == (2048,) * 4


@pytest.mark.parametrize("softmax_scale", [0.0, -1.0, math.inf, math.nan])
def test_sparse_attention_case_rejects_invalid_softmax_scale(softmax_scale):
    case = get_sparse_attention_case("smoke", "tiny_decode_top4")

    with pytest.raises(ValueError, match="softmax_scale must be finite and positive"):
        replace(case, softmax_scale=softmax_scale)


def test_sparse_attention_case_rejects_invalid_topk_lengths():
    case = get_sparse_attention_case("smoke", "tiny_decode_top4")

    with pytest.raises(ValueError, match="topk_lengths must contain one value per query"):
        replace(case, topk_lengths=(1,))

    with pytest.raises(ValueError, match="topk_lengths values must be between"):
        replace(case, topk_lengths=(case.selected_tokens + 1,) * 2)


def test_materializer_marks_entries_after_topk_length_invalid():
    case = replace(
        get_sparse_attention_case("smoke", "tiny_decode_top4"),
        batch=1,
        query_tokens=2,
        context_tokens=8,
        selected_tokens=4,
        causal=False,
        topk_lengths=(2, 0),
    )

    payload = materialize_sparse_attention_inputs(case, dtype="bfloat16", seed=7)

    assert payload["topk_lengths"] == (2, 0)
    assert payload["indices"][2:4] == (-1, -1)
    assert payload["indices"][4:8] == (-1, -1, -1, -1)


def test_reference_masks_invalid_indices_and_returns_canonical_layout():
    torch = pytest.importorskip("torch")
    query = torch.tensor([[[[1.0, 0.0]], [[0.0, 1.0]]]])
    shared_kv = torch.tensor([[[[2.0, 3.0], [20.0, 30.0]]]])
    indices = torch.tensor([[[0, -1]]], dtype=torch.long)
    topk_lengths = torch.tensor([[1]], dtype=torch.int32)

    output, lse = ops._prefill_reference(
        query,
        shared_kv,
        indices,
        value_head_dim=2,
        causal=False,
        softmax_scale=0.5,
        topk_lengths=topk_lengths,
    )

    assert output.shape == (1, 1, 2, 2)
    assert lse.shape == (1, 1, 2)
    assert output.tolist() == [[[[2.0, 3.0], [2.0, 3.0]]]]
    assert lse.tolist() == [[[1.0, 1.5]]]


def test_reference_defines_all_invalid_row_as_zero_and_negative_infinity():
    torch = pytest.importorskip("torch")
    query = torch.zeros((1, 1, 1, 2))
    shared_kv = torch.ones((1, 1, 2, 2))
    indices = torch.tensor([[[-1, 2]]], dtype=torch.long)

    output, lse = ops._prefill_reference(
        query,
        shared_kv,
        indices,
        value_head_dim=2,
        causal=False,
        softmax_scale=0.5,
        topk_lengths=torch.tensor([[2]], dtype=torch.int32),
    )

    assert output.tolist() == [[[[0.0, 0.0]]]]
    assert lse.tolist() == [[[float("-inf")]]]
