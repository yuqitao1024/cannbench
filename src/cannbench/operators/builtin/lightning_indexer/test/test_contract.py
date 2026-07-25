from dataclasses import replace

import pytest

from cannbench.operators.builtin.lightning_indexer.cases import (
    get_lightning_indexer_case,
)
from cannbench.operators.builtin.lightning_indexer.materialize import (
    materialize_lightning_indexer_inputs,
)


def test_v32_case_exposes_indexer_score_contract():
    case = get_lightning_indexer_case(
        "realistic_decode",
        "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
    )

    assert case.score_scale == 1.0
    assert case.tie_policy == "equivalent_score_set"
    assert case.payload["score_scale"] == 1.0
    assert case.payload["tie_policy"] == "equivalent_score_set"


@pytest.mark.parametrize("score_scale", [0.0, -1.0, float("inf"), float("nan")])
def test_indexer_case_rejects_non_positive_or_non_finite_score_scale(score_scale):
    case = get_lightning_indexer_case("smoke", "tiny_decode_top4")

    with pytest.raises(ValueError, match="score_scale must be finite and positive"):
        replace(case, score_scale=score_scale)


def test_indexer_case_rejects_unknown_tie_policy():
    case = get_lightning_indexer_case("smoke", "tiny_decode_top4")

    with pytest.raises(ValueError, match="unsupported tie_policy"):
        replace(case, tie_policy="lowest_index")


def test_materialized_inputs_preserve_indexer_score_contract():
    case = get_lightning_indexer_case("smoke", "tiny_decode_top4")

    payload = materialize_lightning_indexer_inputs(case, dtype="bfloat16", seed=7)

    assert payload["score_scale"] == 1.0
    assert payload["tie_policy"] == "equivalent_score_set"
