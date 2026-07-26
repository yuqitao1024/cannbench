from dataclasses import replace

import pytest

from cannbench.operators.builtin.dsa_decode import _validate_component_pair
from cannbench.operators.builtin.lightning_indexer.cases import (
    get_lightning_indexer_case,
)
from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)


def test_decode_workflow_rejects_mismatched_page_metadata():
    case_id = "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048"
    sparse_case = get_sparse_attention_case("realistic_decode", case_id)
    indexer_case = replace(
        get_lightning_indexer_case("realistic_decode", case_id),
        page_block_size=32,
    )

    with pytest.raises(ValueError, match="page_block_size mismatch"):
        _validate_component_pair(sparse_case, indexer_case)


def test_decode_workflow_rejects_mismatched_rank_local_shape():
    case_id = "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048"
    sparse_case = get_sparse_attention_case("realistic_decode", case_id)
    indexer_case = replace(
        get_lightning_indexer_case("realistic_decode", case_id),
        cp_size=2,
    )

    with pytest.raises(ValueError, match="cp_size mismatch"):
        _validate_component_pair(sparse_case, indexer_case)
