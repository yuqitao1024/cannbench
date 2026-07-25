from __future__ import annotations

from cannbench.operators.builtin.sparse_attention import _select_simt_family


def test_a5_and_v4pro_sparse_attention_cases_map_to_hd512_family():
    assert _select_simt_family(
        {
            "qk_head_dim": 512,
            "value_head_dim": 512,
            "kv_heads": 1,
            "query_heads": 128,
            "selected_tokens": 1024,
        }
    ) == "family_hd512"


def test_v32_sparse_attention_cases_map_to_hd128_family():
    assert _select_simt_family(
        {
            "qk_head_dim": 128,
            "value_head_dim": 128,
            "kv_heads": 1,
            "query_heads": 128,
            "selected_tokens": 2048,
        }
    ) == "family_hd128"
