from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)
from cannbench.operators.builtin.sparse_attention.materialize import (
    materialize_sparse_attention_inputs,
)
from cannbench.operators.builtin.sparse_attention.simt.v1.aten_dsa_sparse_attention import (
    ops,
)


def test_materializer_exposes_one_canonical_shared_kv_tensor():
    case = replace(
        get_sparse_attention_case("smoke", "tiny_decode_top4"),
        batch=1,
        kv_heads=1,
        context_tokens=2,
        selected_tokens=2,
        qk_head_dim=4,
        value_head_dim=2,
    )

    payload = materialize_sparse_attention_inputs(case, dtype="bfloat16", seed=7)

    assert payload["shared_kv_shape"] == (1, 1, 2, 4)
    assert len(payload["shared_kv"]) == 8
    assert "key_shape" not in payload
    assert "value_shape" not in payload
    assert "keys" not in payload
    assert "values" not in payload


def test_simt_wrapper_passes_shared_kv_once_to_registered_op(monkeypatch):
    captured = {}

    def fake_custom_op(
        query,
        shared_kv,
        indices,
        value_head_dim,
        phase,
        family,
        causal,
    ):
        captured.update(
            query=query,
            shared_kv=shared_kv,
            indices=indices,
            value_head_dim=value_head_dim,
            phase=phase,
            family=family,
            causal=causal,
        )
        return "custom"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op)
    query = object()
    shared_kv = object()
    indices = object()

    result = ops.sparse_attention_forward(
        query,
        shared_kv,
        indices,
        value_head_dim=512,
        phase="prefill",
        family="family_hd576",
        causal=True,
    )

    assert result == "custom"
    assert captured == {
        "query": query,
        "shared_kv": shared_kv,
        "indices": indices,
        "value_head_dim": 512,
        "phase": "prefill",
        "family": "family_hd576",
        "causal": True,
    }


def test_registered_simt_schema_has_single_shared_kv_input():
    source = Path(ops.__file__).with_name("csrc").joinpath("sparse_attention.asc")
    text = source.read_text()

    assert "Tensor query, Tensor shared_kv, Tensor indices, int value_head_dim" in text
    assert "Tensor keys, Tensor values" not in text
