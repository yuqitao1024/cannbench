from __future__ import annotations

import pytest

from cannbench.operators.builtin.lightning_indexer.simt.v1.aten_dsa_lightning_indexer import (
    ops,
)


@pytest.mark.parametrize("family", ["family_64x128", "family_4x64"])
def test_lightning_indexer_forward_uses_decode_reference_for_decode_fast_path(
    monkeypatch,
    family,
):
    captured: dict[str, object] = {}

    def fake_decode(query, keys, weights, *, top_k):
        del query, keys, weights
        captured["top_k"] = top_k
        return "decode"

    def unexpected_fallback(query, keys, weights, *, top_k):
        del query, keys, weights, top_k
        raise AssertionError("decode fast path should not use fallback reference")

    monkeypatch.setattr(ops, "_decode_reference", fake_decode, raising=False)
    monkeypatch.setattr(ops, "_fallback_reference", unexpected_fallback)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        top_k=4,
        phase="decode",
        family=family,
    )

    assert actual == "decode"
    assert captured["top_k"] == 4


def test_lightning_indexer_forward_skips_registered_custom_op_for_decode(monkeypatch):
    captured: dict[str, object] = {
        "custom_calls": 0,
        "decode_calls": 0,
    }

    def unexpected_custom(query, keys, weights, top_k, phase, family):
        del query, keys, weights, top_k, phase, family
        captured["custom_calls"] += 1
        raise AssertionError("decode should not use the registered prefill custom op")

    def fake_decode(query, keys, weights, *, top_k):
        del query, keys, weights
        captured["decode_calls"] += 1
        captured["top_k"] = top_k
        return "decode"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: unexpected_custom)
    monkeypatch.setattr(ops, "_decode_reference", fake_decode, raising=False)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        top_k=8,
        phase="decode",
        family="family_4x64",
    )

    assert actual == "decode"
    assert captured["custom_calls"] == 0
    assert captured["decode_calls"] == 1
    assert captured["top_k"] == 8
