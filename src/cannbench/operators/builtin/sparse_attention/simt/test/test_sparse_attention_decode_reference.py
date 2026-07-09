from __future__ import annotations

import pytest

from cannbench.operators.builtin.sparse_attention.simt.v1.aten_dsa_sparse_attention import (
    ops,
)


@pytest.mark.parametrize("family", ["family_hd512", "family_hd128"])
def test_sparse_attention_forward_uses_decode_reference_for_decode_fast_path(
    monkeypatch,
    family,
):
    captured: dict[str, object] = {}

    def fake_decode(query, keys, values, indices, *, causal):
        del query, keys, values, indices
        captured["causal"] = causal
        return "decode"

    def unexpected_fallback(query, keys, values, indices, *, causal):
        del query, keys, values, indices, causal
        raise AssertionError("decode fast path should not use fallback reference")

    monkeypatch.setattr(ops, "_decode_reference", fake_decode, raising=False)
    monkeypatch.setattr(ops, "_fallback_reference", unexpected_fallback)

    actual = ops.sparse_attention_forward(
        object(),
        object(),
        object(),
        object(),
        phase="decode",
        family=family,
        causal=True,
    )

    assert actual == "decode"
    assert captured["causal"] is True


def test_sparse_attention_forward_prefers_registered_custom_op_for_decode_family_hd512(
    monkeypatch,
):
    captured: dict[str, object] = {}

    def fake_custom_op(query, keys, values, indices, phase, family, causal):
        del query, keys, values, indices
        captured["phase"] = phase
        captured["family"] = family
        captured["causal"] = causal
        return "custom"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op, raising=False)

    actual = ops.sparse_attention_forward(
        object(),
        object(),
        object(),
        object(),
        phase="decode",
        family="family_hd512",
        causal=True,
    )

    assert actual == "custom"
    assert captured == {
        "phase": "decode",
        "family": "family_hd512",
        "causal": True,
    }


def test_custom_op_decode_family_hd512_matches_reference_when_registered():
    if ops.torch is None:
        pytest.skip("torch is required for exact custom-op correctness coverage")

    namespace = getattr(ops.torch.ops, "aten_dsa_sparse_attention", None)
    if namespace is None or not hasattr(namespace, "sparse_attention_forward"):
        pytest.skip("registered custom op is required for exact custom-op correctness coverage")

    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")

    device = ops.torch.device("npu")
    query = ops.torch.randn(1, 64, 1, 512, device=device, dtype=ops.torch.float16)
    keys = ops.torch.randn(1, 1, 32, 512, device=device, dtype=ops.torch.float16)
    values = ops.torch.randn(1, 1, 32, 512, device=device, dtype=ops.torch.float16)
    indices = ops.torch.randint(0, 32, (1, 1, 16), device=device, dtype=ops.torch.long)

    reference_out, reference_lse = ops._decode_reference(
        query,
        keys,
        values,
        indices,
        causal=True,
    )
    custom_out, custom_lse = ops.sparse_attention_forward(
        query,
        keys,
        values,
        indices,
        phase="decode",
        family="family_hd512",
        causal=True,
    )

    assert ops.torch.allclose(custom_out.float(), reference_out.float(), atol=5e-2, rtol=5e-2)
    assert ops.torch.allclose(custom_lse.float(), reference_lse.float(), atol=5e-2, rtol=5e-2)
