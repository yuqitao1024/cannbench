from __future__ import annotations

import pytest

from cannbench.operators.builtin.sparse_attention.simt.v1.aten_dsa_sparse_attention import (
    ops,
)


@pytest.mark.parametrize("family", ["family_hd512", "family_hd128"])
def test_sparse_attention_forward_requires_registered_custom_op_for_decode_family(
    monkeypatch,
    family,
):
    monkeypatch.setattr(ops, "_load_registered_op", lambda: None, raising=False)

    with pytest.raises(RuntimeError, match="custom op is not registered"):
        ops.sparse_attention_forward(
            object(),
            object(),
            object(),
            object(),
            phase="decode",
            family=family,
            causal=True,
        )


@pytest.mark.parametrize("family", ["family_hd512", "family_hd128"])
def test_sparse_attention_forward_prefers_registered_custom_op_for_decode_family(
    monkeypatch,
    family,
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
        family=family,
        causal=True,
    )

    assert actual == "custom"
    assert captured == {
        "phase": "decode",
        "family": family,
        "causal": True,
    }


def _require_custom_sparse_attention_op():
    if ops.torch is None:
        pytest.skip("torch is required for exact custom-op correctness coverage")
    namespace = getattr(ops.torch.ops, "aten_dsa_sparse_attention", None)
    if namespace is None or not hasattr(namespace, "sparse_attention_forward"):
        pytest.skip("registered custom op is required for exact custom-op correctness coverage")
    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")


@pytest.mark.parametrize(
    ("family", "query_shape", "kv_shape", "indices_shape"),
    [
        ("family_hd256", (1, 64, 3, 256), (1, 1, 32, 256), (1, 3, 16)),
        ("family_hd512", (1, 64, 1, 512), (1, 1, 32, 512), (1, 1, 16)),
        ("family_hd576", (1, 128, 2, 576), (1, 1, 32, 576), (1, 2, 16)),
        ("family_hd128", (1, 128, 1, 128), (1, 1, 64, 128), (1, 1, 16)),
    ],
    ids=["decode_hd256", "decode_hd512", "decode_hd576", "decode_hd128"],
)
def test_custom_op_decode_matches_reference_when_registered(
    family,
    query_shape,
    kv_shape,
    indices_shape,
):
    _require_custom_sparse_attention_op()

    device = ops.torch.device("npu")
    query = ops.torch.randn(*query_shape, device=device, dtype=ops.torch.bfloat16)
    keys = ops.torch.randn(*kv_shape, device=device, dtype=ops.torch.bfloat16)
    values = ops.torch.randn(*kv_shape, device=device, dtype=ops.torch.bfloat16)
    indices = ops.torch.randint(
        0,
        kv_shape[2],
        indices_shape,
        device=device,
        dtype=ops.torch.long,
    )

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
        family=family,
        causal=True,
    )

    assert ops.torch.allclose(custom_out.float(), reference_out.float(), atol=5e-2, rtol=5e-2)
    assert ops.torch.allclose(custom_lse.float(), reference_lse.float(), atol=5e-2, rtol=5e-2)
