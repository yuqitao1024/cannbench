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
            value_head_dim=512,
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

    def fake_custom_op(
        query,
        shared_kv,
        indices,
        value_head_dim,
        phase,
        family,
        causal,
        head_tile,
        selected_partitions,
    ):
        del query, shared_kv, indices, value_head_dim
        captured["phase"] = phase
        captured["family"] = family
        captured["causal"] = causal
        captured["tuning"] = (head_tile, selected_partitions)
        return "custom"

    monkeypatch.delenv("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", raising=False)
    monkeypatch.delenv(
        "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS", raising=False
    )
    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op, raising=False)

    actual = ops.sparse_attention_forward(
        object(),
        object(),
        object(),
        value_head_dim=512,
        phase="decode",
        family=family,
        causal=True,
    )

    assert actual == "custom"
    assert captured == {
        "phase": "decode",
        "family": family,
        "causal": True,
        "tuning": (1, 1),
    }


def test_sparse_attention_forward_passes_head64_tuning(monkeypatch):
    captured = {}

    def fake_custom_op(*args):
        captured["tuning"] = args[-2:]
        return "custom"

    monkeypatch.setenv("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", "64")
    monkeypatch.setenv("CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS", "1")
    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op)

    result = ops.sparse_attention_forward(
        object(),
        object(),
        object(),
        value_head_dim=512,
        phase="decode",
        family="family_hd576",
        causal=True,
    )

    assert result == "custom"
    assert captured["tuning"] == (64, 1)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", "abc", "must be an integer"),
        ("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", "0", "must be positive"),
        (
            "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS",
            "2",
            "unsupported sparse_attention tuning",
        ),
    ],
)
def test_sparse_attention_forward_rejects_invalid_tuning(
    monkeypatch, name, value, message
):
    monkeypatch.delenv("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", raising=False)
    monkeypatch.delenv(
        "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS", raising=False
    )
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        ops.sparse_attention_forward(
            object(),
            object(),
            object(),
            value_head_dim=512,
            phase="decode",
            family="family_hd576",
            causal=True,
        )


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
    shared_kv = ops.torch.randn(*kv_shape, device=device, dtype=ops.torch.bfloat16)
    indices = ops.torch.randint(
        0,
        kv_shape[2],
        indices_shape,
        device=device,
        dtype=ops.torch.long,
    )

    reference_out, reference_lse = ops._decode_reference(
        query,
        shared_kv,
        indices,
        value_head_dim=512 if family == "family_hd576" else kv_shape[-1],
        causal=False,
    )
    custom_out, custom_lse = ops.sparse_attention_forward(
        query,
        shared_kv,
        indices,
        value_head_dim=512 if family == "family_hd576" else kv_shape[-1],
        phase="decode",
        family=family,
        causal=False,
    )

    assert ops.torch.allclose(custom_out.float(), reference_out.float(), atol=5e-2, rtol=5e-2)
    assert ops.torch.allclose(custom_lse.float(), reference_lse.float(), atol=5e-2, rtol=5e-2)
