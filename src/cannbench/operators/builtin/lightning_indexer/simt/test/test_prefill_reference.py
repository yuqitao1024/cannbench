from __future__ import annotations

import builtins
from types import SimpleNamespace

import pytest

from cannbench.operators.builtin.lightning_indexer.simt.v1.aten_dsa_lightning_indexer import (
    ops,
)


class FakeTensor:
    def __init__(self, data, dtype: str = "float32"):
        self.data = data
        self.dtype = dtype

    def unsqueeze(self, dim: int):
        assert dim == -1
        return FakeTensor(
            [
                [
                    [[head] for head in query]
                    for query in batch
                ]
                for batch in self.data
            ],
            dtype=self.dtype,
        )

    def sum(self, dim: int):
        assert dim == 2
        return FakeTensor(
            [
                [
                    [
                        sum(head[context_index] for head in query)
                        for context_index in range(len(query[0]))
                    ]
                    for query in batch
                ]
                for batch in self.data
            ],
            dtype=self.dtype,
        )

    def to(self, dtype):
        return FakeTensor(self.data, dtype=dtype)

    def __mul__(self, other):
        result = []
        for batch_index, batch in enumerate(self.data):
            batch_rows = []
            for query_index, query in enumerate(batch):
                query_heads = []
                for head_index, context_values in enumerate(query):
                    scale = other.data[batch_index][query_index][head_index][0]
                    query_heads.append([value * scale for value in context_values])
                batch_rows.append(query_heads)
            result.append(batch_rows)
        return FakeTensor(result, dtype=self.dtype)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.data == other.data and self.dtype == other.dtype


class FakeTorch:
    int32 = "int32"

    @staticmethod
    def einsum(pattern, query, keys):
        assert pattern == "bqhd,bcd->bqhc"
        return FakeTensor(
            [
                [
                    [
                        [
                            sum(qv * kv for qv, kv in zip(head_vector, key_vector))
                            for key_vector in keys.data[batch_index]
                        ]
                        for head_vector in query_row
                    ]
                    for query_row in batch
                ]
                for batch_index, batch in enumerate(query.data)
            ]
        )

    @staticmethod
    def relu(tensor):
        return FakeTensor(
            [
                [
                    [
                        [max(0.0, value) for value in head]
                        for head in query
                    ]
                    for query in batch
                ]
                for batch in tensor.data
            ],
            dtype=tensor.dtype,
        )

    @staticmethod
    def topk(tensor, top_k, dim=-1, largest=True, sorted=True):
        assert dim == -1
        assert largest is True
        assert sorted is True
        return SimpleNamespace(
            indices=FakeTensor(
                [
                    [
                        [
                            index
                            for index, _ in builtins.sorted(
                                enumerate(query),
                                key=lambda item: item[1],
                                reverse=True,
                            )[:top_k]
                        ]
                        for query in batch
                    ]
                    for batch in tensor.data
                ]
            )
        )


def _fake_query():
    return FakeTensor(
        [
            [
                [[1.0, -2.0], [0.5, 3.0]],
                [[2.0, 0.0], [-1.0, 1.0]],
            ]
        ]
    )


def _fake_keys():
    return FakeTensor(
        [
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [1.0, 1.0],
            ]
        ]
    )


def _fake_weights():
    return FakeTensor(
        [
            [
                [2.0, 1.0],
                [2.0, 1.0],
            ]
        ]
    )


def test_prefill_reference_matches_manual_topk(monkeypatch):
    monkeypatch.setattr(ops, "torch", FakeTorch)

    actual = ops._prefill_reference(_fake_query(), _fake_keys(), _fake_weights(), top_k=2)

    assert actual == FakeTensor([[[2, 1], [0, 2]]], dtype="int32")


def test_prefill_reference_returns_int32_indices(monkeypatch):
    monkeypatch.setattr(ops, "torch", FakeTorch)

    actual = ops._prefill_reference(_fake_query(), _fake_keys(), _fake_weights(), top_k=2)

    assert actual.dtype == "int32"


def test_lightning_indexer_forward_uses_fallback_reference_outside_fast_path(
    monkeypatch,
):
    monkeypatch.setattr(ops, "torch", FakeTorch)

    captured = {}

    def fake_fallback(query, keys, weights, *, top_k):
        captured["top_k"] = top_k
        return "fallback"

    monkeypatch.setattr(ops, "_fallback_reference", fake_fallback)

    actual = ops.lightning_indexer_forward(
        _fake_query(),
        _fake_keys(),
        _fake_weights(),
        top_k=2,
        phase="decode",
        family="fallback",
    )

    assert actual == "fallback"
    assert captured["top_k"] == 2


def test_lightning_indexer_forward_prefers_registered_custom_op_for_prefill_family_4x64(
    monkeypatch,
):
    captured = {}

    def fake_custom_op(query, keys, weights, top_k, phase, family):
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return "custom"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op, raising=False)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        top_k=4,
        phase="prefill",
        family="family_4x64",
    )

    assert actual == "custom"
    assert captured == {"top_k": 4, "phase": "prefill", "family": "family_4x64"}


def test_lightning_indexer_forward_prefers_registered_custom_op_for_prefill_family_64x128(
    monkeypatch,
):
    captured = {}

    def fake_custom_op(query, keys, weights, top_k, phase, family):
        del query, keys, weights
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return "custom"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op, raising=False)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        top_k=512,
        phase="prefill",
        family="family_64x128",
    )

    assert actual == "custom"
    assert captured == {"top_k": 512, "phase": "prefill", "family": "family_64x128"}


def test_custom_op_prefill_family_4x64_matches_reference_when_registered(monkeypatch):
    if ops.torch is None:
        pytest.skip("torch is required for exact custom-op correctness coverage")

    namespace = getattr(ops.torch.ops, "aten_dsa_lightning_indexer", None)
    if namespace is None or not hasattr(namespace, "lightning_indexer_forward"):
        pytest.skip("registered custom op is required for exact custom-op correctness coverage")

    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")

    device = ops.torch.device("npu")
    query = ops.torch.randn(1, 2, 4, 64, device=device, dtype=ops.torch.bfloat16)
    keys = ops.torch.randn(1, 32, 64, device=device, dtype=ops.torch.bfloat16)
    weights = ops.torch.rand(1, 2, 4, device=device, dtype=ops.torch.bfloat16)

    reference = ops._prefill_reference(query, keys, weights, top_k=8)

    custom = ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        top_k=8,
        phase="prefill",
        family="family_4x64",
    )

    assert ops.torch.equal(custom, reference)
    assert custom.dtype == ops.torch.int32


def test_custom_op_prefill_family_64x128_matches_reference_when_registered():
    if ops.torch is None:
        pytest.skip("torch is required for exact custom-op correctness coverage")

    namespace = getattr(ops.torch.ops, "aten_dsa_lightning_indexer", None)
    if namespace is None or not hasattr(namespace, "lightning_indexer_forward"):
        pytest.skip("registered custom op is required for exact custom-op correctness coverage")

    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")

    device = ops.torch.device("npu")
    query = ops.torch.randn(1, 2, 64, 128, device=device, dtype=ops.torch.bfloat16)
    keys = ops.torch.randn(1, 64, 128, device=device, dtype=ops.torch.bfloat16)
    weights = ops.torch.rand(1, 2, 64, device=device, dtype=ops.torch.bfloat16)

    reference = ops._prefill_reference(query, keys, weights, top_k=16)

    custom = ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        top_k=16,
        phase="prefill",
        family="family_64x128",
    )

    assert ops.torch.equal(custom, reference)
    assert custom.dtype == ops.torch.int32
