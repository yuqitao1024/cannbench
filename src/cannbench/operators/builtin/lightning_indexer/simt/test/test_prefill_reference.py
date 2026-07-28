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


def test_prefill_reference_masks_future_context_before_topk():
    if ops.torch is None:
        pytest.skip("torch is required for right-aligned mask coverage")

    query = ops.torch.ones(1, 2, 1, 1)
    keys = ops.torch.tensor([[[1.0], [2.0], [3.0], [100.0]]])
    weights = ops.torch.ones(1, 2, 1)
    valid_context_lengths = ops.torch.tensor([[3, 4]], dtype=ops.torch.int32)

    actual = ops._prefill_reference(
        query,
        keys,
        weights,
        valid_context_lengths=valid_context_lengths,
        top_k=1,
    )

    assert ops.torch.equal(actual, ops.torch.tensor([[[2], [3]]], dtype=ops.torch.int32))


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

    def fake_custom_op(
        query, keys, weights, valid_context_lengths, top_k, phase, family
    ):
        del query, keys, weights, valid_context_lengths
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return "custom"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op, raising=False)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        valid_context_lengths=object(),
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

    def fake_custom_op(
        query, keys, weights, valid_context_lengths, top_k, phase, family
    ):
        del query, keys, weights, valid_context_lengths
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return "custom"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op, raising=False)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        valid_context_lengths=object(),
        top_k=512,
        phase="prefill",
        family="family_64x128",
    )

    assert actual == "custom"
    assert captured == {"top_k": 512, "phase": "prefill", "family": "family_64x128"}


def test_lightning_indexer_forward_passes_valid_context_lengths_to_custom_op(
    monkeypatch,
):
    captured = {}
    valid_context_lengths = object()

    def fake_custom_op(
        query, keys, weights, native_valid_context_lengths, top_k, phase, family
    ):
        del query, keys, weights, top_k, phase, family
        captured["valid_context_lengths"] = native_valid_context_lengths
        return "custom"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op, raising=False)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        valid_context_lengths=valid_context_lengths,
        top_k=512,
        phase="prefill",
        family="family_64x128",
    )

    assert actual == "custom"
    assert captured["valid_context_lengths"] is valid_context_lengths


def test_lightning_indexer_forward_defaults_custom_op_to_full_context(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        shape = (2, 3, 4, 8)
        device = "npu"

    class FakeKeys:
        shape = (2, 5, 8)

    def fake_full(shape, value, *, dtype, device):
        captured["full"] = (shape, value, dtype, device)
        return "full-context-lengths"

    def fake_custom_op(
        query, keys, weights, valid_context_lengths, top_k, phase, family
    ):
        del query, keys, weights, top_k, phase, family
        captured["valid_context_lengths"] = valid_context_lengths
        return "custom"

    monkeypatch.setattr(
        ops,
        "torch",
        SimpleNamespace(full=fake_full, int32="int32"),
    )
    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op)

    actual = ops.lightning_indexer_forward(
        FakeTensor(),
        FakeKeys(),
        object(),
        top_k=2,
        phase="prefill",
        family="family_4x64",
    )

    assert actual == "custom"
    assert captured == {
        "full": ((2, 3), 5, "int32", "npu"),
        "valid_context_lengths": "full-context-lengths",
    }


def test_lightning_indexer_forward_prefers_registered_custom_op_for_prefill_family_32x128(
    monkeypatch,
):
    captured = {}

    def fake_custom_op(
        query, keys, weights, valid_context_lengths, top_k, phase, family
    ):
        del query, keys, weights, valid_context_lengths
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return "custom"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op, raising=False)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        valid_context_lengths=object(),
        top_k=2048,
        phase="prefill",
        family="family_32x128",
    )

    assert actual == "custom"
    assert captured == {"top_k": 2048, "phase": "prefill", "family": "family_32x128"}


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


def _require_v32_prefill_npu_custom_op():
    if ops.torch is None:
        pytest.skip("torch is required for V3.2 prefill custom-op coverage")
    namespace = getattr(ops.torch.ops, "aten_dsa_lightning_indexer", None)
    if namespace is None or not hasattr(namespace, "lightning_indexer_forward"):
        pytest.skip("registered custom op is required for V3.2 prefill coverage")
    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")
    return ops.torch


def _v32_prefill_target_tensors(torch):
    torch.manual_seed(7)
    device = torch.device("npu")
    query = torch.randn(1, 4096, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(1, 32768, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(1, 4096, 64, device=device, dtype=torch.bfloat16)
    valid = torch.arange(28673, 32769, device=device, dtype=torch.int32).reshape(
        1, 4096
    )
    return query, keys, weights, valid


def _v32_prefill_sampled_scores(torch, query, keys, weights, valid):
    rows = (0, 1365, 2730, 4095)
    sampled_query = query[:, rows]
    sampled_weights = weights[:, rows]
    reduced = torch.einsum("bqhd,bcd->bqhc", sampled_query, keys)
    reduced = torch.relu(reduced)
    reduced = (reduced * sampled_weights.unsqueeze(-1)).sum(dim=2)
    positions = torch.arange(keys.shape[1], device=keys.device).reshape(1, 1, -1)
    reduced = reduced.masked_fill(
        positions >= valid[:, rows].unsqueeze(-1),
        float("-inf"),
    )
    reference_scores = torch.topk(
        reduced, 2048, dim=-1, largest=True, sorted=True
    ).values
    return rows, reduced, reference_scores


def _assert_v32_prefill_sampled_score_sets(
    torch, output, rows, reduced, reference_scores
):
    custom_scores = reduced.gather(-1, output[:, rows].to(torch.int64))
    assert torch.equal(custom_scores, reference_scores)


def test_v32_prefill_matches_sampled_reference_scores():
    torch = _require_v32_prefill_npu_custom_op()
    query, keys, weights, valid = _v32_prefill_target_tensors(torch)
    rows, reduced, reference_scores = _v32_prefill_sampled_scores(
        torch, query, keys, weights, valid
    )

    custom = ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        valid_context_lengths=valid,
        top_k=2048,
        phase="prefill",
        family="family_64x128",
    )
    torch.npu.synchronize()

    assert custom.shape == (1, 4096, 2048)
    assert custom.dtype == torch.int32
    assert not bool((custom[:, 0, :] >= 28673).any().item())
    _assert_v32_prefill_sampled_score_sets(
        torch, custom, rows, reduced, reference_scores
    )


def test_v32_prefill_is_stable_across_repeated_launches():
    torch = _require_v32_prefill_npu_custom_op()
    query, keys, weights, valid = _v32_prefill_target_tensors(torch)
    rows, reduced, reference_scores = _v32_prefill_sampled_scores(
        torch, query, keys, weights, valid
    )

    outputs = [
        ops.lightning_indexer_forward(
            query,
            keys,
            weights,
            valid_context_lengths=valid,
            top_k=2048,
            phase="prefill",
            family="family_64x128",
        )
        for _ in range(3)
    ]
    torch.npu.synchronize()

    for output in outputs:
        _assert_v32_prefill_sampled_score_sets(
            torch, output, rows, reduced, reference_scores
        )
