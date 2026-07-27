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
    monkeypatch.setattr(ops, "_load_registered_op", lambda: None, raising=False)

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
        "fallback_calls": 0,
    }

    def unexpected_custom(query, keys, weights, top_k, phase, family):
        del query, keys, weights, top_k, phase, family
        captured["custom_calls"] += 1
        raise AssertionError("decode should not use the registered prefill custom op")

    def fake_fallback(query, keys, weights, *, top_k):
        del query, keys, weights
        captured["fallback_calls"] += 1
        captured["top_k"] = top_k
        return "fallback"

    monkeypatch.setattr(ops, "_load_registered_op", lambda: unexpected_custom)
    monkeypatch.setattr(ops, "_fallback_reference", fake_fallback, raising=False)

    actual = ops.lightning_indexer_forward(
        object(),
        object(),
        object(),
        top_k=8,
        phase="decode",
        family="fallback",
    )

    assert actual == "fallback"
    assert captured["custom_calls"] == 0
    assert captured["fallback_calls"] == 1
    assert captured["top_k"] == 8


def test_lightning_indexer_forward_prefers_registered_custom_op_for_decode_family_4x64(
    monkeypatch,
):
    captured: dict[str, object] = {}

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
        phase="decode",
        family="family_4x64",
    )

    assert actual == "custom"
    assert captured == {"top_k": 2048, "phase": "decode", "family": "family_4x64"}


def test_lightning_indexer_forward_prefers_registered_custom_op_for_decode_family_64x128(
    monkeypatch,
):
    captured: dict[str, object] = {}

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
        phase="decode",
        family="family_64x128",
    )

    assert actual == "custom"
    assert captured == {"top_k": 512, "phase": "decode", "family": "family_64x128"}


def test_custom_op_decode_family_64x128_matches_reference_when_registered():
    if ops.torch is None:
        pytest.skip("torch is required for exact custom-op correctness coverage")

    namespace = getattr(ops.torch.ops, "aten_dsa_lightning_indexer", None)
    if namespace is None or not hasattr(namespace, "lightning_indexer_forward"):
        pytest.skip("registered custom op is required for exact custom-op correctness coverage")

    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")

    ops.torch.manual_seed(5)
    device = ops.torch.device("npu")
    query = ops.torch.randn(1, 1, 64, 128, device=device, dtype=ops.torch.bfloat16)
    keys = ops.torch.randn(1, 64, 128, device=device, dtype=ops.torch.bfloat16)
    weights = ops.torch.rand(1, 1, 64, device=device, dtype=ops.torch.bfloat16)

    scores = ops.torch.einsum("bqhd,bcd->bqhc", query, keys)
    scores = ops.torch.relu(scores)
    scores = scores * weights.unsqueeze(-1)
    reduced = scores.sum(dim=2)
    reference = ops._decode_reference(query, keys, weights, top_k=16)

    custom = ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        top_k=16,
        phase="decode",
        family="family_64x128",
    )

    reference_scores = reduced.gather(-1, reference.to(ops.torch.int64))
    custom_scores = reduced.gather(-1, custom.to(ops.torch.int64))

    assert ops.torch.equal(custom_scores, reference_scores)
    assert bool((custom_scores[..., :-1] >= custom_scores[..., 1:]).all().item())


def test_custom_op_decode_family_4x64_matches_reference_when_registered():
    if ops.torch is None:
        pytest.skip("torch is required for exact custom-op correctness coverage")

    namespace = getattr(ops.torch.ops, "aten_dsa_lightning_indexer", None)
    if namespace is None or not hasattr(namespace, "lightning_indexer_forward"):
        pytest.skip("registered custom op is required for exact custom-op correctness coverage")

    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")

    device = ops.torch.device("npu")
    query = ops.torch.randn(1, 1, 4, 64, device=device, dtype=ops.torch.bfloat16)
    keys = ops.torch.randn(1, 64, 64, device=device, dtype=ops.torch.bfloat16)
    weights = ops.torch.rand(1, 1, 4, device=device, dtype=ops.torch.bfloat16)

    reference = ops._decode_reference(query, keys, weights, top_k=16)

    custom = ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        top_k=16,
        phase="decode",
        family="family_4x64",
    )

    assert ops.torch.equal(custom, reference)


def _require_context_sharded_npu_custom_op():
    if ops.torch is None:
        pytest.skip("torch is required for context-sharded NPU coverage")

    namespace = getattr(ops.torch.ops, "aten_dsa_lightning_indexer", None)
    if namespace is None or not hasattr(namespace, "lightning_indexer_forward"):
        pytest.skip("registered custom op is required for context-sharded coverage")

    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")
    return ops.torch


def _context_sharded_target_tensors(torch):
    torch.manual_seed(7)
    device = torch.device("npu")
    query = torch.randn(2, 2, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(2, 32768, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(2, 2, 64, device=device, dtype=torch.bfloat16)
    valid = torch.tensor(
        [[32767, 32768], [32767, 32768]],
        device=device,
        dtype=torch.int32,
    )
    return query, keys, weights, valid


def _context_sharded_reference_scores(torch, query, keys, weights, valid):
    scores = torch.einsum("bqhd,bcd->bqhc", query, keys)
    scores = torch.relu(scores)
    scores = scores * weights.unsqueeze(-1)
    reduced = scores.sum(dim=2)
    positions = torch.arange(keys.shape[1], device=keys.device).reshape(1, 1, -1)
    reduced = reduced.masked_fill(
        positions >= valid.unsqueeze(-1),
        float("-inf"),
    )
    reference_indices = torch.topk(
        reduced,
        2048,
        dim=-1,
        largest=True,
        sorted=True,
    ).indices
    reference_scores = reduced.gather(-1, reference_indices)
    return reduced, reference_scores


def test_context_sharded_decode_matches_target_reference_scores():
    torch = _require_context_sharded_npu_custom_op()
    query, keys, weights, valid = _context_sharded_target_tensors(torch)
    reduced, reference_scores = _context_sharded_reference_scores(
        torch, query, keys, weights, valid
    )

    custom = ops.lightning_indexer_forward(
        query,
        keys,
        weights,
        valid_context_lengths=valid,
        top_k=2048,
        phase="decode",
        family="family_64x128",
    )
    custom_scores = reduced.gather(-1, custom.to(torch.int64))

    assert custom.shape == (2, 2, 2048)
    assert custom.dtype == torch.int32
    assert torch.equal(custom_scores, reference_scores)
    assert not bool((custom[:, 0, :] == 32767).any().item())


def test_context_sharded_decode_is_stable_across_repeated_launches():
    torch = _require_context_sharded_npu_custom_op()
    query, keys, weights, valid = _context_sharded_target_tensors(torch)
    reduced, reference_scores = _context_sharded_reference_scores(
        torch, query, keys, weights, valid
    )

    outputs = [
        ops.lightning_indexer_forward(
            query,
            keys,
            weights,
            valid_context_lengths=valid,
            top_k=2048,
            phase="decode",
            family="family_64x128",
        )
        for _ in range(20)
    ]
    torch.npu.synchronize()

    for output in outputs:
        custom_scores = reduced.gather(-1, output.to(torch.int64))
        assert torch.equal(custom_scores, reference_scores)
