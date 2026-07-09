from __future__ import annotations

import math
from types import SimpleNamespace

from cannbench.operators.builtin.sparse_attention.simt.v1.aten_dsa_sparse_attention import (
    ops,
)


class FakeTensor:
    def __init__(self, data, dtype: str = "float32"):
        self.data = data
        self.dtype = dtype

    @property
    def shape(self):
        return _shape_of(self.data)

    def unsqueeze(self, dim: int):
        rank = len(self.shape)
        if dim < 0:
            dim = rank + dim + 1
        return FakeTensor(_unsqueeze_at(self.data, dim, 0), dtype=self.dtype)

    def reshape(self, *shape):
        if len(shape) == 1 and isinstance(shape[0], tuple):
            shape = shape[0]
        flat = _flatten(self.data)
        return FakeTensor(_reshape_flat(flat, shape), dtype=self.dtype)

    def sum(self, dim: int):
        if dim < 0:
            dim = len(self.shape) + dim
        assert dim == 4 or dim == 3
        return FakeTensor(_sum_along_dim(self.data, dim), dtype=self.dtype)

    def to(self, dtype):
        return FakeTensor(self.data, dtype=dtype)

    def float(self):
        return FakeTensor(self.data, dtype="float32")

    def masked_fill(self, mask, fill_value):
        return FakeTensor(
            _masked_fill(self.data, mask.data, fill_value),
            dtype=self.dtype,
        )

    def __mul__(self, other):
        return FakeTensor(_elementwise_binary(self.data, other.data, lambda a, b: a * b), dtype=self.dtype)

    def __truediv__(self, scalar):
        return FakeTensor(_elementwise_scalar(self.data, lambda value: value / scalar), dtype=self.dtype)

    def __gt__(self, other):
        return FakeTensor(_elementwise_binary(self.data, other.data, lambda a, b: a > b), dtype="bool")

    def __getitem__(self, item):
        if not isinstance(item, tuple):
            item = (item,)
        return FakeTensor(_slice_with_spec(self.data, item), dtype=self.dtype)

    def __eq__(self, other):
        return isinstance(other, FakeTensor) and self.data == other.data and self.dtype == other.dtype


def _shape_of(data):
    if not isinstance(data, list):
        return ()
    if not data:
        return (0,)
    return (len(data),) + _shape_of(data[0])


def _flatten(data):
    if isinstance(data, list):
        result = []
        for item in data:
            result.extend(_flatten(item))
        return result
    return [data]


def _reshape_flat(flat, shape):
    iterator = iter(flat)

    def build(remaining_shape):
        if not remaining_shape:
            return next(iterator)
        size = remaining_shape[0]
        return [build(remaining_shape[1:]) for _ in range(size)]

    return build(tuple(shape))


def _elementwise_binary(left, right, fn):
    if isinstance(left, list) and isinstance(right, list):
        if len(left) == len(right):
            return [_elementwise_binary(lhs, rhs, fn) for lhs, rhs in zip(left, right)]
        if len(left) == 1:
            return [_elementwise_binary(left[0], rhs, fn) for rhs in right]
        if len(right) == 1:
            return [_elementwise_binary(lhs, right[0], fn) for lhs in left]
        raise ValueError("incompatible fake tensor shapes for broadcasting")
    if isinstance(left, list):
        return [_elementwise_binary(item, right, fn) for item in left]
    if isinstance(right, list):
        return [_elementwise_binary(left, item, fn) for item in right]
    return fn(left, right)


def _elementwise_scalar(data, fn):
    if isinstance(data, list):
        return [_elementwise_scalar(item, fn) for item in data]
    return fn(data)


def _sum_along_dim(data, dim: int):
    if dim == 0:
        if not isinstance(data[0], list):
            return sum(data)
        return [
            _sum_along_dim([item[index] for item in data], 0)
            for index in range(len(data[0]))
        ]
    return [_sum_along_dim(item, dim - 1) for item in data]


def _masked_fill(data, mask, fill_value):
    if isinstance(data, list) or isinstance(mask, list):
        return _elementwise_binary(
            data,
            mask,
            lambda value, mask_value: fill_value if mask_value else value,
        )
    return fill_value if mask else data


def _slice_with_spec(data, spec):
    if not spec:
        return data
    head, *tail = spec
    if head is None:
        return [_slice_with_spec(data, tuple(tail))]
    if isinstance(head, slice):
        return [_slice_with_spec(item, tuple(tail)) for item in data[head]]
    return _slice_with_spec(data[head], tuple(tail))


def _unsqueeze_at(data, target_axis: int, current_axis: int):
    if target_axis == current_axis:
        return [_clone_nested_value(data)]
    return [_unsqueeze_at(item, target_axis, current_axis + 1) for item in data]


def _clone_nested_value(value):
    if isinstance(value, list):
        return [_clone_nested_value(item) for item in value]
    return value


def _softmax_last_dim(data):
    result = []
    for batch in data:
        batch_rows = []
        for head in batch:
            head_rows = []
            for query in head:
                max_value = max(query)
                exps = [math.exp(value - max_value) for value in query]
                total = sum(exps)
                head_rows.append([value / total for value in exps])
            batch_rows.append(head_rows)
        result.append(batch_rows)
    return result


def _logsumexp_last_dim(data):
    result = []
    for batch in data:
        batch_rows = []
        for head in batch:
            head_rows = []
            for query in head:
                max_value = max(query)
                total = sum(math.exp(value - max_value) for value in query)
                head_rows.append(max_value + math.log(total))
            batch_rows.append(head_rows)
        result.append(batch_rows)
    return result


def _gather_selected(data, indices):
    batch_size = len(indices)
    query_tokens = len(indices[0])
    selected_tokens = len(indices[0][0])
    head_dim = len(data[0][0][0][0])
    gathered = []
    for batch_index in range(batch_size):
        batch_heads = []
        for head_index in range(len(data[batch_index])):
            query_rows = []
            for query_index in range(query_tokens):
                selected_rows = []
                for selected_index in range(selected_tokens):
                    token_index = indices[batch_index][query_index][selected_index]
                    selected_rows.append(
                        list(data[batch_index][head_index][token_index][:head_dim])
                    )
                query_rows.append(selected_rows)
            batch_heads.append(query_rows)
        gathered.append(batch_heads)
    return gathered


class FakeTorch:
    float32 = "float32"

    @staticmethod
    def arange(length, device=None):
        del device
        return FakeTensor(list(range(length)), dtype="int64")

    @staticmethod
    def softmax(tensor, dim=-1):
        assert dim == -1
        return FakeTensor(_softmax_last_dim(tensor.data), dtype="float32")

    @staticmethod
    def logsumexp(tensor, dim=-1):
        assert dim == -1
        return FakeTensor(_logsumexp_last_dim(tensor.data), dtype="float32")


def _fake_query():
    return FakeTensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0]],
                [[0.5, 0.5], [1.0, -1.0]],
            ]
        ]
    )


def _fake_keys():
    return FakeTensor(
        [
            [
                [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
            ]
        ]
    )


def _fake_values():
    return FakeTensor(
        [
            [
                [[10.0, 1.0], [20.0, 2.0], [30.0, 3.0]],
            ]
        ]
    )


def _fake_indices():
    return FakeTensor([[[0, 2], [1, 2]]], dtype="int64")


def test_prefill_reference_returns_output_and_lse(monkeypatch):
    monkeypatch.setattr(ops, "torch", FakeTorch)

    output, lse = ops._prefill_reference(
        _fake_query(),
        _fake_keys(),
        _fake_values(),
        _fake_indices(),
        causal=True,
    )

    assert output.shape == (1, 2, 2, 2)
    assert lse.shape == (1, 2, 2)


def test_sparse_attention_forward_uses_fallback_reference_outside_prefill_fast_path(
    monkeypatch,
):
    monkeypatch.setattr(ops, "torch", FakeTorch)

    captured = {}

    def fake_fallback(query, keys, values, indices, *, causal):
        del query, keys, values, indices
        captured["causal"] = causal
        return "fallback"

    monkeypatch.setattr(ops, "_fallback_reference", fake_fallback)

    actual = ops.sparse_attention_forward(
        _fake_query(),
        _fake_keys(),
        _fake_values(),
        _fake_indices(),
        phase="decode",
        family="fallback",
        causal=False,
    )

    assert actual == "fallback"
    assert captured["causal"] is False


def test_sparse_attention_forward_prefers_registered_custom_op_for_prefill_family_hd512(
    monkeypatch,
):
    captured = {}

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
        phase="prefill",
        family="family_hd512",
        causal=True,
    )

    assert actual == "custom"
    assert captured == {
        "phase": "prefill",
        "family": "family_hd512",
        "causal": True,
    }


def test_custom_op_prefill_family_hd512_matches_reference_when_registered():
    if ops.torch is None:
        return

    namespace = getattr(ops.torch.ops, "aten_dsa_sparse_attention", None)
    if namespace is None or not hasattr(namespace, "sparse_attention_forward"):
        return

    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        return

    device = ops.torch.device("npu")
    query = ops.torch.randn(1, 64, 8, 512, device=device, dtype=ops.torch.float16)
    keys = ops.torch.randn(1, 1, 8, 512, device=device, dtype=ops.torch.float16)
    values = ops.torch.randn(1, 1, 8, 512, device=device, dtype=ops.torch.float16)
    indices = ops.torch.randint(0, 8, (1, 8, 8), device=device, dtype=ops.torch.long)

    reference_out, reference_lse = ops._prefill_reference(
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
        phase="prefill",
        family="family_hd512",
        causal=True,
    )

    assert ops.torch.allclose(custom_out.float(), reference_out.float(), atol=5e-2, rtol=5e-2)
    assert ops.torch.allclose(custom_lse.float(), reference_lse.float(), atol=5e-2, rtol=5e-2)
