from __future__ import annotations

from types import SimpleNamespace

import pytest

from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.operators.builtin.sparse_attention import (
    _build_simt_callable,
    _select_simt_family,
    _simt_module_name,
    get_sparse_attention_case,
)
from cannbench.operators.plugin import TorchOperatorContext


def test_select_simt_family_prefers_hd512():
    payload = {
        "head_dim": 512,
        "kv_heads": 1,
        "query_heads": 128,
        "selected_tokens": 1024,
    }

    assert _select_simt_family(payload) == "family_hd512"


def test_select_simt_family_prefers_hd128():
    payload = {
        "head_dim": 128,
        "kv_heads": 1,
        "query_heads": 128,
        "selected_tokens": 2048,
    }

    assert _select_simt_family(payload) == "family_hd128"


def test_select_simt_family_falls_back_for_unknown_shape():
    payload = {
        "head_dim": 64,
        "kv_heads": 12,
        "query_heads": 12,
        "selected_tokens": 512,
    }

    assert _select_simt_family(payload) == "fallback"


def test_simt_module_name_registers_only_v1():
    assert _simt_module_name(None) == "aten_dsa_sparse_attention"
    assert _simt_module_name("v1") == "aten_dsa_sparse_attention"
    assert _simt_module_name("v2") is None


def test_build_simt_callable_requires_loaded_module():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        seed=7,
        implementation="simt",
    )

    ctx = TorchOperatorContext(
        backend=SimpleNamespace(),
        torch=SimpleNamespace(),
        request=request,
        case=get_sparse_attention_case("smoke", "tiny_decode_top4"),
        device="npu",
        dtype="float16",
        implementation_module=None,
    )

    with pytest.raises(
        RuntimeError,
        match="sparse_attention SIMT implementation module is not loaded",
    ):
        _build_simt_callable(ctx)


def test_build_simt_callable_passes_family_to_operator():
    captured: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, values):
            self.values = values
            self.shape = None

        def reshape(self, shape):
            self.shape = shape
            return self

    class FakeBackend:
        def _tensor(self, torch, values, *, device, dtype):
            del torch
            tensor = FakeTensor(values)
            captured.setdefault("tensors", []).append(
                {
                    "device": device,
                    "dtype": dtype,
                    "values": values,
                    "tensor": tensor,
                }
            )
            return tensor

    def fake_forward(query, keys, values, indices, *, phase, family, causal):
        captured["query_shape"] = query.shape
        captured["key_shape"] = keys.shape
        captured["value_shape"] = values.shape
        captured["indices_shape"] = indices.shape
        captured["phase"] = phase
        captured["family"] = family
        captured["causal"] = causal
        return "ok"

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
        dtype="float16",
        dataset="realistic_decode",
        case_id="deepseek_128k_decode_top2048",
        seed=7,
        implementation="simt",
    )

    ctx = TorchOperatorContext(
        backend=FakeBackend(),
        torch=SimpleNamespace(long="long"),
        request=request,
        case=get_sparse_attention_case("realistic_decode", "deepseek_128k_decode_top2048"),
        device="npu",
        dtype="float16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(sparse_attention_forward=fake_forward)
        ),
    )

    operator = _build_simt_callable(ctx)
    assert operator() == "ok"
    assert captured["query_shape"] == (1, 128, 1, 128)
    assert captured["key_shape"] == (1, 1, 131072, 128)
    assert captured["value_shape"] == (1, 1, 131072, 128)
    assert captured["indices_shape"] == (1, 1, 2048)
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_hd128"
    assert captured["causal"] is True


def test_build_simt_callable_rejects_unsupported_family():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        seed=7,
        implementation="simt",
    )

    ctx = TorchOperatorContext(
        backend=SimpleNamespace(),
        torch=SimpleNamespace(),
        request=request,
        case=get_sparse_attention_case("smoke", "tiny_decode_top4"),
        device="npu",
        dtype="float16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(sparse_attention_forward=lambda *args, **kwargs: None)
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="does not support this shape family",
    ):
        _build_simt_callable(ctx)
