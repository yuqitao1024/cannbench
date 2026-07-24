from __future__ import annotations

from types import SimpleNamespace

import pytest

from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.operators.builtin.lightning_indexer import (
    _build_simt_callable,
    _select_simt_family,
    _simt_module_name,
    get_lightning_indexer_case,
)
from cannbench.operators.plugin import TorchOperatorContext


def test_select_simt_family_prefers_64x128_family():
    payload = {
        "index_heads": 64,
        "index_dim": 128,
        "phase": "prefill",
        "top_k": 1024,
    }

    assert _select_simt_family(payload) == "family_64x128"


def test_select_simt_family_prefers_32x128_family():
    payload = {
        "index_heads": 32,
        "index_dim": 128,
        "phase": "prefill",
        "top_k": 2048,
    }

    assert _select_simt_family(payload) == "family_32x128"


def test_select_simt_family_prefers_4x64_family():
    payload = {
        "index_heads": 4,
        "index_dim": 64,
        "phase": "decode",
        "top_k": 2048,
    }

    assert _select_simt_family(payload) == "family_4x64"


def test_select_simt_family_falls_back_for_unknown_shape():
    payload = {
        "index_heads": 8,
        "index_dim": 96,
        "phase": "prefill",
        "top_k": 256,
    }

    assert _select_simt_family(payload) == "fallback"


def test_simt_module_name_registers_only_v1():
    assert _simt_module_name(None) == "aten_dsa_lightning_indexer"
    assert _simt_module_name("v1") == "aten_dsa_lightning_indexer"
    assert _simt_module_name("v2") is None


def test_build_simt_callable_requires_loaded_module():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="lightning_indexer",
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
        case=get_lightning_indexer_case("smoke", "tiny_decode_top4"),
        device="npu",
        dtype="float16",
        implementation_module=None,
    )

    with pytest.raises(
        RuntimeError,
        match="lightning_indexer SIMT implementation module is not loaded",
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

    def fake_forward(query, keys, weights, *, top_k, phase, family):
        captured["query_shape"] = query.shape
        captured["key_shape"] = keys.shape
        captured["weight_shape"] = weights.shape
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return "ok"

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="lightning_indexer",
        dtype="float16",
        dataset="realistic",
        case_id="llama4_decode_32760_top2048",
        seed=7,
        implementation="simt",
    )

    ctx = TorchOperatorContext(
        backend=FakeBackend(),
        torch=SimpleNamespace(),
        request=request,
        case=get_lightning_indexer_case("realistic", "llama4_decode_32760_top2048"),
        device="npu",
        dtype="float16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(lightning_indexer_forward=fake_forward)
        ),
    )

    operator = _build_simt_callable(ctx)
    assert operator() == "ok"
    assert captured["query_shape"] == (16, 1, 4, 64)
    assert captured["key_shape"] == (16, 32760, 64)
    assert captured["weight_shape"] == (16, 1, 4)
    assert captured["top_k"] == 2048
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_4x64"


def test_plugin_exposes_supported_prefill_and_decode_simt_cases():
    prefill_case = get_lightning_indexer_case("realistic", "opt_prefill_2048_top512")
    decode_case = get_lightning_indexer_case("realistic", "llama4_decode_32760_top2048")

    assert prefill_case.phase == "prefill"
    assert prefill_case.index_heads == 4
    assert prefill_case.index_dim == 64
    assert decode_case.phase == "decode"
    assert decode_case.index_heads == 4
    assert decode_case.index_dim == 64
