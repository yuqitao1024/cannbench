import importlib
import sys
from types import SimpleNamespace

import pytest


def _reload_adapter():
    sys.modules.pop("cannbench_cuda_dsa_flashmla_deepgemm", None)
    return importlib.import_module("cannbench_cuda_dsa_flashmla_deepgemm")


def test_flashmla_deepgemm_adapter_routes_decode_indexer_to_paged_logits(
    monkeypatch,
):
    calls = []
    fake_deep_gemm = SimpleNamespace(
        fp8_paged_mqa_logits=lambda **kwargs: calls.append(
            ("fp8_paged_mqa_logits", kwargs)
        )
        or "decode-indices",
        fp8_mqa_logits=lambda **kwargs: calls.append(("fp8_mqa_logits", kwargs)),
    )
    monkeypatch.setitem(sys.modules, "deep_gemm", fake_deep_gemm)
    adapter = _reload_adapter()
    monkeypatch.setattr(
        adapter,
        "_deep_gemm_decode_indexer_kwargs",
        lambda deep_gemm, kwargs: {"q": "q_native", "kv_cache": "kv_native"},
    )

    result = adapter.lightning_indexer(
        payload={"top_k": 512},
        case=SimpleNamespace(phase="decode"),
        query="q",
        keys="k",
        weights="w",
    )

    assert result == "decode-indices"
    assert calls == [
        (
            "fp8_paged_mqa_logits",
            {"q": "q_native", "kv_cache": "kv_native"},
        )
    ]


def test_flashmla_deepgemm_adapter_routes_prefill_indexer_to_logits(monkeypatch):
    calls = []
    fake_deep_gemm = SimpleNamespace(
        fp8_paged_mqa_logits=lambda **kwargs: calls.append(
            ("fp8_paged_mqa_logits", kwargs)
        ),
        fp8_mqa_logits=lambda **kwargs: calls.append(("fp8_mqa_logits", kwargs))
        or "prefill-indices",
    )
    monkeypatch.setitem(sys.modules, "deep_gemm", fake_deep_gemm)
    adapter = _reload_adapter()
    monkeypatch.setattr(
        adapter,
        "_deep_gemm_prefill_indexer_kwargs",
        lambda kwargs: {"q": "q_native", "kv": "kv_native"},
    )

    result = adapter.lightning_indexer(
        payload={"phase": "prefill"},
        query="q",
        keys="k",
        weights="w",
    )

    assert result == "prefill-indices"
    assert calls == [
        (
            "fp8_mqa_logits",
            {"q": "q_native", "kv": "kv_native"},
        )
    ]


def test_flashmla_deepgemm_adapter_routes_decode_attention_to_flash_mla_decode(
    monkeypatch,
):
    calls = []
    fake_flash_mla = SimpleNamespace(
        flash_mla_with_kvcache=lambda **kwargs: calls.append(
            ("flash_mla_with_kvcache", kwargs)
        )
        or "decode-attn",
        flash_mla_sparse_fwd=lambda **kwargs: calls.append(
            ("flash_mla_sparse_fwd", kwargs)
        ),
    )
    monkeypatch.setitem(sys.modules, "flash_mla", fake_flash_mla)
    adapter = _reload_adapter()
    monkeypatch.setattr(
        adapter,
        "_flash_mla_decode_attention_kwargs",
        lambda flash_mla, kwargs: {
            "q": "q_native",
            "k_cache": "kv_native",
            "indices": "indices_native",
        },
    )

    result = adapter.sparse_attention(
        payload={"phase": "decode"},
        query="q",
        keys="k",
        values="v",
        indices="i",
    )

    assert result == "decode-attn"
    assert calls == [
        (
            "flash_mla_with_kvcache",
            {"q": "q_native", "k_cache": "kv_native", "indices": "indices_native"},
        )
    ]


def test_flashmla_deepgemm_adapter_routes_prefill_attention_to_flash_mla_sparse_fwd(
    monkeypatch,
):
    calls = []
    fake_flash_mla = SimpleNamespace(
        flash_mla_with_kvcache=lambda **kwargs: calls.append(
            ("flash_mla_with_kvcache", kwargs)
        ),
        flash_mla_sparse_fwd=lambda **kwargs: calls.append(
            ("flash_mla_sparse_fwd", kwargs)
        )
        or "prefill-attn",
    )
    monkeypatch.setitem(sys.modules, "flash_mla", fake_flash_mla)
    adapter = _reload_adapter()
    monkeypatch.setattr(
        adapter,
        "_flash_mla_prefill_attention_kwargs",
        lambda kwargs: {"q": "q_native", "kv": "kv_native", "indices": "indices_native"},
    )

    result = adapter.sparse_attention(
        payload={"phase": "prefill"},
        query="q",
        keys="k",
        values="v",
        indices="i",
    )

    assert result == "prefill-attn"
    assert calls == [
        (
            "flash_mla_sparse_fwd",
            {"q": "q_native", "kv": "kv_native", "indices": "indices_native"},
        )
    ]


def test_flashmla_deepgemm_adapter_reports_missing_phase():
    adapter = _reload_adapter()

    with pytest.raises(RuntimeError, match="phase"):
        adapter.sparse_attention(payload={}, query="q")
