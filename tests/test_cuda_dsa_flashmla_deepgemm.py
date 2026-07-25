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


def test_flashmla_prefill_uses_distinct_qk_and_value_dimensions():
    torch = pytest.importorskip("torch")
    adapter = _reload_adapter()
    query = torch.arange(8, dtype=torch.float32).reshape(1, 2, 1, 4)
    keys = torch.tensor(
        [[[[10.0, 11.0, 12.0, 13.0], [20.0, 21.0, 22.0, 23.0]]]]
    )
    values = torch.tensor([[[[30.0, 31.0], [40.0, 41.0]]]])

    result = adapter._flash_mla_prefill_attention_kwargs(
        {
            "torch": torch,
            "payload": {
                "query_shape": (1, 2, 1, 4),
                "key_shape": (1, 1, 2, 4),
                "value_shape": (1, 1, 2, 2),
                "indices_shape": (1, 1, 1),
                "qk_head_dim": 4,
                "value_head_dim": 2,
            },
            "query": query,
            "keys": keys,
            "values": values,
            "indices": torch.tensor([[[1]]]),
        }
    )

    assert result["q"].shape == (1, 2, 4)
    assert result["kv"].shape == (2, 1, 4)
    assert result["kv"].float().tolist() == [
        [[30.0, 31.0, 12.0, 13.0]],
        [[40.0, 41.0, 22.0, 23.0]],
    ]
    assert result["d_v"] == 2


def test_flashmla_v32_decode_cache_uses_656_byte_layout():
    torch = pytest.importorskip("torch")
    adapter = _reload_adapter()
    cache = torch.randn(1, 2, 1, 576, dtype=torch.bfloat16)

    packed = adapter._flash_mla_fp8_k_cache(torch, cache)

    assert packed.shape == (1, 2, 1, 656)
    assert packed.dtype == torch.float8_e4m3fn
