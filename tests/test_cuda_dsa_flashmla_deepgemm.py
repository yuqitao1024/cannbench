import importlib
import sys
from types import SimpleNamespace

import pytest


def _reload_adapter():
    sys.modules.pop("cannbench_cuda_dsa_flashmla_deepgemm", None)
    return importlib.import_module("cannbench_cuda_dsa_flashmla_deepgemm")


class _FakeTensor:
    def __init__(self, shape, values=None):
        self.shape = shape
        self.values = values
        self.device = "cuda"

    def reshape(self, *shape):
        return _FakeTensor(shape, self.values)

    def to(self, dtype):
        return self

    def contiguous(self):
        return self

    def tolist(self):
        return self.values


class _FakeTorch:
    float8_e4m3fn = "float8"
    bfloat16 = "bfloat16"
    float32 = "float32"
    int32 = "int32"

    @staticmethod
    def full(shape, value, *, device, dtype):
        del device, dtype
        return _FakeTensor(shape, [[value] * shape[1] for _ in range(shape[0])])


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
    values = keys[..., :2]

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
                "shared_kv": True,
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
        [[10.0, 11.0, 12.0, 13.0]],
        [[20.0, 21.0, 22.0, 23.0]],
    ]
    assert result["d_v"] == 2


def test_flashmla_shared_kv_reuses_canonical_key_tensor():
    adapter = _reload_adapter()
    keys = object()
    values = object()

    result = adapter._flash_mla_shared_kv(
        None,
        keys,
        values,
        qk_head_dim=4,
        value_head_dim=2,
        shared_kv=True,
    )

    assert result is keys


def test_flashmla_v32_decode_cache_uses_656_byte_layout():
    torch = pytest.importorskip("torch")
    adapter = _reload_adapter()
    cache = torch.randn(1, 2, 1, 576, dtype=torch.bfloat16)

    packed = adapter._flash_mla_fp8_k_cache(torch, cache)

    assert packed.shape == (1, 2, 1, 656)
    assert packed.dtype == torch.float8_e4m3fn


def test_deepgemm_decode_indexer_supports_nextn_queries(monkeypatch):
    adapter = _reload_adapter()
    metadata_calls = []
    torch = _FakeTorch()
    deep_gemm = SimpleNamespace(
        get_num_sms=lambda: 120,
        get_paged_mqa_logits_metadata=lambda *args, **kwargs: metadata_calls.append(
            (args, kwargs)
        )
        or "metadata",
    )
    monkeypatch.setattr(
        adapter,
        "_deep_gemm_fp8_kv_cache",
        lambda torch_module, tensor: tensor,
    )
    monkeypatch.setattr(
        adapter,
        "_sequential_block_table",
        lambda *args, **kwargs: (_FakeTensor((2, 512)), 32768),
    )
    monkeypatch.setattr(
        adapter,
        "_blocked_kv_cache",
        lambda *args, **kwargs: _FakeTensor((1024, 64, 1, 128)),
    )

    result = adapter._deep_gemm_decode_indexer_kwargs(
        deep_gemm,
        {
            "torch": torch,
            "payload": {
                "query_shape": (2, 2, 64, 128),
                "key_shape": (2, 32768, 128),
                "weight_shape": (2, 2, 64),
            },
            "query": _FakeTensor((2, 2, 64, 128)),
            "keys": _FakeTensor((2, 32768, 128)),
            "weights": _FakeTensor((2, 2, 64)),
        },
    )

    assert result["q"].shape == (2, 2, 64, 128)
    assert result["weights"].shape == (4, 64)
    assert result["context_lens"].shape == (2, 2)
    assert result["context_lens"].tolist() == [
        [32768, 32768],
        [32768, 32768],
    ]
    assert metadata_calls[0][0][0] is result["context_lens"]
    assert result["schedule_meta"] == "metadata"


def test_flashmla_decode_attention_supports_multiple_query_tokens(monkeypatch):
    adapter = _reload_adapter()
    torch = _FakeTorch()
    monkeypatch.setattr(
        adapter,
        "_flash_mla_fp8_k_cache",
        lambda torch_module, tensor: tensor,
    )
    monkeypatch.setattr(
        adapter,
        "_bhtd_to_bthd",
        lambda tensor, batch, heads, tokens, dim: _FakeTensor(
            (batch, tokens, heads, dim)
        ),
    )
    monkeypatch.setattr(
        adapter,
        "_sequential_block_table",
        lambda *args, **kwargs: (_FakeTensor((2, 512)), 32768),
    )
    monkeypatch.setattr(
        adapter,
        "_flash_mla_shared_kv",
        lambda *args, **kwargs: _FakeTensor((2, 1, 32768, 576)),
    )
    monkeypatch.setattr(
        adapter,
        "_blocked_kv_cache",
        lambda *args, **kwargs: _FakeTensor((1024, 64, 1, 576)),
    )
    source_indices = _FakeTensor((2, 2, 2048))
    converted_indices = _FakeTensor((2, 2, 2048))
    monkeypatch.setattr(
        adapter,
        "_indices_to_kvcache_indices",
        lambda *args, **kwargs: converted_indices,
    )
    flash_mla = SimpleNamespace(get_mla_metadata=lambda: ("metadata", None))

    result = adapter._flash_mla_decode_attention_kwargs(
        flash_mla,
        {
            "torch": torch,
            "payload": {
                "query_shape": (2, 128, 2, 576),
                "key_shape": (2, 1, 32768, 576),
                "value_shape": (2, 1, 32768, 512),
                "indices_shape": (2, 2, 2048),
                "shared_kv": True,
            },
            "query": _FakeTensor((2, 128, 2, 576)),
            "keys": _FakeTensor((2, 1, 32768, 576)),
            "values": _FakeTensor((2, 1, 32768, 512)),
            "indices": source_indices,
        },
    )

    assert result["q"].shape == (2, 2, 128, 576)
    assert result["indices"] is converted_indices
    assert result["indices"].shape == (2, 2, 2048)
    assert result["tile_scheduler_metadata"] == "metadata"
