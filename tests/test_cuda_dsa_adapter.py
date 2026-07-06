import importlib
from types import SimpleNamespace

import pytest


def test_default_cuda_dsa_adapter_exports_required_callables():
    adapter = importlib.import_module("cannbench_cuda_dsa")

    assert callable(adapter.lightning_indexer)
    assert callable(adapter.sparse_attention)


def test_default_cuda_dsa_adapter_reports_missing_lightning_indexer_dependency(monkeypatch):
    adapter = importlib.import_module("cannbench_cuda_dsa")

    monkeypatch.delenv("CANNBENCH_CUDA_DSA_LIGHTNING_INDEXER", raising=False)

    with pytest.raises(RuntimeError, match="phase"):
        adapter.lightning_indexer()


def test_default_cuda_dsa_adapter_reports_missing_sparse_attention_dependency(monkeypatch):
    adapter = importlib.import_module("cannbench_cuda_dsa")

    monkeypatch.delenv("CANNBENCH_CUDA_DSA_SPARSE_ATTENTION", raising=False)

    with pytest.raises(RuntimeError, match="phase"):
        adapter.sparse_attention()


def test_default_cuda_dsa_adapter_reports_bad_explicit_lightning_indexer_dependency(
    monkeypatch,
):
    adapter = importlib.import_module("cannbench_cuda_dsa")

    monkeypatch.setenv(
        "CANNBENCH_CUDA_DSA_LIGHTNING_INDEXER", "missing.module:lightning_indexer"
    )

    with pytest.raises(RuntimeError, match="CANNBENCH_CUDA_DSA_LIGHTNING_INDEXER"):
        adapter.lightning_indexer()


def test_default_cuda_dsa_adapter_reports_bad_explicit_sparse_attention_dependency(
    monkeypatch,
):
    adapter = importlib.import_module("cannbench_cuda_dsa")

    monkeypatch.setenv(
        "CANNBENCH_CUDA_DSA_SPARSE_ATTENTION", "missing.module:sparse_attention"
    )

    with pytest.raises(RuntimeError, match="CANNBENCH_CUDA_DSA_SPARSE_ATTENTION"):
        adapter.sparse_attention()


def test_default_cuda_dsa_adapter_dispatches_configured_lightning_indexer(monkeypatch):
    adapter = importlib.import_module("cannbench_cuda_dsa")
    calls = []

    def fake_kernel(**kwargs):
        calls.append(kwargs)
        return "indices"

    monkeypatch.setattr(adapter, "_import_symbol", lambda spec, **kwargs: fake_kernel)
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_LIGHTNING_INDEXER", "fake.module:kernel")

    result = adapter.lightning_indexer(query="q", top_k=512)

    assert result == "indices"
    assert calls == [{"query": "q", "top_k": 512}]


def test_default_cuda_dsa_adapter_can_use_module_level_callable(monkeypatch):
    adapter = importlib.import_module("cannbench_cuda_dsa")
    fake_module = SimpleNamespace(sparse_attention=lambda **kwargs: "out")

    monkeypatch.setattr(adapter.importlib, "import_module", lambda name: fake_module)
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_SPARSE_ATTENTION", "fake.module")

    assert adapter.sparse_attention(query="q") == "out"
