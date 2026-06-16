import builtins
import sys
from types import SimpleNamespace

import pytest

from cannbench.backends import get_backend
from cannbench.core.config import OperatorBenchmarkRequest


def test_get_backend_returns_nvidia_backend():
    backend = get_backend("nvidia")
    assert backend.name == "nvidia"


def test_get_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_backend("unknown")


def test_get_backend_rejects_ascend_from_public_factory():
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_backend("ascend")


def test_backend_raises_clear_error_when_torch_is_missing(monkeypatch):
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "torch":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=1,
        iterations=1,
    )

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        backend.run_softmax(request)


def test_backend_raises_clear_error_when_cuda_is_unavailable(monkeypatch):
    from cannbench.backends.pytorch_backend import NvidiaBackend

    fake_torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=1,
        iterations=1,
    )

    with pytest.raises(RuntimeError, match="CUDA is required"):
        backend.run_softmax(request)


def test_backend_rejects_non_positive_iterations():
    with pytest.raises(ValueError, match="iterations must be > 0"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            warmup=1,
            iterations=0,
        )


def test_backend_rejects_negative_warmup():
    with pytest.raises(ValueError, match="warmup must be >= 0"):
        OperatorBenchmarkRequest(
            backend="nvidia",
            op="softmax",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_logits",
            warmup=-1,
            iterations=1,
        )
