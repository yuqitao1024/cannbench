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


def test_backend_materializes_softmax_inputs_from_seed(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def reshape(self, shape):
            captured["shape"] = shape
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake GPU",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.tensor = self._tensor
            self.softmax = lambda tensor, dim: tensor

        def _tensor(self, values, device=None, dtype=None):
            captured["values"] = values
            captured["device"] = device
            captured["dtype"] = dtype
            return FakeTensor()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

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
        seed=7,
    )

    backend.run_softmax(request)

    assert captured["shape"] == (32, 128)
    assert captured["dtype"] == "float16"
    assert captured["values"]


def test_backend_runs_embedding_with_materialized_inputs(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def reshape(self, *shape):
            captured.setdefault("reshapes", []).append(shape)
            return self

    class FakeEmbeddingModule:
        def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
            captured["embedding_ctor"] = {
                "num_embeddings": num_embeddings,
                "embedding_dim": embedding_dim,
                "device": device,
                "dtype": dtype,
            }
            self.weight = None

        def __call__(self, indices):
            captured["indices_tensor"] = indices
            return FakeTensor()

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake GPU",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.long = "long"
            self.tensor = self._tensor
            self.softmax = lambda tensor, dim: tensor
            self.nn = SimpleNamespace(Embedding=FakeEmbeddingModule)

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="embedding",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_token_lookup",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "embedding"
    assert captured["embedding_ctor"]["num_embeddings"] == 128
    assert captured["embedding_ctor"]["embedding_dim"] == 64
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"


def test_backend_runs_gather_with_materialized_inputs(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def reshape(self, *shape):
            captured.setdefault("reshapes", []).append(shape)
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake GPU",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.long = "long"
            self.tensor = self._tensor
            self.softmax = lambda tensor, dim: tensor
            self.nn = SimpleNamespace(Embedding=lambda *args, **kwargs: None)

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

        def gather(self, input_tensor, dim, index_tensor):
            captured["gather_dim"] = dim
            captured["gather_input"] = input_tensor
            captured["gather_index"] = index_tensor
            return input_tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="gather",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_gather",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "gather"
    assert captured["gather_dim"] == 1
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"


def test_backend_runs_index_select_with_materialized_inputs(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def reshape(self, *shape):
            captured.setdefault("reshapes", []).append(shape)
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake GPU",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.long = "long"
            self.tensor = self._tensor
            self.softmax = lambda tensor, dim: tensor
            self.nn = SimpleNamespace(Embedding=lambda *args, **kwargs: None)

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

        def index_select(self, input_tensor, dim, index_tensor):
            captured["index_select_dim"] = dim
            captured["index_select_input"] = input_tensor
            captured["index_select_index"] = index_tensor
            return input_tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="index_select",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_select",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "index_select"
    assert captured["index_select_dim"] == 1
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"


def test_backend_runs_take_along_dim_with_materialized_inputs(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def reshape(self, *shape):
            captured.setdefault("reshapes", []).append(shape)
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake GPU",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.long = "long"
            self.tensor = self._tensor
            self.softmax = lambda tensor, dim: tensor
            self.nn = SimpleNamespace(Embedding=lambda *args, **kwargs: None)

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

        def take_along_dim(self, input_tensor, indices, dim):
            captured["take_along_dim_dim"] = dim
            captured["take_along_dim_input"] = input_tensor
            captured["take_along_dim_indices"] = indices
            return input_tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="take_along_dim",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_take_along_dim",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "take_along_dim"
    assert captured["take_along_dim_dim"] == 1
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"


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
