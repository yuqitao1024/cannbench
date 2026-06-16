import builtins
import sys
from types import SimpleNamespace

import pytest

from cannbench.backends import get_backend
from cannbench.core.config import OperatorBenchmarkRequest


def test_get_backend_returns_nvidia_backend():
    backend = get_backend("nvidia")
    assert backend.name == "nvidia"


def test_get_backend_returns_ascend_backend():
    backend = get_backend("ascend")
    assert backend.name == "ascend"


def test_get_backend_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unsupported backend"):
        get_backend("unknown")


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


def test_ascend_backend_raises_clear_error_when_torch_npu_is_missing(monkeypatch):
    original_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name == "torch_npu":
            raise ModuleNotFoundError(name)
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=1,
        iterations=1,
    )

    with pytest.raises(RuntimeError, match="torch_npu is required"):
        backend.run_softmax(request)


def test_ascend_backend_raises_clear_error_when_npu_is_unavailable(monkeypatch):
    from cannbench.backends.pytorch_backend import AscendBackend

    fake_torch = SimpleNamespace(
        npu=SimpleNamespace(is_available=lambda: False),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    backend = AscendBackend()
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=1,
        iterations=1,
    )

    with pytest.raises(RuntimeError, match="Ascend NPU is required"):
        backend.run_softmax(request)


def test_ascend_backend_materializes_softmax_inputs_from_seed(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def reshape(self, shape):
            captured["shape"] = shape
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.npu = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: captured.setdefault("synchronized", True),
                get_device_name=lambda device: "Fake Ascend",
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
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_softmax(request)

    assert result.backend == "ascend"
    assert result.device_name == "Fake Ascend"
    assert captured["shape"] == (32, 128)
    assert captured["device"] == "npu"
    assert captured["dtype"] == "float16"
    assert captured["synchronized"] is True
    assert captured["values"]


def test_ascend_backend_skips_custom_op_deployment_when_disabled(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def reshape(self, shape):
            del shape
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.npu = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake Ascend",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.softmax = lambda tensor, dim: tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(
        backend,
        "_deploy_custom_op",
        lambda request, op_name: captured.setdefault("called", True),
    )

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=1,
        iterations=1,
        seed=7,
    )

    backend.run_softmax(request)

    assert "called" not in captured


def test_ascend_backend_deploys_default_custom_op_when_enabled(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    op_dir = (
        tmp_path
        / "src"
        / "cannbench"
        / "datasets"
        / "data"
        / "softmax"
        / "custom_ops"
        / "ascend"
        / "default"
    )
    op_dir.mkdir(parents=True)
    install_script = op_dir / "install.sh"
    install_script.write_text("#!/bin/sh\nexit 0\n")

    class FakeTorch:
        def __init__(self) -> None:
            self.npu = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake Ascend",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.tensor = lambda *args, **kwargs: SimpleNamespace(reshape=lambda shape: SimpleNamespace())
            self.softmax = lambda tensor, dim: tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_custom_op_base_dir", lambda op_name: op_dir)
    monkeypatch.setattr(
        backend,
        "_run_custom_op_install",
        lambda script: captured.setdefault("script", script),
    )

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        warmup=1,
        iterations=1,
        seed=7,
        deploy_custom_op=True,
    )

    backend.run_softmax(request)

    assert captured["script"] == install_script


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


def test_backend_runs_masked_select_with_materialized_inputs(monkeypatch):
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
            self.bool = "bool"
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

        def masked_select(self, input_tensor, mask_tensor):
            captured["masked_select_input"] = input_tensor
            captured["masked_select_mask"] = mask_tensor
            return input_tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="masked_select",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_masked_select",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "masked_select"
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "bool"


def test_backend_runs_cross_entropy_with_materialized_inputs(monkeypatch):
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
            self.nn = SimpleNamespace(
                Embedding=lambda *args, **kwargs: None,
                functional=SimpleNamespace(
                    cross_entropy=self.cross_entropy,
                ),
            )

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

        def cross_entropy(self, logits, targets):
            captured["cross_entropy_logits"] = logits
            captured["cross_entropy_targets"] = targets
            return logits

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="cross_entropy",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_token_classification_loss",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "cross_entropy"
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"


def test_backend_runs_scatter_add_with_materialized_inputs(monkeypatch):
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
            self.nn = SimpleNamespace(
                Embedding=lambda *args, **kwargs: None,
                functional=SimpleNamespace(cross_entropy=lambda *args, **kwargs: None),
            )

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

        def scatter_add(self, input_tensor, dim, index_tensor, src_tensor):
            captured["scatter_add_dim"] = dim
            captured["scatter_add_input"] = input_tensor
            captured["scatter_add_index"] = index_tensor
            captured["scatter_add_src"] = src_tensor
            return input_tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="scatter_add",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_scatter_add",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "scatter_add"
    assert captured["scatter_add_dim"] == 1
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"
    assert captured["tensor_calls"][2]["dtype"] == "float16"


def test_backend_runs_index_add_with_materialized_inputs(monkeypatch):
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
            self.nn = SimpleNamespace(
                Embedding=lambda *args, **kwargs: None,
                functional=SimpleNamespace(cross_entropy=lambda *args, **kwargs: None),
            )

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

        def index_add(self, input_tensor, dim, index_tensor, src_tensor):
            captured["index_add_dim"] = dim
            captured["index_add_input"] = input_tensor
            captured["index_add_index"] = index_tensor
            captured["index_add_src"] = src_tensor
            return input_tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="index_add",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_add",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "index_add"
    assert captured["index_add_dim"] == 1
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"
    assert captured["tensor_calls"][2]["dtype"] == "float16"


def test_backend_runs_scatter_with_materialized_inputs(monkeypatch):
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
            self.nn = SimpleNamespace(
                Embedding=lambda *args, **kwargs: None,
                functional=SimpleNamespace(cross_entropy=lambda *args, **kwargs: None),
            )

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

        def scatter(self, input_tensor, dim, index_tensor, src_tensor):
            captured["scatter_dim"] = dim
            captured["scatter_input"] = input_tensor
            captured["scatter_index"] = index_tensor
            captured["scatter_src"] = src_tensor
            return input_tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="scatter",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_scatter",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "scatter"
    assert captured["scatter_dim"] == 1
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"
    assert captured["tensor_calls"][2]["dtype"] == "float16"


def test_backend_runs_index_put_with_materialized_inputs(monkeypatch):
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
            self.nn = SimpleNamespace(
                Embedding=lambda *args, **kwargs: None,
                functional=SimpleNamespace(cross_entropy=lambda *args, **kwargs: None),
            )

        def _tensor(self, values, device=None, dtype=None):
            captured.setdefault("tensor_calls", []).append(
                {
                    "values": values,
                    "device": device,
                    "dtype": dtype,
                }
            )
            return FakeTensor()

        def index_put(self, input_tensor, indices, values, accumulate=False):
            captured["index_put_indices"] = indices
            captured["index_put_values"] = values
            captured["index_put_accumulate"] = accumulate
            return input_tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="index_put",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_put",
        warmup=1,
        iterations=1,
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "index_put"
    assert len(captured["index_put_indices"]) == 2
    assert captured["index_put_accumulate"] is False
    assert captured["tensor_calls"][0]["dtype"] == "float16"
    assert captured["tensor_calls"][1]["dtype"] == "long"
    assert captured["tensor_calls"][2]["dtype"] == "long"
    assert captured["tensor_calls"][3]["dtype"] == "float16"


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
