import builtins
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from cannbench.backends import get_backend
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.core.profile import LocalDeviceProfileResult


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
    )

    with pytest.raises(RuntimeError, match="PyTorch is required"):
        backend.run_operator(request)


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
    )

    with pytest.raises(RuntimeError, match="CUDA is required"):
        backend.run_operator(request)


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
    )

    with pytest.raises(RuntimeError, match="torch_npu is required"):
        backend.run_operator(request)


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
    )

    with pytest.raises(RuntimeError, match="Ascend NPU is required"):
        backend.run_operator(request)


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
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.backend == "ascend"
    assert result.device_name == "Fake Ascend"
    assert captured["shape"] == (32, 128)
    assert captured["device"] == "npu"
    assert captured["dtype"] == "float16"
    assert captured["synchronized"] is True
    assert captured["values"]


def test_ascend_backend_skips_simt_op_deployment_when_disabled(monkeypatch):
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
        "_install_simt_op",
        lambda request, op_name: captured.setdefault("called", True),
    )

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )

    backend.run_operator(request)

    assert "called" not in captured


def test_ascend_backend_deploys_v1_simt_op_when_enabled(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    op_dir = (
        tmp_path
        / "src"
        / "cannbench"
        / "operators"
        / "builtin"
        / "softmax"
        / "simt"
        / "v1"
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
    monkeypatch.setattr(backend, "_simt_op_root", lambda op_name: op_dir.parent)
    monkeypatch.setattr(
        backend,
        "_run_simt_op_install",
        lambda script: captured.setdefault("script", script),
    )
    fake_module = SimpleNamespace(
        ops=SimpleNamespace(spatial_softmax_forward=lambda tensor, dim: tensor)
    )

    def fake_load_simt_op_module(request, op_name):
        captured.setdefault("loaded", (op_name, request.implementation_version or "v1"))
        return fake_module

    monkeypatch.setattr(backend, "_load_simt_op_module", fake_load_simt_op_module)
    monkeypatch.setitem(
        sys.modules,
        "aten_softmax",
        SimpleNamespace(ops=SimpleNamespace(spatial_softmax_forward=lambda tensor, dim: tensor)),
    )

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        implementation="simt",
    )

    backend.run_operator(request)

    assert captured["script"] == install_script
    assert captured["loaded"] == ("softmax", "v1")


def test_ascend_backend_deploys_requested_simt_op_version(monkeypatch, tmp_path):
    captured: dict[str, object] = {}
    root = (
        tmp_path
        / "src"
        / "cannbench"
        / "operators"
        / "builtin"
        / "softmax"
        / "simt"
    )
    op_dir = root / "v2"
    op_dir.mkdir(parents=True)
    install_script = op_dir / "install.sh"
    install_script.write_text("#!/bin/sh\nexit 0\n")

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_simt_op_root", lambda op_name: root)
    monkeypatch.setattr(
        backend,
        "_run_simt_op_install",
        lambda script: captured.setdefault("script", script),
    )
    monkeypatch.setattr(
        backend,
        "_load_simt_op_module",
        lambda request, op_name: captured.setdefault(
            "loaded", (op_name, request.implementation_version)
        ),
    )

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        implementation="simt",
        implementation_version="v2",
    )

    backend._install_simt_op(request, "softmax")

    assert captured["script"] == install_script
    assert captured["loaded"] == ("softmax", "v2")


def test_ascend_backend_resolves_simt_op_under_plugin_directory(monkeypatch, tmp_path):
    plugin_root = tmp_path / "operators" / "builtin" / "softmax" / "simt"
    op_dir = plugin_root / "v1"
    op_dir.mkdir(parents=True)
    install_script = op_dir / "install.sh"
    install_script.write_text("#!/bin/sh\nexit 0\n")

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_simt_op_root", lambda op_name: plugin_root)

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        implementation="simt",
    )

    assert backend._simt_op_base_dir(request, "softmax") == op_dir


def test_ascend_backend_profiles_index_add_with_msprof(monkeypatch):
    captured: dict[str, object] = {}

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(device):
            del device
            return "Ascend 950PR"

    class FakeTorch:
        npu = FakeNpu()
        device = staticmethod(lambda kind: kind)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    monkeypatch.setattr(AscendBackend, "_install_simt_op", lambda self, request, op_name: None)

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None, check=None):
        del text, capture_output, check
        captured["command"] = command
        captured["cwd"] = cwd
        captured["env"] = env
        profile_dir = cwd / "profile"
        perf_dir = cwd / "perf"
        profile_dir.mkdir(parents=True, exist_ok=True)
        perf_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "summary.csv").write_text(
            "Op Name,Task Duration(us)\nindex_add,1000\n",
            encoding="utf-8",
        )
        (perf_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="[msprof]\n", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="index_add",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_add",
        seed=7,
    )

    result = AscendBackend().profile_operator_device_time(request)

    command = captured["command"]
    assert command[:3] == ["msprof", "op", f"--output={captured['cwd'] / 'profile'}"]
    assert "--launch-skip-before-match=1" not in command
    assert "--warm-up=2" not in command
    assert "--launch-count=1" in command
    assert "internal-run" in command
    assert command[command.index("--backend") + 1] == "ascend"
    assert "--warmup" not in command
    assert "--iterations" not in command
    assert command[command.index("--implementation") + 1] == "simt"
    assert result.profile.device_name == "Ascend 950PR"
    assert result.profile.profile_summary.backend == "ascend"
    assert result.profile.profile_summary.latency_ms == 1.0
    assert (
        "summary.csv",
        b"Op Name,Task Duration(us)\nindex_add,1000\n",
    ) in result.profile.profile_artifacts
    assert captured["env"]["PYTHONPATH"]
    assert captured["env"]["CANNBENCH_SKIP_SIMT_INSTALL"] == "1"


def test_ascend_backend_profiles_cann_index_add_past_tensor_move(monkeypatch):
    captured: dict[str, object] = {}

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(device):
            del device
            return "Ascend 950PR"

    class FakeTorch:
        npu = FakeNpu()
        device = staticmethod(lambda kind: kind)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None, check=None):
        del env, text, capture_output, check
        captured["command"] = command
        profile_dir = cwd / "profile"
        perf_dir = cwd / "perf"
        profile_dir.mkdir(parents=True, exist_ok=True)
        perf_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "summary.csv").write_text(
            "Op Name,Task Duration(us)\n"
            "TensorMove_d2db1a80c523e7e59a032c95969880af_high_performance_2,4.588\n"
            "InplaceIndexAdd_20d0b91f852eb04b8f161ab3cc623d32_high_performance_101001_mix_aiv,9.750\n",
            encoding="utf-8",
        )
        (perf_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    from cannbench.backends.pytorch_backend import AscendBackend

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="cann_ops_library",
        op="index_add",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_add",
        seed=7,
    )

    result = AscendBackend().profile_operator_device_time(request)

    command = captured["command"]
    assert "--launch-count=1" in command
    assert result.profile.profile_summary.latency_ms == 0.00975


def test_ascend_backend_installs_simt_before_msprof_without_deploying_inside(monkeypatch):
    captured: dict[str, object] = {}

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def get_device_name(device):
            del device
            return "Ascend 950PR"

    class FakeTorch:
        npu = FakeNpu()
        device = staticmethod(lambda kind: kind)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(
        backend,
        "_install_simt_op",
        lambda request, op_name: captured.setdefault(
            "installed", (op_name, request.implementation_version)
        ),
    )

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None, check=None):
        del env, text, capture_output, check
        captured["command"] = command
        profile_dir = cwd / "profile"
        profile_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "summary.csv").write_text(
            "Op Name,Task Duration(us)\nindex_add,1000\n",
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="index_add",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_add",
    )

    backend.profile_operator_device_time(request)

    command = captured["command"]
    assert captured["installed"] == ("index_add", "v1")
    assert "--launch-skip-before-match=1" not in command
    assert "--warm-up=0" not in command
    assert "--launch-count=1" in command
    assert "--use-simt-op" not in command
    assert command[command.index("--implementation") + 1] == "simt"
    assert command[command.index("--implementation-version") + 1] == "v1"


def test_ascend_backend_runs_simt_softmax_through_registered_op(monkeypatch):
    captured: dict[str, object] = {"torch_softmax_calls": 0, "simt_calls": 0}

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
            self.softmax = self._softmax

        def _softmax(self, tensor, dim):
            del tensor, dim
            captured["torch_softmax_calls"] += 1
            return FakeTensor()

    fake_ops_module = SimpleNamespace(
        spatial_softmax_forward=lambda tensor, dim: captured.__setitem__(
            "simt_calls", captured["simt_calls"] + 1
        )
        or tensor
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "aten_softmax", SimpleNamespace(ops=fake_ops_module))

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        implementation="simt",
    )

    backend.run_operator(request)

    assert captured["simt_calls"] == 1
    assert captured["torch_softmax_calls"] == 0


def test_ascend_backend_runs_simt_index_add_through_registered_op(monkeypatch):
    captured: dict[str, object] = {
        "torch_index_add_calls": 0,
        "simt_calls": 0,
        "tensor_dtypes": [],
    }

    class FakeTensor:
        def reshape(self, *shape):
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
            self.int32 = "int32"
            self.long = "long"
            self.tensor = self._tensor
            self.index_add = self._index_add

        def _tensor(self, values, device=None, dtype=None):
            del values, device
            captured["tensor_dtypes"].append(dtype)
            return FakeTensor()

        def _index_add(self, input_tensor, dim, index_tensor, src_tensor):
            del input_tensor, dim, index_tensor, src_tensor
            captured["torch_index_add_calls"] += 1
            return FakeTensor()

    fake_ops_module = SimpleNamespace(
        index_add_forward=lambda input_tensor, dim, index_tensor, src_tensor: captured.__setitem__(
            "simt_calls", captured["simt_calls"] + 1
        )
        or input_tensor
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "aten_index_add", SimpleNamespace(ops=fake_ops_module))

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="index_add",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_rank2_index_add",
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "index_add"
    assert captured["simt_calls"] == 1
    assert captured["torch_index_add_calls"] == 0
    assert captured["tensor_dtypes"][1] == "int32"


def test_lightning_indexer_simt_v1_prefers_custom_op_for_prefill_family_4x64(
    monkeypatch,
):
    captured: dict[str, object] = {
        "custom_calls": 0,
        "tensor_dtypes": [],
        "family": None,
        "phase": None,
        "top_k": None,
    }

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
            self.tensor = self._tensor

        def _tensor(self, values, device=None, dtype=None):
            del values, device
            captured["tensor_dtypes"].append(dtype)
            return FakeTensor()

    def fake_custom_forward(query, keys, weights, *, top_k, phase, family):
        del query, keys, weights
        captured["custom_calls"] += 1
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return FakeTensor()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    monkeypatch.setattr(
        backend,
        "_load_simt_op_module",
        lambda request, op_name: SimpleNamespace(
            ops=SimpleNamespace(lightning_indexer_forward=fake_custom_forward)
        ),
    )
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="lightning_indexer",
        dtype="float16",
        dataset="realistic",
        case_id="opt_prefill_2048_top512",
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "lightning_indexer"
    assert captured["custom_calls"] == 1
    assert captured["tensor_dtypes"] == ["float16", "float16", "float16"]
    assert captured["top_k"] == 512
    assert captured["phase"] == "prefill"
    assert captured["family"] == "family_4x64"


def test_lightning_indexer_simt_v1_passes_decode_family_to_simt_module(monkeypatch):
    captured: dict[str, object] = {
        "simt_calls": 0,
        "tensor_dtypes": [],
        "family": None,
        "phase": None,
        "top_k": None,
    }

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
            self.tensor = self._tensor

        def _tensor(self, values, device=None, dtype=None):
            del values, device
            captured["tensor_dtypes"].append(dtype)
            return FakeTensor()

    def fake_simt_forward(query, keys, weights, *, top_k, phase, family):
        del query, keys, weights
        captured["simt_calls"] += 1
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return FakeTensor()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    monkeypatch.setattr(
        backend,
        "_load_simt_op_module",
        lambda request, op_name: SimpleNamespace(
            ops=SimpleNamespace(lightning_indexer_forward=fake_simt_forward)
        ),
    )
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="lightning_indexer",
        dtype="float16",
        dataset="realistic",
        case_id="llama4_decode_32760_top2048",
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "lightning_indexer"
    assert captured["simt_calls"] == 1
    assert captured["tensor_dtypes"] == ["float16", "float16", "float16"]
    assert captured["top_k"] == 2048
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_4x64"


def test_lightning_indexer_simt_v1_prefers_custom_op_for_decode_family_64x128(
    monkeypatch,
):
    captured: dict[str, object] = {
        "custom_calls": 0,
        "tensor_dtypes": [],
        "family": None,
        "phase": None,
        "top_k": None,
    }

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
            self.tensor = self._tensor

        def _tensor(self, values, device=None, dtype=None):
            del values, device
            captured["tensor_dtypes"].append(dtype)
            return FakeTensor()

    def fake_custom_forward(query, keys, weights, *, top_k, phase, family):
        del query, keys, weights
        captured["custom_calls"] += 1
        captured["top_k"] = top_k
        captured["phase"] = phase
        captured["family"] = family
        return FakeTensor()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    monkeypatch.setattr(
        backend,
        "_load_simt_op_module",
        lambda request, op_name: SimpleNamespace(
            ops=SimpleNamespace(lightning_indexer_forward=fake_custom_forward)
        ),
    )
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="lightning_indexer",
        dtype="float16",
        dataset="realistic_decode",
        case_id="vllm_ascend_a5_decode_b1_ctx512_top512",
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "lightning_indexer"
    assert captured["custom_calls"] == 1
    assert captured["tensor_dtypes"] == ["float16", "float16", "float16"]
    assert captured["top_k"] == 512
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_64x128"


def test_ascend_backend_runs_simt_sparse_attention_through_registered_op(monkeypatch):
    captured: dict[str, object] = {
        "simt_calls": 0,
        "tensor_dtypes": [],
        "family": None,
        "phase": None,
        "causal": None,
    }

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
            self.long = "long"
            self.tensor = self._tensor

        def _tensor(self, values, device=None, dtype=None):
            del values, device
            captured["tensor_dtypes"].append(dtype)
            return FakeTensor()

    def fake_forward(query, keys, values, indices, *, phase, family, causal):
        del query, keys, values, indices
        captured["simt_calls"] += 1
        captured["phase"] = phase
        captured["family"] = family
        captured["causal"] = causal
        return FakeTensor()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "aten_dsa_sparse_attention",
        SimpleNamespace(ops=SimpleNamespace(sparse_attention_forward=fake_forward)),
    )

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="sparse_attention",
        dtype="float16",
        dataset="realistic_decode",
        case_id="deepseek_128k_decode_top2048",
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "sparse_attention"
    assert captured["simt_calls"] == 1
    assert captured["tensor_dtypes"] == ["float16", "float16", "float16", "long"]
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_hd128"
    assert captured["causal"] is True


def test_ascend_backend_prefers_sparse_attention_custom_op_for_decode_family_hd512(
    monkeypatch,
):
    captured: dict[str, object] = {
        "simt_calls": 0,
        "tensor_dtypes": [],
        "family": None,
        "phase": None,
        "causal": None,
    }

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
            self.long = "long"
            self.tensor = self._tensor

        def _tensor(self, values, device=None, dtype=None):
            del values, device
            captured["tensor_dtypes"].append(dtype)
            return FakeTensor()

    def fake_forward(query, keys, values, indices, *, phase, family, causal):
        del query, keys, values, indices
        captured["simt_calls"] += 1
        captured["phase"] = phase
        captured["family"] = family
        captured["causal"] = causal
        return (FakeTensor(), FakeTensor())

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "aten_dsa_sparse_attention",
        SimpleNamespace(ops=SimpleNamespace(sparse_attention_forward=fake_forward)),
    )

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="sparse_attention",
        dtype="float16",
        dataset="realistic_decode",
        case_id="vllm_ascend_a5_decode_b1_ctx512_top512",
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "sparse_attention"
    assert captured["simt_calls"] == 1
    assert captured["tensor_dtypes"] == ["float16", "float16", "float16", "long"]
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_hd512"
    assert captured["causal"] is True


def test_ascend_backend_prefers_sparse_attention_custom_op_for_decode_family_hd128(
    monkeypatch,
):
    captured: dict[str, object] = {
        "simt_calls": 0,
        "tensor_dtypes": [],
        "family": None,
        "phase": None,
        "causal": None,
    }

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
            self.long = "long"
            self.tensor = self._tensor

        def _tensor(self, values, device=None, dtype=None):
            del values, device
            captured["tensor_dtypes"].append(dtype)
            return FakeTensor()

    def fake_forward(query, keys, values, indices, *, phase, family, causal):
        del query, keys, values, indices
        captured["simt_calls"] += 1
        captured["phase"] = phase
        captured["family"] = family
        captured["causal"] = causal
        return (FakeTensor(), FakeTensor())

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(
        sys.modules,
        "aten_dsa_sparse_attention",
        SimpleNamespace(ops=SimpleNamespace(sparse_attention_forward=fake_forward)),
    )

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="simt",
        implementation_version="v1",
        op="sparse_attention",
        dtype="float16",
        dataset="realistic_decode",
        case_id="deepseek_128k_decode_top2048",
        seed=7,
    )

    result = backend.run_operator(request)

    assert result.op == "sparse_attention"
    assert captured["simt_calls"] == 1
    assert captured["tensor_dtypes"] == ["float16", "float16", "float16", "long"]
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_hd128"
    assert captured["causal"] is True

@pytest.mark.parametrize(
    ("implementation_version", "module_name", "expected_counter"),
    [
        ("v2", "aten_softmax_v2", "simt_v2_calls"),
        ("v3", "aten_softmax_v3", "simt_v3_calls"),
    ],
)
def test_ascend_backend_runs_simt_softmax_through_versioned_module(
    monkeypatch,
    implementation_version,
    module_name,
    expected_counter,
):
    captured: dict[str, object] = {
        "torch_softmax_calls": 0,
        "simt_v1_calls": 0,
        "simt_v2_calls": 0,
        "simt_v3_calls": 0,
    }

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
            self.softmax = self._softmax

        def _softmax(self, tensor, dim):
            del tensor, dim
            captured["torch_softmax_calls"] += 1
            return FakeTensor()

    fake_v1_ops = SimpleNamespace(
        spatial_softmax_forward=lambda tensor, dim: captured.__setitem__(
            "simt_v1_calls", captured["simt_v1_calls"] + 1
        )
        or tensor
    )
    fake_v2_ops = SimpleNamespace(
        spatial_softmax_forward=lambda tensor, dim: captured.__setitem__(
            "simt_v2_calls", captured["simt_v2_calls"] + 1
        )
        or tensor
    )
    fake_v3_ops = SimpleNamespace(
        spatial_softmax_forward=lambda tensor, dim: captured.__setitem__(
            "simt_v3_calls", captured["simt_v3_calls"] + 1
        )
        or tensor
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "aten_softmax", SimpleNamespace(ops=fake_v1_ops))
    monkeypatch.setitem(sys.modules, "aten_softmax_v2", SimpleNamespace(ops=fake_v2_ops))
    monkeypatch.setitem(sys.modules, "aten_softmax_v3", SimpleNamespace(ops=fake_v3_ops))

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        implementation="simt",
        implementation_version=implementation_version,
    )

    backend.run_operator(request)

    assert captured[expected_counter] == 1
    assert captured["simt_v1_calls"] == 0
    if expected_counter != "simt_v2_calls":
        assert captured["simt_v2_calls"] == 0
    if expected_counter != "simt_v3_calls":
        assert captured["simt_v3_calls"] == 0
    assert captured["torch_softmax_calls"] == 0


def test_ascend_backend_captures_simt_softmax_through_registered_op(monkeypatch):
    captured: dict[str, object] = {"torch_softmax_calls": 0, "simt_calls": 0}

    class FakeTensor:
        shape = (2,)

        def reshape(self, shape):
            del shape
            return self

        def detach(self):
            return self

        def cpu(self):
            return self

        def to(self, dtype=None):
            del dtype
            return self

        def flatten(self):
            return self

        def tolist(self):
            return [0.25, 0.75]

    class FakeTorch:
        def __init__(self) -> None:
            self.npu = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: None,
                get_device_name=lambda device: "Fake Ascend",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.float32 = "float32"
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.softmax = self._softmax

        def _softmax(self, tensor, dim):
            del tensor, dim
            captured["torch_softmax_calls"] += 1
            return FakeTensor()

    fake_ops_module = SimpleNamespace(
        spatial_softmax_forward=lambda tensor, dim: captured.__setitem__(
            "simt_calls", captured["simt_calls"] + 1
        )
        or tensor
    )

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())
    monkeypatch.setitem(sys.modules, "aten_softmax", SimpleNamespace(ops=fake_ops_module))

    from cannbench.backends.pytorch_backend import AscendBackend

    backend = AscendBackend()
    monkeypatch.setattr(backend, "_install_simt_op", lambda request, op_name: None)
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
        implementation="simt",
    )

    output = backend.capture_operator_output(request)

    assert output.values == (0.25, 0.75)
    assert captured["simt_calls"] == 1
    assert captured["torch_softmax_calls"] == 0


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
        seed=7,
    )

    backend.run_operator(request)

    assert captured["shape"] == (32, 128)
    assert captured["dtype"] == "float16"
    assert captured["values"]


def test_backend_captures_softmax_output_once(monkeypatch):
    captured: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, values=None, shape=(32, 128)):
            self._values = values or [0.25, 0.75]
            self.shape = shape

        def reshape(self, shape):
            captured["input_shape"] = shape
            return self

        def detach(self):
            return self

        def cpu(self):
            captured["moved_to_cpu"] = True
            return self

        def to(self, dtype=None):
            captured["output_dtype"] = dtype
            return self

        def flatten(self):
            return self

        def tolist(self):
            return self._values

    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(
                is_available=lambda: True,
                synchronize=lambda: captured.setdefault("synchronized", True),
                get_device_name=lambda device: "Fake GPU",
            )
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.float32 = "float32"
            self.tensor = self._tensor
            self.softmax = self._softmax

        def _tensor(self, values, device=None, dtype=None):
            captured["device"] = device
            captured["dtype"] = dtype
            return FakeTensor(values=values)

        def _softmax(self, tensor, dim):
            captured["softmax_dim"] = dim
            return FakeTensor(values=[0.25, 0.75], shape=(2,))

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )

    output = backend.capture_operator_output(request)

    assert output.backend == "nvidia"
    assert output.device_name == "Fake GPU"
    assert output.shape == (2,)
    assert output.values == (0.25, 0.75)
    assert captured["input_shape"] == (32, 128)
    assert captured["moved_to_cpu"] is True
    assert captured["output_dtype"] == "float32"
    assert captured["synchronized"] is True


def test_nvidia_backend_profiles_softmax_with_ncu(monkeypatch, tmp_path, capsys):
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
            captured["device"] = device
            captured["dtype"] = dtype
            return FakeTensor()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    def fake_run(command, **kwargs):
        cwd = kwargs.get("cwd")
        profile_dir = cwd / "profile"
        perf_dir = cwd / "perf"
        if "--target-processes" in command:
            captured["profile_command"] = command
            captured["cwd"] = cwd
            profile_dir.mkdir(parents=True, exist_ok=True)
            perf_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "ncu-report.ncu-rep").write_text("binary-placeholder", encoding="utf-8")
            (perf_dir / "benchmark.json").write_text(
                "{\"backend\":\"nvidia\",\"device_name\":\"Fake GPU\"}\n",
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(command, 0, "", "")
        if "--import" in command:
            captured["render_command"] = command
            return subprocess.CompletedProcess(
                command,
                0,
                (
                    "Kernel Name,Metric Name,Metric Unit,Metric Value\n"
                    "softmax,gpu__time_duration.avg,usecond,1000\n"
                ),
                "render-warning\n",
            )
        raise AssertionError(f"unexpected subprocess invocation: {command}")

    monkeypatch.setattr("subprocess.run", fake_run)

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )

    result = backend.profile_operator_device_time(request)

    assert isinstance(result, LocalDeviceProfileResult)
    assert result.benchmark_result.device_name == "Fake GPU"
    assert result.profile.profile_summary.backend == "nvidia"
    assert result.profile.profile_summary.source_files == ("ncu.csv",)
    assert "ncu.csv" in {name for name, _ in result.profile.profile_artifacts}
    assert "ncu-report.ncu-rep" in {name for name, _ in result.profile.profile_artifacts}
    assert result.profile.perf_artifacts[0][0] == "benchmark.json"
    profile_command = captured["profile_command"]
    assert profile_command[0] == "ncu"
    assert "--launch-skip" not in profile_command
    assert profile_command[profile_command.index("--launch-count") + 1] == "1"
    assert "--export" in profile_command
    assert profile_command[profile_command.index("-m") - 1] == sys.executable
    render_command = captured["render_command"]
    assert render_command[:2] == ["ncu", "--import"]
    assert "--page" in render_command
    assert "raw" in render_command
    assert "--csv" in render_command
    terminal = capsys.readouterr()
    assert "Kernel Name,Metric Name,Metric Unit,Metric Value" in terminal.out
    assert "render-warning" in terminal.err


def test_nvidia_backend_profile_raises_when_ncu_fails(monkeypatch):
    class FakeTensor:
        def reshape(self, shape):
            del shape
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
            self.tensor = lambda values, device=None, dtype=None: FakeTensor()
            self.softmax = lambda tensor, dim: tensor

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setattr(
        "subprocess.run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command,
            1,
            "",
            "ncu failed",
        ),
    )

    from cannbench.backends.pytorch_backend import NvidiaBackend

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )

    with pytest.raises(RuntimeError, match="ncu profiling failed"):
        backend.profile_operator_device_time(request)


def test_ascend_profile_operator_device_time_uses_msprof_launch_controls(monkeypatch):
    captured: dict[str, object] = {}

    class FakeNpu:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def get_device_name(device):
            del device
            return "Ascend 950PR"

    class FakeTorch:
        npu = FakeNpu()
        device = staticmethod(lambda kind: kind)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None, check=None):
        del env, text, capture_output, check
        captured["profile_command"] = command
        captured["cwd"] = cwd
        profile_dir = cwd / "profile"
        perf_dir = cwd / "perf"
        profile_dir.mkdir(parents=True, exist_ok=True)
        perf_dir.mkdir(parents=True, exist_ok=True)
        (profile_dir / "op_summary.csv").write_text(
            "Op Name,Task Duration(us)\nsoftmax,1000\n",
            encoding="utf-8",
        )
        (perf_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
        seed=7,
    )

    result = AscendBackend().profile_operator_device_time(request)

    command = captured["profile_command"]
    assert command[:2] == ["msprof", "op"]
    assert "--warm-up=3" not in command
    assert "--launch-count=1" in command
    assert "--warmup" not in command
    assert "--iterations" not in command
    assert "internal-run" in command
    assert result.profile.device_name == "Ascend 950PR"
    assert result.profile.profile_summary.backend == "ascend"
    assert result.profile.profile_summary.latency_ms == 1.0


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


def test_nvidia_profile_operator_device_time_invokes_internal_run(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def get_device_name(device):
            del device
            return "NVIDIA H800 PCIe"

    class FakeTorch:
        cuda = FakeCuda()
        device = staticmethod(lambda kind: kind)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None, check=None):
        del text, capture_output, check
        profile_dir = cwd / "profile"
        perf_dir = cwd / "perf"
        if "--target-processes" in command:
            captured["profile_command"] = command
            captured["cwd"] = cwd
            captured["env"] = env
            profile_dir.mkdir(parents=True, exist_ok=True)
            perf_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "ncu-report.ncu-rep").write_text("binary-placeholder", encoding="utf-8")
            (perf_dir / "benchmark.json").write_text(
                json.dumps(
                    {
                        "backend": "nvidia",
                        "device_name": "NVIDIA H800 PCIe",
                        "op": "softmax",
                        "dtype": "float16",
                        "case": {
                            "case_id": "tiny_logits",
                            "family": "smoke",
                            "source_kind": "synthetic",
                            "source_project": "CannBench",
                            "source_model": "SmokeModel",
                            "source_file": "smoke.json",
                            "source_op": "aten._softmax.default",
                            "payload": {"dimensions": [32, 128], "dim": -1},
                        },
                        "warmup": 2,
                        "iterations": 3,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "--import" in command:
            captured["render_command"] = command
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Kernel Name,Metric Name,Metric Unit,Metric Value\n"
                    "softmax,gpu__time_duration.avg,usecond,100\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess invocation: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    backend = NvidiaBackend()
    request = OperatorBenchmarkRequest(
        backend="nvidia",
        op="softmax",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_logits",
    )

    result = backend.profile_operator_device_time(request)

    assert result.profile.device_name == "NVIDIA H800 PCIe"
    assert "internal-run" in " ".join(captured["profile_command"])
    assert "--launch-skip" not in captured["profile_command"]
    assert captured["profile_command"][captured["profile_command"].index("--launch-count") + 1] == "1"
    assert " operator " not in f" {' '.join(captured['profile_command'])} "
    assert captured["render_command"][:2] == ["ncu", "--import"]
    assert "--page" in captured["render_command"]
    assert "raw" in captured["render_command"]
    assert "--csv" in captured["render_command"]
    assert captured["env"]["PYTHONPATH"]


def test_nvidia_profile_operator_device_time_preserves_external_implementation(monkeypatch):
    captured: dict[str, object] = {}

    class FakeCuda:
        @staticmethod
        def is_available():
            return True

        @staticmethod
        def synchronize():
            return None

        @staticmethod
        def get_device_name(device):
            del device
            return "NVIDIA H800 PCIe"

    class FakeTorch:
        cuda = FakeCuda()
        device = staticmethod(lambda kind: kind)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

    from cannbench.backends.pytorch_backend import NvidiaBackend

    def fake_run(command, cwd=None, env=None, text=None, capture_output=None, check=None):
        del env, text, capture_output, check
        profile_dir = cwd / "profile"
        perf_dir = cwd / "perf"
        if "--target-processes" in command:
            captured["profile_command"] = command
            profile_dir.mkdir(parents=True, exist_ok=True)
            perf_dir.mkdir(parents=True, exist_ok=True)
            (profile_dir / "ncu-report.ncu-rep").write_text("binary-placeholder", encoding="utf-8")
            (perf_dir / "benchmark.json").write_text("{}\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        if "--import" in command:
            return SimpleNamespace(
                returncode=0,
                stdout=(
                    "Kernel Name,Metric Name,Metric Unit,Metric Value\n"
                    "sparse_attention,gpu__time_duration.avg,usecond,100\n"
                ),
                stderr="",
            )
        raise AssertionError(f"unexpected subprocess invocation: {command}")

    monkeypatch.setattr(subprocess, "run", fake_run)

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        implementation_version="flashmla-main",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
    )

    NvidiaBackend().profile_operator_device_time(request)

    command = captured["profile_command"]
    assert command[command.index("--implementation") + 1] == "cuda_library"
    assert command[command.index("--implementation-version") + 1] == "flashmla-main"
