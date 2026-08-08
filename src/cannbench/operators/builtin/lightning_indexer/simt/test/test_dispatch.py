from __future__ import annotations

from pathlib import Path
import re
from types import SimpleNamespace

import pytest

import cannbench.operators.builtin.lightning_indexer as lightning_indexer
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


def test_simt_module_name_registers_version_isolated_v2():
    assert _simt_module_name(None) == "aten_dsa_lightning_indexer"
    assert _simt_module_name("v1") == "aten_dsa_lightning_indexer"
    assert _simt_module_name("v2") == "aten_dsa_lightning_indexer_v2"
    assert _simt_module_name("vllm") == "aten_dsa_lightning_indexer"


def test_vllm_workflow_alias_delegates_to_v1_indexer_install():
    install_script = Path(__file__).parents[1] / "vllm" / "install.sh"

    assert install_script.is_file()
    source = install_script.read_text(encoding="utf-8")
    assert 'exec "${SCRIPT_DIR}/../v1/install.sh" "$@"' in source


def test_v2_project_has_independent_python_package():
    project_dir = Path(__file__).parents[1] / "v2"

    assert (project_dir / "install.sh").is_file()
    assert (project_dir / "setup.py").is_file()
    assert (project_dir / "aten_dsa_lightning_indexer_v2" / "__init__.py").is_file()


def test_v2_device_libraries_have_independent_sonames():
    setup_source = (Path(__file__).parents[1] / "v2" / "setup.py").read_text(
        encoding="utf-8"
    )
    kernel_libraries = re.findall(r'"(lib[^\"]+_kernel\.so)"\s*:', setup_source)

    assert kernel_libraries
    assert all(name.endswith("_v2_kernel.so") for name in kernel_libraries)


def test_v2_device_launch_abi_is_version_isolated():
    csrc_dir = (
        Path(__file__).parents[1]
        / "v2"
        / "aten_dsa_lightning_indexer_v2"
        / "csrc"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in csrc_dir.rglob("*.asc")
    )
    launch_symbols = re.findall(
        r"\b(launch_lightning_indexer_[A-Za-z0-9_]+)\s*\(", source
    )

    assert launch_symbols
    assert all(name.endswith("_v2") for name in launch_symbols)


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

    def fake_forward(
        query, keys, weights, *, valid_context_lengths, top_k, phase, family
    ):
        captured["valid_context_lengths_shape"] = valid_context_lengths.shape
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
        torch=SimpleNamespace(int32="int32"),
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
    assert captured["valid_context_lengths_shape"] == (16, 1)
    assert captured["top_k"] == 2048
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_4x64"


def test_build_simt_callable_allocates_large_inputs_directly_on_device(monkeypatch):
    captured: dict[str, object] = {"allocations": []}

    class FakeTensor:
        def reshape(self, shape):
            self.shape = shape
            return self

    class FakeTorch:
        bfloat16 = "bfloat16"
        int32 = "int32"

        @staticmethod
        def zeros(shape, *, device, dtype):
            captured["allocations"].append((shape, device, dtype))
            return FakeTensor()

    class FakeBackend:
        @staticmethod
        def _tensor(torch, values, *, device, dtype):
            del torch, values, device, dtype
            return FakeTensor()

    def fake_forward(
        query, keys, weights, *, valid_context_lengths, top_k, phase, family
    ):
        del query, keys, weights
        captured["valid_context_lengths_shape"] = valid_context_lengths.shape
        captured.update(top_k=top_k, phase=phase, family=family)
        return "ok"

    monkeypatch.setattr(
        lightning_indexer,
        "materialize_lightning_indexer_inputs",
        lambda *args, **kwargs: pytest.fail("large inputs must not use host materialization"),
    )
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="lightning_indexer",
        dtype="bfloat16",
        dataset="realistic_decode",
        case_id="deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024",
        seed=7,
        implementation="simt",
    )
    ctx = TorchOperatorContext(
        backend=FakeBackend(),
        torch=FakeTorch(),
        request=request,
        case=get_lightning_indexer_case(
            "realistic_decode",
            "deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024",
        ),
        device="npu",
        dtype="bfloat16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(lightning_indexer_forward=fake_forward)
        ),
    )

    operator = _build_simt_callable(ctx)

    assert operator() == "ok"
    assert captured["allocations"] == [
        ((60, 1, 64, 128), "npu", "bfloat16"),
        ((60, 131072, 128), "npu", "bfloat16"),
        ((60, 1, 64), "npu", "bfloat16"),
    ]
    assert captured["top_k"] == 1024
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_64x128"
    assert captured["valid_context_lengths_shape"] == (60, 1)


def test_plugin_exposes_supported_prefill_and_decode_simt_cases():
    prefill_case = get_lightning_indexer_case("realistic", "opt_prefill_2048_top512")
    decode_case = get_lightning_indexer_case("realistic", "llama4_decode_32760_top2048")

    assert prefill_case.phase == "prefill"
    assert prefill_case.index_heads == 4
    assert prefill_case.index_dim == 64
    assert decode_case.phase == "decode"
    assert decode_case.index_heads == 4
    assert decode_case.index_dim == 64
