from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import re
import stat
from types import SimpleNamespace

import pytest

import cannbench.operators.builtin.sparse_attention as sparse_attention
from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.operators.builtin.sparse_attention import (
    _build_simt_callable,
    _select_simt_family,
    _simt_module_name,
    get_sparse_attention_case,
)
from cannbench.operators.plugin import TorchOperatorContext


def test_simt_install_scripts_are_executable():
    project_dir = Path(__file__).parents[1] / "v1"

    for script in (project_dir / "install.sh", project_dir / "scripts" / "install.sh"):
        assert script.stat().st_mode & stat.S_IXUSR


def test_select_simt_family_prefers_hd512():
    payload = {
        "qk_head_dim": 512,
        "value_head_dim": 512,
        "kv_heads": 1,
        "query_heads": 128,
        "selected_tokens": 1024,
    }

    assert _select_simt_family(payload) == "family_hd512"


def test_select_simt_family_prefers_hd128():
    payload = {
        "qk_head_dim": 128,
        "value_head_dim": 128,
        "kv_heads": 1,
        "query_heads": 128,
        "selected_tokens": 2048,
    }

    assert _select_simt_family(payload) == "family_hd128"


@pytest.mark.parametrize(
    ("qk_head_dim", "value_head_dim", "query_heads", "expected"),
    [
        (256, 256, 64, "family_hd256"),
        (576, 512, 128, "family_hd576"),
    ],
)
def test_select_simt_family_prefers_new_wide_head_families(
    qk_head_dim,
    value_head_dim,
    query_heads,
    expected,
):
    payload = {
        "qk_head_dim": qk_head_dim,
        "value_head_dim": value_head_dim,
        "kv_heads": 1,
        "query_heads": query_heads,
        "selected_tokens": 2048,
    }

    assert _select_simt_family(payload) == expected


def test_select_simt_family_falls_back_for_unknown_shape():
    payload = {
        "qk_head_dim": 64,
        "value_head_dim": 64,
        "kv_heads": 12,
        "query_heads": 12,
        "selected_tokens": 512,
    }

    assert _select_simt_family(payload) == "fallback"


def test_select_simt_family_rejects_unknown_qk_value_dimension_pair():
    payload = {
        "qk_head_dim": 576,
        "value_head_dim": 576,
        "kv_heads": 1,
        "query_heads": 128,
        "selected_tokens": 2048,
    }

    assert _select_simt_family(payload) == "fallback"


def test_simt_module_name_registers_version_isolated_v2():
    assert _simt_module_name(None) == "aten_dsa_sparse_attention"
    assert _simt_module_name("v1") == "aten_dsa_sparse_attention"
    assert _simt_module_name("v2") == "aten_dsa_sparse_attention_v2"
    assert _simt_module_name("vllm") == "aten_dsa_sparse_attention_vllm"


def test_vllm_project_has_copied_mla_source_and_independent_python_package():
    project_dir = Path(__file__).parents[1] / "vllm"
    source_dir = (
        project_dir
        / "vendor"
        / "vllm_ascend_a5b0ce"
        / "csrc"
        / "attention"
        / "sparse_flash_attention"
        / "op_kernel"
    )
    arch35_dir = source_dir / "arch35"

    assert (project_dir / "install.sh").is_file()
    assert (project_dir / "setup.py").is_file()
    assert (
        project_dir / "aten_dsa_sparse_attention_vllm" / "__init__.py"
    ).is_file()
    assert (project_dir / "vendor" / "PROVENANCE.md").is_file()
    assert (source_dir / "sparse_flash_attention.cpp").is_file()
    assert (source_dir / "sparse_flash_attention_common.h").is_file()
    assert (source_dir / "sparse_flash_attention_template_tiling_key.h").is_file()
    assert (arch35_dir / "sparse_flash_attention_kernel_mla.h").is_file()
    assert (arch35_dir / "sparse_flash_attention_service_cube_mla.h").is_file()
    assert (arch35_dir / "sparse_flash_attention_service_vector_mla.h").is_file()


def test_vllm_project_vendors_reproducible_ascend950_operator_project():
    project_dir = Path(__file__).parents[1] / "vllm"
    vendor_dir = project_dir / "vendor" / "vllm_ascend_a5b0ce"
    operator_dir = vendor_dir / "csrc" / "attention" / "sparse_flash_attention"

    assert (vendor_dir / "csrc" / "build.sh").is_file()
    assert (operator_dir / "op_kernel" / "sparse_flash_attention.cpp").is_file()
    op_def = operator_dir / "op_host" / "sparse_flash_attention_def.cpp"
    assert op_def.is_file()
    assert 'AddConfig("ascend950", aicore_config)' in op_def.read_text(
        encoding="utf-8"
    )


def test_v2_project_has_independent_python_package():
    project_dir = Path(__file__).parents[1] / "v2"

    assert (project_dir / "install.sh").is_file()
    assert (project_dir / "setup.py").is_file()
    assert (project_dir / "aten_dsa_sparse_attention_v2" / "__init__.py").is_file()


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
        / "aten_dsa_sparse_attention_v2"
        / "csrc"
    )
    source = "\n".join(
        path.read_text(encoding="utf-8") for path in csrc_dir.rglob("*.asc")
    )
    launch_symbols = re.findall(
        r"\b(launch_sparse_attention_[A-Za-z0-9_]+)\s*\(", source
    )

    assert launch_symbols
    assert all(name.endswith("_v2") for name in launch_symbols)


def test_build_simt_callable_requires_loaded_module():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
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
        case=get_sparse_attention_case("smoke", "tiny_decode_top4"),
        device="npu",
        dtype="float16",
        implementation_module=None,
    )

    with pytest.raises(
        RuntimeError,
        match="sparse_attention SIMT implementation module is not loaded",
    ):
        _build_simt_callable(ctx)


def test_build_simt_callable_routes_vllm_version_to_v32_adapter(monkeypatch):
    captured = {}
    expected = object()

    def fake_builder(ctx):
        captured["ctx"] = ctx
        return expected

    monkeypatch.setattr(
        sparse_attention,
        "build_vllm_simt_callable",
        fake_builder,
        raising=False,
    )
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
        dtype="bfloat16",
        dataset="realistic_decode",
        case_id="deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
        implementation="simt",
        implementation_version="vllm",
    )
    ctx = TorchOperatorContext(
        backend=SimpleNamespace(),
        torch=SimpleNamespace(),
        request=request,
        case=get_sparse_attention_case(request.dataset, request.case_id),
        device="npu",
        dtype="bfloat16",
        implementation_module=SimpleNamespace(ops=SimpleNamespace()),
    )

    assert _build_simt_callable(ctx) is expected
    assert captured["ctx"] is ctx


def test_vllm_simt_adapter_uses_copied_three_output_lse_abi(monkeypatch):
    import cannbench.operators.builtin.sparse_attention.external as external

    captured = {}
    expected = object()
    attention_op = lambda **kwargs: kwargs

    def fake_builder(ctx, *, attention_op, return_lse=True):
        captured.update(
            ctx=ctx,
            attention_op=attention_op,
            return_lse=return_lse,
        )
        return expected

    monkeypatch.setattr(
        external,
        "_build_vllm_sparse_flash_attention_callable",
        fake_builder,
    )
    ctx = SimpleNamespace(
        case=SimpleNamespace(qk_head_dim=576, value_head_dim=512),
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(sparse_flash_attention_forward=attention_op)
        ),
    )

    assert external.build_vllm_simt_callable(ctx) is expected
    assert captured == {
        "ctx": ctx,
        "attention_op": attention_op,
        "return_lse": True,
    }


def test_vllm_simt_output_only_path_accepts_three_output_abi():
    import cannbench.operators.builtin.sparse_attention.external as external

    class FakeTensor:
        device = "npu"
        dtype = "bfloat16"

        def __init__(self):
            self.shapes = []

        def reshape(self, *shape):
            self.shapes.append(shape)
            return self

    output = FakeTensor()
    result = external._reshape_or_pad_tnd_output(
        SimpleNamespace(),
        (output, object(), object()),
        batch=1,
        query_tokens=2,
        query_heads=128,
        value_head_dim=512,
        query_lens=(2,),
    )

    assert result is output
    assert output.shapes == [(2, 128, 512), (1, 2, 128, 512)]


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
        query,
        shared_kv,
        indices,
        *,
        value_head_dim,
        phase,
        family,
        causal,
    ):
        captured["query_shape"] = query.shape
        captured["shared_kv_shape"] = shared_kv.shape
        captured["indices_shape"] = indices.shape
        captured["value_head_dim"] = value_head_dim
        captured["phase"] = phase
        captured["family"] = family
        captured["causal"] = causal
        return "ok"

    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
        dtype="float16",
        dataset="stress",
        case_id="deepseek_64k_decode_top2048",
        seed=7,
        implementation="simt",
    )

    case = replace(
        get_sparse_attention_case("stress", "deepseek_64k_decode_top2048"),
        batch=1,
        context_tokens=32,
        selected_tokens=16,
    )
    ctx = TorchOperatorContext(
        backend=FakeBackend(),
        torch=SimpleNamespace(long="long"),
        request=request,
        case=case,
        device="npu",
        dtype="float16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(sparse_attention_forward=fake_forward)
        ),
    )

    operator = _build_simt_callable(ctx)
    assert operator() == "ok"
    assert captured["query_shape"] == (1, 128, 1, 128)
    assert captured["shared_kv_shape"] == (1, 1, 32, 128)
    assert captured["indices_shape"] == (1, 1, 16)
    assert captured["value_head_dim"] == 128
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_hd128"
    assert captured["causal"] is True


def test_build_simt_callable_validates_topk_lengths_per_batch():
    class FakeTensor:
        def reshape(self, shape):
            del shape
            return self

    class FakeBackend:
        @staticmethod
        def _tensor(torch, values, *, device, dtype):
            del torch, values, device, dtype
            return FakeTensor()

    case = replace(
        get_sparse_attention_case("stress", "deepseek_64k_decode_top2048"),
        batch=2,
        query_tokens=3,
        context_tokens=8,
        selected_tokens=4,
        query_lens=(1, 3),
        context_lens=(4, 8),
        query_start_positions=(3, 5),
        page_block_size=4,
    )
    ctx = TorchOperatorContext(
        backend=FakeBackend(),
        torch=SimpleNamespace(long="long"),
        request=OperatorBenchmarkRequest(
            backend="ascend",
            op="sparse_attention",
            dtype="float16",
            dataset="stress",
            case_id="deepseek_64k_decode_top2048",
            seed=7,
            implementation="simt",
        ),
        case=case,
        device="npu",
        dtype="float16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(sparse_attention_forward=lambda *args, **kwargs: None)
        ),
    )

    operator = _build_simt_callable(ctx)

    assert callable(operator)


def test_build_simt_callable_allocates_large_inputs_directly_on_device(monkeypatch):
    captured: dict[str, object] = {"allocations": []}

    class FakeTensor:
        pass

    class FakeTorch:
        bfloat16 = "bfloat16"
        long = "long"

        @staticmethod
        def zeros(shape, *, device, dtype):
            captured["allocations"].append((shape, device, dtype))
            return FakeTensor()

    def fake_forward(
        query,
        shared_kv,
        indices,
        *,
        value_head_dim,
        phase,
        family,
        causal,
    ):
        del query, shared_kv, indices, value_head_dim
        captured.update(phase=phase, family=family, causal=causal)
        return "ok"

    monkeypatch.setattr(
        sparse_attention,
        "materialize_sparse_attention_inputs",
        lambda *args, **kwargs: pytest.fail("large inputs must not use host materialization"),
    )
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
        dtype="bfloat16",
        dataset="realistic_decode",
        case_id="deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024",
        seed=7,
        implementation="simt",
    )
    ctx = TorchOperatorContext(
        backend=SimpleNamespace(),
        torch=FakeTorch(),
        request=request,
        case=get_sparse_attention_case(
            "realistic_decode",
            "deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024",
        ),
        device="npu",
        dtype="bfloat16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(sparse_attention_forward=fake_forward)
        ),
    )

    operator = _build_simt_callable(ctx)

    assert operator() == "ok"
    assert captured["allocations"] == [
        ((60, 128, 1, 512), "npu", "bfloat16"),
        ((60, 1, 131072, 512), "npu", "bfloat16"),
        ((60, 1, 1024), "npu", "long"),
    ]
    assert captured["phase"] == "decode"
    assert captured["family"] == "family_hd512"
    assert captured["causal"] is True


def test_build_simt_callable_uses_bound_indices_for_large_inputs(monkeypatch):
    captured: dict[str, object] = {"allocations": []}

    class FakeTensor:
        def __init__(self, name):
            self.name = name

        def reshape(self, shape):
            captured["bound_shape"] = shape
            return self

    class FakeTorch:
        bfloat16 = "bfloat16"
        long = "long"

        @staticmethod
        def zeros(shape, *, device, dtype):
            captured["allocations"].append((shape, device, dtype))
            return FakeTensor("allocated")

    bound_indices = FakeTensor("indexer-output")

    def fake_forward(
        query,
        shared_kv,
        indices,
        *,
        value_head_dim,
        phase,
        family,
        causal,
    ):
        del query, shared_kv, value_head_dim, phase, family, causal
        captured["indices"] = indices
        return "ok"

    monkeypatch.setattr(
        sparse_attention,
        "materialize_sparse_attention_inputs",
        lambda *args, **kwargs: pytest.fail("large inputs must not use host materialization"),
    )
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
        dtype="bfloat16",
        dataset="realistic_decode",
        case_id="deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024",
        seed=7,
        implementation="simt",
    )
    ctx = TorchOperatorContext(
        backend=SimpleNamespace(),
        torch=FakeTorch(),
        request=request,
        case=get_sparse_attention_case(
            "realistic_decode",
            "deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024",
        ),
        device="npu",
        dtype="bfloat16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(sparse_attention_forward=fake_forward)
        ),
        bound_inputs={"indices": bound_indices},
    )

    assert _build_simt_callable(ctx)() == "ok"
    assert captured["indices"] is bound_indices
    assert captured["bound_shape"] == (60, 1, 1024)
    assert len(captured["allocations"]) == 2


def test_build_simt_callable_rejects_unsupported_family():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        op="sparse_attention",
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
        case=get_sparse_attention_case("smoke", "tiny_decode_top4"),
        device="npu",
        dtype="float16",
        implementation_module=SimpleNamespace(
            ops=SimpleNamespace(sparse_attention_forward=lambda *args, **kwargs: None)
        ),
    )

    with pytest.raises(
        RuntimeError,
        match="does not support this shape family",
    ):
        _build_simt_callable(ctx)
