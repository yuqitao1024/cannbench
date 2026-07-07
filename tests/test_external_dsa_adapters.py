import sys
from types import SimpleNamespace

import pytest

from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.operators import TorchOperatorContext, get_operator_plugin
from cannbench.operators.builtin.sparse_attention.cases import SparseAttentionCase


def _build_sparse_attention_vllm_callable(*, backend, torch, request, case, device, dtype):
    plugin = get_operator_plugin("sparse_attention")
    return plugin.build_vllm_ascend_callable(
        TorchOperatorContext(
            backend=backend,
            torch=torch,
            request=request,
            case=case,
            device=device,
            dtype=dtype,
        )
    )


def test_operator_request_preserves_external_implementation():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="lightning_indexer",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    assert request.implementation == "vllm_ascend"


def test_operator_request_rejects_unknown_implementation():
    with pytest.raises(ValueError, match="Unsupported implementation"):
        OperatorBenchmarkRequest(
            backend="ascend",
            implementation="unknown",
            op="lightning_indexer",
            dtype="float16",
            dataset="smoke",
            case_id="tiny_decode_top4",
            warmup=0,
            iterations=1,
        )


def test_ascend_vllm_adapter_calls_torch_npu_lightning_indexer(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeTensor:
        shape = ()

        def __init__(self, name="tensor"):
            self.name = name

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def to(self, *args, **kwargs):
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
            self.tensor = lambda *args, **kwargs: FakeTensor()

    def fake_lightning_indexer(**kwargs):
        calls.append(kwargs)
        return FakeTensor("indices"), FakeTensor("scores")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_lightning_indexer=fake_lightning_indexer),
    )

    from cannbench.backends.pytorch_backend import AscendBackend

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="lightning_indexer",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls
    assert calls[0]["layout_query"] == "TND"
    assert calls[0]["layout_key"] == "BSND"
    assert calls[0]["sparse_count"] == 4
    assert calls[0]["sparse_mode"] == 3


def test_ascend_vllm_adapter_prefers_a5_quant_lightning_indexer(monkeypatch):
    calls: dict[str, dict[str, object]] = {}

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def to(self, *args, **kwargs):
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
            self.int8 = "int8"
            self.int32 = "int32"
            self.long = "long"
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.ops = SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_vllm_quant_lightning_indexer_metadata=self._metadata,
                    npu_vllm_quant_lightning_indexer=self._indexer,
                )
            )

        def _metadata(self, **kwargs):
            calls["metadata"] = kwargs
            return FakeTensor("metadata", (1024,))

        def _indexer(self, **kwargs):
            calls["indexer"] = kwargs
            return FakeTensor("indices"), FakeTensor("values")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="lightning_indexer",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls["metadata"]["layout_query"] == "TND"
    assert calls["metadata"]["layout_key"] == "PA_BSND"
    assert calls["metadata"]["sparse_count"] == 4
    assert calls["indexer"]["metadata"].name == "metadata"
    assert calls["indexer"]["query_quant_mode"] == 0
    assert calls["indexer"]["key_quant_mode"] == 0
    assert calls["indexer"]["cmp_ratio"] == 4
    assert calls["indexer"]["return_value"] is False


def test_ascend_vllm_custom_op_loader_bootstraps_vendor_env(monkeypatch):
    calls: list[dict[str, object]] = []

    def fake_bootstrap_custom_op_env(**kwargs):
        calls.append(kwargs)

    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.utils",
        SimpleNamespace(bootstrap_custom_op_env=fake_bootstrap_custom_op_env),
    )
    monkeypatch.setitem(
        sys.modules,
        "vllm_ascend.vllm_ascend_C",
        SimpleNamespace(),
    )

    from cannbench.backends.pytorch_backend import AscendBackend

    AscendBackend()._ensure_vllm_ascend_custom_ops_loaded()

    assert calls == [{"include_vendor_lib": True}]


def test_ascend_vllm_sparse_attention_calls_sharedkv_metadata_and_op(monkeypatch):
    calls: dict[str, dict[str, object]] = {}

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            self.permuted_dims = dims
            return self

        def contiguous(self):
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
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.ops = SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_sparse_attn_sharedkv_metadata=self._metadata,
                    npu_sparse_attn_sharedkv=self._attention,
                )
            )

        def _metadata(self, **kwargs):
            calls["metadata"] = kwargs
            return FakeTensor("metadata", (1024,))

        def _attention(self, q, **kwargs):
            calls["attention"] = {"q": q, **kwargs}
            return FakeTensor("out"), FakeTensor("lse")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls["metadata"]["num_heads_q"] == 2
    assert calls["metadata"]["num_heads_kv"] == 2
    assert calls["metadata"]["head_dim"] == 16
    assert calls["metadata"]["batch_size"] == 2
    assert calls["metadata"]["max_seqlen_q"] == 1
    assert calls["metadata"]["max_seqlen_kv"] == 32
    assert calls["metadata"]["cmp_topk"] == 4
    assert calls["metadata"]["has_ori_kv"] is False
    assert calls["metadata"]["has_cmp_kv"] is True
    assert calls["attention"]["layout_q"] == "TND"
    assert calls["attention"]["layout_kv"] == "PA_ND"
    assert calls["attention"]["cmp_ratio"] == 1
    assert calls["attention"]["ori_kv"] is None
    assert calls["attention"]["cmp_sparse_indices"].shape == (2, 2, 4)


def test_ascend_vllm_sparse_attention_prefers_a5_quant_sharedkv_ops(monkeypatch):
    calls: dict[str, dict[str, object]] = {}

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            self.permuted_dims = dims
            return self

        def contiguous(self):
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
            self.float32 = "float32"
            self.float8_e4m3fn = "float8_e4m3fn"
            self.int32 = "int32"
            self.long = "long"
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.ops = SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_kv_quant_sparse_attn_sharedkv_metadata=self._metadata,
                    npu_kv_quant_sparse_attn_sharedkv=self._attention,
                )
            )

        def _metadata(self, **kwargs):
            calls["metadata"] = kwargs
            return FakeTensor("metadata", (1024,))

        def _attention(self, q, **kwargs):
            calls["attention"] = {"q": q, **kwargs}
            return FakeTensor("out"), FakeTensor("lse")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls["metadata"]["kv_quant_mode"] == 1
    assert calls["metadata"]["tile_size"] == 64
    assert calls["metadata"]["rope_head_dim"] == 64
    assert calls["metadata"]["layout_q"] == "TND"
    assert calls["metadata"]["layout_kv"] == "PA_ND"
    assert calls["attention"]["metadata"].name == "metadata"
    assert calls["attention"]["kv_quant_mode"] == 1
    assert calls["attention"]["tile_size"] == 64
    assert calls["attention"]["rope_head_dim"] == 64
    assert calls["attention"]["cmp_sparse_indices"].shape == (2, 2, 4)


def test_ascend_vllm_sparse_attention_uses_a5_kv_physical_dim(monkeypatch):
    calls: dict[str, dict[str, object]] = {}

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            self.permuted_dims = dims
            return self

        def contiguous(self):
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.float16 = "float16"
            self.bfloat16 = "bfloat16"
            self.float8_e4m3fn = "float8_e4m3fn"
            self.int32 = "int32"
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.ops = SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_kv_quant_sparse_attn_sharedkv_metadata=self._metadata,
                    npu_kv_quant_sparse_attn_sharedkv=self._attention,
                )
            )

        def _metadata(self, **kwargs):
            calls["metadata"] = kwargs
            return FakeTensor("metadata", (1024,))

        def _attention(self, q, **kwargs):
            calls["attention"] = {"q": q, **kwargs}
            return FakeTensor("out"), FakeTensor("lse")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    case = SparseAttentionCase(
        case_id="a5_decode_b1_ctx512_top512",
        family="decode_sparse_attention",
        batch=1,
        query_heads=64,
        kv_heads=1,
        query_tokens=1,
        context_tokens=512,
        selected_tokens=512,
        head_dim=512,
        causal=True,
        phase="decode",
        source_kind="unit",
        source_project="cannbench",
        source_model="ascend950_a5",
        source_file="unit",
        source_op="sparse_attention",
    )
    request = SimpleNamespace(dtype="bfloat16", seed=0)

    operator = _build_sparse_attention_vllm_callable(
        backend=AscendBackend(),
        torch=FakeTorch(),
        request=request,
        case=case,
        device="npu",
        dtype="bfloat16",
    )
    operator()

    assert calls["metadata"]["head_dim"] == 512
    assert calls["metadata"]["cmp_ratio"] == 4
    assert calls["metadata"]["has_ori_kv"] is True
    assert calls["metadata"]["has_cmp_kv"] is True
    assert calls["attention"]["q"].shape == (1, 64, 512)
    assert calls["attention"]["ori_kv"].shape == (4, 128, 1, 640)
    assert calls["attention"]["cmp_kv"].shape == (1, 128, 1, 640)
    assert calls["attention"]["ori_block_table"].shape == (1, 4)
    assert calls["attention"]["cmp_block_table"].shape == (1, 1)
    assert calls["attention"]["cu_seqlens_ori_kv"] is None
    assert calls["attention"]["cu_seqlens_cmp_kv"] is None
    assert calls["attention"]["sinks"].shape == (64,)
    assert calls["attention"]["cmp_ratio"] == 4


def test_ascend_vllm_sparse_attention_a5_setup_avoids_device_permute(monkeypatch):
    calls: dict[str, dict[str, object]] = {}

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            raise AssertionError("vllm_ascend sparse_attention setup must not launch device permute")

        def contiguous(self):
            return self

    class FakeTorch:
        def __init__(self) -> None:
            self.float16 = "float16"
            self.bfloat16 = "bfloat16"
            self.float8_e4m3fn = "float8_e4m3fn"
            self.int32 = "int32"
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.ops = SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_kv_quant_sparse_attn_sharedkv_metadata=self._metadata,
                    npu_kv_quant_sparse_attn_sharedkv=self._attention,
                )
            )

        def _metadata(self, **kwargs):
            calls["metadata"] = kwargs
            return FakeTensor("metadata", (1024,))

        def _attention(self, q, **kwargs):
            calls["attention"] = {"q": q, **kwargs}
            return FakeTensor("out"), FakeTensor("lse")

    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace())

    from cannbench.backends.pytorch_backend import AscendBackend

    case = SparseAttentionCase(
        case_id="a5_prefill_b1_q512_ctx512_top512",
        family="prefill_sparse_attention",
        batch=1,
        query_heads=64,
        kv_heads=1,
        query_tokens=512,
        context_tokens=512,
        selected_tokens=512,
        head_dim=512,
        causal=True,
        phase="prefill",
        source_kind="unit",
        source_project="cannbench",
        source_model="ascend950_a5",
        source_file="unit",
        source_op="sparse_attention",
    )
    request = SimpleNamespace(dtype="bfloat16", seed=0)

    operator = _build_sparse_attention_vllm_callable(
        backend=AscendBackend(),
        torch=FakeTorch(),
        request=request,
        case=case,
        device="npu",
        dtype="bfloat16",
    )
    operator()

    assert calls["attention"]["q"].shape == (512, 64, 512)
    assert calls["attention"]["ori_kv"].shape == (4, 128, 1, 640)
    assert calls["attention"]["cmp_kv"].shape == (1, 128, 1, 640)


def test_nvidia_cuda_library_uses_external_lightning_indexer_adapter(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
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
            self.int32 = "int32"
            self.tensor = lambda *args, **kwargs: FakeTensor()

    def fake_lightning_indexer(**kwargs):
        calls.append(kwargs)
        return FakeTensor("indices")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "fake_cuda_dsa_adapter",
        SimpleNamespace(lightning_indexer=fake_lightning_indexer),
    )
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_ADAPTER", "fake_cuda_dsa_adapter")

    from cannbench.backends.pytorch_backend import NvidiaBackend

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        op="lightning_indexer",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = NvidiaBackend().run_operator(request)

    assert result.backend == "nvidia"
    assert calls
    assert calls[0]["request"] is request
    assert calls[0]["payload"]["top_k"] == 4
    assert calls[0]["query"].shape == (2, 1, 2, 16)
    assert calls[0]["keys"].shape == (2, 32, 16)
    assert calls[0]["weights"].shape == (2, 1, 2)


def test_nvidia_cuda_library_uses_external_sparse_attention_adapter(monkeypatch):
    calls: list[dict[str, object]] = []

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
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
            self.int32 = "int32"
            self.tensor = lambda *args, **kwargs: FakeTensor()

    def fake_sparse_attention(**kwargs):
        calls.append(kwargs)
        return FakeTensor("out")

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(
        sys.modules,
        "fake_cuda_dsa_adapter",
        SimpleNamespace(sparse_attention=fake_sparse_attention),
    )
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_ADAPTER", "fake_cuda_dsa_adapter")

    from cannbench.backends.pytorch_backend import NvidiaBackend

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    result = NvidiaBackend().run_operator(request)

    assert result.backend == "nvidia"
    assert calls
    assert calls[0]["request"] is request
    assert calls[0]["payload"]["phase"] == "decode"
    assert calls[0]["query"].shape == (2, 2, 1, 16)
    assert calls[0]["keys"].shape == (2, 2, 32, 16)
    assert calls[0]["values"].shape == (2, 2, 32, 16)
    assert calls[0]["indices"].shape == (2, 1, 4)


def test_nvidia_cuda_library_default_dsa_adapter_requires_flash_mla(monkeypatch):
    class FakeTensor:
        def reshape(self, *shape):
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
            self.int32 = "int32"
            self.tensor = lambda *args, **kwargs: FakeTensor()

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.delenv("CANNBENCH_CUDA_DSA_ADAPTER", raising=False)
    monkeypatch.delenv("CANNBENCH_CUDA_DSA_SPARSE_ATTENTION", raising=False)

    from cannbench.backends.pytorch_backend import NvidiaBackend

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    with pytest.raises(RuntimeError, match="flash_mla"):
        NvidiaBackend().run_operator(request)


def test_nvidia_cuda_library_rejects_adapter_without_required_callable(monkeypatch):
    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(is_available=lambda: True)
            self.device = lambda kind: kind
            self.float16 = "float16"
            self.int32 = "int32"

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    monkeypatch.setitem(sys.modules, "fake_cuda_dsa_adapter", SimpleNamespace())
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_ADAPTER", "fake_cuda_dsa_adapter")

    from cannbench.backends.pytorch_backend import NvidiaBackend

    request = OperatorBenchmarkRequest(
        backend="nvidia",
        implementation="cuda_library",
        op="sparse_attention",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
        warmup=0,
        iterations=1,
    )

    with pytest.raises(RuntimeError, match="callable sparse_attention"):
        NvidiaBackend().run_operator(request)
