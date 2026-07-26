import sys
from types import SimpleNamespace

import pytest

from cannbench.core.config import OperatorBenchmarkRequest
from cannbench.operators import TorchOperatorContext, get_operator_plugin
from cannbench.operators.builtin.sparse_attention.cases import SparseAttentionCase


def _build_sparse_attention_vllm_callable(
    *, backend, torch, request, case, device, dtype, bound_inputs=None
):
    plugin = get_operator_plugin("sparse_attention")
    return plugin.build_vllm_ascend_callable(
        TorchOperatorContext(
            backend=backend,
            torch=torch,
            request=request,
            case=case,
            device=device,
            dtype=dtype,
            bound_inputs=bound_inputs or {},
        )
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "cannbench.operators.builtin.lightning_indexer.external",
        "cannbench.operators.builtin.sparse_attention.external",
    ),
)
def test_cuda_dynamic_callable_is_scoped_by_nvtx(module_name):
    module = __import__(module_name, fromlist=["_with_cuda_nvtx_range"])
    calls = []
    torch = SimpleNamespace(
        cuda=SimpleNamespace(
            nvtx=SimpleNamespace(
                range_push=lambda name: calls.append(("push", name)),
                range_pop=lambda: calls.append("pop"),
            )
        )
    )
    operator = module._with_cuda_nvtx_range(
        torch,
        "dynamic-range",
        lambda: calls.append("operator") or "result",
    )

    assert operator() == "result"
    assert calls == [("push", "dynamic-range"), "operator", "pop"]


def test_ascend_vllm_sparse_attention_uses_sparse_flash_attention_for_v32(
    monkeypatch,
):
    import cannbench.operators.builtin.sparse_attention.external as external

    calls: dict[str, dict[str, object]] = {}

    class FakeTensor:
        def __init__(self, name="tensor", shape=()):
            self.name = name
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            self.shape = tuple(self.shape[index] for index in dims)
            return self

        def __getitem__(self, key):
            last = key[-1]
            if isinstance(last, slice):
                start = last.start or 0
                stop = last.stop or self.shape[-1]
                return FakeTensor(self.name, (*self.shape[:-1], stop - start))
            return self

        def contiguous(self):
            return self

        def __add__(self, other):
            del other
            return FakeTensor("lse", self.shape)

    class FakeTorch:
        int32 = "int32"

        @staticmethod
        def log(tensor):
            return tensor

    def fake_sparse_flash_attention(
        query,
        key,
        value,
        sparse_indices,
        scale_value,
        sparse_block_size,
        *,
        block_table=None,
        actual_seq_lengths_query=None,
        actual_seq_lengths_kv=None,
        query_rope=None,
        key_rope=None,
        layout_query="BSND",
        layout_kv="BSND",
        sparse_mode=3,
        attention_mode=2,
        return_softmax_lse=False,
    ):
        calls["attention"] = {
            "query": query,
            "key": key,
            "value": value,
            "sparse_indices": sparse_indices,
            "scale_value": scale_value,
            "sparse_block_size": sparse_block_size,
            "block_table": block_table,
            "actual_seq_lengths_query": actual_seq_lengths_query,
            "actual_seq_lengths_kv": actual_seq_lengths_kv,
            "query_rope": query_rope,
            "key_rope": key_rope,
            "layout_query": layout_query,
            "layout_kv": layout_kv,
            "sparse_mode": sparse_mode,
            "attention_mode": attention_mode,
            "return_softmax_lse": return_softmax_lse,
        }
        return (
            FakeTensor("out", (1, 128, 512)),
            FakeTensor("max", (1, 1, 128)),
            FakeTensor("sum", (1, 1, 128)),
        )

    payload = {
        "query_shape": (1, 128, 1, 576),
        "shared_kv_shape": (1, 1, 16, 576),
        "value_head_dim": 512,
        "indices_shape": (1, 1, 4),
        "query": tuple(0.0 for _ in range(128 * 576)),
        "shared_kv": tuple(0.0 for _ in range(16 * 576)),
        "indices": (0, 1, 2, 3),
    }
    monkeypatch.setattr(
        external,
        "materialize_sparse_attention_inputs",
        lambda *args, **kwargs: payload,
    )
    backend = SimpleNamespace(
        _ascend_custom_ops=lambda torch: SimpleNamespace(
            npu_sparse_flash_attention=fake_sparse_flash_attention
        ),
        _tensor=lambda torch, values, *, device, dtype: FakeTensor(
            shape=(len(values),)
        ),
        _custom_op_pair=lambda *args: pytest.fail("sharedkv ops must not be selected"),
    )
    case = SimpleNamespace(
        batch=1,
        query_heads=128,
        kv_heads=1,
        query_tokens=1,
        context_tokens=16,
        selected_tokens=4,
        qk_head_dim=576,
        value_head_dim=512,
        shared_kv=True,
    )

    operator = _build_sparse_attention_vllm_callable(
        backend=backend,
        torch=FakeTorch(),
        request=SimpleNamespace(dtype="bfloat16", seed=0),
        case=case,
        device="npu",
        dtype="bfloat16",
    )
    output, lse = operator()

    assert output.name == "out"
    assert output.shape == (1, 1, 128, 512)
    assert lse.shape == (1, 1, 128)
    assert calls["attention"]["query"].shape == (1, 128, 512)
    assert calls["attention"]["query_rope"].shape == (1, 128, 64)
    assert calls["attention"]["key"].shape == (16, 1, 512)
    assert calls["attention"]["key_rope"].shape == (16, 1, 64)
    assert calls["attention"]["value"].shape == (16, 1, 512)
    assert calls["attention"]["sparse_indices"].shape == (1, 1, 4)
    assert calls["attention"]["block_table"] is None
    assert calls["attention"]["actual_seq_lengths_query"].shape == (1,)
    assert calls["attention"]["actual_seq_lengths_kv"].shape == (1,)
    assert calls["attention"]["scale_value"] == pytest.approx(576**-0.5)
    assert calls["attention"]["layout_query"] == "TND"
    assert calls["attention"]["layout_kv"] == "TND"
    assert calls["attention"]["attention_mode"] == 2
    assert calls["attention"]["return_softmax_lse"] is True


def test_ascend_vllm_v32_attention_lowers_dynamic_inputs_per_call(monkeypatch):
    import cannbench.operators.builtin.sparse_attention.external as external

    calls: list[object] = []

    class FakeTensor:
        def __init__(self, shape=()):
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            self.shape = tuple(self.shape[index] for index in dims)
            return self

        def __getitem__(self, key):
            last = key[-1]
            if isinstance(last, slice):
                start = last.start or 0
                stop = last.stop or self.shape[-1]
                return FakeTensor((*self.shape[:-1], stop - start))
            return self

        def contiguous(self):
            return self

        def __add__(self, other):
            del other
            return FakeTensor(self.shape)

    class FakeTorch:
        int32 = "int32"

        @staticmethod
        def log(tensor):
            return tensor

    payload = {
        "query_shape": (1, 1, 1, 576),
        "shared_kv_shape": (1, 1, 16, 576),
        "value_head_dim": 512,
        "indices_shape": (1, 1, 1),
        "query": tuple(0.0 for _ in range(576)),
        "shared_kv": tuple(0.0 for _ in range(16 * 576)),
        "indices": (0,),
    }
    monkeypatch.setattr(
        external,
        "materialize_sparse_attention_inputs",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(
        external,
        "_lower_vllm_sparse_attention_query",
        lambda *args, **kwargs: calls.append("lower-query")
        or (FakeTensor((1, 1, 512)), FakeTensor((1, 1, 64))),
        raising=False,
    )
    monkeypatch.setattr(
        external,
        "_lower_vllm_sparse_attention_indices",
        lambda *args, **kwargs: calls.append("lower-indices")
        or FakeTensor((1, 1, 1)),
        raising=False,
    )
    backend = SimpleNamespace(
        _ascend_custom_ops=lambda torch: SimpleNamespace(
            npu_sparse_flash_attention=lambda **kwargs: calls.append(
                ("attention", kwargs["query"], kwargs["sparse_indices"])
            )
            or (
                FakeTensor((1, 1, 512)),
                FakeTensor((1, 1, 1)),
                FakeTensor((1, 1, 1)),
            )
        ),
        _tensor=lambda *args, **kwargs: FakeTensor(),
    )
    case = SimpleNamespace(
        batch=1,
        query_heads=1,
        kv_heads=1,
        query_tokens=1,
        context_tokens=16,
        selected_tokens=1,
        qk_head_dim=576,
        value_head_dim=512,
        shared_kv=True,
    )
    ctx = SimpleNamespace(
        backend=backend,
        torch=FakeTorch(),
        request=SimpleNamespace(dtype="bfloat16", seed=0),
        case=case,
        bound_inputs={},
        device="npu",
        dtype="bfloat16",
    )

    operator = external.build_vllm_ascend_callable(ctx)

    assert calls == []
    operator()
    operator()
    assert [entry if isinstance(entry, str) else entry[0] for entry in calls] == [
        "lower-query",
        "lower-indices",
        "attention",
        "lower-query",
        "lower-indices",
        "attention",
    ]


def test_ascend_vllm_v32_attention_falls_back_to_torch_npu_op(monkeypatch):
    import cannbench.operators.builtin.sparse_attention.external as external

    calls = []

    class FakeTensor:
        def __init__(self, shape=()):
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def __add__(self, other):
            del other
            return FakeTensor(self.shape)

    class FakeTorch:
        int32 = "int32"

        @staticmethod
        def log(tensor):
            return tensor

    payload = {
        "query_shape": (1, 1, 1, 576),
        "shared_kv_shape": (1, 1, 16, 576),
        "value_head_dim": 512,
        "indices_shape": (1, 1, 1),
        "query": tuple(0.0 for _ in range(576)),
        "shared_kv": tuple(0.0 for _ in range(16 * 576)),
        "indices": (0,),
    }
    monkeypatch.setattr(
        external,
        "materialize_sparse_attention_inputs",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(
        external,
        "_lower_vllm_sparse_attention_query",
        lambda *args, **kwargs: (FakeTensor(), FakeTensor()),
    )
    monkeypatch.setattr(
        external,
        "_lower_vllm_sparse_attention_indices",
        lambda *args, **kwargs: FakeTensor(),
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_sparse_flash_attention=lambda **kwargs: calls.append(kwargs)
            or (FakeTensor(), FakeTensor(), FakeTensor())
        ),
    )
    ctx = SimpleNamespace(
        backend=SimpleNamespace(
            _ascend_custom_ops=lambda torch: SimpleNamespace(),
            _tensor=lambda *args, **kwargs: FakeTensor(),
        ),
        torch=FakeTorch(),
        request=SimpleNamespace(dtype="bfloat16", seed=0),
        case=SimpleNamespace(
            batch=1,
            query_heads=1,
            kv_heads=1,
            query_tokens=1,
            context_tokens=16,
            selected_tokens=1,
            qk_head_dim=576,
            value_head_dim=512,
            shared_kv=True,
        ),
        bound_inputs={},
        device="npu",
        dtype="bfloat16",
    )

    operator = external.build_vllm_ascend_callable(ctx)
    operator()

    assert len(calls) == 1
    assert calls[0]["layout_query"] == "TND"
    assert calls[0]["layout_kv"] == "TND"
    assert calls[0]["block_table"] is None


def test_ascend_vllm_sparse_attention_uses_bound_indexer_output(monkeypatch):
    import cannbench.operators.builtin.sparse_attention.external as external

    captured: dict[str, object] = {}

    class FakeTensor:
        def __init__(self, shape=()):
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            self.shape = tuple(self.shape[index] for index in dims)
            return self

        def __getitem__(self, key):
            last = key[-1]
            if isinstance(last, slice):
                start = last.start or 0
                stop = last.stop or self.shape[-1]
                return FakeTensor((*self.shape[:-1], stop - start))
            return self

        def contiguous(self):
            return self

        def __add__(self, other):
            del other
            return FakeTensor(self.shape)

    class FakeTorch:
        int32 = "int32"

        @staticmethod
        def log(tensor):
            return tensor

    payload = {
        "query_shape": (1, 128, 1, 576),
        "shared_kv_shape": (1, 1, 16, 576),
        "value_head_dim": 512,
        "indices_shape": (1, 1, 4),
        "query": tuple(0.0 for _ in range(128 * 576)),
        "shared_kv": tuple(0.0 for _ in range(16 * 576)),
        "indices": (0, 1, 2, 3),
    }
    monkeypatch.setattr(
        external,
        "materialize_sparse_attention_inputs",
        lambda *args, **kwargs: payload,
    )
    backend = SimpleNamespace(
        _ascend_custom_ops=lambda torch: SimpleNamespace(
            npu_sparse_flash_attention=lambda **kwargs: captured.update(kwargs)
            or (FakeTensor((1, 128, 512)), FakeTensor((1, 1, 128)), FakeTensor((1, 1, 128)))
        ),
        _tensor=lambda torch, values, *, device, dtype: FakeTensor((len(values),)),
        _custom_op_pair=lambda *args: pytest.fail("sharedkv ops must not be selected"),
    )
    case = SimpleNamespace(
        batch=1,
        query_heads=128,
        kv_heads=1,
        query_tokens=1,
        context_tokens=16,
        selected_tokens=4,
        qk_head_dim=576,
        value_head_dim=512,
        shared_kv=True,
    )
    indexer_output = FakeTensor((1, 1, 4))

    operator = _build_sparse_attention_vllm_callable(
        backend=backend,
        torch=FakeTorch(),
        request=SimpleNamespace(dtype="bfloat16", seed=0),
        case=case,
        device="npu",
        dtype="bfloat16",
        bound_inputs={"indices": indexer_output},
    )
    operator()

    assert captured["sparse_indices"] is indexer_output


def test_ascend_vllm_v32_prefill_allocates_large_inputs_on_device(monkeypatch):
    import cannbench.operators.builtin.sparse_attention.external as external

    allocations: list[tuple[tuple[int, ...], object]] = []

    class FakeTensor:
        def __init__(self, shape):
            self.shape = shape

        def reshape(self, *shape):
            self.shape = shape[0] if len(shape) == 1 else shape
            return self

        def permute(self, *dims):
            self.shape = tuple(self.shape[index] for index in dims)
            return self

        def __getitem__(self, key):
            last = key[-1]
            if isinstance(last, slice):
                start = last.start or 0
                stop = last.stop or self.shape[-1]
                return FakeTensor((*self.shape[:-1], stop - start))
            return self

        def contiguous(self):
            return self

        def __add__(self, other):
            del other
            return FakeTensor(self.shape)

    class FakeTorch:
        int32 = "int32"

        @staticmethod
        def log(tensor):
            return tensor

        @staticmethod
        def zeros(shape, *, device, dtype):
            del device
            allocations.append((shape, dtype))
            return FakeTensor(shape)

    def fake_sparse_flash_attention(*args, **kwargs):
        del args, kwargs
        return (
            FakeTensor((4096, 128, 512)),
            FakeTensor((1, 4096, 128)),
            FakeTensor((1, 4096, 128)),
        )

    monkeypatch.setattr(
        external,
        "materialize_sparse_attention_inputs",
        lambda *args, **kwargs: pytest.fail("large inputs must not use host materialization"),
    )
    backend = SimpleNamespace(
        _ascend_custom_ops=lambda torch: SimpleNamespace(
            npu_sparse_flash_attention=fake_sparse_flash_attention
        ),
        _tensor=lambda torch, values, *, device, dtype: FakeTensor((len(values),)),
    )
    case = SimpleNamespace(
        batch=1,
        query_heads=128,
        kv_heads=1,
        query_tokens=4096,
        context_tokens=32768,
        selected_tokens=2048,
        qk_head_dim=576,
        value_head_dim=512,
        shared_kv=True,
    )

    operator = _build_sparse_attention_vllm_callable(
        backend=backend,
        torch=FakeTorch(),
        request=SimpleNamespace(dtype="bfloat16", seed=0),
        case=case,
        device="npu",
        dtype="bfloat16",
    )
    output, lse = operator()

    assert output.shape == (1, 4096, 128, 512)
    assert lse.shape == (1, 4096, 128)
    assert allocations == [
        ((1, 128, 4096, 576), "bfloat16"),
        ((32768, 1, 512), "bfloat16"),
        ((32768, 1, 64), "bfloat16"),
        ((32768, 1, 512), "bfloat16"),
        ((1, 4096, 2048), "int32"),
    ]


def test_operator_request_preserves_external_implementation():
    request = OperatorBenchmarkRequest(
        backend="ascend",
        implementation="vllm_ascend",
        op="lightning_indexer",
        dtype="float16",
        dataset="smoke",
        case_id="tiny_decode_top4",
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
            self.tensor = lambda values, **kwargs: FakeTensor().reshape(len(values))

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
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls
    assert calls[0]["layout_query"] == "TND"
    assert calls[0]["layout_key"] == "TND"
    assert calls[0]["key"].shape == (64, 1, 16)
    assert calls[0]["weights"].shape == (2, 2)
    assert calls[0]["sparse_count"] == 4
    assert calls[0]["sparse_mode"] == 3
    assert calls[0]["actual_seq_lengths_query"].shape == 2
    assert calls[0]["actual_seq_lengths_key"].shape == 2


def test_ascend_vllm_indexer_lowers_dynamic_inputs_per_call(monkeypatch):
    import cannbench.operators.builtin.lightning_indexer.external as external

    calls: list[object] = []

    class FakeTensor:
        def reshape(self, *shape):
            del shape
            return self

    payload = {
        "query": (0.0,),
        "keys": (0.0,),
        "weights": (0.0,),
        "query_shape": (1, 1, 1, 1),
        "key_shape": (1, 1, 1),
        "weight_shape": (1, 1, 1),
        "query_lens": (1,),
        "context_lens": (1,),
        "cu_seqlens_q": (0, 1),
        "cu_seqlens_kv": (0, 1),
        "top_k": 1,
    }
    monkeypatch.setattr(
        external,
        "materialize_lightning_indexer_inputs",
        lambda *args, **kwargs: payload,
    )
    monkeypatch.setattr(
        external,
        "_lower_vllm_indexer_query",
        lambda *args, **kwargs: calls.append("lower-query") or "query-tnd",
        raising=False,
    )
    monkeypatch.setattr(
        external,
        "_lower_vllm_indexer_weights",
        lambda *args, **kwargs: calls.append("lower-weights") or "weights-tnd",
        raising=False,
    )
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            npu_lightning_indexer=lambda **kwargs: calls.append(
                ("indexer", kwargs["query"], kwargs["weights"])
            )
            or FakeTensor()
        ),
    )
    ctx = SimpleNamespace(
        backend=SimpleNamespace(_tensor=lambda *args, **kwargs: FakeTensor()),
        torch=SimpleNamespace(int32="int32"),
        request=SimpleNamespace(dtype="bfloat16", seed=0),
        case=SimpleNamespace(),
        device="npu",
        dtype="bfloat16",
    )

    operator = external.build_vllm_ascend_callable(ctx)

    assert calls == []
    operator()
    operator()
    assert calls == [
        "lower-query",
        "lower-weights",
        ("indexer", "query-tnd", "weights-tnd"),
        "lower-query",
        "lower-weights",
        ("indexer", "query-tnd", "weights-tnd"),
    ]


def test_ascend_vllm_adapter_ignores_quant_lightning_indexer(monkeypatch):
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
            self.bfloat16 = "bfloat16"
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
            raise AssertionError("quant lightning indexer metadata must not be used")

        def _indexer(self, **kwargs):
            raise AssertionError("quant lightning indexer must not be used")

    def fake_lightning_indexer(**kwargs):
        calls["indexer"] = kwargs
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
        dtype="bfloat16",
        dataset="smoke",
        case_id="tiny_decode_top4",
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls["indexer"]["layout_query"] == "TND"
    assert calls["indexer"]["layout_key"] == "TND"
    assert calls["indexer"]["key"].shape == (64, 1, 16)
    assert calls["indexer"]["weights"].shape == (2, 2)
    assert calls["indexer"]["sparse_count"] == 4


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


def test_ascend_vllm_sparse_attention_ignores_quant_sharedkv_ops(monkeypatch):
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
            self.bfloat16 = "bfloat16"
            self.float32 = "float32"
            self.float8_e4m3fn = "float8_e4m3fn"
            self.int32 = "int32"
            self.long = "long"
            self.tensor = lambda *args, **kwargs: FakeTensor()
            self.ops = SimpleNamespace(
                _C_ascend=SimpleNamespace(
                    npu_kv_quant_sparse_attn_sharedkv_metadata=self._quant_metadata,
                    npu_kv_quant_sparse_attn_sharedkv=self._quant_attention,
                    npu_sparse_attn_sharedkv_metadata=self._metadata,
                    npu_sparse_attn_sharedkv=self._attention,
                )
            )

        def _quant_metadata(self, **kwargs):
            raise AssertionError("quant sparse attention metadata must not be used")

        def _quant_attention(self, q, **kwargs):
            raise AssertionError("quant sparse attention must not be used")

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
        dtype="bfloat16",
        dataset="smoke",
        case_id="tiny_decode_top4",
    )

    result = AscendBackend().run_operator(request)

    assert result.backend == "ascend"
    assert calls["metadata"]["layout_q"] == "TND"
    assert calls["metadata"]["layout_kv"] == "PA_ND"
    assert calls["metadata"]["cmp_ratio"] == 1
    assert calls["attention"]["metadata"].name == "metadata"
    assert calls["attention"]["cmp_ratio"] == 1
    assert calls["attention"]["cmp_sparse_indices"].shape == (2, 2, 4)


def test_ascend_vllm_sparse_attention_uses_bf16_wide_head_layout(monkeypatch):
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
            self.int32 = "int32"
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
        qk_head_dim=512,
        value_head_dim=512,
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
    assert calls["metadata"]["cmp_ratio"] == 1
    assert calls["metadata"]["has_ori_kv"] is False
    assert calls["metadata"]["has_cmp_kv"] is True
    assert calls["attention"]["q"].shape == (1, 64, 512)
    assert calls["attention"]["ori_kv"] is None
    assert calls["attention"]["cmp_kv"].shape == (4, 128, 1, 512)
    assert calls["attention"]["ori_block_table"] is None
    assert calls["attention"]["cmp_block_table"].shape == (1, 4)
    assert calls["attention"]["cu_seqlens_ori_kv"] is None
    assert calls["attention"]["cu_seqlens_cmp_kv"] is None
    assert calls["attention"]["sinks"] is None
    assert calls["attention"]["cmp_ratio"] == 1


def test_ascend_vllm_sparse_attention_bf16_setup_avoids_device_permute(monkeypatch):
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
            self.int32 = "int32"
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
        qk_head_dim=512,
        value_head_dim=512,
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
    assert calls["attention"]["ori_kv"] is None
    assert calls["attention"]["cmp_kv"].shape == (4, 128, 1, 512)


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
    )

    result = NvidiaBackend().run_operator(request)

    assert result.backend == "nvidia"
    assert calls
    assert calls[0]["request"] is request
    assert calls[0]["payload"]["top_k"] == 4
    assert calls[0]["query"].shape == (2, 1, 2, 16)
    assert calls[0]["keys"].shape == (2, 32, 16)
    assert calls[0]["weights"].shape == (2, 1, 2)


def test_nvidia_cuda_library_prepares_lightning_indexer_once(monkeypatch):
    import cannbench.operators.builtin.lightning_indexer.external as external

    calls: list[object] = []

    class FakeTensor:
        def reshape(self, *shape):
            del shape
            return self

    payload = {
        "query": (0.0,),
        "keys": (0.0,),
        "weights": (0.0,),
        "query_shape": (1, 1, 1, 1),
        "key_shape": (1, 1, 1),
        "weight_shape": (1, 1, 1),
        "top_k": 1,
        "score_scale": 1.0,
        "tie_policy": "equivalent_score_set",
    }
    monkeypatch.setattr(
        external,
        "materialize_lightning_indexer_inputs",
        lambda *args, **kwargs: payload,
    )

    def prepare_lightning_indexer(**kwargs):
        calls.append(("prepare", kwargs))

        def operator():
            calls.append("dynamic")
            return "indices"

        return operator

    monkeypatch.setitem(
        sys.modules,
        "fake_cuda_dsa_adapter",
        SimpleNamespace(
            lightning_indexer=lambda **kwargs: pytest.fail(
                "legacy adapter entry point must not run"
            ),
            prepare_lightning_indexer=prepare_lightning_indexer,
        ),
    )
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_ADAPTER", "fake_cuda_dsa_adapter")
    ctx = SimpleNamespace(
        backend=SimpleNamespace(_tensor=lambda *args, **kwargs: FakeTensor()),
        torch=SimpleNamespace(),
        request=SimpleNamespace(dtype="bfloat16", seed=0),
        case=SimpleNamespace(),
        device="cuda",
        dtype="bfloat16",
    )

    operator = external.build_cuda_library_callable(ctx)

    assert [entry[0] for entry in calls] == ["prepare"]
    assert operator() == "indices"
    assert operator() == "indices"
    assert [entry if isinstance(entry, str) else entry[0] for entry in calls] == [
        "prepare",
        "dynamic",
        "dynamic",
    ]


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
    )

    result = NvidiaBackend().run_operator(request)

    assert result.backend == "nvidia"
    assert calls
    assert calls[0]["request"] is request
    assert calls[0]["payload"]["phase"] == "decode"
    assert calls[0]["query"].shape == (2, 2, 1, 16)
    assert calls[0]["shared_kv"].shape == (2, 2, 32, 16)
    assert "keys" not in calls[0]
    assert "values" not in calls[0]
    assert calls[0]["indices"].shape == (2, 1, 4)


def test_nvidia_cuda_library_prepares_sparse_attention_once(monkeypatch):
    import cannbench.operators.builtin.sparse_attention.external as external

    calls: list[object] = []

    class FakeTensor:
        def reshape(self, *shape):
            del shape
            return self

    payload = {
        "query": (0.0,),
        "shared_kv": (0.0,),
        "indices": (0,),
        "query_shape": (1, 1, 1, 1),
        "shared_kv_shape": (1, 1, 1, 1),
        "indices_shape": (1, 1, 1),
        "selected_tokens": 1,
        "topk_lengths": (1,),
        "softmax_scale": 1.0,
        "causal": False,
        "phase": "decode",
    }
    monkeypatch.setattr(
        external,
        "materialize_sparse_attention_inputs",
        lambda *args, **kwargs: payload,
    )

    def prepare_sparse_attention(**kwargs):
        calls.append(("prepare", kwargs))

        def operator():
            calls.append("dynamic")
            return "output"

        return operator

    monkeypatch.setitem(
        sys.modules,
        "fake_cuda_dsa_adapter",
        SimpleNamespace(
            sparse_attention=lambda **kwargs: pytest.fail(
                "legacy adapter entry point must not run"
            ),
            prepare_sparse_attention=prepare_sparse_attention,
        ),
    )
    monkeypatch.setenv("CANNBENCH_CUDA_DSA_ADAPTER", "fake_cuda_dsa_adapter")
    ctx = SimpleNamespace(
        backend=SimpleNamespace(_tensor=lambda *args, **kwargs: FakeTensor()),
        torch=SimpleNamespace(int32="int32"),
        request=SimpleNamespace(dtype="bfloat16", seed=0),
        case=SimpleNamespace(),
        bound_inputs={},
        device="cuda",
        dtype="bfloat16",
    )

    operator = external.build_cuda_library_callable(ctx)

    assert [entry[0] for entry in calls] == ["prepare"]
    assert operator() == "output"
    assert operator() == "output"
    assert [entry if isinstance(entry, str) else entry[0] for entry in calls] == [
        "prepare",
        "dynamic",
        "dynamic",
    ]


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
    )

    with pytest.raises(RuntimeError, match="callable sparse_attention"):
        NvidiaBackend().run_operator(request)
