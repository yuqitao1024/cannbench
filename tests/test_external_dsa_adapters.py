import sys
from types import SimpleNamespace

import pytest

from cannbench.core.config import OperatorBenchmarkRequest


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


def test_ascend_vllm_sparse_attention_requires_metadata_adapter(monkeypatch):
    class FakeTorch:
        def __init__(self) -> None:
            self.npu = SimpleNamespace(is_available=lambda: True)
            self.device = lambda kind: kind
            self.float16 = "float16"

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

    with pytest.raises(RuntimeError, match="paged-KV metadata adapter"):
        AscendBackend().run_operator(request)


def test_nvidia_cuda_library_dsa_requires_flashmla_adapter(monkeypatch):
    class FakeTorch:
        def __init__(self) -> None:
            self.cuda = SimpleNamespace(is_available=lambda: True)
            self.device = lambda kind: kind
            self.float16 = "float16"

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())

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

    with pytest.raises(RuntimeError, match="FlashMLA/DeepGEMM adapter"):
        NvidiaBackend().run_operator(request)
