from pathlib import Path
import sys
from types import SimpleNamespace


def test_vllm_bhtd_wrapper_keeps_key_and_value_storage_independent(monkeypatch):
    project_dir = Path(__file__).parents[1] / "vllm"
    monkeypatch.syspath_prepend(str(project_dir))
    from aten_dsa_sparse_attention_vllm import ops

    class FakeTensor:
        next_storage = 0

        def __init__(self, shape, *, storage=None, dtype="bfloat16"):
            self.shape = shape
            self.dtype = dtype
            self.device = "npu"
            if storage is None:
                storage = self.next_storage
                type(self).next_storage += 1
            self.storage = storage

        def _view(self, shape):
            return FakeTensor(shape, storage=self.storage, dtype=self.dtype)

        def permute(self, *dims):
            return self._view(tuple(self.shape[index] for index in dims))

        def reshape(self, *shape):
            return self._view(shape)

        def __getitem__(self, key):
            last = key[-1]
            start = last.start or 0
            stop = last.stop or self.shape[-1]
            return self._view((*self.shape[:-1], stop - start))

        def contiguous(self):
            return self

        def clone(self):
            return FakeTensor(self.shape, dtype=self.dtype)

        def to(self, *, dtype):
            return FakeTensor(self.shape, dtype=dtype)

        def data_ptr(self):
            return self.storage

        def __add__(self, other):
            del other
            return self._view(self.shape)

    fake_torch = SimpleNamespace(
        int32="int32",
        tensor=lambda values, **kwargs: FakeTensor(
            (len(values),), dtype=kwargs.get("dtype", "int32")
        ),
        log=lambda tensor: tensor,
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    captured = {}

    def fake_attention(**kwargs):
        captured.update(kwargs)
        return (
            FakeTensor((1, 64, 512)),
            FakeTensor((1, 1, 64), dtype="float32"),
            FakeTensor((1, 1, 64), dtype="float32"),
        )

    monkeypatch.setattr(ops, "sparse_flash_attention_forward", fake_attention)
    query = FakeTensor((1, 64, 1, 576))
    shared_kv = FakeTensor((1, 1, 4, 576))
    indices = FakeTensor((1, 1, 2), dtype="int64")

    output, lse = ops.sparse_attention_forward(
        query,
        shared_kv,
        indices,
        value_head_dim=512,
        phase="decode",
        family="family_hd576",
        causal=True,
    )

    assert captured["key"].data_ptr() != captured["value"].data_ptr()
    assert captured["attention_mode"] == 2
    assert captured["return_softmax_lse"] is True
    assert output.shape == (1, 1, 64, 512)
    assert lse.shape == (1, 1, 64)


def test_vllm_wrapper_uses_torch_npu_three_output_binding(monkeypatch):
    project_dir = Path(__file__).parents[1] / "vllm"
    monkeypatch.syspath_prepend(str(project_dir))
    from aten_dsa_sparse_attention_vllm import ops

    expected = object()
    fake_torch = SimpleNamespace(ops=SimpleNamespace())
    fake_torch_npu = SimpleNamespace(npu_sparse_flash_attention=expected)
    imports = []

    def fake_import_module(name):
        imports.append(name)
        if name == "torch":
            return fake_torch
        if name == "torch_npu":
            return fake_torch_npu
        if name == "vllm_ascend.vllm_ascend_C":
            raise AssertionError("the installed vLLM 0.18 binding has an old ABI")
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(ops, "import_module", fake_import_module)

    assert ops._resolve_attention_op() is expected
    assert imports == ["torch", "torch_npu"]


def test_vllm_wrapper_rejects_torch_npu_without_three_output_binding(monkeypatch):
    project_dir = Path(__file__).parents[1] / "vllm"
    monkeypatch.syspath_prepend(str(project_dir))
    from aten_dsa_sparse_attention_vllm import ops

    fake_torch = SimpleNamespace(ops=SimpleNamespace())
    imports = []

    def fake_import_module(name):
        imports.append(name)
        if name == "torch":
            return fake_torch
        if name == "torch_npu":
            return SimpleNamespace()
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(ops, "import_module", fake_import_module)

    try:
        ops._resolve_attention_op()
    except RuntimeError as exc:
        assert "torch_npu.npu_sparse_flash_attention" in str(exc)
    else:
        raise AssertionError("missing three-output binding must be rejected")
    assert imports == ["torch", "torch_npu"]
