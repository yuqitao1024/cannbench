from dataclasses import replace

import pytest

from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)
from cannbench.operators.builtin.sparse_attention.materialize import (
    materialize_sparse_attention_inputs,
)
from cannbench.operators.builtin.sparse_attention import (
    _build_ragged_simt_operator,
)


def test_v32_sparse_attention_exposes_real_sequence_metadata():
    case = get_sparse_attention_case(
        "realistic_prefill",
        "deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048",
    )

    assert case.resolved_query_lens == (4096,)
    assert case.resolved_context_lens == (32768,)
    assert case.resolved_query_start_positions == (28672,)
    assert case.cu_seqlens_q == (0, 4096)
    assert case.cu_seqlens_kv == (0, 32768)
    assert case.resolved_page_block_size == 64


def test_sparse_attention_ragged_metadata_invalidates_padding_rows():
    case = replace(
        get_sparse_attention_case("smoke", "tiny_decode_top4"),
        batch=2,
        query_tokens=3,
        context_tokens=8,
        selected_tokens=4,
        causal=True,
        query_lens=(1, 3),
        context_lens=(4, 8),
        query_start_positions=(3, 5),
        page_block_size=4,
    )

    payload = materialize_sparse_attention_inputs(case, dtype="bfloat16", seed=7)

    assert case.resolved_topk_lengths == (4, 0, 0, 4, 4, 4)
    assert payload["query_lens"] == (1, 3)
    assert payload["context_lens"] == (4, 8)
    assert payload["query_start_positions"] == (3, 5)
    assert payload["cu_seqlens_q"] == (0, 1, 4)
    assert payload["cu_seqlens_kv"] == (0, 4, 12)
    assert payload["block_tables"] == ((0, -1), (2, 3))
    assert payload["indices"][4:12] == (-1,) * 8


def test_sparse_attention_rejects_nonzero_topk_length_for_padding_query():
    with pytest.raises(ValueError, match="padding query rows must have zero"):
        replace(
            get_sparse_attention_case("smoke", "tiny_decode_top4"),
            batch=2,
            query_tokens=3,
            context_tokens=8,
            selected_tokens=4,
            query_lens=(1, 3),
            context_lens=(4, 8),
            query_start_positions=(3, 5),
            topk_lengths=(4, 4, 4, 4, 4, 4),
        )


def test_sparse_attention_rejects_non_right_aligned_causal_query_positions():
    with pytest.raises(ValueError, match="right-aligned"):
        replace(
            get_sparse_attention_case("smoke", "tiny_decode_top4"),
            query_start_positions=(0, 0),
        )


def test_cuda_prefill_ragged_packing_preserves_invalid_indices():
    from cannbench_cuda_dsa_flashmla_deepgemm import (
        _offset_prefill_indices,
    )

    class FakeTensor:
        device = "cuda"

        def __init__(self, values):
            self.values = values

        def __lt__(self, value):
            return FakeTensor(
                [
                    [[item < value for item in row] for row in batch]
                    for batch in self.values
                ]
            )

        def __add__(self, offsets):
            return FakeTensor(
                [
                    [
                        [item + offsets.values[batch_index] for item in row]
                        for row in batch
                    ]
                    for batch_index, batch in enumerate(self.values)
                ]
            )

        def masked_fill(self, mask, value):
            return FakeTensor(
                [
                    [
                        [
                            value if mask.values[batch_index][row_index][item_index]
                            else item
                            for item_index, item in enumerate(row)
                        ]
                        for row_index, row in enumerate(batch)
                    ]
                    for batch_index, batch in enumerate(self.values)
                ]
            )

    class FakeOffsets:
        def __init__(self, values):
            self.values = values

        def view(self, *shape):
            del shape
            return self

        def __mul__(self, value):
            return FakeOffsets([item * value for item in self.values])

    class FakeTorch:
        int32 = "int32"

        @staticmethod
        def arange(batch, *, device, dtype):
            del device, dtype
            return FakeOffsets(list(range(batch)))

    indices = FakeTensor(
        [
            [[0, 1], [-1, -1], [-1, -1]],
            [[-1, 1], [2, 3], [4, -1]],
        ]
    )

    result = _offset_prefill_indices(
        FakeTorch(), indices, batch=2, context_tokens=5
    )

    assert result.values == [
        [[0, 1], [-1, -1], [-1, -1]],
        [[-1, 6], [7, 8], [9, -1]],
    ]


def test_sparse_attention_ragged_simt_lowering_slices_each_request_on_device():
    calls = []
    outputs = []

    class FakeTensor:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, key):
            return self.name, key

    class FakeOutput:
        def __init__(self):
            self.assignments = []

        def __setitem__(self, key, value):
            self.assignments.append((key, value))

    class FakeTorch:
        float32 = "float32"

        @staticmethod
        def zeros(shape, *, device, dtype):
            del shape, device, dtype
            output = FakeOutput()
            outputs.append(output)
            return output

        @staticmethod
        def full(shape, value, *, device, dtype):
            del shape, value, device, dtype
            output = FakeOutput()
            outputs.append(output)
            return output

    def fake_forward(query, shared_kv, indices, **kwargs):
        calls.append((query, shared_kv, indices, kwargs))
        return f"output-{len(calls)}", f"lse-{len(calls)}"

    ctx = type(
        "Context",
        (),
        {
            "torch": FakeTorch(),
            "device": "npu",
            "dtype": "bfloat16",
            "implementation_module": type(
                "Module",
                (),
                {
                    "ops": type(
                        "Ops",
                        (),
                        {"sparse_attention_forward": staticmethod(fake_forward)},
                    )()
                },
            )(),
        },
    )()
    operator = _build_ragged_simt_operator(
        ctx,
        {
            "query_shape": (2, 128, 3, 128),
            "query_lens": (1, 3),
            "context_lens": (4, 8),
            "value_head_dim": 128,
            "phase": "decode",
            "causal": True,
        },
        query=FakeTensor("query"),
        shared_kv=FakeTensor("shared_kv"),
        indices=FakeTensor("indices"),
        family="family_hd128",
    )

    result = operator()

    assert len(calls) == 2
    assert calls[0][0][1][2] == slice(None, 1)
    assert calls[0][1][1][2] == slice(None, 4)
    assert calls[1][0][1][2] == slice(None, 3)
    assert calls[1][1][1][2] == slice(None, 8)
    assert result == tuple(outputs)
    assert [value for _, value in outputs[0].assignments] == [
        "output-1",
        "output-2",
    ]
    assert [value for _, value in outputs[1].assignments] == [
        "lse-1",
        "lse-2",
    ]
