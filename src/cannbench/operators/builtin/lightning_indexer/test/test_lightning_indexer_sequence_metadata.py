from dataclasses import replace

import pytest

from cannbench.operators.builtin.lightning_indexer.cases import (
    get_lightning_indexer_case,
)
from cannbench.operators.builtin.lightning_indexer.materialize import (
    materialize_lightning_indexer_inputs,
    valid_context_lengths,
)
from cannbench.operators.builtin.lightning_indexer import (
    _build_ragged_simt_operator,
)


def test_v32_indexer_exposes_real_sequence_metadata():
    case = get_lightning_indexer_case(
        "realistic_decode",
        "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
    )

    assert case.resolved_query_lens == (2, 2)
    assert case.resolved_context_lens == (32768, 32768)
    assert case.resolved_query_start_positions == (32766, 32766)
    assert case.cu_seqlens_q == (0, 2, 4)
    assert case.cu_seqlens_kv == (0, 32768, 65536)
    assert case.resolved_page_block_size == 64
    assert len(case.block_tables) == 2
    assert len(case.block_tables[0]) == 512


def test_indexer_ragged_metadata_controls_padded_query_rows():
    case = replace(
        get_lightning_indexer_case("smoke", "tiny_decode_top4"),
        batch=2,
        query_tokens=3,
        context_tokens=8,
        top_k=4,
        causal=True,
        query_lens=(1, 3),
        context_lens=(4, 8),
        query_start_positions=(3, 5),
        page_block_size=4,
    )

    payload = materialize_lightning_indexer_inputs(case, dtype="bfloat16", seed=7)

    assert valid_context_lengths(case) == (4, 0, 0, 6, 7, 8)
    assert payload["query_lens"] == (1, 3)
    assert payload["context_lens"] == (4, 8)
    assert payload["cu_seqlens_q"] == (0, 1, 4)
    assert payload["cu_seqlens_kv"] == (0, 4, 12)
    assert payload["block_tables"] == ((0, -1), (2, 3))


def test_indexer_rejects_non_right_aligned_causal_query_positions():
    with pytest.raises(ValueError, match="right-aligned"):
        replace(
            get_lightning_indexer_case("smoke", "tiny_decode_top4"),
            causal=True,
            query_start_positions=(0, 0),
        )


def test_indexer_ragged_simt_lowering_slices_each_request_on_device():
    calls = []

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
        int32 = "int32"

        @staticmethod
        def full(shape, value, *, device, dtype):
            del shape, value, device, dtype
            return FakeOutput()

    def fake_forward(query, keys, weights, **kwargs):
        calls.append((query, keys, weights, kwargs))
        return f"request-{len(calls)}"

    ctx = type(
        "Context",
        (),
        {
            "torch": FakeTorch(),
            "device": "npu",
            "implementation_module": type(
                "Module",
                (),
                {
                    "ops": type(
                        "Ops",
                        (),
                        {"lightning_indexer_forward": staticmethod(fake_forward)},
                    )()
                },
            )(),
        },
    )()
    operator = _build_ragged_simt_operator(
        ctx,
        {
            "query_shape": (2, 3, 64, 128),
            "query_lens": (1, 3),
            "context_lens": (4, 8),
            "top_k": 4,
            "phase": "decode",
        },
        query=FakeTensor("query"),
        keys=FakeTensor("keys"),
        weights=FakeTensor("weights"),
        valid_context_lengths=FakeTensor("valid_lengths"),
        family="family_64x128",
    )

    output = operator()

    assert len(calls) == 2
    assert calls[0][0][1][1] == slice(None, 1)
    assert calls[0][1][1][1] == slice(None, 4)
    assert calls[1][0][1][1] == slice(None, 3)
    assert calls[1][1][1][1] == slice(None, 8)
    assert [value for _, value in output.assignments] == [
        "request-1",
        "request-2",
    ]
