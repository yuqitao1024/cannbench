from __future__ import annotations

from cannbench.operators.builtin.sparse_attention.cases import (
    SparseAttentionCase,
    get_sparse_attention_case,
)
from cannbench.operators.builtin.sparse_attention.materialize import (
    materialize_sparse_attention_inputs,
)
from cannbench.operators.materialize import materialized_values_to_buffer
from cannbench.operators.builtin.sparse_attention.simt.v1.aten_dsa_sparse_attention import (
    ops,
)

import pytest


def _require_custom_sparse_attention_op():
    if ops.torch is None:
        pytest.skip("torch is required for exact custom-op correctness coverage")
    namespace = getattr(ops.torch.ops, "aten_dsa_sparse_attention", None)
    if namespace is None or not hasattr(namespace, "sparse_attention_forward"):
        pytest.skip("registered custom op is required for exact custom-op correctness coverage")
    npu_namespace = getattr(ops.torch, "npu", None)
    if npu_namespace is None or not npu_namespace.is_available():
        pytest.skip("torch.npu with an available PrivateUse1 device is required")


def _build_npu_inputs(case: SparseAttentionCase, *, seed: int = 7):
    payload = materialize_sparse_attention_inputs(case, dtype="float16", seed=seed)
    device = ops.torch.device("npu")
    query = ops.torch.tensor(
        materialized_values_to_buffer(payload["query"]),
        device=device,
        dtype=ops.torch.float16,
    ).reshape(payload["query_shape"])
    keys = ops.torch.tensor(
        materialized_values_to_buffer(payload["keys"]),
        device=device,
        dtype=ops.torch.float16,
    ).reshape(payload["key_shape"])
    values = ops.torch.tensor(
        materialized_values_to_buffer(payload["values"]),
        device=device,
        dtype=ops.torch.float16,
    ).reshape(payload["value_shape"])
    indices = ops.torch.tensor(
        payload["indices"],
        device=device,
        dtype=ops.torch.long,
    ).reshape(payload["indices_shape"])
    return query, keys, values, indices


def _run_reference(case: SparseAttentionCase, query, keys, values, indices):
    reference_fn = ops._decode_reference if case.phase == "decode" else ops._prefill_reference
    return reference_fn(query, keys, values, indices, causal=case.causal)


def _run_custom_op(case: SparseAttentionCase, query, keys, values, indices):
    return ops.torch.ops.aten_dsa_sparse_attention.sparse_attention_forward(
        query,
        keys,
        values,
        indices,
        case.phase,
        "family_hd512" if case.head_dim == 512 else "family_hd128",
        case.causal,
    )


def _assert_matches_reference(case: SparseAttentionCase, query, keys, values, indices):
    reference_out, reference_lse = _run_reference(case, query, keys, values, indices)
    custom_out, custom_lse = _run_custom_op(case, query, keys, values, indices)
    assert ops.torch.allclose(custom_out.float(), reference_out.float(), atol=5e-2, rtol=5e-2)
    assert ops.torch.allclose(
        custom_lse.float(),
        reference_lse.float(),
        atol=5e-2,
        rtol=5e-2,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    ("dataset_name", "case_id"),
    [
        ("smoke", "deepseek_a5_decode_b1_ctx512_top512"),
        ("smoke", "deepseek_a5_decode_b1_ctx16384_top1024"),
        ("realistic_prefill", "deepseek_a5_prefill_b1_q64_ctx512_top512"),
        ("realistic_prefill", "deepseek_a5_prefill_b1_q512_ctx1024_top1024"),
        ("realistic_decode", "deepseek_128k_decode_top2048"),
        ("smoke", "tiny_hd128_prefill_top8"),
        ("realistic_prefill", "deepseek_v32_prefill_b1_q128_ctx16384_top2048"),
    ],
    ids=[
        "hd512_decode_smoke_top512",
        "hd512_decode_top1024",
        "hd512_prefill_q64_top512",
        "hd512_prefill_q512_top1024",
        "hd128_decode_top2048",
        "hd128_prefill_smoke",
        "hd128_prefill_top2048",
    ],
)
def test_custom_op_matches_reference_for_existing_sparse_attention_cases(
    dataset_name,
    case_id,
):
    _require_custom_sparse_attention_op()
    case = get_sparse_attention_case(dataset_name, case_id)
    query, keys, values, indices = _build_npu_inputs(case)
    _assert_matches_reference(case, query, keys, values, indices)


def test_custom_op_prefill_matches_reference_for_all_invalid_rows():
    _require_custom_sparse_attention_op()
    case = get_sparse_attention_case(
        "realistic_prefill",
        "deepseek_a5_prefill_b1_q64_ctx512_top512",
    )
    query, keys, values, indices = _build_npu_inputs(case, seed=17)
    indices = indices.clone()
    indices[:, :4, :] = case.context_tokens - 1
    _assert_matches_reference(case, query, keys, values, indices)


def test_custom_op_prefill_lse_uses_negative_infinity_not_nan_for_all_invalid_rows():
    _require_custom_sparse_attention_op()
    case = get_sparse_attention_case(
        "realistic_prefill",
        "deepseek_a5_prefill_b1_q64_ctx512_top512",
    )
    query, keys, values, indices = _build_npu_inputs(case, seed=17)
    indices = indices.clone()
    indices[:, :4, :] = case.context_tokens - 1

    reference_out, reference_lse = _run_reference(case, query, keys, values, indices)
    custom_out, custom_lse = _run_custom_op(case, query, keys, values, indices)

    del reference_out, custom_out

    reference_lse = reference_lse.float()
    custom_lse = custom_lse.float()

    assert not ops.torch.isnan(reference_lse).any()
    assert not ops.torch.isnan(custom_lse).any()
    assert ops.torch.equal(ops.torch.isinf(custom_lse), ops.torch.isinf(reference_lse))

    finite_mask = ops.torch.isfinite(reference_lse) & ops.torch.isfinite(custom_lse)
    assert finite_mask.any()
    assert ops.torch.allclose(
        custom_lse[finite_mask],
        reference_lse[finite_mask],
        atol=5e-2,
        rtol=5e-2,
    )
