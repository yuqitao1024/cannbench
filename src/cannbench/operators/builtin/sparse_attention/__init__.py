from __future__ import annotations

from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_sparse_attention_inputs
from .cases import (
    get_sparse_attention_case,
    get_sparse_attention_dataset,
)
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec
from .external import build_cuda_library_callable, build_vllm_ascend_callable


def _build_torch_callable(ctx):
    payload = materialize_sparse_attention_inputs(
        ctx.case, dtype=ctx.request.dtype, seed=ctx.request.seed
    )
    query = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["query"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["query_shape"])
    keys = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["keys"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["key_shape"])
    values = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["values"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["value_shape"])
    indices = ctx.backend._tensor(
        ctx.torch,
        payload["indices"],
        device=ctx.device,
        dtype=ctx.torch.long,
    ).reshape(payload["indices_shape"])
    return lambda: ctx.backend._sparse_attention(
        ctx.torch,
        query,
        keys,
        values,
        indices,
        causal=payload["causal"],
        phase=payload["phase"],
    )


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="sparse_attention",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="sparse_attention",
        runner_name="sparse_attention",
    ),
    get_dataset=get_sparse_attention_dataset,
    get_case=get_sparse_attention_case,
    materialize_inputs=materialize_sparse_attention_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=13,
    build_cuda_library_callable=build_cuda_library_callable,
    build_vllm_ascend_callable=build_vllm_ascend_callable,
)
