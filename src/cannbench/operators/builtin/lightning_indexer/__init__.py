from __future__ import annotations

from .cases import (
    get_lightning_indexer_case,
    get_lightning_indexer_dataset,
)
from cannbench.operators.materialize import materialized_values_to_buffer
from .materialize import materialize_lightning_indexer_inputs
from cannbench.operators.plugin import OperatorPlugin
from cannbench.operators.spec import OperatorSpec
from .external import build_cuda_library_callable, build_vllm_ascend_callable


def _build_torch_callable(ctx):
    payload = materialize_lightning_indexer_inputs(
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
    weights = ctx.backend._tensor(
        ctx.torch,
        materialized_values_to_buffer(payload["weights"]),
        device=ctx.device,
        dtype=ctx.dtype,
    ).reshape(payload["weight_shape"])
    return lambda: ctx.backend._lightning_indexer(
        ctx.torch,
        query,
        keys,
        weights,
        top_k=payload["top_k"],
    )


PLUGIN = OperatorPlugin(
    spec=OperatorSpec(
        name="lightning_indexer",
        supported_dtypes=("float32", "float16", "bfloat16"),
        dataset_namespace="lightning_indexer",
        runner_name="lightning_indexer",
    ),
    get_dataset=get_lightning_indexer_dataset,
    get_case=get_lightning_indexer_case,
    materialize_inputs=materialize_lightning_indexer_inputs,
    build_torch_callable=_build_torch_callable,
    sort_order=12,
    build_cuda_library_callable=build_cuda_library_callable,
    build_vllm_ascend_callable=build_vllm_ascend_callable,
)
