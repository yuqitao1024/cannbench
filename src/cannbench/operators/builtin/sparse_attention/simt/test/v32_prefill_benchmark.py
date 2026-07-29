from __future__ import annotations

import argparse
import json
import statistics
import time

from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)
from v32_full_accuracy import _build_indices, _fill_deterministic


_DATASET = "realistic_prefill"
_CASE_ID = "deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048"


def summarize_samples(samples_ms: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }


def _materialize_inputs(torch, case, *, seed: int):
    device = torch.device("npu")
    query = torch.empty(
        (case.batch, case.query_heads, case.query_tokens, case.qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    shared_kv = torch.empty(
        (case.batch, case.kv_heads, case.context_tokens, case.qk_head_dim),
        dtype=torch.bfloat16,
        device=device,
    )
    _fill_deterministic(torch, query, seed=seed + 1, nonnegative=False)
    _fill_deterministic(torch, shared_kv, seed=seed + 2, nonnegative=False)
    indices = _build_indices(torch, case, seed=seed + 3, device=device)
    return query, shared_kv, indices


def _validate_outputs(torch, case, output, lse) -> None:
    expected_output_shape = (
        case.batch,
        case.query_tokens,
        case.query_heads,
        case.value_head_dim,
    )
    expected_lse_shape = (case.batch, case.query_tokens, case.query_heads)
    if output.shape != expected_output_shape:
        raise RuntimeError(
            f"unexpected output shape: {tuple(output.shape)} != {expected_output_shape}"
        )
    if lse.shape != expected_lse_shape:
        raise RuntimeError(
            f"unexpected LSE shape: {tuple(lse.shape)} != {expected_lse_shape}"
        )
    if output.dtype != torch.bfloat16:
        raise RuntimeError(f"unexpected output dtype: {output.dtype}")
    if lse.dtype != torch.float32:
        raise RuntimeError(f"unexpected LSE dtype: {lse.dtype}")


def run_benchmark(*, warmups: int, iters: int, seed: int) -> dict[str, object]:
    import torch
    import torch_npu  # noqa: F401
    from aten_dsa_sparse_attention import ops

    case = get_sparse_attention_case(_DATASET, _CASE_ID)
    query, shared_kv, indices = _materialize_inputs(torch, case, seed=seed)
    torch.npu.synchronize()

    def run_once():
        return ops.sparse_attention_forward(
            query,
            shared_kv,
            indices,
            value_head_dim=case.value_head_dim,
            phase=case.phase,
            family="family_hd576",
            causal=case.causal,
        )

    for _ in range(warmups):
        run_once()
        torch.npu.synchronize()

    samples_ms = []
    output = lse = None
    for _ in range(iters):
        started = time.perf_counter_ns()
        output, lse = run_once()
        torch.npu.synchronize()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)
        _validate_outputs(torch, case, output, lse)

    assert output is not None and lse is not None
    return {
        "case_id": case.case_id,
        "phase": case.phase,
        "query_shape": list(query.shape),
        "shared_kv_shape": list(shared_kv.shape),
        "indices_shape": list(indices.shape),
        "output_shape": list(output.shape),
        "output_dtype": str(output.dtype),
        "lse_shape": list(lse.shape),
        "lse_dtype": str(lse.dtype),
        "warmups": warmups,
        "iters": iters,
        "seed": seed,
        **summarize_samples(samples_ms),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.iters <= 0:
        parser.error("warmups must be nonnegative and iters must be positive")

    print(
        json.dumps(run_benchmark(**vars(args)), indent=2, sort_keys=True),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
