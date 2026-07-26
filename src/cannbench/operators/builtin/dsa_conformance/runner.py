from __future__ import annotations

import argparse
import json
import time
from functools import lru_cache
from pathlib import Path

from cannbench.operators.builtin.lightning_indexer.cases import (
    get_lightning_indexer_case,
)
from cannbench.operators.builtin.lightning_indexer.materialize import (
    valid_context_lengths,
)
from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)

from .artifact import validation_query_rows, write_manifest


_SCHEMA_VERSION = 1
_PATTERN_PERIOD = 65521
_CASE_IDS = {
    "prefill": "deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048",
    "decode": "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
}
_DATASETS = {
    "prefill": "realistic_prefill",
    "decode": "realistic_decode",
}


@lru_cache(maxsize=None)
def deterministic_pattern(*, seed: int, nonnegative: bool) -> tuple[float, ...]:
    values = list(range(_PATTERN_PERIOD))
    state = seed & 0xFFFFFFFFFFFFFFFF
    mask = 0xFFFFFFFFFFFFFFFF
    for index in range(_PATTERN_PERIOD - 1, 0, -1):
        state = (state + 0x9E3779B97F4A7C15) & mask
        mixed = state
        mixed = ((mixed ^ (mixed >> 30)) * 0xBF58476D1CE4E5B9) & mask
        mixed = ((mixed ^ (mixed >> 27)) * 0x94D049BB133111EB) & mask
        mixed ^= mixed >> 31
        swap_index = mixed % (index + 1)
        values[index], values[swap_index] = values[swap_index], values[index]
    if nonnegative:
        return tuple(value / (_PATTERN_PERIOD - 1) for value in values)
    return tuple((value - 32760.0) / 32768.0 for value in values)


def load_v32_cases(phase: str):
    try:
        dataset = _DATASETS[phase]
        case_id = _CASE_IDS[phase]
    except KeyError as exc:
        raise ValueError(f"unsupported V3.2 phase: {phase}") from exc
    indexer_case = get_lightning_indexer_case(dataset, case_id)
    sparse_case = get_sparse_attention_case(dataset, case_id)
    _validate_component_cases(indexer_case, sparse_case)
    return indexer_case, sparse_case


def canonical_input_descriptor(indexer_case, sparse_case) -> dict:
    return {
        "generator": "splitmix64-period-v2",
        "pattern_period": _PATTERN_PERIOD,
        "query_tokens": indexer_case.query_tokens,
        "context_tokens": indexer_case.context_tokens,
        "top_k": indexer_case.top_k,
        "index_shape": [
            indexer_case.batch,
            indexer_case.query_tokens,
            indexer_case.index_heads,
            indexer_case.index_dim,
        ],
        "index_key_shape": [
            indexer_case.batch,
            indexer_case.context_tokens,
            indexer_case.index_dim,
        ],
        "index_weight_shape": [
            indexer_case.batch,
            indexer_case.query_tokens,
            indexer_case.index_heads,
        ],
        "attention_query_shape": [
            sparse_case.batch,
            sparse_case.query_heads,
            sparse_case.query_tokens,
            sparse_case.qk_head_dim,
        ],
        "shared_kv_shape": [
            sparse_case.batch,
            sparse_case.kv_heads,
            sparse_case.context_tokens,
            sparse_case.qk_head_dim,
        ],
        "value_head_dim": sparse_case.value_head_dim,
        "query_lens": list(sparse_case.resolved_query_lens),
        "context_lens": list(sparse_case.resolved_context_lens),
        "query_start_positions": list(
            sparse_case.resolved_query_start_positions
        ),
        "page_block_size": sparse_case.resolved_page_block_size,
        "seed_offsets": {
            "index_query": 11,
            "index_keys": 12,
            "index_weights": 13,
            "attention_query": 21,
            "shared_kv": 22,
            "attention_indices": 23,
        },
    }


def run_backend(
    *,
    backend: str,
    phase: str,
    seed: int,
    output_directory: Path,
) -> dict:
    import torch

    indexer_case, sparse_case = load_v32_cases(phase)
    device = _resolve_device(torch, backend)
    if backend in {"simt", "vllm_ascend"}:
        import torch_npu  # noqa: F401

    index_query = _empty_and_fill(
        torch,
        (
            indexer_case.batch,
            indexer_case.query_tokens,
            indexer_case.index_heads,
            indexer_case.index_dim,
        ),
        device=device,
        seed=seed + 11,
        nonnegative=False,
    )
    index_keys = _empty_and_fill(
        torch,
        (
            indexer_case.batch,
            indexer_case.context_tokens,
            indexer_case.index_dim,
        ),
        device=device,
        seed=seed + 12,
        nonnegative=False,
    )
    index_weights = _empty_and_fill(
        torch,
        (
            indexer_case.batch,
            indexer_case.query_tokens,
            indexer_case.index_heads,
        ),
        device=device,
        seed=seed + 13,
        nonnegative=True,
    )
    attention_query = _empty_and_fill(
        torch,
        (
            sparse_case.batch,
            sparse_case.query_heads,
            sparse_case.query_tokens,
            sparse_case.qk_head_dim,
        ),
        device=device,
        seed=seed + 21,
        nonnegative=False,
    )
    shared_kv = _empty_and_fill(
        torch,
        (
            sparse_case.batch,
            sparse_case.kv_heads,
            sparse_case.context_tokens,
            sparse_case.qk_head_dim,
        ),
        device=device,
        seed=seed + 22,
        nonnegative=False,
    )
    component_indices = _build_component_indices(
        torch, sparse_case, seed=seed + 23, device=device
    )
    valid_lengths = torch.tensor(
        valid_context_lengths(indexer_case),
        dtype=torch.int32,
        device=device,
    ).reshape(indexer_case.batch, indexer_case.query_tokens)
    _synchronize(torch, backend)

    started = time.monotonic()
    indexer_indices = _run_indexer(
        torch,
        backend=backend,
        case=indexer_case,
        query=index_query,
        keys=index_keys,
        weights=index_weights,
        valid_lengths=valid_lengths,
    ).to(torch.int32)
    _synchronize(torch, backend)
    indexer_seconds = time.monotonic() - started

    started = time.monotonic()
    attention_output, attention_lse = _run_attention(
        torch,
        backend=backend,
        case=sparse_case,
        query=attention_query,
        shared_kv=shared_kv,
        indices=component_indices,
    )
    _synchronize(torch, backend)
    attention_seconds = time.monotonic() - started

    started = time.monotonic()
    workflow_output, workflow_lse = _run_attention(
        torch,
        backend=backend,
        case=sparse_case,
        query=attention_query,
        shared_kv=shared_kv,
        indices=indexer_indices,
    )
    _synchronize(torch, backend)
    workflow_attention_seconds = time.monotonic() - started

    rows = validation_query_rows(sparse_case.query_tokens, phase=phase)
    output_directory.mkdir(parents=True, exist_ok=True)
    _write_tensor(
        output_directory / "indexer_indices.i32",
        indexer_indices.to(torch.int32),
    )
    _write_sampled_result(
        torch,
        output_directory,
        prefix="attention",
        output=attention_output,
        lse=attention_lse,
        query_rows=rows,
    )
    _write_sampled_result(
        torch,
        output_directory,
        prefix="workflow",
        output=workflow_output,
        lse=workflow_lse,
        query_rows=rows,
    )
    manifest = {
        "schema_version": _SCHEMA_VERSION,
        "backend": backend,
        "case_id": sparse_case.case_id,
        "phase": phase,
        "seed": seed,
        "canonical_input": canonical_input_descriptor(indexer_case, sparse_case),
        "validation_query_rows": list(rows),
        "outputs": {
            "indexer": {"dtype": "int32", "shape": list(indexer_indices.shape)},
            "attention": {
                "output": "available",
                "lse": "available" if attention_lse is not None else "unavailable",
            },
            "workflow": {
                "output": "available",
                "lse": "available" if workflow_lse is not None else "unavailable",
            },
        },
        "backend_details": (
            {"attention_lse_layout": "TND-equivalent"}
            if backend == "vllm_ascend"
            else {}
        ),
        "timing_seconds": {
            "indexer": indexer_seconds,
            "attention": attention_seconds,
            "workflow_attention": workflow_attention_seconds,
            "workflow_total": indexer_seconds + workflow_attention_seconds,
        },
    }
    write_manifest(output_directory, manifest)
    return manifest


def _validate_component_cases(indexer_case, sparse_case) -> None:
    for name in ("batch", "query_tokens", "context_tokens"):
        if getattr(indexer_case, name) != getattr(sparse_case, name):
            raise ValueError(f"V3.2 component {name} mismatch")
    if indexer_case.top_k != sparse_case.selected_tokens:
        raise ValueError("V3.2 component Top-K mismatch")
    if indexer_case.resolved_query_lens != sparse_case.resolved_query_lens:
        raise ValueError("V3.2 component query_lens mismatch")
    if indexer_case.resolved_context_lens != sparse_case.resolved_context_lens:
        raise ValueError("V3.2 component context_lens mismatch")
    if (
        indexer_case.resolved_query_start_positions
        != sparse_case.resolved_query_start_positions
    ):
        raise ValueError("V3.2 component query positions mismatch")


def _resolve_device(torch, backend: str):
    if backend in {"simt", "vllm_ascend"}:
        return torch.device("npu")
    if backend == "cuda":
        return torch.device("cuda")
    raise ValueError(f"unsupported conformance backend: {backend}")


def _empty_and_fill(
    torch,
    shape,
    *,
    device,
    seed: int,
    nonnegative: bool,
):
    tensor = torch.empty(shape, dtype=torch.bfloat16, device=device)
    flat = tensor.reshape(-1)
    pattern = torch.tensor(
        deterministic_pattern(seed=seed, nonnegative=nonnegative),
        dtype=torch.bfloat16,
        device=device,
    )
    chunk_elements = 4 * 1024 * 1024
    for start in range(0, flat.numel(), chunk_elements):
        count = min(chunk_elements, flat.numel() - start)
        offset = start % _PATTERN_PERIOD
        repeats = (offset + count + _PATTERN_PERIOD - 1) // _PATTERN_PERIOD
        flat[start : start + count].copy_(
            pattern.repeat(repeats)[offset : offset + count]
        )
    return tensor


def _build_component_indices(torch, case, *, seed: int, device):
    selected = torch.arange(case.selected_tokens, dtype=torch.int32, device=device)
    rows = []
    for batch_index in range(case.batch):
        context_len = case.resolved_context_lens[batch_index]
        query_start = case.resolved_query_start_positions[batch_index]
        for query_index in range(case.query_tokens):
            valid_context = (
                min(context_len, query_start + query_index + 1)
                if case.causal
                else context_len
            )
            row_number = batch_index * case.query_tokens + query_index
            rows.append((selected * 17 + row_number * 31 + seed) % valid_context)
    return torch.stack(rows).reshape(
        case.batch, case.query_tokens, case.selected_tokens
    )


def _run_indexer(torch, *, backend, case, query, keys, weights, valid_lengths):
    if backend == "simt":
        from aten_dsa_lightning_indexer import ops

        return ops.lightning_indexer_forward(
            query,
            keys,
            weights,
            valid_context_lengths=valid_lengths,
            top_k=case.top_k,
            phase=case.phase,
            family="family_64x128",
        )
    if backend == "vllm_ascend":
        import torch_npu

        total_query = sum(case.resolved_query_lens)
        total_context = sum(case.resolved_context_lens)
        result = torch_npu.npu_lightning_indexer(
            query=query.reshape(total_query, case.index_heads, case.index_dim),
            key=keys.reshape(total_context, 1, case.index_dim),
            weights=weights.reshape(total_query, case.index_heads),
            actual_seq_lengths_query=torch.tensor(
                case.cu_seqlens_q[1:], device=query.device, dtype=torch.int32
            ),
            actual_seq_lengths_key=torch.tensor(
                case.cu_seqlens_kv[1:], device=query.device, dtype=torch.int32
            ),
            block_table=None,
            layout_query="TND",
            layout_key="TND",
            sparse_count=case.top_k,
            sparse_mode=3,
        )
        result = result[0] if isinstance(result, tuple) else result
        return result.reshape(case.batch, case.query_tokens, case.top_k)
    if backend == "cuda":
        from cannbench_cuda_dsa_flashmla_deepgemm import lightning_indexer

        return lightning_indexer(
            torch=torch,
            case=case,
            payload=_indexer_payload(case),
            query=query,
            keys=keys,
            weights=weights,
            top_k=case.top_k,
            score_scale=case.score_scale,
            tie_policy=case.tie_policy,
        )
    raise ValueError(f"unsupported conformance backend: {backend}")


def _run_attention(torch, *, backend, case, query, shared_kv, indices):
    if backend == "simt":
        from aten_dsa_sparse_attention import ops

        return ops.sparse_attention_forward(
            query,
            shared_kv,
            indices,
            value_head_dim=case.value_head_dim,
            phase=case.phase,
            family="family_hd576",
            causal=case.causal,
        )
    if backend == "vllm_ascend":
        return _run_vllm_attention(torch, case, query, shared_kv, indices)
    if backend == "cuda":
        from cannbench_cuda_dsa_flashmla_deepgemm import sparse_attention

        return sparse_attention(
            torch=torch,
            case=case,
            payload=_sparse_payload(case),
            query=query,
            shared_kv=shared_kv,
            indices=indices,
            causal=case.causal,
            phase=case.phase,
            softmax_scale=case.softmax_scale,
            topk_lengths=None,
        )
    raise ValueError(f"unsupported conformance backend: {backend}")


def _run_vllm_attention(torch, case, query, shared_kv, indices):
    import torch_npu

    batch = case.batch
    query_tokens = case.query_tokens
    total_query = sum(case.resolved_query_lens)
    block_size = case.resolved_page_block_size
    blocks_per_batch = case.context_tokens // block_size
    query_tnd = query.permute(0, 2, 1, 3).reshape(
        total_query, case.query_heads, case.qk_head_dim
    )
    query_nope = query_tnd[..., : case.value_head_dim].contiguous()
    query_rope = query_tnd[..., case.value_head_dim :].contiguous()
    kv = shared_kv.permute(0, 2, 1, 3).contiguous()
    key = kv[..., : case.value_head_dim].reshape(
        batch * blocks_per_batch,
        block_size,
        case.kv_heads,
        case.value_head_dim,
    )
    key_rope = kv[..., case.value_head_dim :].reshape(
        batch * blocks_per_batch,
        block_size,
        case.kv_heads,
        case.qk_head_dim - case.value_head_dim,
    )
    sparse_indices = indices.reshape(
        total_query, case.kv_heads, case.selected_tokens
    ).to(torch.int32)
    block_table = torch.tensor(
        tuple(value for row in case.block_tables for value in row),
        device=query.device,
        dtype=torch.int32,
    ).reshape(batch, blocks_per_batch)
    actual_query = torch.tensor(
        case.cu_seqlens_q[1:], device=query.device, dtype=torch.int32
    )
    actual_kv = torch.tensor(
        case.resolved_context_lens, device=query.device, dtype=torch.int32
    )
    paged_kwargs = {
        "query": query_nope,
        "key": key,
        "value": key,
        "sparse_indices": sparse_indices,
        "scale_value": float(case.softmax_scale),
        "block_table": block_table,
        "actual_seq_lengths_query": actual_query,
        "actual_seq_lengths_kv": actual_kv,
        "query_rope": query_rope,
        "key_rope": key_rope,
        "sparse_block_size": 1,
        "layout_query": "TND",
        "layout_kv": "PA_BSND",
        "sparse_mode": 3,
        "attention_mode": 2,
    }
    paged_result = torch_npu.npu_sparse_flash_attention(
        **paged_kwargs, return_softmax_lse=False
    )
    output = paged_result[0] if isinstance(paged_result, tuple) else paged_result

    total_context = sum(case.resolved_context_lens)
    tnd_kwargs = {
        **paged_kwargs,
        "key": kv[..., : case.value_head_dim]
        .reshape(total_context, case.kv_heads, case.value_head_dim)
        .contiguous(),
        "value": kv[..., : case.value_head_dim]
        .reshape(total_context, case.kv_heads, case.value_head_dim)
        .contiguous(),
        "key_rope": kv[..., case.value_head_dim :]
        .reshape(
            total_context,
            case.kv_heads,
            case.qk_head_dim - case.value_head_dim,
        )
        .contiguous(),
        "block_table": None,
        "actual_seq_lengths_kv": torch.tensor(
            case.cu_seqlens_kv[1:], device=query.device, dtype=torch.int32
        ),
        "layout_kv": "TND",
    }
    tnd_output, softmax_max, softmax_sum = torch_npu.npu_sparse_flash_attention(
        **tnd_kwargs, return_softmax_lse=True
    )[:3]
    if not torch.allclose(
        output.float(), tnd_output.float(), atol=0.05, rtol=0.05
    ):
        raise RuntimeError("vLLM paged and TND sparse attention outputs diverged")
    lse = softmax_max + torch.log(softmax_sum)
    output = output.reshape(
        batch, query_tokens, case.query_heads, case.value_head_dim
    )
    if lse is not None:
        lse = lse.reshape(batch, query_tokens, case.query_heads)
    return output, lse


def _indexer_payload(case) -> dict:
    return {
        "phase": case.phase,
        "query_shape": (
            case.batch,
            case.query_tokens,
            case.index_heads,
            case.index_dim,
        ),
        "key_shape": (case.batch, case.context_tokens, case.index_dim),
        "weight_shape": (case.batch, case.query_tokens, case.index_heads),
        "top_k": case.top_k,
        "score_scale": case.score_scale,
        "query_lens": case.resolved_query_lens,
        "context_lens": case.resolved_context_lens,
        "valid_context_lengths": valid_context_lengths(case),
        "page_block_size": case.resolved_page_block_size,
        "block_tables": case.block_tables,
    }


def _sparse_payload(case) -> dict:
    return {
        "phase": case.phase,
        "query_shape": (
            case.batch,
            case.query_heads,
            case.query_tokens,
            case.qk_head_dim,
        ),
        "shared_kv_shape": (
            case.batch,
            case.kv_heads,
            case.context_tokens,
            case.qk_head_dim,
        ),
        "indices_shape": (
            case.batch,
            case.query_tokens,
            case.selected_tokens,
        ),
        "value_head_dim": case.value_head_dim,
        "softmax_scale": case.softmax_scale,
        "query_lens": case.resolved_query_lens,
        "context_lens": case.resolved_context_lens,
        "page_block_size": case.resolved_page_block_size,
        "block_tables": case.block_tables,
    }


def _write_sampled_result(
    torch,
    directory: Path,
    *,
    prefix: str,
    output,
    lse,
    query_rows,
) -> None:
    sampled_output = output[:, query_rows].to(torch.float32).contiguous()
    _write_tensor(directory / f"{prefix}_output.f32", sampled_output)
    if lse is not None:
        sampled_lse = lse[:, query_rows].to(torch.float32).contiguous()
        _write_tensor(directory / f"{prefix}_lse.f32", sampled_lse)


def _write_tensor(path: Path, tensor) -> None:
    cpu_tensor = tensor.detach().contiguous().cpu()
    path.write_bytes(cpu_tensor.numpy().tobytes(order="C"))


def _synchronize(torch, backend: str) -> None:
    if backend in {"simt", "vllm_ascend"}:
        torch.npu.synchronize()
    else:
        torch.cuda.synchronize()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("simt", "vllm_ascend", "cuda"), required=True)
    parser.add_argument("--phase", choices=("prefill", "decode"), required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    manifest = run_backend(
        backend=args.backend,
        phase=args.phase,
        seed=args.seed,
        output_directory=args.output,
    )
    print(json.dumps(manifest, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
