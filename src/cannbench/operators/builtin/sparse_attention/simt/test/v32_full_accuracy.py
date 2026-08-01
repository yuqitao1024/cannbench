from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from cannbench.operators.builtin.sparse_attention.cases import (
    get_sparse_attention_case,
)


_CASES = {
    "decode": (
        "realistic_decode",
        "deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048",
    ),
    "prefill": (
        "realistic_prefill",
        "deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048",
    ),
}


def validation_query_rows(query_tokens: int, *, phase: str) -> tuple[int, ...]:
    if phase == "decode":
        return tuple(range(query_tokens))
    if phase != "prefill":
        raise ValueError(f"unsupported phase: {phase}")
    if query_tokens <= 4:
        return tuple(range(query_tokens))
    step = (query_tokens - 1) // 3
    return (0, step, step * 2, query_tokens - 1)


def deterministic_pattern(*, seed: int, nonnegative: bool) -> tuple[float, ...]:
    values = tuple(float((index * 17 + seed * 29) % 257) for index in range(257))
    if nonnegative:
        return tuple(value / 256.0 for value in values)
    return tuple((value - 128.0) / 128.0 for value in values)


def resize_prefill_case(case, query_tokens: int):
    if case.phase != "prefill":
        raise ValueError("query-token override is only valid for prefill")
    if query_tokens <= 0 or query_tokens > case.context_tokens:
        raise ValueError("query_tokens must be between 1 and context_tokens")
    query_lens = (query_tokens,) * case.batch
    query_start_positions = tuple(
        context_len - query_tokens for context_len in case.resolved_context_lens
    )
    return replace(
        case,
        query_tokens=query_tokens,
        query_lens=query_lens,
        query_start_positions=query_start_positions,
        topk_lengths=None,
    )


def _fill_deterministic(torch, tensor, *, seed: int, nonnegative: bool) -> None:
    flat = tensor.reshape(-1)
    chunk_elements = 4 * 1024 * 1024
    pattern = torch.tensor(
        deterministic_pattern(seed=seed, nonnegative=nonnegative),
        dtype=tensor.dtype,
        device=tensor.device,
    )
    for start in range(0, flat.numel(), chunk_elements):
        end = min(start + chunk_elements, flat.numel())
        count = end - start
        pattern_offset = start % len(pattern)
        repeats = (pattern_offset + count + len(pattern) - 1) // len(pattern)
        values = pattern.repeat(repeats)[pattern_offset : pattern_offset + count]
        flat[start:end].copy_(values)


def causal_boundary_values(*, context: int, causal_limit: int) -> tuple[int, ...]:
    boundary = max(0, causal_limit - 1)
    future_or_end = causal_limit if causal_limit < context else context - 1
    return (0, boundary, future_or_end, -1, context)


def index_category_counts(
    row: list[int], *, context: int, causal_limit: int
) -> dict[str, int]:
    return {
        "negative": sum(index < 0 for index in row),
        "out_of_range": sum(index >= context for index in row),
        "valid_past": sum(0 <= index < causal_limit for index in row),
        "valid_future": sum(causal_limit <= index < context for index in row),
    }


def _build_indices(
    torch,
    case,
    *,
    seed: int,
    device,
    inject_causal_boundaries: bool = False,
):
    selected = torch.arange(case.selected_tokens, dtype=torch.int64, device="cpu")
    rows = []
    for batch_index in range(case.batch):
        for query_index in range(case.query_tokens):
            if case.causal:
                valid_context = min(
                    case.resolved_context_lens[batch_index],
                    case.resolved_query_start_positions[batch_index]
                    + query_index
                    + 1,
                )
            else:
                valid_context = case.resolved_context_lens[batch_index]
            row_number = batch_index * case.query_tokens + query_index
            rows.append((selected * 17 + row_number * 31 + seed) % valid_context)
    indices = torch.stack(rows).reshape(
        case.batch, case.query_tokens, case.selected_tokens
    )
    if inject_causal_boundaries and case.causal:
        for batch_index in range(case.batch):
            for query_index in validation_query_rows(
                case.query_tokens, phase=case.phase
            ):
                context = case.resolved_context_lens[batch_index]
                causal_limit = min(
                    context,
                    case.resolved_query_start_positions[batch_index]
                    + query_index
                    + 1,
                )
                values = causal_boundary_values(
                    context=context, causal_limit=causal_limit
                )[: case.selected_tokens]
                indices[batch_index, query_index, : len(values)] = torch.tensor(
                    values, dtype=torch.int64, device="cpu"
                )
    return indices.to(device=device)


def _compare_chunk(torch, actual, expected, *, atol: float, rtol: float):
    actual = actual.float()
    expected = expected.float()
    abs_error = (actual - expected).abs()
    tolerance = atol + rtol * expected.abs()
    relative = abs_error / expected.abs().clamp_min(1e-12)
    return {
        "mismatch_count": int((abs_error > tolerance).sum().item()),
        "max_abs_error": float(abs_error.max().item()),
        "max_rel_error": float(relative.max().item()),
        "numel": abs_error.numel(),
    }


def _merge_metrics(target: dict[str, float | int], current) -> None:
    target["mismatch_count"] += current["mismatch_count"]
    target["max_abs_error"] = max(
        target["max_abs_error"], current["max_abs_error"]
    )
    target["max_rel_error"] = max(
        target["max_rel_error"], current["max_rel_error"]
    )
    target["numel"] += current["numel"]


def _chunked_reference_metrics(
    torch,
    case,
    query,
    shared_kv,
    indices,
    actual_output,
    actual_lse,
    *,
    atol: float,
    rtol: float,
):
    output_metrics: dict[str, float | int] = {
        "mismatch_count": 0,
        "max_abs_error": 0.0,
        "max_rel_error": 0.0,
        "numel": 0,
    }
    lse_metrics = dict(output_metrics)
    query_rows = validation_query_rows(case.query_tokens, phase=case.phase)
    head_chunk = 8
    scale = float(case.softmax_scale)
    for batch_index in range(case.batch):
        kv = shared_kv[batch_index, 0]
        for query_index in query_rows:
            row_indices = indices[batch_index, query_index]
            safe_indices = row_indices.clamp(min=0, max=max(kv.shape[0] - 1, 0))
            selected_keys = kv.index_select(0, safe_indices)
            selected_values = selected_keys[:, : case.value_head_dim]
            context = case.resolved_context_lens[batch_index]
            valid_indices = (row_indices >= 0) & (row_indices < context)
            if case.causal:
                causal_limit = min(
                    context,
                    case.resolved_query_start_positions[batch_index]
                    + query_index
                    + 1,
                )
                valid_indices &= row_indices < causal_limit
            for head_start in range(0, case.query_heads, head_chunk):
                head_end = min(head_start + head_chunk, case.query_heads)
                query_chunk = query[
                    batch_index, head_start:head_end, query_index
                ]
                scores = (
                    query_chunk[:, None, :] * selected_keys[None, :, :]
                ).sum(dim=-1)
                scores = scores.float() * scale
                scores = scores.masked_fill(
                    ~valid_indices[None, :], float("-inf")
                )
                if bool(valid_indices.any().item()):
                    probabilities = torch.softmax(scores, dim=-1)
                    expected_lse = torch.logsumexp(scores, dim=-1)
                else:
                    probabilities = torch.zeros_like(scores)
                    expected_lse = torch.full(
                        (head_end - head_start,),
                        float("-inf"),
                        dtype=scores.dtype,
                        device=scores.device,
                    )
                expected_output = (
                    probabilities.to(query.dtype).unsqueeze(-1)
                    * selected_values[None, :, :]
                ).sum(dim=1)
                _merge_metrics(
                    output_metrics,
                    _compare_chunk(
                        torch,
                        actual_output[
                            batch_index,
                            query_index,
                            head_start:head_end,
                        ],
                        expected_output,
                        atol=atol,
                        rtol=rtol,
                    ),
                )
                _merge_metrics(
                    lse_metrics,
                    _compare_chunk(
                        torch,
                        actual_lse[
                            batch_index,
                            query_index,
                            head_start:head_end,
                        ],
                        expected_lse,
                        atol=atol,
                        rtol=rtol,
                    ),
                )
    return query_rows, output_metrics, lse_metrics


def run_case(
    *,
    phase: str,
    implementation_version: str,
    seed: int,
    atol: float,
    rtol: float,
    query_tokens: int | None = None,
    runtime_only: bool = False,
):
    import torch
    import torch_npu  # noqa: F401

    if implementation_version == "v2":
        from aten_dsa_sparse_attention_v2 import ops
    else:
        from aten_dsa_sparse_attention import ops

    dataset, case_id = _CASES[phase]
    case = get_sparse_attention_case(dataset, case_id)
    if query_tokens is not None:
        case = resize_prefill_case(case, query_tokens)
    device = torch.device("npu")
    query = torch.empty(
        (
            case.batch,
            case.query_heads,
            case.query_tokens,
            case.qk_head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    shared_kv = torch.empty(
        (
            case.batch,
            case.kv_heads,
            case.context_tokens,
            case.qk_head_dim,
        ),
        dtype=torch.bfloat16,
        device=device,
    )
    _fill_deterministic(torch, query, seed=seed + 1, nonnegative=False)
    _fill_deterministic(torch, shared_kv, seed=seed + 2, nonnegative=False)
    indices = _build_indices(
        torch,
        case,
        seed=seed + 3,
        device=device,
        inject_causal_boundaries=not runtime_only,
    )
    torch.npu.synchronize()

    started = time.monotonic()
    output, lse = ops.sparse_attention_forward(
        query,
        shared_kv,
        indices,
        value_head_dim=case.value_head_dim,
        phase=case.phase,
        family="family_hd576",
        causal=case.causal,
    )
    torch.npu.synchronize()
    kernel_wall_seconds = time.monotonic() - started

    common_result = {
        "case_id": case.case_id,
        "phase": case.phase,
        "query_shape": list(query.shape),
        "shared_kv_shape": list(shared_kv.shape),
        "indices_shape": list(indices.shape),
        "kernel_wall_seconds": kernel_wall_seconds,
    }
    if runtime_only:
        return {
            **common_result,
            "validation": "skipped",
            "passed": True,
        }

    query_rows, output_metrics, lse_metrics = _chunked_reference_metrics(
        torch,
        case,
        query,
        shared_kv,
        indices,
        output,
        lse,
        atol=atol,
        rtol=rtol,
    )
    passed = (
        output_metrics["mismatch_count"] == 0
        and lse_metrics["mismatch_count"] == 0
    )
    causal_index_categories = {}
    if case.causal:
        for batch_index in range(case.batch):
            context = case.resolved_context_lens[batch_index]
            for query_index in query_rows:
                causal_limit = min(
                    context,
                    case.resolved_query_start_positions[batch_index]
                    + query_index
                    + 1,
                )
                causal_index_categories[f"{batch_index}:{query_index}"] = (
                    index_category_counts(
                        indices[batch_index, query_index].detach().cpu().tolist(),
                        context=context,
                        causal_limit=causal_limit,
                    )
                )
    return {
        **common_result,
        "validated_query_rows": list(query_rows),
        "validated_all_heads": True,
        "causal_index_categories": causal_index_categories,
        "atol": atol,
        "rtol": rtol,
        "output": output_metrics,
        "lse": lse_metrics,
        "passed": passed,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("decode", "prefill", "both"), default="both"
    )
    parser.add_argument(
        "--implementation-version", choices=("v1", "v2"), default="v1"
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--query-tokens", type=int)
    parser.add_argument("--runtime-only", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.query_tokens is not None and args.phase != "prefill":
        parser.error("--query-tokens requires --phase prefill")

    phases = ("decode", "prefill") if args.phase == "both" else (args.phase,)
    results = [
        run_case(
            phase=phase,
            implementation_version=args.implementation_version,
            seed=args.seed,
            atol=args.atol,
            rtol=args.rtol,
            query_tokens=args.query_tokens,
            runtime_only=args.runtime_only,
        )
        for phase in phases
    ]
    payload = {"results": results, "passed": all(row["passed"] for row in results)}
    rendered = json.dumps(payload, indent=2)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
