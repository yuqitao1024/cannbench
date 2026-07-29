from __future__ import annotations

import argparse
import json
import os


PARTITIONS = (1, 2, 4)

CASES = (
    {
        "name": "all_invalid_s17",
        "context": 256,
        "selected": 17,
        "mode": "invalid",
    },
    {
        "name": "int64_overflow_s17",
        "context": 256,
        "selected": 17,
        "mode": "int64_overflow",
    },
    {"name": "valid_s64", "context": 256, "selected": 64, "mode": "valid"},
    {"name": "tail_s70", "context": 256, "selected": 70, "mode": "valid"},
    {
        "name": "invalid_causal_s70",
        "context": 256,
        "selected": 70,
        "mode": "mixed",
    },
    {
        "name": "causal_q4_c256_s64",
        "query_tokens": 4,
        "context": 256,
        "selected": 64,
        "mode": "mixed",
        "causal": True,
    },
    {"name": "valid_s128", "context": 256, "selected": 128, "mode": "valid"},
    {
        "name": "multi_task_b2_q9_s70",
        "batch": 2,
        "query_tokens": 9,
        "context": 256,
        "selected": 70,
        "mode": "valid",
    },
    {
        "name": "valid_s2048",
        "context": 32768,
        "selected": 2048,
        "mode": "valid",
    },
)


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


def index_rows(case) -> tuple[tuple[int, ...], ...]:
    query_tokens = case.get("query_tokens", 1)
    rows = []
    for query_index in range(query_tokens):
        row = [
            (selected_index * 13 + 7) % case["context"]
            for selected_index in range(case["selected"])
        ]
        if case["mode"] == "mixed":
            row[::5] = [-1] * len(row[::5])
            row[3::7] = [case["context"]] * len(row[3::7])
            causal_limit = case["context"] - query_tokens + query_index + 1
            boundary_values = causal_boundary_values(
                context=case["context"], causal_limit=causal_limit
            )
            row[: len(boundary_values)] = boundary_values
        elif case["mode"] == "invalid":
            row = [-1] * len(row)
        elif case["mode"] == "int64_overflow":
            row[::2] = [1 << 40] * len(row[::2])
            row[1::2] = [-(1 << 40)] * len(row[1::2])
        rows.append(tuple(row))
    return tuple(rows)


def _pattern(torch, shape, *, offset: int, device):
    count = 1
    for extent in shape:
        count *= extent
    values = torch.arange(count, dtype=torch.int32, device=device)
    values = ((values * 17 + offset) % 251 - 125).float() / 128.0
    return values.reshape(shape).to(torch.bfloat16)


def _indices(torch, case, *, device):
    batch = case.get("batch", 1)
    query_tokens = case.get("query_tokens", 1)
    indices = torch.tensor(index_rows(case), dtype=torch.int64, device=device)
    indices = indices.reshape(1, query_tokens, case["selected"])
    indices = indices.expand(batch, query_tokens, -1).clone()
    return indices


def _max_finite_error(torch, actual, expected) -> float:
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not bool(finite.any().item()):
        return 0.0
    return float((actual[finite].float() - expected[finite].float()).abs().max().item())


def _run_case(torch, ops, case, *, phase: str):
    device = torch.device("npu")
    batch = case.get("batch", 1)
    query_tokens = case.get("query_tokens", 1)
    query = _pattern(
        torch,
        (batch, 128, query_tokens, 576),
        offset=11,
        device=device,
    )
    shared_kv = _pattern(
        torch,
        (batch, 1, case["context"], 576),
        offset=29,
        device=device,
    )
    indices = _indices(torch, case, device=device)
    causal = case.get("causal", case["mode"] == "mixed")

    reference = (
        ops._prefill_reference if phase == "prefill" else ops._decode_reference
    )
    expected_output, expected_lse = reference(
        query,
        shared_kv,
        indices,
        value_head_dim=512,
        causal=causal,
    )
    actual_output, actual_lse = ops.sparse_attention_forward(
        query,
        shared_kv,
        indices,
        value_head_dim=512,
        phase=phase,
        family="family_hd576",
        causal=causal,
    )
    torch.npu.synchronize()

    output_passed = torch.allclose(
        actual_output.float(),
        expected_output.float(),
        atol=0.05,
        rtol=0.05,
        equal_nan=True,
    )
    lse_passed = torch.allclose(
        actual_lse.float(),
        expected_lse.float(),
        atol=0.05,
        rtol=0.05,
        equal_nan=True,
    )
    boundary_passed = True
    if case["mode"] in {"invalid", "int64_overflow"}:
        boundary_passed = (
            int(torch.count_nonzero(actual_output).item()) == 0
            and bool(torch.isneginf(actual_lse).all().item())
            and not bool(torch.isnan(actual_output).any().item())
            and not bool(torch.isnan(actual_lse).any().item())
        )

    result = {
        "name": case["name"],
        "output_max_abs_error": _max_finite_error(
            torch, actual_output, expected_output
        ),
        "lse_max_abs_error": _max_finite_error(torch, actual_lse, expected_lse),
        "output_passed": bool(output_passed),
        "lse_passed": bool(lse_passed),
        "boundary_passed": boundary_passed,
        "passed": bool(output_passed and lse_passed and boundary_passed),
    }
    if causal:
        result["causal_index_categories"] = {
            str(query_index): index_category_counts(
                indices[0, query_index].detach().cpu().tolist(),
                context=case["context"],
                causal_limit=case["context"] - query_tokens + query_index + 1,
            )
            for query_index in range(query_tokens)
        }
    return result


def _run_empty_contract(
    torch, ops, *, name: str, context: int, selected: int, phase: str
):
    device = torch.device("npu")
    query = _pattern(torch, (1, 128, 1, 576), offset=11, device=device)
    shared_kv = _pattern(
        torch,
        (1, 1, context, 576),
        offset=29,
        device=device,
    )
    indices = torch.full(
        (1, 1, selected),
        -1,
        dtype=torch.int64,
        device=device,
    )
    output, lse = ops.sparse_attention_forward(
        query,
        shared_kv,
        indices,
        value_head_dim=512,
        phase=phase,
        family="family_hd576",
        causal=False,
    )
    torch.npu.synchronize()
    passed = (
        int(torch.count_nonzero(output).item()) == 0
        and bool(torch.isneginf(lse).all().item())
        and not bool(torch.isnan(output).any().item())
        and not bool(torch.isnan(lse).any().item())
    )
    return {"name": name, "passed": passed}


def _run_width_rejection(torch, ops, width: int, *, phase: str = "decode"):
    device = torch.device("npu")
    query = _pattern(torch, (1, 128, 1, 576), offset=11, device=device)
    shared_kv = _pattern(
        torch,
        (1, 1, 1, width),
        offset=29,
        device=device,
    )
    indices = torch.zeros((1, 1, 1), dtype=torch.int64, device=device)
    error = ""
    try:
        ops.sparse_attention_forward(
            query,
            shared_kv,
            indices,
            value_head_dim=512,
            phase=phase,
            family="family_hd576",
            causal=False,
        )
        torch.npu.synchronize()
    except RuntimeError as exc:
        error = str(exc)
    return {
        "name": f"shared_kv_width_{width}_rejected",
        "passed": "head64 requires shared_kv_head_dim=576" in error,
    }


def _run_concurrent_reuse(torch, ops, *, phase: str):
    case = next(case for case in CASES if case["name"] == "tail_s70")
    device = torch.device("npu")
    query = _pattern(torch, (1, 128, 1, 576), offset=11, device=device)
    shared_kv = _pattern(
        torch,
        (1, 1, case["context"], 576),
        offset=29,
        device=device,
    )
    indices = _indices(torch, case, device=device)
    expected_output, expected_lse = ops._prefill_reference(
        query,
        shared_kv,
        indices,
        value_head_dim=512,
        causal=False,
    )
    streams = [torch.npu.Stream() for _ in range(2)]
    for worker_stream in streams:
        worker_stream.wait_stream(torch.npu.current_stream())
    results = []

    for launch_index in range(8):
        with torch.npu.stream(streams[launch_index % len(streams)]):
            results.append(
                ops.sparse_attention_forward(
                    query,
                    shared_kv,
                    indices,
                    value_head_dim=512,
                    phase=phase,
                    family="family_hd576",
                    causal=False,
                )
            )
    torch.npu.synchronize()

    matches = [
        (
            bool(
                torch.allclose(
                    actual_output.float(),
                    expected_output.float(),
                    atol=0.05,
                    rtol=0.05,
                    equal_nan=True,
                )
            ),
            bool(
                torch.allclose(
                    actual_lse.float(),
                    expected_lse.float(),
                    atol=0.05,
                    rtol=0.05,
                    equal_nan=True,
                )
            ),
        )
        for actual_output, actual_lse in results
    ]
    passed = all(output_match and lse_match for output_match, lse_match in matches)
    return {
        "name": "concurrent_reuse_valid_s70",
        "launches": len(results),
        "passed": bool(passed),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase", choices=("decode", "prefill"), default="decode"
    )
    args = parser.parse_args(argv)

    import torch
    import torch_npu  # noqa: F401
    from aten_dsa_sparse_attention import ops

    results = []
    os.environ["CANNBENCH_SPARSE_ATTENTION_HEAD_TILE"] = "64"
    phase = args.phase
    partitions = (1,) if phase == "prefill" else PARTITIONS
    for selected_partitions in partitions:
        partition_value = str(selected_partitions)
        os.environ[
            "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS"
        ] = partition_value
        partition_results = []
        for case in CASES:
            result = _run_case(torch, ops, case, phase=phase)
            partition_results.append(result)
        partition_results.extend(
            (
                _run_empty_contract(
                    torch,
                    ops,
                    name="empty_selected_s0",
                    context=256,
                    selected=0,
                    phase=phase,
                ),
                _run_empty_contract(
                    torch,
                    ops,
                    name="empty_context_c0",
                    context=0,
                    selected=17,
                    phase=phase,
                ),
            )
        )
        if phase == "decode":
            partition_results.extend(
                (
                    _run_width_rejection(torch, ops, 512),
                    _run_width_rejection(torch, ops, 640),
                )
            )
        else:
            partition_results.extend(
                (
                    _run_width_rejection(torch, ops, 512, phase=phase),
                    _run_width_rejection(torch, ops, 640, phase=phase),
                )
            )
        if phase == "prefill":
            partition_results.append(_run_concurrent_reuse(torch, ops, phase=phase))
        for result in partition_results:
            result["selected_partitions"] = selected_partitions
        results.extend(partition_results)
    passed = all(result["passed"] for result in results)
    print(json.dumps({"passed": passed, "cases": results}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
