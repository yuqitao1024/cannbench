from __future__ import annotations

import json
import os


CASES = (
    {"name": "valid_s64", "context": 256, "selected": 64, "mode": "valid"},
    {"name": "tail_s70", "context": 256, "selected": 70, "mode": "valid"},
    {
        "name": "invalid_causal_s70",
        "context": 256,
        "selected": 70,
        "mode": "mixed",
    },
    {
        "name": "all_invalid_s17",
        "context": 256,
        "selected": 17,
        "mode": "invalid",
    },
)


def _pattern(torch, shape, *, offset: int, device):
    count = 1
    for extent in shape:
        count *= extent
    values = torch.arange(count, dtype=torch.int32, device=device)
    values = ((values * 17 + offset) % 251 - 125).float() / 128.0
    return values.reshape(shape).to(torch.bfloat16)


def _indices(torch, case, *, device):
    selected = torch.arange(case["selected"], dtype=torch.int64, device=device)
    indices = ((selected * 13 + 7) % case["context"]).reshape(1, 1, -1)
    if case["mode"] == "mixed":
        indices[:, :, ::5] = -1
        indices[:, :, 3::7] = case["context"]
    elif case["mode"] == "invalid":
        indices.fill_(-1)
    return indices


def _max_finite_error(torch, actual, expected) -> float:
    finite = torch.isfinite(actual) & torch.isfinite(expected)
    if not bool(finite.any().item()):
        return 0.0
    return float((actual[finite].float() - expected[finite].float()).abs().max().item())


def _run_case(torch, ops, case):
    device = torch.device("npu")
    query = _pattern(torch, (1, 128, 1, 576), offset=11, device=device)
    shared_kv = _pattern(
        torch,
        (1, 1, case["context"], 576),
        offset=29,
        device=device,
    )
    indices = _indices(torch, case, device=device)
    causal = case["mode"] == "mixed"

    expected_output, expected_lse = ops._decode_reference(
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
        phase="decode",
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
    if case["mode"] == "invalid":
        boundary_passed = (
            int(torch.count_nonzero(actual_output).item()) == 0
            and bool(torch.isneginf(actual_lse).all().item())
            and not bool(torch.isnan(actual_output).any().item())
            and not bool(torch.isnan(actual_lse).any().item())
        )

    return {
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


def main() -> int:
    import torch
    import torch_npu  # noqa: F401
    from aten_dsa_sparse_attention import ops

    os.environ["CANNBENCH_SPARSE_ATTENTION_HEAD_TILE"] = "64"
    os.environ["CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS"] = "1"
    results = [_run_case(torch, ops, case) for case in CASES]
    passed = all(result["passed"] for result in results)
    print(json.dumps({"passed": passed, "cases": results}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
