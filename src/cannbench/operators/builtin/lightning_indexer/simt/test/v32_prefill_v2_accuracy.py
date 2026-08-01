from __future__ import annotations

import argparse
import json


def _build_case(torch, *, seed: int, case_name: str):
    torch.manual_seed(seed)
    device = torch.device("npu")
    if case_name == "tied_threshold":
        query = torch.zeros(
            1, 4096, 64, 128, device=device, dtype=torch.bfloat16
        )
        keys = torch.zeros(1, 32768, 128, device=device, dtype=torch.bfloat16)
        weights = torch.ones(1, 4096, 64, device=device, dtype=torch.bfloat16)
    else:
        query = torch.randn(
            1, 4096, 64, 128, device=device, dtype=torch.bfloat16
        )
        keys = torch.randn(1, 32768, 128, device=device, dtype=torch.bfloat16)
        if case_name == "near_threshold":
            weights = torch.randn(
                1, 4096, 64, device=device, dtype=torch.bfloat16
            ) * 0.015625
        elif case_name == "negative_scores":
            weights = -torch.rand(
                1, 4096, 64, device=device, dtype=torch.bfloat16
            )
        else:
            weights = torch.rand(
                1, 4096, 64, device=device, dtype=torch.bfloat16
            )

    if case_name == "masked_tail":
        valid = torch.arange(24576, 28672, device=device, dtype=torch.int32).reshape(
            1, 4096
        )
    else:
        valid = torch.full((1, 4096), 32768, device=device, dtype=torch.int32)
    return query, keys, weights, valid


def _sampled_reference(torch, query, keys, weights, valid):
    rows = (0, 1365, 2730, 4095)
    scores = torch.einsum("bqhd,bcd->bqhc", query[:, rows], keys)
    reduced = (torch.relu(scores) * weights[:, rows].unsqueeze(-1)).sum(dim=2)
    positions = torch.arange(keys.shape[1], device=keys.device).reshape(1, 1, -1)
    reduced = reduced.masked_fill(
        positions >= valid[:, rows].unsqueeze(-1), float("-inf")
    )
    reference_scores = torch.topk(
        reduced, 2048, dim=-1, largest=True, sorted=True
    ).values
    return rows, reduced, reference_scores


def _validate_output(torch, output, rows, reduced, reference_scores, valid):
    if output.shape != (1, 4096, 2048) or output.dtype != torch.int32:
        raise RuntimeError(f"unexpected output metadata: {output.shape}, {output.dtype}")
    if not bool(((output >= 0) & (output < 32768)).all().item()):
        raise RuntimeError("output contains an out-of-range index")
    if not bool((output < valid.unsqueeze(-1)).all().item()):
        raise RuntimeError("output contains an index beyond valid context")

    sorted_indices = torch.sort(output, dim=-1).values
    if not bool(
        (sorted_indices[..., :-1] != sorted_indices[..., 1:]).all().item()
    ):
        raise RuntimeError("output contains duplicate indices")

    sampled_output = output[:, rows].to(torch.int64)
    selected_scores = reduced.gather(-1, sampled_output)
    selected_scores = torch.sort(selected_scores, dim=-1, descending=True).values
    reference_scores = torch.sort(
        reference_scores, dim=-1, descending=True
    ).values
    if not torch.equal(selected_scores, reference_scores):
        raise RuntimeError("unordered Top-K score multiset differs from reference")
    return sorted_indices


def run_accuracy(*, seed: int, repeats: int):
    import torch
    import torch_npu  # noqa: F401
    from aten_dsa_lightning_indexer_v2 import ops

    results = {}
    case_names = (
        "canonical",
        "masked_tail",
        "tied_threshold",
        "near_threshold",
        "negative_scores",
    )
    for case_offset, case_name in enumerate(case_names):
        query, keys, weights, valid = _build_case(
            torch, seed=seed + case_offset, case_name=case_name
        )
        rows, reduced, reference_scores = _sampled_reference(
            torch, query, keys, weights, valid
        )
        first_index_set = None
        for _ in range(repeats):
            output = ops.lightning_indexer_forward(
                query,
                keys,
                weights,
                valid_context_lengths=valid,
                top_k=2048,
                phase="prefill",
                family="family_64x128",
            )
            index_set = _validate_output(
                torch, output, rows, reduced, reference_scores, valid
            )
            if first_index_set is None:
                first_index_set = index_set
            elif not torch.equal(first_index_set, index_set):
                raise RuntimeError(f"{case_name} selected index set is unstable")
        torch.npu.synchronize()
        results[case_name] = {
            "repeats": repeats,
            "sampled_score_multiset_match": True,
            "valid_unique_indices": True,
            "stable_index_set": True,
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeats <= 0:
        parser.error("repeats must be positive")
    print(
        json.dumps(
            run_accuracy(seed=args.seed, repeats=args.repeats),
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
