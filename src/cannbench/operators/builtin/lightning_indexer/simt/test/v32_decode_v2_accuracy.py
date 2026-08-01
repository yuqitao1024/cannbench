from __future__ import annotations

import argparse
import json


def _build_random_case(torch, *, seed: int, masked_tail: bool):
    torch.manual_seed(seed)
    device = torch.device("npu")
    query = torch.randn(2, 2, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(2, 32768, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(2, 2, 64, device=device, dtype=torch.bfloat16)
    valid = torch.tensor(
        [[24576, 28672], [30720, 32767]]
        if masked_tail
        else [[32767, 32768], [32766, 32765]],
        device=device,
        dtype=torch.int32,
    )
    return query, keys, weights, valid


def _build_tied_threshold_case(torch):
    device = torch.device("npu")
    query = torch.zeros(2, 2, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.zeros(2, 32768, 128, device=device, dtype=torch.bfloat16)
    weights = torch.ones(2, 2, 64, device=device, dtype=torch.bfloat16)
    valid = torch.full((2, 2), 32768, device=device, dtype=torch.int32)
    return query, keys, weights, valid


def _build_reduction_order_case(torch, *, seed: int, negative_weights: bool):
    torch.manual_seed(seed)
    device = torch.device("npu")
    query = torch.randn(2, 2, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(2, 32768, 128, device=device, dtype=torch.bfloat16)
    if negative_weights:
        weights = -torch.rand(
            2, 2, 64, device=device, dtype=torch.bfloat16
        )
    else:
        weights = torch.randn(
            2, 2, 64, device=device, dtype=torch.bfloat16
        ) * 0.015625
    valid = torch.tensor(
        [[32768, 32767], [32766, 32765]],
        device=device,
        dtype=torch.int32,
    )
    return query, keys, weights, valid


def _reference_scores(torch, query, keys, weights, valid):
    scores = torch.einsum("bqhd,bcd->bqhc", query, keys)
    reduced = (torch.relu(scores) * weights.unsqueeze(-1)).sum(dim=2)
    positions = torch.arange(keys.shape[1], device=keys.device).reshape(1, 1, -1)
    reduced = reduced.masked_fill(positions >= valid.unsqueeze(-1), float("-inf"))
    topk_scores = torch.topk(
        reduced, 2048, dim=-1, largest=True, sorted=True
    ).values
    return reduced, topk_scores


def _validate_output(torch, output, reduced, reference_scores, valid):
    if output.shape != (2, 2, 2048) or output.dtype != torch.int32:
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

    selected_scores = reduced.gather(-1, output.to(torch.int64))
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

    cases = {
        "canonical": _build_random_case(torch, seed=seed, masked_tail=False),
        "masked_tail": _build_random_case(
            torch, seed=seed + 1, masked_tail=True
        ),
        "tied_threshold": _build_tied_threshold_case(torch),
        "near_threshold": _build_reduction_order_case(
            torch, seed=seed + 2, negative_weights=False
        ),
        "negative_scores": _build_reduction_order_case(
            torch, seed=seed + 3, negative_weights=True
        ),
    }
    results = {}
    for case_name, (query, keys, weights, valid) in cases.items():
        reduced, reference_scores = _reference_scores(
            torch, query, keys, weights, valid
        )
        index_sets = []
        for _ in range(repeats):
            output = ops.lightning_indexer_forward(
                query,
                keys,
                weights,
                valid_context_lengths=valid,
                top_k=2048,
                phase="decode",
                family="family_64x128",
            )
            index_sets.append(
                _validate_output(
                    torch, output, reduced, reference_scores, valid
                )
            )
        torch.npu.synchronize()
        stable_set = all(torch.equal(index_sets[0], item) for item in index_sets[1:])
        if not stable_set:
            raise RuntimeError(f"{case_name} selected index set is unstable")
        results[case_name] = {
            "repeats": repeats,
            "score_multiset_match": True,
            "stable_index_set": True,
        }
    return results


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--repeats", type=int, default=5)
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
