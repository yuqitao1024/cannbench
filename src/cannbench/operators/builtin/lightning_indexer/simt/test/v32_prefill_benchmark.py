from __future__ import annotations

import argparse
import json
import statistics
import time


SAMPLED_ROWS = (0, 1365, 2730, 4095)


def build_target_tensors(torch, *, seed: int):
    torch.manual_seed(seed)
    device = torch.device("npu")
    query = torch.randn(1, 4096, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(1, 32768, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(1, 4096, 64, device=device, dtype=torch.bfloat16)
    valid = torch.arange(28673, 32769, device=device, dtype=torch.int32).reshape(
        1, 4096
    )
    return query, keys, weights, valid


def sampled_reference(torch, query, keys, weights, valid):
    sampled_query = query[:, SAMPLED_ROWS]
    sampled_weights = weights[:, SAMPLED_ROWS]
    reduced = torch.einsum("bqhd,bcd->bqhc", sampled_query, keys)
    reduced = torch.relu(reduced)
    reduced = (reduced * sampled_weights.unsqueeze(-1)).sum(dim=2)
    positions = torch.arange(keys.shape[1], device=keys.device).reshape(1, 1, -1)
    reduced = reduced.masked_fill(
        positions >= valid[:, SAMPLED_ROWS].unsqueeze(-1),
        float("-inf"),
    )
    reference_scores = torch.topk(
        reduced, 2048, dim=-1, largest=True, sorted=True
    ).values
    return reduced, reference_scores


def sampled_score_sets_match(torch, output, reduced, reference_scores) -> bool:
    custom_scores = reduced.gather(
        -1, output[:, SAMPLED_ROWS].to(torch.int64)
    )
    return bool(torch.equal(custom_scores, reference_scores))


def run_benchmark(*, warmups: int, iters: int, seed: int, stability_runs: int):
    import torch
    import torch_npu  # noqa: F401
    from aten_dsa_lightning_indexer import ops

    query, keys, weights, valid = build_target_tensors(torch, seed=seed)

    def run_once():
        return ops.lightning_indexer_forward(
            query,
            keys,
            weights,
            valid_context_lengths=valid,
            top_k=2048,
            phase="prefill",
            family="family_64x128",
        )

    for _ in range(warmups):
        run_once()
        torch.npu.synchronize()

    samples_ms = []
    timed_output = None
    for _ in range(iters):
        started = time.perf_counter_ns()
        timed_output = run_once()
        torch.npu.synchronize()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    reduced, reference_scores = sampled_reference(
        torch, query, keys, weights, valid
    )
    timed_match = sampled_score_sets_match(
        torch, timed_output, reduced, reference_scores
    )
    stability_outputs = [run_once() for _ in range(stability_runs)]
    torch.npu.synchronize()
    stability_match = all(
        sampled_score_sets_match(torch, output, reduced, reference_scores)
        for output in stability_outputs
    )
    if not timed_match or not stability_match:
        raise RuntimeError("V3.2 prefill sampled score-set validation failed")

    return {
        "shape": [1, 4096, 32768, 64, 128, 2048],
        "warmups": warmups,
        "iters": iters,
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "sampled_rows": list(SAMPLED_ROWS),
        "sampled_score_sets_match": timed_match,
        "stability_runs": stability_runs,
        "stability_match": stability_match,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--stability-runs", type=int, default=3)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.iters <= 0 or args.stability_runs <= 0:
        parser.error("warmups must be nonnegative; iters and stability-runs must be positive")
    result = run_benchmark(
        warmups=args.warmups,
        iters=args.iters,
        seed=args.seed,
        stability_runs=args.stability_runs,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
