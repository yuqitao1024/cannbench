from __future__ import annotations

import argparse
import json
import statistics
import time


def build_target_tensors(torch, *, seed: int):
    torch.manual_seed(seed)
    device = torch.device("npu")
    query = torch.randn(2, 2, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(2, 32768, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(2, 2, 64, device=device, dtype=torch.bfloat16)
    valid = torch.tensor(
        [[32767, 32768], [32766, 32765]], device=device, dtype=torch.int32
    )
    return query, keys, weights, valid


def reference_score_sets(torch, query, keys, weights, valid):
    scores = torch.einsum("bqhd,bcd->bqhc", query, keys)
    reduced = (torch.relu(scores) * weights.unsqueeze(-1)).sum(dim=2)
    positions = torch.arange(keys.shape[1], device=keys.device).reshape(1, 1, -1)
    reduced = reduced.masked_fill(positions >= valid.unsqueeze(-1), float("-inf"))
    reference_scores = torch.topk(
        reduced, 2048, dim=-1, largest=True, sorted=True
    ).values
    return reduced, reference_scores


def run_benchmark(*, warmups: int, iters: int, seed: int):
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
            phase="decode",
            family="family_64x128",
        )

    for _ in range(warmups):
        run_once()
        torch.npu.synchronize()

    samples_ms = []
    output = None
    for _ in range(iters):
        started = time.perf_counter_ns()
        output = run_once()
        torch.npu.synchronize()
        samples_ms.append((time.perf_counter_ns() - started) / 1_000_000)

    reduced, reference_scores = reference_score_sets(
        torch, query, keys, weights, valid
    )
    custom_scores = reduced.gather(-1, output.to(torch.int64))
    score_sets_match = bool(torch.equal(custom_scores, reference_scores))
    if not score_sets_match:
        raise RuntimeError("V3.2 decode score-set validation failed")

    return {
        "shape": [2, 2, 32768, 64, 128, 2048],
        "warmups": warmups,
        "iters": iters,
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
        "score_sets_match": score_sets_match,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmups", type=int, default=5)
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args(argv)
    if args.warmups < 0 or args.iters <= 0:
        parser.error("warmups must be nonnegative and iters must be positive")
    result = run_benchmark(
        warmups=args.warmups,
        iters=args.iters,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
