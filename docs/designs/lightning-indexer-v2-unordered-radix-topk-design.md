# Lightning Indexer V2 Unordered Radix Top-K Design

## Status

Approved for implementation on 2026-08-01.

## Background

The canonical DeepSeek V3.2 decode case is BF16 `B=2`, `Q=2`, `C=32768`,
`H=64`, `D=128`, and `K=2048`. The SIMT V1 context-sharded kernel spends
approximately 1.66 ms in four long Top-K VFs, about 84% of the 1.98 ms main
Indexer kernel. Each output row repeatedly merges 2048 retained candidates
with 2048 new candidates and fully bitonic-sorts all 4096 entries. Across 16
rounds, that performs about 2.56 million compare pairs and approximately 1265
block barriers per row.

PyTorch CUDA and vLLM-Ascend both avoid this repeated full sort. CUDA finds the
Kth threshold by radix passes, compacts the selected candidates, and sorts only
the final K values when `sorted=True`. vLLM-Ascend converts BF16 scores to
sortable 16-bit keys, finds the threshold using high-byte and low-byte
histograms, and returns the selected indices without a final sort.

## Goals

- Preserve V1 unchanged as the first published implementation.
- Add independently installable Lightning Indexer V2 and Sparse Attention V2
  packages.
- Make `--implementation-version v2` compose the existing DSA decode workflow
  from the two V2 component packages.
- Replace decode's repeated 4096-entry bitonic merge with unordered BF16 radix
  threshold selection.
- Measure the remaining phase split before deciding whether to implement a
  distributed context-shard histogram path.

## Non-Goals

- This milestone does not optimize Sparse Attention V2; it starts as a
  version-isolated copy of V1.
- It does not change prefill dispatch or the existing V1 prefill radix path.
- It does not add a concrete `dsa_decode_v2` operator or version branch to a
  public backend, CLI, or core module.
- It does not require Top-K indices to be ordered by score.

## Version Isolation

The V2 packages follow the established multi-version operator pattern:

```text
src/cannbench/operators/builtin/lightning_indexer/simt/v2/
  aten_dsa_lightning_indexer_v2/

src/cannbench/operators/builtin/sparse_attention/simt/v2/
  aten_dsa_sparse_attention_v2/
```

Python package names, extension names, device-library SONAMEs, C launch ABI
symbols, and Torch registration namespaces carry the `_v2` suffix. The
operator plugins map `v2` to those module names. This prevents Python import
caching, shared-library loading, C symbol interposition, and Torch registration
collisions between V1 and V2.

The workflow plugin remains `dsa_decode`. CannBench propagates one
`implementation_version` to every workflow component, so a V2 workflow is
naturally composed from Lightning Indexer V2 and Sparse Attention V2.

## Output Semantics

Lightning Indexer V2 returns `int32` indices with shape `[B,Q,K]`. For every
valid query row:

- all indices are in the visible context range;
- no index is repeated;
- every returned score is greater than or equal to every unreturned score;
- any valid subset at a tied Kth-score boundary is accepted;
- output order is unspecified.

Sparse Attention consumes the selected KV positions as a set and does not
depend on their order. Tests therefore compare selected score multisets and
validity properties instead of requiring descending output.

## Phase 1: Full-Row Radix Selection

### Data Flow

```text
query + keys + weights
        |
        v
16 context-sharded mixed score tasks
        |
        v
[4,32768] BF16 score workspace (256 KiB)
        |
        v
4 independent Vector radix-selection blocks
        |
        v
[4,2048] unordered int32 indices
```

The score launch retains V1's validated context sharding, Cube computation,
BF16 rounding order, ReLU, weighting, masking, and per-shard GM ownership. It
removes local shard Top-K, the shard-candidate workspace, final-candidate
inter-core synchronization, and the shard-zero merge. Every task terminates
after writing its non-overlapping score range.

Stream ordering makes the completed score workspace visible to the second
launch. No cross-task device synchronization is required.

### Radix Threshold

One pure-Vector block owns each output row. A BF16 value is mapped to a
monotonic unsigned 16-bit key:

```text
negative value -> bitwise complement
non-negative value -> sign bit flip
```

The selector performs:

1. a 256-bin histogram of the high byte and descending bucket selection;
2. a 256-bin low-byte histogram restricted to the chosen high-byte prefix;
3. threshold construction from the two selected bytes;
4. compaction of every key greater than the threshold;
5. deterministic low-index compaction of enough threshold-equal entries to
   fill exactly K outputs.

The output tensor itself is the compaction destination. V2 does not allocate a
candidate-score array and contains no final bitonic network. Entries strictly
above the threshold may appear in any order. The equal-threshold policy is
deterministic for repeatable testing, but lowest-index tie behavior is not part
of the public contract.

The new selector uses only C API, Tensor API, and SIMT API. V1 score sources
contain transitional Basic API code; copied score code is retained only to
establish the function-first V2 baseline. New Top-K and synchronization logic
must not add Basic API dependencies, and later score cleanup must converge to
the repository's target API boundary.

## Phase 2 Candidate: Distributed Context-Shard Histograms

Phase 2 is not implemented until Phase 1 is profiled. Its purpose is to use
the existing 16 context shards per batch to parallelize selection when four
row-owner Vector blocks leave Top-K as a material bottleneck.

The proposed multi-launch data flow is:

```text
1. score shards write BF16 scores and local high-byte histograms
2. row reducers select the high-byte bucket
3. shard blocks scan only that prefix and write local low-byte histograms
4. row reducers select the low byte and compute per-shard compact offsets
5. shard blocks compact greater/equal indices into disjoint output ranges
```

Histogram workspace is `[row_count, shard_count, 256]` `uint32` entries. For
four rows and 16 shards it occupies 64 KiB per radix digit. Row reducers also
store the selected prefix, remaining rank, and per-shard greater/equal counts
and offsets. Compaction uses those offsets, so it needs no global atomics and
produces exactly K unique indices.

This path trades additional launches and GM histogram traffic for up to 16-way
context parallelism per row. It stays within the C API, Tensor API, and SIMT
API boundary and does not use inter-core flags or spin waiting.

### Phase 2 Decision Gate

Implement Phase 2 only when the Phase 1 Default profile shows either:

- unordered radix Top-K is at least 20% of V2 Indexer device time; or
- unordered radix Top-K median duration is at least 100 us.

Before full implementation, microbenchmark the histogram/reducer launch chain.
Proceed only when its repeated median predicts at least a 20% Top-K-stage gain
after including every added launch. Otherwise move optimization effort to the
new dominant Indexer score stage or Sparse Attention V2.

## Validation

Source and dispatch tests verify V2 namespaces, version mapping, the absence
of final bitonic sorting, two 8-bit radix passes, and score-then-Top-K launch
order. Device correctness covers the canonical case, masked tails, tied Kth
scores, uniqueness, valid range, repeated launches, and V1 regression.

Remote validation uses the repository's CannBench commands on `dav-3510` for
build, accuracy, synchronized latency, and profiling. Reports must separate
score-kernel time, radix Top-K time, Lightning Indexer total time, Sparse
Attention time, and DSA workflow component sum. Phase 1 is retained only when
it is correct and its repeated median improves on V1.
