# DSA Inference Fusion Benchmark Spec

This spec defines the inference-side DeepSeek Sparse Attention (DSA) fusion
operators that CannBench should benchmark. It is scoped to realistic serving
cases and to fair comparisons across CUDA H800, CANN ops, and SIMT operator
implementations.

## Source Baseline

Use the public DeepSeek-V3.2-Exp inference path as the reference model shape and
fusion guide:

- DeepSeek-V3.2-Exp inference demo: indexer, TopK selection, and MLA decode
  orchestration.
- DeepGEMM indexer kernels: FP8 MQA index logits, including paged variants.
- FlashMLA sparse kernels: token-level sparse MLA prefill and decode.

Do not treat an unverified model name or private serving stack as the source of
truth. The public baseline already exposes the important inference boundaries:
index selection and sparse MLA attention.

## Main Decision

Prefill and decode are separate benchmark operators. They have different input
rank, batching, cache layout, scheduling, memory pressure, and practical fusion
boundaries.

Decode should be the first-class target because it is the repeated serving hot
path. Prefill should be included, but with chunked cases and a reference path
that does not materialize huge selected-KV tensors.

## Why Prefill And Decode Differ

### Decode

Decode normally processes one or a few query tokens per request:

- Query shape is batched: `[batch, seq_q, heads_q, dim_qk]`.
- `seq_q` is usually `1`; speculative or MTP paths may use `2` or more.
- KV is a paged cache.
- Sparse decode consumes token indices in KV-cache address space.
- The production kernel may use FP8 KV cache and dequantize inside attention.
- Split-KV scheduling and a combine stage are common on H800.

The realistic fused decode attention contract is:

```text
sparse_mla_decode(
    q,
    fp8_kv_cache,
    indices,
    topk_length?,
    attn_sink?,
    metadata?
) -> out, lse
```

This fuses selected-token QK, softmax, PV, and FP8 KV cache dequantization. It
does not compute indexer logits or TopK internally.

### Prefill

Prefill processes many query tokens and has different parallelism:

- Query shape is often flattened or chunked: `[seq_q, heads_q, dim_qk]`.
- Batch support may be absent in the low-level sparse prefill kernel.
- KV can be a contiguous prefill buffer rather than a paged serving cache.
- Work is tiled over query tokens and selected tokens instead of mainly over
  batch and split-KV.
- A dense-equivalent fallback can be better for short contexts or low sparsity.

The realistic fused prefill attention contract is:

```text
sparse_mla_prefill(
    q,
    kv,
    indices,
    topk_length?,
    attn_sink?
) -> out, max_logits, lse
```

Prefill should not reuse the decode kernel contract unless the backend
implementation actually does so.

## Operator Set

### `dsa_index_select`

Purpose: measure the DSA index selection path.

Contract:

```text
dsa_index_select(
    q_index,
    index_k_cache,
    weights,
    context_lens,
    block_table?
) -> indices
```

Expected fusion:

- FP8 index Q/K score computation.
- Per-token weight application.
- Reduction over index heads.
- TopK selection.

This operator produces token indices consumed by sparse attention. It should be
benchmarked separately from attention because TopK is a discrete selection
boundary and because production implementations expose index logits and sparse
attention as separate high-performance components.

### `sparse_mla_decode`

Purpose: measure the serving decode sparse attention hot path.

Contract:

```text
sparse_mla_decode(
    q,
    fp8_kv_cache,
    indices,
    topk_length?,
    attn_sink?,
    softmax_scale,
    scheduler_metadata?
) -> out, lse
```

Expected fusion:

- Decode KV cache addressing through `indices`.
- FP8 KV cache dequantization when the backend supports it.
- Selected-token QK.
- Mask or invalid-index handling.
- Softmax.
- PV.
- Optional split-KV combine.

This is the priority fused operator for CUDA H800, CANN ops, and SIMT op
comparison.

### `sparse_mla_prefill`

Purpose: measure sparse attention in the prefill phase.

Contract:

```text
sparse_mla_prefill(
    q,
    kv,
    indices,
    topk_length?,
    attn_sink?,
    softmax_scale
) -> out, max_logits, lse
```

Expected fusion:

- Selected-token gather through `indices`.
- QK.
- Max logits and log-sum-exp.
- Softmax.
- PV.

This operator is separate from decode. For large contexts, the benchmark
reference must avoid explicit materialization of `[seq_q, heads_q, topk,
dim_qk]` and `[seq_q, heads_q, topk, dim_v]`.

### Optional `dsa_decode_step`

Purpose: orchestration-level measurement after component baselines are stable.

Contract:

```text
dsa_decode_step(
    x,
    mla_q,
    index_q,
    index_k_cache,
    fp8_kv_cache,
    block_table,
    context_lens
) -> out, indices?, lse?
```

This may combine index selection and sparse MLA decode. Do not use it as the
first fairness target, because backend libraries may not expose the same
end-to-end fusion boundary.

## Backend Equivalence Rules

All backends must compare the same operator contract. A result is not comparable
if one backend includes index selection while another only runs sparse
attention.

For each case, CUDA H800, CANN ops, and SIMT op must align on:

- Operator name and fusion boundary.
- Input and output tensors.
- Dtype policy.
- KV cache format.
- Index format and invalid-index convention.
- `topk` and optional `topk_length` semantics.
- Causal or non-causal behavior.
- Whether `attn_sink` is enabled.
- Whether scheduler metadata and combine kernels are counted in latency.

Recommended initial comparison matrix:

| Operator | CUDA H800 reference | CANN ops target | SIMT target |
| --- | --- | --- | --- |
| `dsa_index_select` | DeepGEMM-style FP8 paged MQA logits + TopK | same contract | same contract |
| `sparse_mla_decode` | FlashMLA sparse decode | same contract | same contract |
| `sparse_mla_prefill` | FlashMLA sparse prefill | same contract | same contract |

## Shape Plan

### Decode Priority Cases

Start with production-like DeepSeek-V3.2 sparse decode shapes:

| case family | batch | seq_q | context | heads_q | heads_kv | dim_qk | dim_v | topk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| small serving | 2 | 1 or 2 | 32768 | 128 | 1 | 576 | 512 | 2048 |
| common serving | 64 | 1 or 2 | 32768 | 128 | 1 | 576 | 512 | 2048 |
| production batch | 74 | 1 or 2 | 32768 | 128 | 1 | 576 | 512 | 2048 |
| high batch | 128 | 1 or 2 | 32768 | 128 | 1 | 576 | 512 | 2048 |
| long context | 1 or 8 | 1 | 65536 or 131072 | 128 | 1 | 576 | 512 | 2048 |

The first benchmark wave should prefer `seq_q = 1` for standard decode and add
`seq_q = 2` for speculative or MTP-like decode.

### Prefill Priority Cases

Use chunked prefill first:

| case family | seq_q | context | heads_q | heads_kv | dim_qk | dim_v | topk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| smoke chunk | 64 | 8192 | 128 | 1 | 576 | 512 | 2048 |
| practical chunk | 128 | 32768 | 128 | 1 | 576 | 512 | 2048 |
| large chunk | 256 | 65536 | 128 | 1 | 576 | 512 | 2048 |
| stress chunk | 512 | 131072 | 128 | 1 | 576 | 512 | 2048 |

Avoid using `seq_q = 4096` as a default correctness case unless the reference
implementation is streaming or kernel-native. A naive reference that gathers
selected K/V explicitly can require hundreds of GiB or more.

## Single-Card H800 Feasibility

Sparse decode production cases are single-card feasible. For example, with
`batch = 128`, `seq_q = 2`, `context = 32768`, and FP8 KV cache at 656 bytes per
token, the KV cache payload is roughly 2.6 GiB. Q, output, and indices are much
smaller.

Sparse prefill kernel cases can also be feasible, but naive references are not.
For `seq_q = 4096`, `topk = 2048`, `heads_q = 128`, `dim_qk = 576`, and
`dim_v = 512`, explicitly materializing selected K and selected V would require
more than 2 TiB. CannBench should therefore gate large prefill cases behind a
kernel-native reference, a streaming reference, or performance-only mode.

## Implementation Order

1. Add `sparse_mla_decode` as the first fused inference operator.
2. Add DeepSeek-V3.2 decode shape manifests with `smoke`, `realistic`, and
   `stress` splits.
3. Implement CUDA H800 using the FlashMLA contract as the behavior reference.
4. Implement CANN ops and SIMT op against the same `sparse_mla_decode`
   contract.
5. Add `dsa_index_select` once indexer logits and TopK semantics are fixed.
6. Add `sparse_mla_prefill` with chunked shapes and memory-safe references.
7. Consider orchestration-level `dsa_decode_step` only after component
   comparisons are stable.

## Acceptance Criteria

A DSA inference benchmark case is acceptable only when:

- The phase is explicit: `decode` or `prefill`.
- The fusion boundary is explicit and matches the backend implementation.
- The case records the model/source provenance.
- The memory footprint is documented for large cases.
- Correctness tolerances account for FP8 KV cache and BF16 accumulation.
- CUDA H800, CANN ops, and SIMT op can run the same case without changing the
  operator contract.
