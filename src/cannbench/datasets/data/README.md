# Operator Dataset Coverage

This directory contains CannBench built-in operator dataset manifests. Each
operator directory owns its `smoke`, `realistic`, and `stress` splits. The
sections below define the DeepSeek DSA benchmark coverage model so new operator
cases can be added consistently.

DSA inference workflow benchmarks use workflow-level manifests under
`dsa_inference_workflow/` as the case-selection source. The matching
`lightning_indexer/` and `sparse_attention/` cases are component inputs for the
current execution path, not the primary benchmark grouping.

The workflow manifests define phase-specific realistic splits:

- `realistic_decode`: 16 decode workflow cases, expanded to 32 component runs
  because each case runs `lightning_indexer` and `sparse_attention`.
- `realistic_prefill`: 15 prefill workflow cases, expanded to 30 component
  runs for the same two components.

These splits are budgeted independently. A single scenario should fit the
single-card H800 run window; decode and prefill are not expected to run together
inside one two-hour budget.

Frontend grouping should follow the same workflow split names. The default
view for DSA should be `DSA Decode` / `DSA Prefill` workflow results, with
component timings available only as drilldown data.

For the inference fusion operator plan, see
[DSA_INFERENCE_FUSION_SPEC.md](DSA_INFERENCE_FUSION_SPEC.md).

## DSA Coverage Model

DSA can be benchmarked from three views, but the current priority is workflow
performance:

1. Inference workflow mode for the actual serving path, especially decode.
2. Single-operator mode for later bottleneck diagnosis and backend bring-up.
3. Training scenario mode for later forward/backward coverage and saved-tensor
   costs.

The minimal DSA data path currently covered by CannBench has three primitive
operators:

- `lightning_indexer`: computes sparse candidate scores for each query.
- `topk`: selects the candidate token blocks or tokens.
- `sparse_attention`: runs attention over the selected KV range.

If an implementation materializes selected KV before attention, add
`sparse_kv_gather` as an explicit diagnostic operator. In MLA deployments,
fused operator names may also include `quant_lightning_indexer`,
`sparse_flash_attention`, `sparse_flash_mla`,
`kv_quant_sparse_flash_attention`, and `lightning_indexer_grad`.

## Single-Operator Mode

Single-operator mode should isolate the primitive cost and correctness contract
without assuming a specific fusion policy.

### `topk`

Test coverage:

- Score tensor shape, reduction dimension, `k`, `largest`, and `sorted`.
- DSA score shapes from short prefill, long prefill, and single-token decode.
- Small `k`, medium `k`, and large `k` relative to context length.
- Return-value and return-index correctness, including deterministic ordering
  when scores tie or nearly tie.
- Dtypes used by the backend path, at minimum `float32` reference and the
  inference/training dtypes used by the target backend.

Current datasets: `topk/smoke.json`, `topk/realistic.json`,
`topk/stress.json`.

### `lightning_indexer`

Test coverage:

- Batch size, query token count, context token count, index head count,
  index dimension, and `top_k`.
- Prefill cases where `query_tokens > 1`.
- Decode cases where `query_tokens = 1` and `context_tokens` is large.
- GQA/MLA-style head ratios when the index head count differs from attention
  head count.
- Quantized indexer variants when the production path uses quantized KV or
  quantized index features.

Current datasets: `lightning_indexer/smoke.json`,
`lightning_indexer/realistic.json`, `lightning_indexer/stress.json`.

### `sparse_attention`

Test coverage:

- Batch size, query heads, KV heads, query token count, context token count,
  selected token count, head dimension, causal mask, and phase.
- Prefill sparse attention and decode sparse attention as separate families.
- Dense-equivalent edge cases where `selected_tokens == context_tokens`.
- Very sparse cases where `selected_tokens` is small enough for launch overhead
  and index handling to dominate.
- GQA/MLA head layouts and long-context decode layouts.

Current datasets: `sparse_attention/smoke.json`,
`sparse_attention/realistic.json`, `sparse_attention/stress.json`.

### Optional `sparse_kv_gather`

Add this operator when the target backend has a standalone gather stage before
attention. Test contiguous, random, block-sparse, and duplicate index patterns.
Record both latency and temporary memory traffic, because this stage is often
removed by decode fusion.

## Training Scenario

Training coverage should preserve enough separation to diagnose gradients,
saved tensors, and recomputation tradeoffs. Do not assume the training path uses
the same fusion boundary as inference.

Operators and blocks to test:

- `lightning_indexer` forward for score construction.
- `topk` selection cost in the forward path.
- `sparse_attention` forward.
- `sparse_attention` backward, or the backend equivalent sparse flash attention
  backward block.
- `lightning_indexer_grad` when the indexer parameters or index features are
  trainable.
- Alignment-loss or dense-reference paths used to train or validate the sparse
  selection policy.

Recommended training fusion boundaries:

- Keep indexer forward, selection, sparse attention forward, and backward blocks
  separately measurable first.
- Fuse only after the separated blocks have stable correctness and performance
  baselines.
- Treat TopK as a discrete selection boundary. Backward coverage should document
  whether gradients flow through the indexer, through an auxiliary loss, or only
  through sparse attention.

Training case families should include short sequence training, long sequence
training, GQA/MLA head layouts, and stress cases where saved activation size is
the limiting factor.

## Inference And Fusion Scenario

Inference should benchmark the fused serving path in addition to the primitive
operators. Decode is the highest-priority fusion target because intermediate
score, index, and gathered-KV writes are expensive relative to one generated
token.

For the current workflow-first plan, benchmark and report at scenario level:

- `dsa_decode`: `dsa_index_select -> sparse_mla_decode`
- `dsa_prefill`: `dsa_index_select -> sparse_mla_prefill`

The two component records are still emitted because the existing adapter
interfaces run those fused components separately. Result consumers should
aggregate same-`case_id` component records into a workflow result before
presenting DSA performance.

### Decode Fusion

Blocks to test:

- MLA prolog or projection preparation when the model uses MLA.
- Indexer prolog plus `lightning_indexer`.
- TopK or block selection.
- Sparse KV gather when it is not eliminated.
- Fused QK, mask, softmax, and PV over selected tokens.
- End-to-end fused `dsa_attention`, `sparse_flash_attention`, or
  `sparse_flash_mla` operator.

Important decode cases:

- `query_tokens = 1`.
- Long `context_tokens`.
- Large `selected_tokens` such as 1024 or 2048.
- Batch sizes used by serving, including continuous batching.
- KV cache layouts and quantized KV cache variants.

Current realistic workflow coverage:

- `realistic_decode` covers 4K, 8K, 16K, 32K, and 64K contexts.
- Batch coverage is deliberately capped at 16 for 8K/16K and 8 for 32K in the
  default realistic split, because the current CannBench materializer still
  builds host-side Python tuples. Batch 64/128 DeepSeek serving shapes belong in
  `stress` or a future device-native input generation flow.
- The split includes one TritonBench Llama4 decode case to keep a non-DeepSeek
  real-model decode shape in the comparison set.

### Prefill Fusion

Prefill should be benchmarked separately from decode because it has a larger
query dimension and different parallelism.

Blocks to test:

- Chunked indexer and TopK selection.
- Chunked sparse attention.
- Full prefill fused sparse attention when supported.
- Dense-equivalent fallback when sparsity is disabled or not profitable.

Current realistic workflow coverage:

- `realistic_prefill` contains 15 paired cases from TritonBench/HF-style model
  shapes, covering 50-token CLIP text, 77-token CLIP vision, 512-token BERT/GPT
  families, 1K BART/MBART/BigBird, 2K OPT, and 4K Longformer chunks.
- Large DeepSeek MLA prefill chunks with 128 query heads and TopK 2048 should
  stay out of the default realistic split until the reference path is
  kernel-native or streaming. Otherwise the host-side or naive reference work
  can dominate the measurement.

## Recommended Benchmark Order

1. Keep the current primitive DSA operators green: `lightning_indexer`, `topk`,
   and `sparse_attention`.
2. Add `sparse_kv_gather` only if profiling shows the implementation has a real
   standalone gather stage.
3. Add decode fused operators first: `dsa_attention`, `sparse_flash_attention`,
   or `sparse_flash_mla`.
4. Add quantized inference variants, especially `quant_lightning_indexer` and
   `kv_quant_sparse_flash_attention`.
5. Add training backward coverage after forward primitive and fused inference
   coverage are stable.

For each new DSA operator, keep `smoke` for wiring, `realistic` for traced or
derived model shapes, and `stress` for operator-specific pressure points. Do
not mechanically reuse another operator's stress cases unless the workload
semantics match.
