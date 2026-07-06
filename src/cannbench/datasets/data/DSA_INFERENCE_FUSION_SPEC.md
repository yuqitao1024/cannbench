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

## Open-Source Operator Reuse

CannBench should not implement CUDA or CANN kernels for the initial DSA
benchmark. It should wrap proven open-source operator libraries and use the same
operator contract across backends. CannBench-owned code should be limited to
shape manifests, dependency detection, tensor preparation, adapter calls,
correctness checks, and result reporting.

### CUDA H800

Use DeepSeek's public CUDA libraries as the primary backend:

- FlashMLA for sparse MLA attention.
  - Decode: `get_mla_metadata` plus `flash_mla_with_kvcache`.
  - Prefill: `flash_mla_sparse_fwd`.
  - The decode path supports sparse `indices`, invalid index `-1`, and FP8 KV
    cache dequantization inside the attention kernel.
- DeepGEMM for DSA indexer logits.
  - Prefill-style contiguous logits: `fp8_mqa_logits`.
  - Decode-style paged KV logits: `get_paged_mqa_logits_metadata` plus
    `fp8_paged_mqa_logits`.
  - TopK remains an explicit boundary unless a selected backend exposes logits
    and TopK as a single production operator with the same outputs.

The CUDA adapter should call these libraries directly when installed. It should
not add a CannBench CUDA implementation of sparse MLA or FP8 MQA logits.

CannBench's `cuda_library` flow loads a thin external adapter module rather
than vendoring CUDA kernels. The backend checks `CANNBENCH_CUDA_DSA_ADAPTER`
first and then falls back to importing `cannbench_cuda_dsa`. The module must
expose callable `lightning_indexer` and `sparse_attention` entry points. Those
entry points receive the materialized CannBench tensors, request metadata, case
metadata, and payload, and are responsible for calling the installed
FlashMLA/DeepGEMM APIs with the library-native layout.

The built-in `cannbench_cuda_dsa` module is only this thin dispatch layer. It
does not implement CUDA kernels. Configure it with:

- `CANNBENCH_CUDA_DSA_LIGHTNING_INDEXER=cannbench_cuda_dsa_flashmla_deepgemm:lightning_indexer`
- `CANNBENCH_CUDA_DSA_SPARSE_ATTENTION=cannbench_cuda_dsa_flashmla_deepgemm:sparse_attention`

The standard `cannbench_cuda_dsa_flashmla_deepgemm` wrapper exposes the same
two CannBench callable names for both workflow phases. It routes decode indexer
cases to `deep_gemm.fp8_paged_mqa_logits`, prefill indexer cases to
`deep_gemm.fp8_mqa_logits`, decode sparse attention cases to
`flash_mla.flash_mla_with_kvcache`, and prefill sparse attention cases to
`flash_mla.flash_mla_sparse_fwd`. Without those environment variables,
installed CUDA libraries, or importable fallback symbols, the adapter fails
with an explicit dependency error.

### Ascend / CANN

The generic `ascend/cann-ops` repository exposes useful primitives such as
`top_k_v3`, `moe_soft_max_topk`, and large-head flash attention, but the current
public tree does not expose a complete DSA sparse MLA decode/prefill operator
matching the DeepSeek-V3.2 fusion boundary.

Use the Ascend serving-oriented open-source stacks first:

- `vllm-project/vllm-ascend` for DSA serving integration.
  - Indexer: `torch_npu.npu_lightning_indexer`,
    `torch_npu.npu_quant_lightning_indexer`, or the temporary
    `_C_ascend` equivalents used by vLLM Ascend.
  - Sparse decode attention: `torch.ops._C_ascend.npu_sparse_attn_sharedkv`
    with `npu_sparse_attn_sharedkv_metadata`.
  - Quantized KV sparse decode attention:
    `torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv` with
    `npu_kv_quant_sparse_attn_sharedkv_metadata`.
- `sgl-project/sgl-kernel-npu` for standalone NPU kernels.
  - `torch.ops.npu.lightning_indexer` for DSA sparse TopK indexing.
  - `torch.ops.npu.mla_preprocess` for the RMSNorm -> Dequant -> MatMul ->
    RoPE -> ReshapeAndCache preprocessing fusion.
  - Its MLA paged-KV attention path is useful for adapter validation, but the
    benchmark must still align to the same `sparse_mla_decode` contract.

The CANN adapter should prefer public `torch_npu` or exported PyTorch custom op
entry points when available. If only vLLM Ascend private `_C_ascend` symbols are
available in the installed stack, the adapter must record that provenance in the
benchmark result.

Current CannBench status:

- `bench --backend ascend --implementation vllm_ascend --op lightning_indexer`
  calls `torch_npu.npu_lightning_indexer` with vLLM Ascend-compatible TND/BSND
  inputs.
- `bench --backend ascend --implementation vllm_ascend --op sparse_attention`
  calls vLLM Ascend `npu_sparse_attn_sharedkv_metadata` and
  `npu_sparse_attn_sharedkv`. The current adapter maps CannBench
  `sparse_attention` cases to a compressed sparse-KV PA_ND layout, uses
  128-token pages when the context length permits, and passes case `indices` as
  `cmp_sparse_indices`.
- `bench --backend nvidia --implementation cuda_library --op ...` loads a
  CUDA DSA adapter from `CANNBENCH_CUDA_DSA_ADAPTER` or `cannbench_cuda_dsa`.
  Missing adapters fail fast with an actionable dependency error instead of
  falling back to the PyTorch reference implementation.

### SIMT

SIMT remains the CannBench-owned implementation target. It should implement the
same contracts only for comparison and portability experiments. SIMT timing must
not include extra work that CUDA or CANN external libraries do not include.

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

## Current Test Workflow

The current CannBench inference benchmark is workflow-first. Case selection is
owned by `dsa_inference_workflow/<split>.json`; the matching
`lightning_indexer/<split>.json` and `sparse_attention/<split>.json` entries
are component inputs used by the current execution path.

The current workflow is a two-step component execution. It keeps the selection
and sparse attention boundaries explicit internally while allowing decode and
prefill cases to be benchmarked through one workflow API:

```text
build_dsa_inference_workflow(dataset, case_id, dtype, seed)
```

Decode workflow:

```text
dsa_index_select  -> sparse_mla_decode
lightning_indexer -> sparse_attention
```

Prefill workflow:

```text
dsa_index_select  -> sparse_mla_prefill
lightning_indexer -> sparse_attention
```

Each workflow step owns a prepared CannBench single-operator input. A workflow
case is selected only when it is present in the workflow manifest. It is
considered runnable only when `lightning_indexer/<split>.json` and
`sparse_attention/<split>.json` also contain the same `case_id` and agree on
batch, query tokens, context tokens, selected token count, and phase.

The workflow is exposed through `bench`:

```bash
python -m cannbench bench \
  --backend ascend \
  --implementation vllm_ascend \
  --workflow dsa_decode \
  --dataset smoke \
  --case-id tiny_decode_top4
```

```bash
python -m cannbench bench \
  --backend ascend \
  --implementation vllm_ascend \
  --workflow dsa_prefill \
  --dataset smoke \
  --case-id tiny_prefill_top8
```

CUDA H800 uses the same workflow token with the CUDA library implementation:

```bash
CANNBENCH_CUDA_DSA_ADAPTER=cannbench_cuda_dsa \
CANNBENCH_CUDA_DSA_LIGHTNING_INDEXER=cannbench_cuda_dsa_flashmla_deepgemm:lightning_indexer \
CANNBENCH_CUDA_DSA_SPARSE_ATTENTION=cannbench_cuda_dsa_flashmla_deepgemm:sparse_attention \
python -m cannbench bench \
  --backend nvidia \
  --implementation cuda_library \
  --workflow dsa_decode \
  --dataset realistic_decode \
  --case-id deepseek_a5_decode_b1_ctx4096_top512
```

Omitting `--case-id` runs every paired workflow case in the selected dataset and
phase. Omitting `--dataset` on a workflow command selects the phase-specific
realistic split:

- `--workflow dsa_decode` -> `realistic_decode`
- `--workflow dsa_prefill` -> `realistic_prefill`

The command expands each workflow into two prepared inputs and records both
component results under one run directory. The automatic run name uses the
workflow token, for example
`opbench-ascend-950pr-vllm-ascend-dsa_decode-realistic_decode-float16`.

This flow is intentionally reported as a workflow benchmark, not as two
independent single-operator benchmark groups. The two component results should
be aggregated by `case_id` for primary DSA performance reporting, with the
component split kept as drilldown data. It is still not an end-to-end fused
`dsa_decode_step`; that contract should wait until CUDA, Ascend, and SIMT expose
the same boundary.

Current implementation status:

- Ascend `--implementation vllm_ascend` can dispatch the workflow through the
  vLLM Ascend `lightning_indexer` and sparse shared-KV attention adapter when
  the target environment provides the corresponding `torch_npu` symbols.
- Ascend `--implementation cann_ops_library` remains the generic CANN ops path;
  it is not yet the real DSA sparse shared-KV fused path.
- Nvidia `--implementation cuda_library` dispatches DSA operators through an
  external adapter module. A real H800 benchmark requires the environment to
  provide a FlashMLA/DeepGEMM wrapper module matching the adapter contract.

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
| `dsa_index_select` | DeepGEMM FP8 MQA logits + TopK | vLLM Ascend or SGLang NPU lightning indexer | same contract |
| `sparse_mla_decode` | FlashMLA sparse decode | vLLM Ascend sparse shared-KV op | same contract |
| `sparse_mla_prefill` | FlashMLA sparse prefill | available NPU sparse MLA/prefill library op, if contract-compatible | same contract |

## Shape Plan

### Realistic Split Sizing

The realistic DSA workflow datasets are split by inference phase because decode
and prefill use different fused attention contracts and should be budgeted
separately:

| split | phase | workflow cases | component runs | intent |
| --- | --- | ---: | ---: | --- |
| `realistic_decode` | decode | 8 | 16 | A5 fused-compatible serving buckets across context, TopK, and moderate batch scaling |
| `realistic_prefill` | prefill | 8 | 16 | A5 fused-compatible chunked prefill buckets across query length, context, TopK, and batch |

This is sized for a per-scenario single-card H800 budget, not for one combined
decode-plus-prefill run. The split is intentionally broad enough to expose
batch scaling, context scaling, and TopK scaling while keeping every default
case compatible with the Ascend vLLM A5 fused operator contract. Very
high-batch DeepSeek serving cases such as batch 64/128 belong in `stress` or a
future lazy/materialized-on-device input path. With the current CannBench input
materializer, those cases would be dominated by host-side tuple generation
rather than by the fused CUDA/CANN/SIMT kernels being measured.

The workflow manifests are the frontend grouping contract:

- `realistic_decode` is displayed as `DSA Decode`.
- `realistic_prefill` is displayed as `DSA Prefill`.
- Same-case component records are summed for workflow latency and retained for
  drilldown.

### Decode Priority Cases

Start with A5-compatible sparse decode serving buckets. The `realistic_decode`
split currently covers context scaling at batch 1, moderate continuous-batch
scaling at short/mid context, and the TopK 512/1024 transition:

| case family | batch | seq_q | context | index_heads | index_dim | heads_q | heads_kv | head_dim | topk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| context sweep | 1 | 1 | 512, 2048, 4096, 8192 | 64 | 128 | 64 | 1 | 512 | 512 |
| TopK transition | 1 | 1 | 16384 | 64 | 128 | 64 | 1 | 512 | 1024 |
| batch sweep | 2, 4 | 1 | 4096 | 64 | 128 | 64 | 1 | 512 | 512 |
| batch stresslet | 8 | 1 | 2048 | 64 | 128 | 64 | 1 | 512 | 512 |

Reserve larger batch 16/64/128 and 32K-plus contexts for `stress` until the
adapter path can generate or load inputs directly on device.

The first benchmark wave should prefer `seq_q = 1` for standard decode and add
`seq_q = 2` for speculative or MTP-like decode.

### Prefill Priority Cases

Use chunked prefill first. The `realistic_prefill` split currently contains 8
paired workflow cases that keep the same A5 fused contract as decode while
varying chunk length, context, batch, and TopK:

| case family | batch | seq_q | context | index_heads | index_dim | heads_q | heads_kv | head_dim | topk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| chunk sweep | 1 | 64, 128, 256, 512 | 512 | 64 | 128 | 64 | 1 | 512 | 512 |
| context sweep | 1 | 512 | 1024 | 64 | 128 | 64 | 1 | 512 | 512 |
| TopK transition | 1 | 512 | 1024 | 64 | 128 | 64 | 1 | 512 | 1024 |
| batch sweep | 2 | 128, 256 | 512 | 64 | 128 | 64 | 1 | 512 | 512 |

Avoid using large `seq_q = 4096` and TopK-2048 prefill as a default correctness
case unless the reference implementation is streaming or kernel-native. A naive
reference that gathers selected K/V explicitly can require hundreds of GiB or
more.

## Single-Card H800 Feasibility

The default A5-compatible decode cases are single-card feasible on H800. The
largest default decode bucket is `batch = 1`, `seq_q = 1`, `context = 16384`,
and `topk = 1024`; the largest default batch bucket is `batch = 8`,
`seq_q = 1`, `context = 2048`, and `topk = 512`. Larger production decode
probes such as batch 64/128 and 32K-plus context should be measured as `stress`
after input generation is moved out of the Python tuple path.

The default A5-compatible prefill cases are also single-card feasible because
they cap chunk size at 512, context at 1024, and TopK at 1024. Much larger
prefill chunks can be kernel-feasible, but naive references are not. CannBench
should therefore gate large prefill cases behind a kernel-native reference, a
streaming reference, or performance-only mode.

## Implementation Order

1. Add `sparse_mla_decode` as the first fused inference operator.
2. Add inference workflow shape manifests with `smoke`, `realistic_decode`,
   `realistic_prefill`, and `stress` splits.
3. Report DSA performance at workflow level first, aggregating the existing
   component records by workflow case.
4. Add a CUDA H800 adapter that calls FlashMLA for `sparse_mla_decode`.
5. Add an Ascend adapter that calls vLLM Ascend or SGLang NPU sparse attention
   ops for the same `sparse_mla_decode` contract.
6. Implement the SIMT op against the same `sparse_mla_decode` contract.
7. Add single-operator reporting only when workflow drilldown shows that
   component-level diagnosis is needed.
8. Add `sparse_mla_prefill` with chunked shapes and memory-safe references.
9. Consider orchestration-level `dsa_decode_step` only after component
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
