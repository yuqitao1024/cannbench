# DSA Profile Kernel Aggregation

Status: approved for implementation on 2026-07-30.

## Goal

Make DSA workflow latency equal the sum of the device-side kernels selected by
each component plugin. Correct the current V3.2 SIMT decode record and refresh
the other V3.2 Ascend workflow records from raw profiler artifacts.

## Root Cause

The profile parser already supports two distinct contracts:

- multiple files as repeated samples, summarized by the timing statistic;
- multiple files as stages of one operation, summed when
  `aggregate_across_files=True`.

The SIMT sparse-attention plugin selects both
`sparse_attention_head64_fused_kernel` and
`sparse_attention_head64_combine_kernel`, but does not request cross-file
aggregation. For the successful V3.2 decode profile, the parser therefore
treated `369.535004 us` and `36.964001 us` as two samples and reported their
median, `203.2495025 us`, instead of their sum, `406.499005 us`.

The vLLM Ascend lightning-indexer and sparse-attention selections already set
`aggregate_across_files=True`. Their refreshed artifacts still require direct
inspection because matching patterns may include several lowering, operator,
and LSE kernels.

## Design

Keep the fix inside the sparse-attention operator plugin. Set
`aggregate_across_files=True` on its SIMT `ProfileKernelSelection`; do not
special-case DSA or concrete kernel names in the common parser, CLI, backend,
or workflow aggregation code.

The existing workflow aggregator remains unchanged. It correctly sums the
component `latency_ms` values after each component profile has been reduced
according to its plugin-owned kernel-selection contract.

The SIMT lightning indexer remains a single selected kernel and needs no policy
change. The existing vLLM Ascend selections remain unchanged unless the new
raw artifacts demonstrate a mismatch between their selected kernel set and the
intended dynamic timing boundary.

## Verification Contract

Run the following V3.2 BF16 workflows on Ascend 950PR through CannBench:

- SIMT v1 prefill;
- vLLM Ascend prefill;
- vLLM Ascend decode.

Reuse the successful SIMT v1 decode profiler artifacts rather than recollecting
that workflow. Its correct values are:

- lightning indexer: `1.986091064 ms`;
- sparse-attention fused kernel: `0.369535004 ms`;
- sparse-attention Combine kernel: `0.036964001 ms`;
- sparse-attention total: `0.406499005 ms`;
- workflow total: `2.392590069 ms`.

For each of the four workflow records:

1. enumerate every raw `OpBasicInfo*.csv` selected by the component plugin;
2. sum the selected `Task Duration(us)` values for staged implementations;
3. verify the component `profile-summary.json` equals that sum in milliseconds;
4. verify the workflow benchmark record equals the sum of the two component
   summaries;
5. verify both workflow components succeeded and accuracy passed.

The profile artifacts, not wall time or host-side timing, are authoritative.

## Publication

Publish these canonical runs while preserving the existing schema and run IDs:

- `opbench-ascend-950pr-simt-v1-dsa_decode-realistic-bfloat16`;
- `opbench-ascend-950pr-simt-v1-dsa_prefill-realistic-bfloat16`;
- `opbench-ascend-950pr-vllm-ascend-dsa_decode-realistic-bfloat16`;
- `opbench-ascend-950pr-vllm-ascend-dsa_prefill-realistic-bfloat16`.

Raw profiler directories remain outside `published/`. Commit the plugin/test
fix separately from the refreshed published data.

## Non-Goals

- Changing common profile parsing semantics.
- Adding operator-name branches to public layers.
- Reprofiling the already successful SIMT decode workflow.
- Changing kernel implementation, dispatch, accuracy tolerances, or published
  data schema.
