# DSA Decode TopK comparison specification

## Contract

Each executable consumes finite BF16 scores [4, 32768] in contiguous row-major
layout and produces INT32 indices [4, 2048]. Each executable performs one
kernel launch on device 0. Launch ownership follows each implementation's
actual V3.2 decode stage: four vLLM-Ascend TopK workers and 4 x 16 = 64 SIMT v2
distributed context-shard blocks. This fixed decode-only shape is the entire
supported surface of the example.

The host generates deterministic negative and positive values with controlled
ties. Per row, the oracle computes the 2,048th score threshold and requires:
all indices are in range, every index is unique, every selected score is at
least the threshold, all scores above threshold are represented, and precisely
`2048 - greater_count` threshold-equal values are selected. Output ordering is
intentionally not compared, and equal-score index identity is intentionally not
compared.

## Source provenance

The vLLM-Ascend migration comes from
`/root/aiagent/vllm-ascend-upstream/csrc/attention/lightning_indexer/op_kernel/arch35/`:

- `lightning_indexer_service_vector.h`: `bf9f4aae92041f3b300e64646f6261dfe1da53799944dc4ab13a00ead5966fa6`
- `vf/lightning_indexer_topk.h`: `06f83b05f1b879793089cdc9d2f8a865abdace88306f4eae6f4f723c27f60bce`
- `vf/lightning_indexer_vector1.h`: `f35a4deea49ac37792e87191af5aa4f9e1d54e5b9702889e339aa10d439e8a7e`
- `vf/vf_topk_16_gather.h`: `6d1ceb35e2e9d7a9f961ab42043ce0f2bce2cd84adba447a130fb6e9e1d2a641`

The example uses the explicitly authorized verbatim upstream Basic API
histogram/filter functions, the 16,384-element first trunk, and the second
trunk merge/gather algorithm. The wrapper only converts direct BF16 input to
the sortable key representation that the upstream score producer supplies and
provides the standalone kernel/host boundary.

The SIMT migration comes from
`src/cannbench/operators/builtin/lightning_indexer/simt/v2/aten_dsa_lightning_indexer_v2/csrc/simt/lightning_indexer_decode_distributed_topk_bfloat16.asc`,
SHA-256 `c33332a87b88b9c7e0b76fca16a10c7cdd7545ff54cf8c97aafbf6b2604f553e`.
The V3.2 dispatch condition is
`batch_size == 2 && query_count == 2 && context_shard_count == 16`; it selects
this distributed path. The five production stages and four device-wide
synchronization points are preserved under a unique standalone 64-block
distributed kernel symbol.

## Measurement

The target is Ascend 950, compiled for `dav-3510`. Correctness must pass before
profiling. Collect five independent `msopprof` directories per executable with
the `Default` metric set and launch-count one. Parse exactly one target kernel
row per directory and report only `Task Duration`; ACL Event and host wall time
are outside the boundary. Current/rated frequency parity is mandatory, each
implementation must match its production launch dimension (4 and 64), and all
raw profiler data is retained. Report all samples, median, min, max, and the
ratio of vLLM-Ascend median to SIMT v2 median.
