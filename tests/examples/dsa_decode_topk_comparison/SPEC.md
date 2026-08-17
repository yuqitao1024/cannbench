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

The SIMT algorithm baseline comes from
`src/cannbench/operators/builtin/lightning_indexer/simt/v2/aten_dsa_lightning_indexer_v2/csrc/simt/lightning_indexer_decode_distributed_topk_bfloat16.asc`,
SHA-256 `c33332a87b88b9c7e0b76fca16a10c7cdd7545ff54cf8c97aafbf6b2604f553e`.
The V3.2 dispatch condition is
`batch_size == 2 && query_count == 2 && context_shard_count == 16`; it selects
this distributed path. The five production stages, their algorithms, and four
device-wide synchronization points are preserved under a unique standalone
64-block distributed kernel symbol.

The example applies five retained fixed-shape optimizations on top of that
baseline:

1. Each block issues one 4 KiB C API MTE2 copy for its contiguous 2,048 BF16
   score shard. The high-byte histogram, low-byte histogram, and compact pass
   reuse that dedicated UB tile.
2. Each of the four reducer blocks issues one 16 KiB GM-to-UB copy for its
   complete 16 x 256 high histogram matrix and, after the low producer stage,
   one 16 KiB copy for the corresponding low histogram matrix. Both reducers
   consume the UB matrices.
3. High-byte selection computes 16 group totals in parallel. The owning group
   scans only its 16 bins in descending order instead of one thread scanning
   all 256 bins.
4. Compact classifies each score once. Each warp uses ballot and population
   count, lane zero atomically reserves a contiguous greater/equal range in
   block-private UB, and a shuffle broadcasts the returned start. This removes
   the ten-round block-wide ping-pong scan and all but the counter-initialization
   block barrier from the compact VF.
5. The low reducer computes the 16 shard greater/equal prefixes with four
   width-16 `asc_shfl_up` steps. Warp 0 executes the scan uniformly, and its
   first 16 lanes convert the inclusive register prefixes to the existing
   exclusive GM offsets.

Score, high-histogram, low-histogram, and shared stage-scratch UB regions do not
overlap. Kernel launch geometry, five VFs, and four `SyncAll` boundaries remain
unchanged. The retained standalone candidate source SHA-256 is
`1b11b72738b789228db1230f9f1369b980a0f7da5599db27fe4fd6c510b7fc13`.

An earlier candidate allocated 32 private 256-bin histograms per block, updated
the owning warp's bins, and reduced the 32 counters for every final bin in both
histogram stages. It was correct but increased median task duration by
`12.8417%` in five alternating pairs, so its source-contract test and kernel
changes are deliberately absent from the retained example. All changes are
example-only; the production SIMT v2 source remains unchanged.

## Measurement

The target is Ascend 950, compiled for `dav-3510`. Correctness must pass before
profiling. Collect five independent `msopprof` directories per executable with
the `Default` metric set and launch-count one. Parse exactly one target kernel
row per directory and report only `Task Duration`; ACL Event and host wall time
are outside the boundary. Current/rated frequency parity is mandatory, each
implementation must match its production launch dimension (4 and 64), and all
raw profiler data is retained. Report all samples, median, min, max, and the
ratio of vLLM-Ascend median to SIMT v2 median.

For optimization retention, compare each candidate with the currently retained
stack using alternating-order pairs under the same constraints. Reject a
correct candidate when it does not improve the measured device boundary. The
final retained combination must also be compared with commit `887eb3d` using
ten alternating-order pairs, with source/executable hashes and every raw
OPPROF tree preserved outside the repository.

## Rejected four-block SIMT experiment

The rejected experiment used four row blocks and copied vLLM-Ascend's logical
`16384 -> 18432` score split into SIMT VF code. Its two high/low-byte radix
stages used UB atomic 256-bin histograms. Greater/equal compaction used
1024-element chunks with ballot, warp-prefix calculation, and repeated block
barriers; the two compact stages processed 68 chunks in total. Although the
radix phases and split matched vLLM-Ascend, the SIMD primitives and dataflow
did not.

Ten paired `Default` collections established a `212.9844895 us` median, versus
`7.6760 us` for four-block vLLM-Ascend and `20.8389995 us` for retained
64-block distributed SIMT v2. The rejected candidate won no pair against
either baseline and regressed the medians by `2674.6807%` and `922.0476%`,
respectively. All implementations passed the score-set oracle under the same
one-launch, frequency-parity, unmodified-warmup measurement boundary.

Consequently, the rejected source, executable target, run-script integration,
and source-contract tests are intentionally absent. Do not recreate or tune
the exact combination of four row blocks, the `16384 -> 18432` split, UB atomic
histograms, and per-chunk ballot/warp-prefix/block-barrier compaction. Future
SIMT TopK performance work must start from the retained distributed SIMT v2
path. Four-block ownership requires a materially different inner
implementation and a written bottleneck hypothesis before it may be tested
again, such as warp-atomic output reservation that removes per-chunk block
synchronization.

## Warp-atomic compact optimization

The retained optimization applies only to the 64-block
distributed SIMT v2 path. It preserves the fixed shape, one 2,048-element
score shard per block, single GM-to-UB score copy, radix histograms, reducer
stages, launch geometry, and output oracle. Only the final compact VF changes.

The baseline compact packs each thread's greater/equal counts and performs a
ten-round 1,024-thread ping-pong prefix scan. Every round has two block
barriers. The candidate instead processes the shard's two 1,024-element
iterations with warp-local ballot and population count. Lane zero reserves one
contiguous greater or equal output range for the warp with a UB atomic add,
then broadcasts the returned starting offset with a warp shuffle. Each selected
lane adds its ballot-derived lane prefix to that start. Greater and equal
reservations use separate block-private UB counters; the existing per-shard
global offsets keep blocks' output ranges disjoint. Output ordering and the
identity of threshold-equal indices remain outside the contract.

This mechanism is adapted from the warp-compaction variant in PyTorch
`aten/src/ATen/native/cuda/TensorTopK.cu` at commit
`893b6406afc1a6384ab6fae8a2247d03cc230d87`. That variant is guarded by
PyTorch's ROCm branch; the NVIDIA CUDA branch in the same source still uses
`exclusiveBinaryPrefixScan`. The experiment therefore claims reuse of the
warp-reservation mechanism, not migration of PyTorch's current NVIDIA CUDA
default path.

The algorithm change was measured without reducing the baseline 8 KiB compact
UB request. In ten alternating-order `--aic-metrics=Default` pairs on Ascend
950PR, the block-scan baseline median was `20.5035 us` and the warp-atomic
candidate median was `18.8459995 us`. The candidate reduced task duration by
`8.0840%`, won `10/10` pairs, and passed every direct and profiled score-set
oracle. All accepted rows used unmodified profiler warmup, `--launch-count=1`,
no `--kernel-name`, 64 blocks, and current/rated `1650/1650 MHz`.

Removing the unused scan reservation was then measured independently. Reducing
total dynamic UB from about 44 KiB to 37.125 KiB changed the median from
`18.739 us` to `18.874001 us`, a `0.7204%` regression, and won only `1/5`
pairs. That capacity reduction is rejected, so the retained warp-atomic source
keeps the original 8 KiB compact UB request. Full raw evidence is archived at
`/tmp/dsa-topk-warp-atomic-evidence-20260817.tar.gz`, SHA-256
`22e4ce773d40559262426f3236cddc51b811adb7344f5226666c44b09f83d3e0`.

## Follow-up distributed SIMT optimization campaign

Further tuning remains restricted to the retained 64-block distributed path.
The four-block `16384 -> 18432` experiment and the reduced-scratch candidate
must not be reused. Five candidates are evaluated independently, in this
order, against the currently retained source:

1. Only the four row-owner blocks invoke the high and low reducer VFs. All 64
   physical blocks still reach all four device-wide `SyncAll` boundaries in
   the same order, and every producer still publishes its GM writes before the
   following barrier.
2. Warp output reservation returns immediately when its ballot mask is zero.
   A nonempty warp retains the same UB atomic reservation, lane-zero shuffle,
   and lane-prefix population count as the accepted compact algorithm.
3. The high-histogram VF converts each staged BF16 score to its sortable key
   once and writes the key back to the same UB shard. The low-histogram and
   compact VFs consume those keys directly; no original score bits are needed
   after the first radix pass.
4. The low reducer replaces the first 16 threads' serial prior-shard offset
   loops with a four-step, width-16 warp shuffle scan. The first warp executes
   the collective uniformly and only its first 16 lanes publish shard offsets.
5. Each score-pass thread loads two adjacent staged BF16 bit patterns through
   one aligned 32-bit UB access. The fixed 1,024-thread launch covers the 2,048
   element shard exactly; histogram and compact semantics remain unchanged.

Each candidate must first pass the source-contract test and direct four-row
score-set oracle. Performance uses alternating-order baseline/candidate
`msopprof --aic-metrics=Default --launch-count=1` pairs with unmodified
profiler warmup, no `--kernel-name`, exactly one 64-block target row, and
current/rated `1650/1650 MHz`. A candidate is retained only if its median is
lower, it wins a clear majority of pairs, and fresh combined-stack validation
preserves correctness. Source, executable, commands, individual samples, and
all raw OPPROF trees are retained outside the repository.

The reducer warp-shuffle candidate is the only retained result. Across two
independent five-pair rounds, the frozen baseline median was `18.9505005 us`
and the retained median was `18.2335005 us`, a `3.7835%` reduction with
`10/10` pair wins. Row-owner-only reducer VF invocation regressed `0.1900%`;
the empty-ballot fast path improved `0.9580%` once but regressed `0.2806%` on
repeat; in-place sortable-key reuse regressed `0.2779%`; and adjacent BF16
32-bit packed loads regressed `2.3554%`. Those four source changes and their
source-contract assertions are absent from the retained example.

The retained source/executable SHA-256 pair is
`1b11b72738b789228db1230f9f1369b980a0f7da5599db27fe4fd6c510b7fc13` /
`c22f7c34b400978fe8db224148106d8662a446dd4cc150445f1ec583002b3b3c`.
All follow-up evidence is archived at
`/tmp/dsa-topk-followup-evidence-20260817.tar.gz`, SHA-256
`4bbe1350491d3be7218345a62ec4dffc36a137916c77fa701d9e0416bf417619`.
