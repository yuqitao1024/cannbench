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

The example applies six retained fixed-shape optimizations on top of that
baseline:

1. Each block issues one 4 KiB C API MTE2 copy for its contiguous 2,048 BF16
   score shard. The high-byte histogram, low-byte histogram, and compact pass
   reuse that dedicated UB tile.
2. Each of the four reducer blocks issues one 16 KiB GM-to-UB copy for its
   complete 16 x 256 high histogram matrix and, after the low producer stage,
   one 16 KiB copy for the corresponding low histogram matrix. Both reducers
   consume the UB matrices.
3. Each high/low producer converts its 256 raw UB counters into an inclusive
   suffix histogram with eight warp-local scans and one warp-total merge. The
   reducers select thresholds from adjacent suffix positions. The low reducer
   also obtains per-shard greater/equal counts from constant-position suffix
   reads instead of rescanning all high/low bins.
4. The warp-local suffix remains in a thread register across the warp-total
   merge and is written directly to GM. The publication helper's existing
   barrier replaces the removed final suffix UB round trip and helper barrier.
5. Compact classifies each score once. Each warp uses ballot and population
   count, lane zero atomically reserves a contiguous greater/equal range in
   block-private UB, and a shuffle broadcasts the returned start. This removes
   the ten-round block-wide ping-pong scan and all but the counter-initialization
   block barrier from the compact VF.
6. The low reducer computes the 16 shard greater/equal prefixes with four
   width-16 `asc_shfl_up` steps. Warp 0 executes the scan uniformly, and its
   first 16 lanes convert the inclusive register prefixes to the existing
   exclusive GM offsets.

Score, high-histogram, low-histogram, and shared stage-scratch UB regions do not
overlap. Kernel launch geometry, five VFs, and four `SyncAll` boundaries remain
unchanged. The retained standalone candidate source SHA-256 is
`0d2afd39acab8106b37acf86e15966783b0440507487df84de48ffaacc0da88f`.

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

Within those five follow-up candidates, the reducer warp-shuffle candidate is
the only retained result. Across two independent five-pair rounds, the frozen
baseline median was `18.9505005 us` and the retained median was `18.2335005 us`,
a `3.7835%` reduction with `10/10` pair wins. Row-owner-only reducer VF
invocation regressed `0.1900%`; the empty-ballot fast path improved `0.9580%`
once but regressed `0.2806%` on repeat; in-place sortable-key reuse regressed
`0.2779%`; and adjacent BF16 32-bit packed loads regressed `2.3554%`. Those
four source changes and their source-contract assertions are absent from the
retained example.

The retained source/executable SHA-256 pair is
`1b11b72738b789228db1230f9f1369b980a0f7da5599db27fe4fd6c510b7fc13` /
`c22f7c34b400978fe8db224148106d8662a446dd4cc150445f1ec583002b3b3c`.
All follow-up evidence is archived at
`/tmp/dsa-topk-followup-evidence-20260817.tar.gz`, SHA-256
`4bbe1350491d3be7218345a62ec4dffc36a137916c77fa701d9e0416bf417619`.

## Inclusive suffix histogram retention

The next retained candidate changes the high/low histogram values, but not the
workspace shape, from raw bin counts to inclusive suffix counts. Producer VFs
perform a fixed 256-bin hierarchical scan using eight warps. A reducer selects
bucket `b` when `suffix[b]` reaches the requested rank and `suffix[b + 1]` does
not. For the final threshold, the low reducer reads per-shard high-greater,
low-greater, and equal counts directly from adjacent suffix positions before
the existing width-16 shuffle prefix scan. Five VF calls, four `SyncAll`
boundaries, 64-block launch geometry, score traversal, GM publication, and
compact semantics remain unchanged.

On the same Ascend 950PR/CANN 9.2 boundary, ten alternating-order Default pairs
changed the median from `18.1195 us` to `14.6075 us`, a `19.3824%` reduction
with `10/10` pair wins. All rows reported block dimension 64 and current/rated
`1650/1650 MHz`; direct and profiled executions passed the score-set oracle.
The retained source/executable SHA-256 pair is
`df9424a9399b274390d638eff77208d6e9e25c02357480cda950ecd6383a5f1c` /
`a6d7b98bfc9c66187c03b95adeb18255a9cd077da503bcb68bea6b576bf8b689`.
Raw evidence remains at remote `/tmp/dsa-topk-suffix-histogram-exp-9PYzom` and
is archived on both hosts at
`/tmp/dsa-topk-suffix-histogram-evidence-20260817.tar.gz`, SHA-256
`65e83bf9501c51a400a4203ffe6d50acffaa6799b7a06d78dcc6955ba9481ba2`.

## Post-suffix attribution and warp-offset rejection

Five-prefix attribution was recollected on the retained suffix source. Ten
rotated-order Default rounds measured cumulative medians of `6.188`, `7.423`,
`10.614`, `12.498`, and `14.603 us`. Median increments were therefore `6.188`
us for input DMA plus high histogram, `1.235 us` for high threshold, `3.191 us`
for low histogram, `1.884 us` for low threshold/offsets, and `2.105 us` for
compact. Adjacent-boundary paired medians for stages two through five were
`1.183`, `3.103`, `1.8095`, and `2.074 us`; every paired increment was
positive. All 50 rows passed their stage oracle at block dimension 64 and
current/rated `1650/1650 MHz`.

A single-factor follow-up made lane zero of each warp sum the later warp totals
inside `histogram_to_inclusive_suffix`, then broadcast the sum with `asc_shfl`.
Five VF calls, four `SyncAll` boundaries, workspace, launch geometry, and all
other source remained unchanged. Although this removes repeated source-level
UB reads, ten alternating Default pairs changed the median from `14.6325 us`
to `15.1665 us`, a `3.6494%` regression. The candidate won `2/10` pairs and
its paired delta median was `+0.6265 us`. Direct and all profiled executions
passed the score-set oracle at `1650/1650 MHz`.

The exact lane-zero loop plus warp-broadcast form is rejected and absent from
the retained source. Baseline/candidate source SHA-256 values are
`df9424a9399b274390d638eff77208d6e9e25c02357480cda950ecd6383a5f1c` /
`4f331ecc3a5fd88393483d65d035f8cd1f6ae7e302977e50a5312113539f9ff8`.
Stage evidence is archived at
`/tmp/dsa-topk-suffix-stage-evidence-20260817.tar.gz`, SHA-256
`a6753b8a974f9b30d7192e38ddbcf3ba422adceaaf6dca4e994b2b6734eaeae3`;
A/B evidence is archived at
`/tmp/dsa-topk-warp-offset-evidence-20260817.tar.gz`, SHA-256
`e37bedbca78308e2b8a10ab897a94153dab068ff4b122c8eb9916de07c913021`.

## Register-resident suffix retention

The next candidate keeps the warp-local inclusive suffix in a register through
the existing higher-warp accumulation and returns the final value directly to
each producer's GM store. This removes the local suffix write, later UB
read-modify-write, final producer UB read, and the helper's second block
barrier. The following `publish_gm_for_next_stage()` retains its barrier before
DCCI, so every thread's GM publication remains ordered. Histogram algorithms,
workspace layout and types, VF calls, four `SyncAll` boundaries, reducers, and
compact semantics remain unchanged.

Two independent ten-pair Default rounds changed the medians from `14.5335 us`
to `14.3280 us` (`1.4140%`, `8/10` wins) and from `14.7225 us` to `14.4950 us`
(`1.5453%`, `7/10` wins plus one tie). Paired candidate-minus-baseline medians
were `-0.2215 us` and `-0.2065 us`. Across all 20 pairs, the raw medians were
`14.6080 us` and `14.3375 us`, a `1.8517%` reduction with `15/20` wins and one
tie. Both direct and all 40 profiled executions passed the score-set oracle;
every profile used block dimension 64 and current/rated `1650/1650 MHz`.

The retained source/executable SHA-256 pair is
`0d2afd39acab8106b37acf86e15966783b0440507487df84de48ffaacc0da88f` /
`3ed946705d333e44aff0f28f80b2fc5f805b3ff758d6e888580237d5035df880`.
Raw evidence remains at remote `/tmp/dsa-topk-register-suffix-exp-gN1x5h` and
is archived on both hosts at
`/tmp/dsa-topk-register-suffix-evidence-20260817.tar.gz`, SHA-256
`1fec59bd6b0827ce3621a540bc919917bb49289324c5d8c18e29a5e6639e23a9`.

## Rejected striped partial histograms

The next controlled experiment distributed the 32 producer warps over either
two or four 256-bin UB histogram replicas. Threads 0-255 merged corresponding
replica counters into registers before the retained inclusive-suffix scan.
The two layouts consumed 2,080 and 4,128 bytes of the existing 8 KiB scratch;
all score, workspace, launch, VF, synchronization, reducer, publication, and
compact contracts remained fixed.

Ten rotated-order Default triples changed the `14.3320 us` baseline median to
`14.8800 us` for two replicas and `14.7055 us` for four replicas. Those are
regressions of `3.8236%` and `2.6061%`; the candidates won `0/10` and `1/10`
pairs, with paired delta medians of `+0.5045 us` and `+0.3590 us`. Three direct
and all 30 profiled executions passed the score-set oracle, and every profile
reported 64 blocks at current/rated `1650/1650 MHz`.

Both candidates are rejected because replica initialization and per-bin merge
work cost more than the reduction in UB atomic contention. The repository
retains source SHA-256
`0d2afd39acab8106b37acf86e15966783b0440507487df84de48ffaacc0da88f`.
The rejected two-way/four-way source hashes are
`f2bc5f75635d99425839b22d43760b041fb2fa4dbca1b9cdd71d1d0041c2c2d7` /
`d8c9f1bdb008b47ed4aa5928af3a1fddf03680b72c72957a6b399409de244cfa`.
Raw evidence remains at remote
`/tmp/dsa-topk-striped-histogram-exp-20260817` and is archived on both hosts at
`/tmp/dsa-topk-striped-histogram-evidence-20260817.tar.gz`, SHA-256
`c4536899da8ba087bc32b19599215562961c7c5719fba41027f539ff9c29f375`.
Do not retry this exact two-way/four-way design without a materially changed
contention or compiler premise.

## Adjacent-bin reducer retention

The retained candidate removes the 256-bin combined-histogram UB round trip
from both threshold reducers. Each thread keeps its sum of the 16 shard suffix
values in a `uint32_t` register and executes one uniform width-32
`asc_shfl_down`. Lane zero of each of the eight warps publishes its inclusive
count to an eight-word UB boundary array before the existing block barrier.
Lanes 0-30 use the shuffle result, seven lane-31 threads read the next warp's
published value, and thread 255 uses zero. The low reducer retains
`shard_greater_counts = combined_histogram + kHistogramWords`, so its later
greater/equal arrays and synchronization remain unchanged.

The baseline and candidate were built independently from repository HEAD
`5a560c22424b9c93a9829c618230110aef87dfc6` on Ascend 950PR with CANN 9.2.0,
Bisheng clang 15.0.5, `dav-3510`, and `msopprof` 26.2.0. Both direct executions
passed the four-row score-set oracle. One pilot pair validated the collection
path and was excluded from statistics. Ten formal alternating-order pairs were
then collected under one lock with default profiler warmup,
`--aic-metrics=Default --launch-count=1`, and no `--kernel-name`. All 20 formal
profiles passed the oracle and contained exactly one target row with block
dimension 64 and current/rated `1650/1650 MHz`.

| Variant | Ten Task Duration samples (us) | Median (us) | Min (us) | Max (us) |
| --- | --- | ---: | ---: | ---: |
| Register-suffix baseline | 14.345, 14.078, 14.542, 14.117, 14.238, 14.135, 14.511, 14.427, 14.403, 14.151 | 14.2915 | 14.078 | 14.542 |
| Adjacent-bin exchange | 14.054, 14.186, 13.994, 14.195, 14.095, 14.203, 14.113, 14.211, 14.179, 14.203 | 14.1825 | 13.994 | 14.211 |

The ten candidate-minus-baseline deltas were `-0.291`, `+0.108`, `-0.548`,
`+0.078`, `-0.143`, `+0.068`, `-0.398`, `-0.216`, `-0.224`, and `+0.052 us`.
Their median was `-0.1795 us`; the candidate won `6/10` pairs with no ties and
reduced the unpaired median by `0.7627%`. It therefore meets both predefined
retention gates.

Baseline/candidate source SHA-256 values are
`0d2afd39acab8106b37acf86e15966783b0440507487df84de48ffaacc0da88f` /
`d0f5821eeb739d00c2ab783c0bc19a401a1500e3b5918746b0c8d07ded13a0ae`;
their executable hashes are
`811298087d46264f1f0212ec33e9f108578b0885957201d60c1636f2d630bfb1` /
`ad85d89f66cddb256cb1dbcf1a58956c8e48c3fa89d6c3f66e1420a7a0fdb418`.
Raw evidence remains at remote
`/tmp/dsa-topk-adjacent-bin-reducer-exp-20260817`. The 795-entry archive is
available on both hosts at
`/tmp/dsa-topk-adjacent-bin-reducer-evidence-20260817.tar.gz`, SHA-256
`7bce8463a050432810c8444245d1d25e9d03753dce16e738918c4eb2dfe3eff1`.

## Three-direction follow-up results

Commit `5611bbc` is the immutable baseline for three independent follow-ups.
Every candidate remains confined to this standalone fixed-shape example and
preserves BF16 `[4, 32768]` to INT32 `[4, 2048]` score-set semantics, one
kernel launch, host-generated inputs, five logical VF stages, and the existing
GM workspace contract. Production SIMT v2 is not modified.

### Experiment 1: context-shard geometry

Compare the retained 16-shard geometry with 8 and 32 shards while keeping the
same distributed radix algorithm. The host and kernel shard constants change
together. The three variants launch 32, 64, and 128 blocks; each block owns
4,096, 2,048, or 1,024 BF16 scores respectively. Their calculated dynamic UB
requests are 32,768, 45,056, and 75,776 bytes:

| Shards | Blocks | Score tile | Two reducer matrices | Shared scratch | Total UB |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 32 | 8,192 B | 16,384 B | 8,192 B | 32,768 B |
| 16 | 64 | 4,096 B | 32,768 B | 8,192 B | 45,056 B |
| 32 | 128 | 2,048 B | 65,536 B | 8,192 B | 75,776 B |

The reducer scan width follows the compile-time shard count. No four-block
ownership, `16384 -> 18432` split, or per-chunk barrier algorithm from the
rejected experiment is reused. This experiment isolates the tradeoff between
producer parallelism, per-block score work, reducer matrix traffic, and the
number of participating blocks.

### Experiment 2: selected-high candidate offsets

Keep 16 shards and append one block-private UB array containing at most 2,048
`uint16_t` local offsets plus an aligned `uint32_t` count. During the existing
low-histogram traversal, every warp uses ballot plus one lane-zero UB atomic
reservation to append offsets whose high radix byte is greater than or equal
to `selected_high`. The low histogram itself still counts only the equal
high-byte bucket. After the low threshold and shard offsets are published,
compact traverses only the retained offsets and reloads the corresponding
score from the resident score tile.

For the deterministic example input, each row has 12,296 retained candidates
and each 16-shard block has 756 through 781. The candidate therefore
reduces compact classification from 2,048 scores per block to roughly 37% of
that work, while adding 4,100 bytes of persistent UB and 64 warp reservations
per block. The candidate array has full-shard capacity, so correctness does not
depend on the measured distribution.

### Experiment 3: MTE3 histogram publication

Keep 16 shards and all four `AscendC::SyncAll()` dependency boundaries. The
high and low producer VFs write their final 256 inclusive suffix counts to the
existing 1 KiB local-histogram UB region. After each VF returns, the outer
kernel orders `PIPE_V -> PIPE_MTE3`, performs one aligned 1 KiB UB-to-GM copy,
waits for MTE3 completion, and then enters the existing device-wide barrier.
This replaces 256 scalar GM stores and one entire-data-cache DCCI per producer
block. Reducer state and offset publication remains unchanged.

The four device-wide barriers are required by the dependency chain `high
histogram -> high threshold -> low histogram -> low threshold/offsets ->
compact`. Removing one would require a new inter-core protocol, which is
outside the allowed C API, Tensor API, SIMT API, and existing transitional
barrier boundary. This experiment therefore tests the removable producer
publication cost without weakening cross-core visibility.

### Measurement and retention

Build every variant independently for Ascend 950PR `dav-3510` with CANN 9.2.0
and Bisheng clang 15.0.5. Direct execution must pass the four-row oracle before
profiling. Use only default profiler warmup and
`msopprof --aic-metrics=Default --launch-count=1`, without `--kernel-name`.
Every accepted profile must contain exactly one target kernel row, the expected
32/64/128 block dimension for its geometry, and current/rated `1650/1650 MHz`.

Experiment 1 planned ten rotated-order baseline/8-shard/32-shard triples. The
128-block 32-shard binary instead reproducibly hung at the existing hardware
`SyncAll()` boundary, while a diagnostic that retained the 32-shard data layout
but launched only 64 blocks exited. The invalid 128-block candidate was not
profiled. Baseline and 8-shard then used ten alternating-order pairs.
Experiment 2 also used ten alternating-order pairs. Experiment 3 failed direct
correctness and was not profiled. A candidate was independently retained only
for a lower median plus at least 6 of 10 paired wins.

All source trees, build logs, direct outputs, commands, parsed rows, raw OPPROF
trees, rejected candidates, and summary statistics are retained remotely and
in a local evidence archive. The implementation plan remains under `/tmp` and
is excluded from that archive. Experimental variants are created outside the
repository; a winning combined result receives a source-contract RED/GREEN
cycle before it is applied to this example. No factor passed, so no combined
candidate was constructed and the repository source remains the `5611bbc`
baseline.

### Measured outcome

All accepted profiles used unmodified profiler warmup,
`--aic-metrics=Default --launch-count=1`, and no `--kernel-name`. Every parsed
row contained exactly one target launch at current/rated `1650/1650 MHz`, with
64 blocks for baseline/candidate-offset and 32 blocks for 8 shards. Direct and
profiled accepted runs passed the four-row score-set oracle.

| Comparison | Baseline samples (us) | Candidate samples (us) | Medians (us) | Paired result |
| --- | --- | --- | --- | --- |
| 16 shards vs 8 shards | 14.246, 14.335, 14.084, 14.102, 14.002, 14.110, 14.094, 14.101, 14.047, 14.209 | 16.358999, 16.478001, 16.516001, 16.549, 16.413, 16.381001, 16.476, 16.480, 16.594, 16.420 | 14.1015 vs 16.4770005 | candidate `+2.3805 us`, 0/10 wins |
| baseline vs candidate offsets | 14.151, 14.080, 14.174, 14.037, 14.277, 14.187, 13.972, 14.236, 14.227, 14.097 | 14.933, 14.899, 14.951, 14.871, 14.883, 14.775, 14.937, 14.875, 15.034, 14.966 | 14.1625 vs 14.916 | candidate `+0.7945 us`, 0/10 wins |

The 32-shard source/executable pair is
`e8534aa00b926657777f31a0785dc044b920f6904f6504a176d6d6382238897e` /
`245c10662d3b25ad4ee1d18deec8c5b19545b3caafd8b93e9c10b6746999b776`.
The corrected candidate-offset pair is
`0476d46fcdba931fe6e7ea87f562c941ef9d2f065eec47763b400971325c0627` /
`f91f187785d7db26941fc6821fa2a9f0dd55f7f05b24a3972994fc5ae888219f`.
The MTE3 pair is
`344182237997e08d8826fa931c4d31b760a2da4d0954c43a3f84cc0dceb02844` /
`cc6c9ce98417ccf65bc75d027806b8672d8da62d5a1d1ff57c3e17c545e75c74`.

MTE3 publication produced the same final radix state and shard offsets as the
baseline, but compact emitted below-threshold indices. Restoring producer
thread convergence, reloading score UB, adding compact-output DCCI, and adding
consumer-side DCCI did not recover correctness. Do not retry the same MTE3
publication structure without a changed synchronization or buffer-lifetime
premise. Do not retry 8-shard geometry or the selected-high candidate cache
without a changed work-distribution or reservation-cost premise.

Raw evidence remains at remote
`/tmp/dsa-topk-three-directions-exp-20260817`. The 1,939-entry archive is on
both hosts at `/tmp/dsa-topk-three-directions-evidence-20260817.tar.gz`,
SHA-256
`007125808dee427f0257004de521fd23510a64edf4acc3f70ea1411588ca1e74`.
The implementation plan remains only at
`/tmp/dsa-topk-three-directions-plan.md` and is not archived.
