# DSA Decode TopK comparison

This standalone Ascend 950 experiment compares vLLM-Ascend's Arch35 BF16 16K
trunk TopK against the distributed TopK selected by CannBench SIMT v2 for V3.2
decode. Both map BF16 scores [4, 32768] to INT32 indices [4, 2048] with one
kernel launch and use only ACL Runtime on the host. Output ordering is
intentionally not compared, and equal-score index identity is intentionally not
compared.

The current SIMT executable includes seven retained example-only fixed-shape
optimizations. Each of its 64 blocks copies one 2,048-element BF16 shard from
GM to a dedicated 4 KiB UB tile once; the high-byte histogram, low-byte
histogram, and compact VFs all reuse it. The four reducer blocks also stage
their 16 x 256 high and low histogram matrices in UB. Each histogram producer
converts its 256 raw counters to inclusive suffix counts with an eight-warp
hierarchical scan. Each suffix result remains in a register across the
warp-total merge and is written directly to GM; the following publication
barrier replaces a second helper barrier and final UB round trip. The reducers
keep their combined inclusive counts in registers, exchange adjacent lanes with
a warp shuffle, and use only eight UB words at warp boundaries. They select
thresholds from adjacent suffix positions, and the low reducer obtains each
shard's greater/equal counts from
constant-position suffix reads instead of scanning both matrices again. The
compact VF uses warp-level ballot and UB atomic output reservations instead of
a block-wide ping-pong prefix scan. The low reducer uses a width-16 warp
shuffle scan for shard output offsets. Separate per-warp-private histogram and
follow-up tuning candidates were implemented and validated but are not retained
when they regressed or produced unstable device task time. Production SIMT v2
is not modified.

Build and collect with:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
flock -x /tmp/cannbench-dsa-stage-comparison.lock bash -lc 'RESULT_ROOT="$PWD/evidence/<unique-run>" bash scripts/run.sh'
```

The script configures `dav-3510`, verifies both executables first, then creates
five independent `msopprof` `Default` collections per implementation. It uses
only the single target row's `Task Duration`, requires launch-count one,
production block dimensions (vLLM-Ascend 4, SIMT v2 64), and frequency parity,
and retains every raw collection below `evidence/`.

## Warp-atomic compact evidence

Collected on 2026-08-17 on `root@121.41.199.170:20002`, using Ascend 950PR
device 0, CANN 9.2.0, Bisheng clang 15.0.5, `dav-3510`, and `msopprof` 26.2.0.
The baseline is the retained `4446005` stack. The accepted candidate changes
only its compact VF: it replaces the ten-round 1,024-thread ping-pong scan with
two block-private UB counters, warp ballots, one UB atomic reservation per
nonempty warp/category, and a lane-zero shuffle broadcast. It classifies and
writes each 2,048-element shard in one pass and retains the baseline 8 KiB
compact UB request.

The mechanism is adapted from the warp-compaction variant in PyTorch
`aten/src/ATen/native/cuda/TensorTopK.cu` at commit
`893b6406afc1a6384ab6fae8a2247d03cc230d87`. That code is under PyTorch's ROCm
branch; the NVIDIA CUDA branch in the same file still uses
`exclusiveBinaryPrefixScan`.

- Profiler mode: ten alternating-order baseline/candidate pairs,
  `--aic-metrics=Default`, unmodified warmup (reported as 5),
  `--launch-count=1`, and no `--kernel-name`
- Launch and frequency parity: every accepted row reports 64 blocks and
  current/rated `1650/1650 MHz`
- Correctness: both direct executions and all 20 profiled executions passed the
  four-row score-set oracle
- Baseline source/executable SHA-256:
  `0b2a13a2feac2fa2d353a23e17a1570b22de7f9602e8a0d8d951ab8365e7cf51` /
  `4339779baa096350ba9a7f96be8e096e6cd596e0e1f3c1dbec931d11984b4a42`
- Accepted source/executable SHA-256:
  `a98d73005eac24cf6f7bd0be3dc13677c3819eb80e8e77cd68c285971f032731` /
  `d4d9da5ae54792f1c629b5ef470888503a540914777b38760fa144e8a19c9025`

| Variant | Ten Task Duration samples (us) | Median (us) | Min (us) | Max (us) | Pair wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Block-scan baseline | 20.552, 20.587999, 20.608, 20.308001, 20.840, 20.455, 20.027, 20.632, 20.062, 20.398001 | 20.5035 | 20.027 | 20.840 | 0/10 |
| Warp-atomic compact | 18.686001, 18.677999, 18.868, 18.959, 18.885, 18.691, 19.278, 19.134001, 18.736, 18.823999 | 18.8459995 | 18.677999 | 19.278 | 10/10 |

The accepted candidate reduced median task duration by `8.0840%` and won all
ten pairs.

After that result, a separate candidate reduced total dynamic UB from about
44 KiB to 37.125 KiB by removing the now-unused 8 KiB scan reservation. It was
correct but did not improve the accepted algorithm:

| Variant | Five Task Duration samples (us) | Median (us) | Pair wins |
| --- | --- | ---: | ---: |
| Accepted 8 KiB reservation | 18.495001, 18.739, 19.139999, 18.930, 18.719 | 18.739 | 4/5 |
| Reduced scratch candidate | 18.663, 18.993, 18.849001, 19.115, 18.874001 | 18.874001 | 1/5 |

The reduced-scratch candidate regressed the median by `0.7204%` and is not
retained. Its source/executable SHA-256 pair is
`187d6d7ce2668e1b4773811283718d2d884a1bebabf07c44d161bfb8121741e7` /
`caf119c71e56f3ab40781ab70b4b35935a764f99bc79a4010b826d5a5d306e74`.

Raw evidence remains at remote
`/tmp/cannbench-topk-warp-atomic-Qyq59K`. The 1,016-entry archive is available
on both hosts at `/tmp/dsa-topk-warp-atomic-evidence-20260817.tar.gz`, SHA-256
`22e4ce773d40559262426f3236cddc51b811adb7344f5226666c44b09f83d3e0`.

## Warp-shuffle reducer evidence

Follow-up tuning on 2026-08-17 used the retained warp-atomic source as its
frozen baseline and kept the same Ascend 950PR device, CANN 9.2.0, Bisheng
clang 15.0.5, `dav-3510`, and `msopprof` 26.2.0 boundary. Every accepted row
used unmodified profiler warmup, `--aic-metrics=Default`,
`--launch-count=1`, no `--kernel-name`, one 64-block target launch, and
current/rated `1650/1650 MHz`. Direct runs and all profiled runs passed the
four-row score-set oracle.

The retained candidate replaces the low reducer's 16 serial prior-shard
greater/equal prefix loops with four width-16 `asc_shfl_up` steps. Warp 0
executes the collective uniformly; lanes 0-15 publish exclusive offsets and
the total-greater count. All histogram algorithms, score traversal, compact
logic, five VF calls, four `SyncAll` boundaries, and GM visibility operations
remain unchanged.

Two independent five-pair rounds produced:

| Variant | Ten Task Duration samples (us) | Median (us) | Pair wins |
| --- | --- | ---: | ---: |
| Warp-atomic baseline | 19.115999, 18.922001, 18.903, 19.086, 18.979, 19.209, 18.849001, 18.712, 18.791, 19.351999 | 18.9505005 | 0/10 |
| Warp-shuffle reducer | 18.146, 18.235001, 18.739, 18.179001, 18.389999, 17.993999, 18.100, 18.232, 18.371, 18.365999 | 18.2335005 | 10/10 |

The retained candidate reduced the combined median by `3.7835%`. Relative to
the earlier, separately collected block-scan compact median of `20.5035 us`,
the current `18.2335005 us` median is `11.0713%` lower. That historical ratio
is not a paired A/B result and is reported only for orientation.

Four independently measured follow-ups are rejected:

| Candidate | Baseline median (us) | Candidate median (us) | Delta | Pair wins | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Invoke reducer VFs only on four row-owner blocks | 18.948999 | 18.985001 | +0.1900% | 2/5 | Rejected |
| Return early for an empty compact ballot, first round | 19.205999 | 19.021999 | -0.9580% | 3/5 | Repeated |
| Return early for an empty compact ballot, repeat | 19.247999 | 19.302000 | +0.2806% | 3/5 | Rejected as unstable |
| Transform each BF16 sortable key once in UB | 19.073000 | 19.125999 | +0.2779% | 1/5 | Rejected |
| Load two adjacent BF16 scores per 32-bit UB access | 18.171000 | 18.599001 | +2.3554% | 0/5 | Rejected |

Baseline source/executable SHA-256 is
`a98d73005eac24cf6f7bd0be3dc13677c3819eb80e8e77cd68c285971f032731` /
`d4d9da5ae54792f1c629b5ef470888503a540914777b38760fa144e8a19c9025`.
Retained source/executable SHA-256 is
`1b11b72738b789228db1230f9f1369b980a0f7da5599db27fe4fd6c510b7fc13` /
`c22f7c34b400978fe8db224148106d8662a446dd4cc150445f1ec583002b3b3c`.
Raw evidence remains at remote `/tmp/cannbench-topk-followup-hWn76l`; its
2,286-entry archive is available on both hosts at
`/tmp/dsa-topk-followup-evidence-20260817.tar.gz`, SHA-256
`4bbe1350491d3be7218345a62ec4dffc36a137916c77fa701d9e0416bf417619`.

## Inclusive suffix histogram evidence

Stage-prefix attribution on 2026-08-17 identified the low threshold/offset
boundary as the largest separable post-producer stage. The retained candidate
changes both shard histogram workspaces from raw bin counts to inclusive suffix
counts without changing their size. Each producer performs a 256-bin scan in
eight warps and writes the same 16 x 256 matrix shape. High and low reducers
then select their threshold where `suffix[b]` reaches the rank while
`suffix[b + 1]` does not. The low reducer also reads each shard's high-greater,
low-greater, and equal counts from those two adjacent suffix positions. This
removes its full high/low bin rescan and partial-group reduction.

The frozen baseline is commit `4b99b0a`. Both sources retain one kernel launch,
64 blocks, five VF calls, four `SyncAll` boundaries, the existing GM workspace,
and the warp-atomic compact implementation. Ten alternating-order pairs used
unmodified profiler warmup, `--aic-metrics=Default`, `--launch-count=1`, no
`--kernel-name`, and current/rated `1650/1650 MHz`. Direct runs and all 20
profiled runs passed the four-row score-set oracle.

| Variant | Ten Task Duration samples (us) | Median (us) | Min (us) | Max (us) | Pair wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Raw histogram baseline | 18.271, 18.100, 18.259001, 17.979, 18.546, 18.132999, 18.106001, 18.242001, 18.002001, 18.083 | 18.1195 | 17.979 | 18.546 | 0/10 |
| Inclusive suffix histogram | 14.370, 14.925, 15.250, 14.185, 14.833, 14.304, 14.811, 14.547, 14.668, 14.313 | 14.6075 | 14.185 | 15.250 | 10/10 |

The retained suffix candidate reduced median task duration by `19.3824%` and
won all ten pairs. Its source/executable SHA-256 pair is
`df9424a9399b274390d638eff77208d6e9e25c02357480cda950ecd6383a5f1c` /
`a6d7b98bfc9c66187c03b95adeb18255a9cd077da503bcb68bea6b576bf8b689`.
Raw evidence remains at remote
`/tmp/dsa-topk-suffix-histogram-exp-9PYzom`; the archive is available on both
hosts at `/tmp/dsa-topk-suffix-histogram-evidence-20260817.tar.gz`, SHA-256
`65e83bf9501c51a400a4203ffe6d50acffaa6799b7a06d78dcc6955ba9481ba2`.

## Post-suffix stage attribution and rejected warp-offset broadcast

The five-prefix attribution was repeated after retaining suffix histograms,
because the earlier raw-histogram attribution no longer represented the
kernel. Ten rotated-order Default rounds produced a stage-5 median of
`14.603 us`, consistent with the standalone retained result. All 50 rows used
64 blocks at current/rated `1650/1650 MHz` and passed the appropriate stage
oracle.

| Cumulative boundary | Median (us) | Increment from prior median (us) | Paired increment median (us) |
| --- | ---: | ---: | ---: |
| Input DMA + high histogram + publication + `SyncAll` | 6.188 | 6.188 | n/a |
| High threshold | 7.423 | 1.235 | 1.183 |
| Low histogram | 10.614 | 3.191 | 3.103 |
| Low threshold and offsets | 12.498 | 1.884 | 1.8095 |
| Compact | 14.603 | 2.105 | 2.074 |

The next experiment changed only the second half of
`histogram_to_inclusive_suffix`. Instead of all 32 lanes in each warp summing
higher-warp totals, lane zero performed that loop and broadcast the result with
`asc_shfl`. This reduced the source-level UB reads but regressed device time,
so it is not retained.

| Variant | Ten Task Duration samples (us) | Median (us) | Min (us) | Max (us) | Pair wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Retained per-lane accumulation | 14.613, 14.360, 14.710, 14.652, 14.808, 14.290, 14.536, 14.328, 14.854, 14.670 | 14.6325 | 14.290 | 14.854 | 8/10 |
| Rejected lane-zero broadcast | 15.353, 15.189, 14.854, 15.350, 14.491, 15.144, 14.579, 15.369, 14.815, 15.225 | 15.1665 | 14.491 | 15.369 | 2/10 |

The rejected candidate was `3.6494%` slower, with a paired candidate-minus-
baseline median of `+0.6265 us`. Baseline/candidate source SHA-256 values are
`df9424a9399b274390d638eff77208d6e9e25c02357480cda950ecd6383a5f1c` /
`4f331ecc3a5fd88393483d65d035f8cd1f6ae7e302977e50a5312113539f9ff8`;
binary SHA-256 values are
`224c836088f990c046bd3911b0b78acd62cf62d8124dd827658c856323261124` /
`8400d431d77702b7c9393c9a8c97ea0e233553ea01d70b1f86c45cf0547357c6`.
Do not retry this exact lane-zero loop plus warp-broadcast transformation
without a materially different scan or compiler hypothesis.

Raw stage evidence remains at remote
`/tmp/dsa-topk-suffix-stage-exp-PxqTcF` and is archived on both hosts at
`/tmp/dsa-topk-suffix-stage-evidence-20260817.tar.gz`, SHA-256
`a6753b8a974f9b30d7192e38ddbcf3ba422adceaaf6dca4e994b2b6734eaeae3`.
Raw A/B evidence remains at remote `/tmp/dsa-topk-warp-offset-exp-f8hdY6`
and is archived on both hosts at
`/tmp/dsa-topk-warp-offset-evidence-20260817.tar.gz`, SHA-256
`e37bedbca78308e2b8a10ab897a94153dab068ff4b122c8eb9916de07c913021`.

## Register-resident suffix evidence

The retained follow-up keeps each thread's warp-local suffix in a register,
adds the existing per-thread higher-warp total after the required warp-total
barrier, and returns the final value directly to the high/low producer's GM
store. It removes the two final UB accesses per bin and the helper's second
block barrier. The immediately following `publish_gm_for_next_stage()` barrier
still orders every GM store before thread zero performs DCCI publication. No
VF call, `SyncAll`, workspace, histogram type, reducer, or compact logic
changes.

Two independent ten-pair Default rounds used the same frozen binaries,
alternating order, unmodified profiler warmup, `--launch-count=1`, no
`--kernel-name`, 64 blocks, and current/rated `1650/1650 MHz`. Both direct
executions and all 40 profiled executions passed the four-row score-set oracle.

| Round and variant | Ten Task Duration samples (us) | Median (us) | Pair wins |
| --- | --- | ---: | ---: |
| Round 1 baseline | 14.506, 14.690, 14.404, 14.704, 14.345, 14.403, 14.578, 14.561, 14.447, 14.645 | 14.5335 | 2/10 |
| Round 1 register suffix | 14.443, 14.339, 14.647, 14.235, 14.227, 14.620, 14.139, 14.055, 14.336, 14.320 | 14.3280 | 8/10 |
| Round 2 baseline | 14.872, 14.795, 14.532, 16.587, 14.638, 14.477, 14.826, 14.703, 14.742, 14.497 | 14.7225 | 3/10 plus one tie |
| Round 2 register suffix | 15.108, 14.482, 14.532, 14.256, 14.325, 14.259, 14.308, 14.508, 14.600, 14.631 | 14.4950 | 7/10 plus one tie |

The two rounds reduced their medians by `1.4140%` and `1.5453%`; paired delta
medians were `-0.2215 us` and `-0.2065 us`. Across all 20 pairs, baseline and
candidate medians were `14.6080 us` and `14.3375 us`, a `1.8517%` reduction,
with `15/20` candidate wins and one tie. The candidate is retained because both
independent rounds passed the median and paired-win gates.

Baseline/candidate source SHA-256 values are
`df9424a9399b274390d638eff77208d6e9e25c02357480cda950ecd6383a5f1c` /
`0d2afd39acab8106b37acf86e15966783b0440507487df84de48ffaacc0da88f`;
binary SHA-256 values are
`224c836088f990c046bd3911b0b78acd62cf62d8124dd827658c856323261124` /
`3ed946705d333e44aff0f28f80b2fc5f805b3ff758d6e888580237d5035df880`.
Raw evidence remains at remote `/tmp/dsa-topk-register-suffix-exp-gN1x5h`
and is archived on both hosts at
`/tmp/dsa-topk-register-suffix-evidence-20260817.tar.gz`, SHA-256
`1fec59bd6b0827ce3621a540bc919917bb49289324c5d8c18e29a5e6639e23a9`.

## Rejected striped partial-histogram experiment

A three-way follow-up tested whether reducing UB atomic contention could
improve the two histogram producer stages. The two candidates distributed the
32 producer warps over either two or four 256-bin `uint32_t` histograms. After
the atomic pass, threads 0-255 merged the corresponding replica counters into
registers and fed those values directly into the retained inclusive-suffix
scan. The two-way and four-way layouts used 2,080 and 4,128 bytes of the
unchanged 8 KiB stage scratch. Score traversal, radix buckets, GM workspace,
64 blocks, five VF calls, four `SyncAll` boundaries, reducers, publication,
and compaction remained unchanged.

Ten rotated-order triples used `--aic-metrics=Default --launch-count=1`, the
default profiler warmup, and no `--kernel-name`. All three direct executions
and all 30 profiled executions passed the four-row score-set oracle. Every
parsed target row reported 64 blocks and current/rated `1650/1650 MHz`.

| Variant | Ten Task Duration samples (us) | Median (us) | Min (us) | Max (us) | Wins vs baseline |
| --- | --- | ---: | ---: | ---: | ---: |
| Register-suffix baseline | 14.195, 14.402, 14.211, 14.359, 14.532, 14.548, 14.270, 14.332, 14.332, 14.157 | 14.3320 | 14.157 | 14.548 | n/a |
| Two-way striped | 14.664, 14.724, 14.924, 15.278, 15.016, 14.836, 14.795, 15.103, 14.792, 15.178 | 14.8800 | 14.664 | 15.278 | 0/10 |
| Four-way striped | 14.926, 14.698, 14.806, 14.578, 14.627, 14.543, 14.796, 14.713, 14.769, 14.494 | 14.7055 | 14.494 | 14.926 | 1/10 |

The two-way candidate regressed the median by `3.8236%`, with a paired delta
median of `+0.5045 us`. The four-way candidate regressed by `2.6061%`, with a
paired delta median of `+0.3590 us`. Both are rejected: the reduced atomic
contention did not repay replica initialization and per-bin merge work. The
repository source is restored to the register-suffix baseline, and this exact
striped experiment must not be retried without a changed contention or
compiler premise.

Baseline/two-way/four-way source SHA-256 values are
`0d2afd39acab8106b37acf86e15966783b0440507487df84de48ffaacc0da88f` /
`f2bc5f75635d99425839b22d43760b041fb2fa4dbca1b9cdd71d1d0041c2c2d7` /
`d8c9f1bdb008b47ed4aa5928af3a1fddf03680b72c72957a6b399409de244cfa`;
their executable hashes are
`811298087d46264f1f0212ec33e9f108578b0885957201d60c1636f2d630bfb1` /
`f38adcdc2d4464fadf1fb27a3e81b46eabbe1349bb3245f203eafaf5975acae2` /
`66b60f45edd0bfb3dace754804d2f415c11ed340f8dc2659bcbfcafaf0c173f0`.
Raw evidence remains at remote
`/tmp/dsa-topk-striped-histogram-exp-20260817` and is archived on both hosts at
`/tmp/dsa-topk-striped-histogram-evidence-20260817.tar.gz`, SHA-256
`c4536899da8ba087bc32b19599215562961c7c5719fba41027f539ff9c29f375`.

## Adjacent-bin reducer retention

The retained reducer follow-up keeps each thread's sum of the 16 shard suffix
values in a register. A uniform width-32 shuffle supplies the next lane's
count; lane zero of each warp publishes one value to an eight-word UB boundary
array, so only seven warp-tail threads read the next warp's value. Thread 255
uses zero. This replaces each reducer's 256 UB writes and adjacent-bin rereads
without changing the histogram producers, GM workspace, 64 blocks, five VF
calls, four `SyncAll` boundaries, DCCI publication, shard offsets, or compact
path.

Ten alternating-order pairs used the default profiler warmup,
`--aic-metrics=Default --launch-count=1`, and no `--kernel-name`. All 20
profiled executions and both direct executions passed the four-row score-set
oracle; every parsed target row reported one 64-block launch at current/rated
`1650/1650 MHz`.

| Variant | Ten Task Duration samples (us) | Median (us) | Min (us) | Max (us) | Pair wins |
| --- | --- | ---: | ---: | ---: | ---: |
| Register-suffix baseline | 14.345, 14.078, 14.542, 14.117, 14.238, 14.135, 14.511, 14.427, 14.403, 14.151 | 14.2915 | 14.078 | 14.542 | 4/10 |
| Adjacent-bin exchange | 14.054, 14.186, 13.994, 14.195, 14.095, 14.203, 14.113, 14.211, 14.179, 14.203 | 14.1825 | 13.994 | 14.211 | 6/10 |

The median decreased by `0.7627%`; the paired candidate-minus-baseline median
was `-0.1795 us`. The candidate meets the predefined lower-median and `6/10`
win gates and is retained. Baseline/candidate source SHA-256 values are
`0d2afd39acab8106b37acf86e15966783b0440507487df84de48ffaacc0da88f` /
`d0f5821eeb739d00c2ab783c0bc19a401a1500e3b5918746b0c8d07ded13a0ae`;
their executable hashes are
`811298087d46264f1f0212ec33e9f108578b0885957201d60c1636f2d630bfb1` /
`ad85d89f66cddb256cb1dbcf1a58956c8e48c3fa89d6c3f66e1420a7a0fdb418`.
Raw evidence remains at remote
`/tmp/dsa-topk-adjacent-bin-reducer-exp-20260817` and is archived on both hosts
at `/tmp/dsa-topk-adjacent-bin-reducer-evidence-20260817.tar.gz`, SHA-256
`7bce8463a050432810c8444245d1d25e9d03753dce16e738918c4eb2dfe3eff1`.

## Rejected four-block SIMT experiment

An unpublished experiment tested whether SIMT VF code could benefit from the
same four-row-block ownership and logical score partition as the vLLM-Ascend
implementation. It retained the fixed BF16 `[4, 32768]` to INT32 `[4, 2048]`
contract and one-kernel boundary, assigned one block to each row, processed the
first `16384` scores and then merged the remaining `2048 + 16384 = 18432`
scores, and used two high/low-byte radix-threshold stages. Each stage built a
256-bin histogram with UB atomics; greater/equal compaction used ballot,
warp-prefix calculation, and repeated block barriers for each 1024-element
chunk. The two compact stages processed 68 chunks in total.

This matched vLLM-Ascend's logical radix phases and data split, but not its SIMD
primitives or dataflow. On Ascend 950PR it was more than ten times slower than
the retained 64-block distributed SIMT v2 implementation, so the source,
target, run-script integration, and source-contract tests were removed rather
than retained as an alternative example.

The comparison was collected on 2026-08-17 at
`root@121.41.199.170:20002`, device 0, with CANN 9.2.0, Bisheng clang 15.0.5,
`dav-3510`, and `msopprof` 26.2.0. All implementations used one kernel launch;
vLLM-Ascend and the rejected candidate used four blocks, while retained SIMT
v2 used 64. Every row used unmodified profiler warmup (reported as 5),
`--aic-metrics=Default`, `--launch-count=1`, no `--kernel-name`, and
current/rated frequency `1650/1650 MHz`. All direct and profiled executions
passed the score-set oracle.

| Implementation | Blocks | Ten Task Duration samples (us) | Median (us) |
| --- | ---: | --- | ---: |
| vLLM-Ascend | 4 | 7.772, 7.776, 7.632, 7.690, 7.814, 7.646, 7.769, 7.578, 7.662, 7.563 | 7.6760 |
| Rejected four-block SIMT | 4 | 215.292999, 212.688004, 212.984985, 212.983994, 212.022003, 215.410004, 215.532990, 211.776001, 212.951004, 212.990005 | 212.9844895 |
| Retained distributed SIMT v2 | 64 | 20.849001, 20.868000, 20.840000, 20.662001, 20.782000, 20.454000, 20.875000, 21.077999, 20.837999, 20.809999 | 20.8389995 |

The rejected candidate won `0/10` paired measurements against either baseline.
Its median was `27.7468x` vLLM-Ascend (`+2674.6807%`) and `10.2205x` retained
distributed SIMT v2 (`+922.0476%`). Source/executable SHA-256 pairs are:

- vLLM-Ascend: `df4b560fe8809dbb2b5099bc8e21ff4c25bfc466b8abf8f61a7ab188912e06d8` /
  `855ce09844b8c846f857ae2a4c2e434b90d1068afa98664f5db713851ce9bcb9`
- Rejected four-block SIMT:
  `a21d11c7abb09c1ab70592d6059c8f97d629dbc2f2b4817e98b437cc68483f62` /
  `e89c115e35df58968a373eb47427cc4baafe023376dff8781d21ce82f69736ba`
- Retained distributed SIMT v2:
  `0b2a13a2feac2fa2d353a23e17a1570b22de7f9602e8a0d8d951ab8365e7cf51` /
  `4339779baa096350ba9a7f96be8e096e6cd596e0e1f3c1dbec931d11984b4a42`

Raw evidence remains at remote
`/tmp/cannbench-topk-simt-vllm-partition-iTc9ET`. The 869-entry archive is
available on both hosts at
`/tmp/dsa-topk-simt-vllm-partition-evidence-20260817.tar.gz`, SHA-256
`bc2d63f2f7f495f8b638c4ccaec6205cdaadf8a5b7d4cfd967abe55c1451352f`.

Do not recreate or tune this exact combination: four row blocks, the
`16384 -> 18432` split, UB atomic histograms, and per-chunk
ballot/warp-prefix/block-barrier compaction. Future SIMT TopK work must start
from the retained 64-block distributed SIMT v2 path. Four-block ownership may
be reconsidered only with a materially different inner implementation and a
written bottleneck hypothesis, for example warp-atomic output reservation that
eliminates per-chunk block synchronization.

## Four-candidate tuning evidence

Collected on 2026-08-17 on `root@121.41.199.170:20002`, using Ascend 950PR
device 0, CANN 9.2.0, Bisheng clang 15.0.5, `dav-3510`, and `msopprof` 26.2.0.
Every accepted row used the unmodified profiler warmup (reported as 5),
`--aic-metrics=Default`, `--launch-count=1`, 64 blocks, and current/rated
`1650/1650 MHz`. Both direct runs and every profiled run passed the four-row
score-set oracle.

Each candidate was compared with the currently retained stack. The fourth was
then removed after measurement.

| Candidate | Baseline median (us) | Candidate median (us) | Delta | Pair wins | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Compact reuses resident score UB | 24.9689995 | 24.6849995 | -1.1374% | 7/10 | Retained |
| Reducers stage high/low histogram matrices in UB | 24.591999 | 22.148001 | -9.9382% | 5/5 | Retained |
| High-byte scan uses 16 parallel groups | 22.073999 | 20.891001 | -5.3592% | 5/5 | Retained |
| High/low stages use 32 per-warp private histograms | 20.705999 | 23.365000 | +12.8417% | 0/5 | Rejected |

The final A+B+C stack was rebuilt and compared with commit `887eb3d` in ten
alternating-order pairs. The baseline samples were `25.278999, 25.209999,
24.493999, 25.921000, 25.559999, 25.295000, 25.473000, 25.643000, 25.408001,
25.292999 us`; the retained samples were `20.577999, 21.289000, 20.700001,
20.671000, 21.127001, 20.593000, 20.600000, 20.672001, 20.746000,
21.055000 us`. Median task duration changed from `25.3515005 us` to
`20.686001 us`, a reduction of `18.4032%`, and the retained stack won all ten
pairs.

Source/executable SHA-256 pairs are:

- `887eb3d` baseline: `879e1c85f8bf0f33cf4102deec2ca76b7aa1e6af911ad4a46ae0a7b81acd0dd3` /
  `5d59c7d6f19976fc88b6815f072beb08e9bbb22fc6da18ef40c5a36b63f0c241`
- Compact UB candidate: `a2731a3d6f8aa381e89ff0fa969eb7bc3f491acdc996edcd852e9ffc6bacfc24` /
  `75b9473f22d225ac154abf5499a62b9abac9e1ded9c2e69e56d9e6aad9355cab`
- Reducer UB candidate: `b3cf5b277fa93e0234f42f91f9c7b8a3f7a0168bffbb7f453a188f82a4329d12` /
  `3552fcfc2a1b01220af3384c90878df147f84ccb4241680599613801ff00932d`
- Retained grouped-scan candidate: `0b2a13a2feac2fa2d353a23e17a1570b22de7f9602e8a0d8d951ab8365e7cf51` /
  `4339779baa096350ba9a7f96be8e096e6cd596e0e1f3c1dbec931d11984b4a42`
- Rejected private-histogram candidate: `9b678b76a4cb5d17e7fc9bc57c5efff1cd549d8c09236ea33bdcd3df15871670` /
  `ecedbe4068b30623462d985eab1f56a1a4d5db6d930942180b4da8b423c2eb74`

Raw evidence remains at remote
`/tmp/cannbench-topk-four-opt-bOAJ0C`. One first attempt at the grouped-scan
collection failed before kernel launch with `aclrtSetDevice error=507033` and
is preserved under `profiles/candidate_c`; the complete locked retry under
`profiles/candidate_c_retry_1` is the accepted evidence. The full 2,205-entry
archive is available on both hosts at
`/tmp/dsa-topk-four-optimizations-evidence-20260817.tar.gz`, SHA-256
`c99d2bbf0fbc4fba9abb479a8598503cdb62f5ae7aef429e6b61f35a01859ad3`.

## Single-DMA UB staging evidence

Collected on 2026-08-17 on `root@121.41.199.170:20002`, using Ascend 950PR
device 0, CANN 9.2.0, Bisheng clang 15.0.5, `dav-3510`, and `msopprof` 26.2.0.
The baseline is the SIMT source at commit `8c63488`; the candidate changes only
the histogram input staging described above.

- Profiler mode: `Default`, unmodified default warmup (reported as 5), exactly
  one application launch per collection
- Sampling: ten alternating-order baseline/candidate pairs
- Launch and frequency parity: every row reports 64 blocks and current/rated
  `1650/1650 MHz`
- Correctness: both direct runs and every profiled run passed the four-row
  score-set oracle
- Baseline source/executable SHA-256:
  `5ee3ff8559b824866da939c9cff8446c1efacf29525ff6713623029fae157be4` /
  `671bdf3d22171e4844aaf194602c5559a4a6bfbb424e58515a81aed938d9be83`
- Candidate source/executable SHA-256:
  `879e1c85f8bf0f33cf4102deec2ca76b7aa1e6af911ad4a46ae0a7b81acd0dd3` /
  `5d59c7d6f19976fc88b6815f072beb08e9bbb22fc6da18ef40c5a36b63f0c241`
- Remote evidence root: `/tmp/cannbench-topk-single-dma-D4s0RV`
- Local archive: `/tmp/dsa-topk-single-dma-evidence-20260817.tar.gz`
  (667 entries, SHA-256
  `866684c29c6a70cb1baa682a9339d17f0290c631fda9f3f2d8ecb210b9f306a8`)

| Variant | Ten Task Duration samples (us) | Median (us) | Min (us) | Max (us) |
| --- | --- | ---: | ---: | ---: |
| Direct-GM baseline | 27.601999, 27.743999, 27.898001, 27.865, 27.934999, 27.690001, 28.011999, 28.046, 28.023001, 28.056999 | 27.9165 | 27.601999 | 28.056999 |
| Single-DMA UB candidate | 25.186001, 25.257, 24.714001, 25.035999, 25.181999, 24.944, 25.082001, 25.086, 24.938999, 25.101 | 25.0840 | 24.714001 | 25.257 |

The candidate won all ten pairs and reduced median task duration by `10.146%`.
This result is limited to the fixed standalone BF16 decode shape and does not
claim a production workflow gain until the same change is integrated and
profiled inside the actual workflow.

## Original production-path evidence

Collected on 2026-08-13 on the task-specified Ascend 950 device 0 at
`root@121.41.199.170:20002`. The target was compiled for `dav-3510` with CANN
9.2.0 and Bisheng clang 15.0.5
(`clang-5c68a1cb1231 flang-5c68a1cb1231`). The profiler was `msopprof` 26.2.0.

- Profiler mode: `Default`, default profiler warmup (reported as 5), exactly
  one application launch selected per independent collection
- Frequency: every selected row reports current/rated `1650/1650 MHz`
- Production launch ownership: vLLM-Ascend 4 blocks; SIMT v2 64 blocks
- Remote root:
  `/tmp/cannbench-dsa-topk-actual-20260813-200615/dsa_decode_topk_comparison/evidence/actual-distributed-20260813-200615`
- Retained local root:
  `/tmp/dsa-decode-topk-actual-evidence-20260813-200615/actual-distributed-20260813-200615`
  (262 files, including 10 raw OPPROF trees, parser output, correctness logs,
  and build logs). Raw profiler trees remain outside the repository.
- Standalone source hashes: SIMT v2
  `5ee3ff8559b824866da939c9cff8446c1efacf29525ff6713623029fae157be4`;
  vLLM-Ascend
  `df4b560fe8809dbb2b5099bc8e21ff4c25bfc466b8abf8f61a7ab188912e06d8`
- Executable hashes: SIMT v2
  `4c8f0d34b56153bd4816f4ca4c457598671c58ae25ff16fb609d5bf1623bb09e`;
  vLLM-Ascend
  `691dcb2275e23cd0f92934254b28a34cab505a9d7b0b0f63e78e5924a59b1553`

Both direct correctness runs passed. For every row the oracle found threshold
`112`, `2008` scores above threshold, and required/observed `40`
threshold-equal selections; bounds and uniqueness also passed.

| Implementation | Blocks | Five Task Duration samples (us) | Median (us) | Min (us) | Max (us) |
| --- | ---: | --- | ---: | ---: | ---: |
| vLLM-Ascend | 4 | 7.807, 7.946, 7.913, 7.709, 7.722 | 7.807 | 7.709 | 7.946 |
| SIMT v2 distributed | 64 | 27.792, 27.617001, 28.406, 27.916, 27.882 | 27.882 | 27.617001 | 28.406 |

The median ratio `vLLM-Ascend / SIMT v2` is `0.280001`; equivalently, the SIMT
v2 distributed stage takes `3.5714x` the vLLM-Ascend stage time for this
one-kernel boundary. The implementations deliberately retain their production
launch ownership, so their AIV block dimensions are not equal: forcing both to
the same block count would no longer measure either actual decode stage.

The supplied V3.2 workflow profile reports the SIMT production kernel
`lightning_indexer_decode_distributed_topk_bfloat16_v2_kernel` at `28.629` and
`28.742001 us`, both with block dimension 64. The standalone SIMT samples are
within about 1-4% of those fused-workflow rows, which is consistent with having
extracted the actual distributed decode path.

These measurements are device task time, not end-to-end host dispatch latency,
and do not generalize beyond the fixed BF16 decode shape and recorded Ascend
950/CANN environment.

## Three-direction follow-up results

Three follow-ups were measured independently from frozen commit `5611bbc` on
2026-08-18. All accepted profiles used the default profiler warmup,
`--aic-metrics=Default --launch-count=1`, no `--kernel-name`, exactly one target
row, and current/rated `1650/1650 MHz`.

| Candidate | Baseline median (us) | Candidate median (us) | Paired delta median | Pair wins | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| 8 context shards / 32 blocks | 14.1015 | 16.4770005 | +2.3805 us | 0/10 | Rejected |
| selected-high candidate offsets / 64 blocks | 14.1625 | 14.916 | +0.7945 us | 0/10 | Rejected |
| MTE3 histogram publication / 64 blocks | n/a | n/a | n/a | n/a | Direct correctness failed |

The 32-shard / 128-block geometry reproducibly failed to exit at the existing
`AscendC::SyncAll()` boundaries. A diagnostic retaining its 32-shard data
layout but launching only 64 blocks exited and produced the expected first two
rows, confirming that 128 logical tasks are not valid participants in this
dav-3510 all-AIV hardware barrier structure. It was not profiled.

The corrected candidate-offset implementation caches all offsets whose high
radix byte is greater than or equal to `selected_high`; the low histogram still
counts only the equal bucket. It passed correctness but lost every pair, so the
added ballot, UB atomic reservation, and UB stores cost more than the reduced
compact rescan for this fixed shape.

The MTE3 candidate reached the same final radix state and shard offsets as the
baseline but compact emitted below-threshold indices. Four isolated diagnostics
did not recover it: restoring producer `asc_syncthreads()`, reloading score UB,
publishing compact output with DCCI, and invalidating consumer DCache before
compact. It is rejected before profiling. Do not retry this exact publication
structure without a changed synchronization or buffer-lifetime premise.

Source/executable SHA-256 pairs are:

- 32 shards: `e8534aa00b926657777f31a0785dc044b920f6904f6504a176d6d6382238897e` /
  `245c10662d3b25ad4ee1d18deec8c5b19545b3caafd8b93e9c10b6746999b776`
- candidate offsets: `0476d46fcdba931fe6e7ea87f562c941ef9d2f065eec47763b400971325c0627` /
  `f91f187785d7db26941fc6821fa2a9f0dd55f7f05b24a3972994fc5ae888219f`
- MTE3 publication: `344182237997e08d8826fa931c4d31b760a2da4d0954c43a3f84cc0dceb02844` /
  `cc6c9ce98417ccf65bc75d027806b8672d8da62d5a1d1ff57c3e17c545e75c74`

No factor passed the retention gate, so no combined candidate was created and
the example kernel remains the `5611bbc` baseline. Raw evidence remains at
remote `/tmp/dsa-topk-three-directions-exp-20260817`. The 1,939-entry archive
is present on both hosts at
`/tmp/dsa-topk-three-directions-evidence-20260817.tar.gz`, SHA-256
`007125808dee427f0257004de521fd23510a64edf4acc3f70ea1411588ca1e74`.
The plan remains outside the archive at
`/tmp/dsa-topk-three-directions-plan.md`.
