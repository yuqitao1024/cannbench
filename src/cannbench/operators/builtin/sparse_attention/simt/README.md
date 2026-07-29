# sparse_attention SIMT

This directory contains Ascend SIMT integration for `sparse_attention`.

Fast-path families:

- `family_hd256`
- `family_hd512`
- `family_hd576`
- `family_hd128`

Unsupported shapes are rejected by the SIMT plugin path.

## Current Coverage

The current custom-op fast path covers these shape families:

- `family_hd128`: `H = 128`, `KV_H = 1`, `Dqk = Dv = 128`, `selected_tokens <= 2048`
- `family_hd256`: `Dqk = Dv = 256`, `KV_H = 1`, `selected_tokens <= 2048`
- `family_hd512`: `Dqk = Dv = 512`, `KV_H = 1`, `selected_tokens <= 2048`
- `family_hd576`: `Dqk = 576`, `Dv = 512`, `KV_H = 1`, `selected_tokens <= 2048`

Additional `family_hd128` decode restriction:

- `query_tokens = 1`

For the current case set in this repository, that means:

- `51` cases are covered by the implemented fast paths
- `7` cases are not implemented by the current SIMT custom op

## Realistic Case Classification

For the current realistic case sets in this repository:

- `realistic`: `0 / 4` supported
- `realistic_decode`: `3 / 3` DSA workflow cases supported
- `realistic_prefill`: `3 / 3` DSA workflow cases supported

All `6 / 6` current DSA workflow realistic cases route through a sparse-attention
custom-op fast path.

Atlas 350 validation for the corrected V3.2 workflows:

| Phase | Case ID | Family | Status | Accuracy / runtime |
| --- | --- | --- | --- | --- |
| Decode | `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` | `family_hd576` | Full BF16 decode pass for Head64 P=1/P=2/P=4 | All `262144` output and `512` LSE elements passed at `atol=rtol=0.05`. P=1 output/LSE max abs errors were `0.009765625`/`0.0092773438`; P=2 errors were `0.0078125`/`0.0092763901`; P=4 errors were `0.009765625`/`0.0092763901`. |
| Prefill | `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` | `family_hd576` | Full automatic BF16 Head64/P=1 pass | Reduced P=1 passed `14 / 14`, including Q4 causal-future rejection; the full seeded run validated four causal-boundary rows and every head with zero mismatches at `atol=rtol=0.05`. Full output/LSE max abs errors were `0.0078125 / 0.0082502365`. |

## V3.2 Prefill Head64/P=1 Device Record

This record was collected on 2026-07-29 on `Ascend950PR_9589`, with CANN
9.2.0 at `/usr/local/Ascend/cann-9.2.0`, `dav-3510`, and clang/bisheng 15.0.5
build `5c68a1cb1231`. The generic baseline is source
`1297d3eb4ac5af62b0113f318136fdfae8ad52ea` in
`/tmp/cannbench-sa-v32-prefill-baseline-W1p49H`. The final profiled candidate
is source `fe376f33130c3757d552a95e8337c1e9c024fa18`, based on `2c4b7aa`, in
`/tmp/cannbench-sa-v32-prefill-candidate-rebased-wsnZON`. Its Head64 device
and Host source SHA-256 values are
`0e61fa35102a088b3f2d6ae482bff0678b524f27f352f7944803a54c2d44842d` and
`48ee3b1e2bed64a4eb5a697ffae08c24189909c790dd86a0934fbd106c458fda`.

The exact automatically selected case is:

```text
B=1 Q=4096 H=128 KV_H=1 context=32768 selected=2048 Dqk=576 Dv=512
dtype=BF16 phase=prefill family=family_hd576 seed=7
```

Default tuning maps its 8192 logical `(batch, query_token, head_group64)`
tasks persistently across 32 mixed tasks. One Head64/P=1 fused launch writes
public BF16 output and FP32 LSE directly; it allocates no partial output and
launches no Combine or output Cast. Explicit P=1 remains available for reduced
prefill shapes, while non-target default shapes and existing decode P=1/P=2/P=4
routes remain unchanged.

### Accuracy And Stability

The final rebuilt candidate passed all `14 / 14` reduced prefill P=1 cases,
including `S=0/17/64/70/128/2048`, invalid and overflowing indices, `C=0`,
width rejection, physical-core reuse at `B=2,Q=9`, repeated launches, and
eight alternating launches retained across two streams. The added
`B=1,H=128,Q=4,C=256,S=64` causal case exercised valid-past, negative, and
out-of-range indices on every row; rows `0/1/2` contained `2/2/1` in-range
future indices and row 3 exercised the causal end boundary. Maximum reduced
output/LSE absolute errors were `0.017578125 / 0.020476341247558594`.

The unchanged decode path passed all `36 / 36` P=1/P=2/P=4 results, with
maximum output/LSE absolute errors of
`0.0185546875 / 0.01997852325439453`. The full automatic prefill run used
seed 7 and `atol=rtol=0.05`; rows `0,1365,2730,4095` and all 128 heads per row
had zero mismatches. Rows `0/1365/2730` each included one deterministic
in-range future index, while row `4095` included the causal end boundary; all
four also included one negative and one out-of-range index. Its output/LSE
maximum absolute errors were `0.0078125 / 0.008250236511230469`.

The fresh final-review rerun used the preserved runner-only copy
`/tmp/cannbench-sa-v32-prefill-final-review-ACYDoO`. It again passed the
reduced `14 / 14` and full causal checks with the counts above and zero full
output/LSE mismatches. The Head64 device, Host, and built `_C` SHA-256 values
remained
`0e61fa35102a088b3f2d6ae482bff0678b524f27f352f7944803a54c2d44842d`,
`48ee3b1e2bed64a4eb5a697ffae08c24189909c790dd86a0934fbd106c458fda`, and
`433da93827b198a03eb0c7b8b9d396353f1023e96677bed9d3c95d58fa628a8b`.
The benchmark runner SHA-256 remained
`c41fdcc4defc8a4f5c42859b4683f17df58bdd04c33351429e4fd63e1db850de`, and its
default seed-10 index tensor was byte-identical to the historical candidate
(`9b2a3011fbe8a05c64c21f856aa469dcbd1857393f4194d7cac1a82b17d8cdfd`),
so the existing wall-time and profile evidence remains applicable and was not
rerun.

### Wall Time

Every run used the same deterministic seed-7 inputs, one warmup, three timed
calls, unset tuning variables, and synchronization after every call. The
baseline predates the benchmark runner, so it executed the candidate runner
file while `PYTHONPATH` and extension loading selected only the baseline
implementation. Both candidate runs are retained: the first predates the
upstream Head64 gather-pipeline rebase, while the final row is the primary
comparison because it measures the rebuilt, revalidated source that was
profiled and kept.

| Tree | Three samples (ms) | Median (ms) |
| --- | --- | ---: |
| Generic baseline `1297d3e` | `272551.605669, 272569.785186, 272568.230243` | `272568.230243` |
| Initial candidate `81b7165` | `358.798329, 359.500686, 358.652544` | `358.798329` |
| Final profiled candidate `fe376f3` | `335.586126, 336.215180, 335.327427` | `335.586126` |

The primary final median is `812.215432x` faster than the same-input generic
baseline, a `99.876880%` reduction (`272232.644117 ms`). The initial candidate
was `759.669731x` faster; it is shown rather than discarded so the result does
not depend on selecting the faster of the two candidate runs.

### Candidate Exact-Kernel Profile

The final candidate used `msopprof --aic-metrics=Default --warm-up=5
--launch-count=1` with the exact glob
`*sparse_attention_head64_fused_kernel*`. The isolated profiler raised the
operator and synchronization timeouts to 600 seconds. Its executable SHA-256
is `9648516a3404b7162c8044d7d9ff9c5bcddf9762d27d68808f00c824fdcf7f0a`;
the injection library SHA-256 is
`abcd89c2651c6c457be10cdc83f61f347d02709455a8b829b64bd7437c387831`.
The complete result is:

`/tmp/msopprof-sa-v32-prefill-head64-candidate-rebased-wsnZON-timeout600/OPPROF_20260729062747_IMRDMKLOXXKVQZVU`

`OpBasicInfo.csv` reports `334959.437500 us`, `Block Dim = 32`, and
`Mix Block Dim = 64`. The 96 sub-block rows contain 32/32 AIC rows with
`aic_cube_total_instr_number > 0` and 64/64 AIV rows with
`aiv_vec_time(us) > 0`; every AIC row records 106496 Cube instructions.
The application trace contains no Combine. Its only Cast is immediately
before the selected kernel and belongs to deterministic input materialization;
there is no kernel after the selected kernel, so output Cast launches are zero.

The following utilization and wait values are arithmetic means over finite
AIC or AIV rows:

| Cube % | Vector % | Scalar % AIC/AIV | AIC MTE1/MTE2/MTE3 % | AIV MTE2/MTE3 % |
| ---: | ---: | ---: | ---: | ---: |
| `1.674416` | `86.654003` | `0.895138 / 1.324269` | `4.410788 / 0 / 0` | `0 / 1.530755` |

| AIC wait Cube/MTE1/MTE2/MTE3 % | AIV wait Vector/MTE2/MTE3 % |
| ---: | ---: |
| `93.574156 / 93.077944 / 0 / 0` | `99.947698 / 0 / 99.954256` |

Summed profiler traffic fields, in the CSV's `KB` unit, are:

| GM read/write | GM-to-L1 | L0C-to-L1 | L0C-to-GM | GM-to-UB | UB-to-GM |
| ---: | ---: | ---: | ---: | ---: | ---: |
| `19878032.375 / 1342591.750` | `12.000` | `37748736.000` | `0` | `90187.750` | `n/a` |

L2 hit rates are recomputed from summed close/far hit, miss, and victim
counters. AIC writes recorded zero events, so their hit rate is unavailable:

| AIC read/write hit % | AIV read/write hit % |
| ---: | ---: |
| `96.875000 / n/a` | `96.271313 / 68.104965` |

Mean L0A read/write, L0B read/write, and L0C Cube read/write bandwidths were
`1.586013 / 1.865898`, `1.586013 / 3.172026`, and
`2.985436 / 6.344052 GB/s`. Mean AIV UB Vector read/write bandwidths were
`45.321101 / 37.093105 GB/s`; UB read-from-GM and UB-to-GM traffic are `NA`
in this schema, while UB write-to-GM bandwidth was `0.004012 GB/s`.
The Default metric schema does not provide direct FLOP/s or occupancy, and its
Cube FP instruction field is zero despite nonzero total Cube instructions, so
no FLOP/s value is inferred.

### Baseline Profile Gap And Dispatch Decision

The original 100-second baseline capture is preserved at
`/tmp/msopprof-sa-v32-prefill-head64-baseline-W1p49H/OPPROF_20260729051202_ECEOIVYVHMWJMVMR`.
The application exceeded the profiler's AI Core timeout: stdout reports
`Get op basic info [Task Duration] failed` and `0.000000 us`, while
`OpBasicInfo.csv` stores `NA`; its 32-row CSVs are truncated post-timeout data
and are not used. A requested 600-second retry was stopped during warmup. It
left a partial capture hierarchy under
`/tmp/msopprof-sa-v32-prefill-head64-baseline-W1p49H-timeout600/OPPROF_20260729063339_VNOZAPWFXZNBSNXC`,
containing only the setup dumps `pc_start_addr.txt` and `aicore_binary.o`. It
has no BasicInfo, metric CSV, analyzed duration, or exit marker. The partial
hierarchy and the baseline tree's `task-5-artifacts/profile-timeout600.*` logs
are preserved.

Consequently, the strict selected-kernel-duration comparison against the
generic baseline was not passed or evaluated; it is an explicit evidence gap,
not an inferred success. Automatic dispatch is nevertheless retained under
the approved waiver because reduced/full accuracy, decode regression,
stability, the same-input wall-time gate, the valid candidate duration,
32/32 Cube work, 64/64 Vector work, and zero Combine/output-Cast gates all
passed. No baseline profiler counter is used to support that decision.

## Fused P=4 Ping-Pong Device Record

This record was measured on 2026-07-29 on `Ascend950PR_9589`, with CANN 9.2.0,
`dav-3510`, and the exact V3.2 realistic decode shape shown below. The final
implementation is commit `9cca869`; the single-buffer baseline is `a38d5cd`.

```text
B=2 Q=2 H=128 KV_H=1 context=32768 selected=2048 Dqk=576 Dv=512 causal=true
head_tile=64 selected_partitions=4
```

The fused MIX kernel keeps online softmax, 1024 SIMT threads per AIV,
`CrossCoreFlag` mode 2, and the independent Combine launch. K and V gather each
use two L1 slots. The next gather is requested after the current L1-to-L0 copy,
so AIV gather overlaps the current AIC MMAD. The output-update VF completes
before AIV publishes that the current PV buffer may be reused. The full design
and the PV lifetime root cause are recorded in
[`HEAD64_P4_PINGPONG_DESIGN.zh-CN.md`](../HEAD64_P4_PINGPONG_DESIGN.zh-CN.md).

### Accuracy

The rebuilt final source passed:

- five consecutive P=4 `valid_s64` runs;
- all `36 / 36` reduced, boundary, invalid-index, empty-contract and physical
  core-reuse results;
- all `262144` realistic output and `512` LSE elements with zero mismatches at
  `atol=rtol=0.05`.

Reduced maximum output/LSE absolute errors were
`0.0185546875 / 0.0199785233`. Full realistic maximum errors were
`0.009765625 / 0.0092763901`.

### Wall Time

Each variant used three warmups and seven measured rounds of five calls. Every
call was followed by `torch.npu.synchronize()`. The K+V result was repeated
after reinstalling the K-only binary to expose device-state drift.

| Variant | Commit | Seven round means (ms) | Median (ms) |
| --- | --- | --- | ---: |
| Single buffer | `a38d5cd` | `0.583123, 0.576859, 0.581521, 0.581873, 0.593459, 0.594727, 0.588158` | `0.583123` |
| K-only ping-pong | `c40c78c` | `0.574136, 0.572341, 0.572862, 0.568187, 0.572342, 0.586400, 0.578845` | `0.572862` |
| K+V ping-pong, run 1 | `9cca869` | `0.557186, 0.553204, 0.550888, 0.554370, 0.555444, 0.551369, 0.555174` | `0.554370` |
| K+V ping-pong, run 2 | `9cca869` | `0.576293, 0.564699, 0.571445, 0.565802, 0.564048, 0.567052, 0.565348` | `0.565802` |

The two K+V runs show about `0.01 ms` of device-state drift. The conservative
end-to-end conclusion is therefore an approximately `3%` improvement over the
single-buffer fused baseline, with an observed range of roughly `3%` to `5%`.

### Exact-Kernel Profile

Single-buffer and K+V binaries were rebuilt from their isolated source trees
and collected with `msopprof --aic-metrics=Default --warm-up=5
--launch-count=1`. Both fused profiles reported `32 / 64` launch dimensions,
32 AIC rows with Cube work, and 64 AIV rows with Vector work.

| Variant | Fused (us) | Combine (us) | Sum (us) |
| --- | ---: | ---: | ---: |
| Single buffer `a38d5cd` | `388.953979` | `36.410000` | `425.363979` |
| K+V ping-pong `9cca869` | `365.545990` | `36.264000` | `401.809990` |

The fused kernel improved by `6.02%`; fused plus Combine improved by `5.54%`.
Combine was unchanged within measurement noise. The smaller wall-time gain is
consistent with Host, launch and synchronization overhead outside these two
profiled kernels.

Exact source and artifact identities:

- final fused source SHA-256:
  `f7d573ae26b846e47f9bc9b45e4c0b59d57cb396f1c99ef3f0af5bbbca9c75bb`;
- single-buffer source SHA-256:
  `9d9fa4c231ba5d926fda3ada1b9884b7a1ec857075900acd57cd3f687c2a4942`;
- final remote tree: `/tmp/cannbench-head64-pingpong-v-output-sync-20260729`;
- single-buffer remote tree:
  `/tmp/cannbench-head64-single-buffer-a38d5cd-20260729`.

## Split-KV Device Record

This record was measured on 2026-07-28 from source commit
`b7655df016832a0776b5ed8dca9bc33a22eb6248` on `Ascend950PR_9589`, with
CANN 9.2.0 (`/usr/local/Ascend/cann-9.2.0`), `dav-3510`, and BF16 inputs.
The exact realistic decode shape was:

```text
B=2 Q=2 H=128 KV_H=1 context=32768 selected=2048 Dqk=576 Dv=512 causal=true
```

Throughout this record, P means `selected_partitions`, and every P=1/2/4
measurement fixes `head_tile=64`. P=1 is therefore the Head64 `(64,1)`
baseline, not the default legacy `(head_tile=1, selected_partitions=1)` path.

### Wall Time

Each P used three warmups and seven measured rounds of five calls. Every call
was followed by `torch.npu.synchronize()`. The 21 measured rounds repeated the
order `P=1,2,4,4,2,1`; each round value below is the mean of its five calls,
and the reported latency is the median of the seven round means.

| P | Seven round means (ms) | Median (ms) | Speedup vs P=1 |
| ---: | --- | ---: | ---: |
| 1 | `1.383487, 1.389238, 1.393152, 1.392084, 1.395272, 1.394378, 1.396504` | `1.393152` | `1.000x` |
| 2 | `0.855860, 0.853451, 0.849703, 0.857994, 0.858658, 0.859603, 0.857709` | `0.857709` | `1.624x` |
| 4 | `0.545723, 0.543444, 0.559336, 0.548229, 0.553167, 0.556005, 0.547071` | `0.548229` | `2.541x` |

P=4 was the measured fastest configuration. It passed the primary gate because
`0.548229 ms < 1.393152 ms` and `0.548229 ms < 1.33 ms`.

### Kernel Profile

Each row came from a separate exact-kernel `msopprof` replay with
`--aic-metrics=Default`, `--launch-count=1`, and five profiler warmups.
`Launch` is `Block Dim / Mix Block Dim`. `Rows` counts finite per-subblock
profiler rows; `work` counts rows with Cube instructions or nonzero Vector
time, so it is independent evidence of effective use rather than an inference
from host block dimensions.

| P | Kernel | Duration (us) | Launch | Rows (AIC/AIV) | Work (Cube/Vector) |
| ---: | --- | ---: | ---: | ---: | ---: |
| 1 | QK | `483.575` | `8 / 16` | `8 / 16` | `8 / 16` |
| 1 | PV | `844.832` | `8 / 16` | `8 / 16` | `8 / 16` |
| 2 | QK | `246.179` | `16 / 32` | `16 / 32` | `16 / 32` |
| 2 | PV | `420.535` | `16 / 32` | `16 / 32` | `16 / 32` |
| 2 | Combine | `20.319` | `8 / 16` | `8 / 16` | `0 / 16` |
| 4 | QK | `128.868` | `32 / 64` | `32 / 64` | `32 / 64` |
| 4 | PV | `211.732` | `32 / 64` | `32 / 64` | `32 / 64` |
| 4 | Combine | `36.054` | `8 / 16` | `8 / 16` | `0 / 16` |

P=1 has no combine launch. For P=4, both QK and PV recorded 32 `cube0`,
32 `vector0`, and 32 `vector1` rows with work. Combine remained a separate
8-AIC/16-AIV MIX launch; its AIC branches returned without Cube instructions,
while all 16 AIV rows performed Vector work.

The utilization values below are arithmetic means across finite profiler rows.
MTE columns list AIC `MTE1/MTE2/MTE3` and AIV `MTE2/MTE3`, respectively.

| P | Kernel | Cube % | Vector % | Scalar % (AIC/AIV) | AIC MTE % | AIV MTE % |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | QK | `2.417` | `80.574` | `1.122 / 1.171` | `3.437 / 0.000 / 0.000` | `0.001 / 2.355` |
| 1 | PV | `1.483` | `93.683` | `0.448 / 0.635` | `1.426 / 3.439 / 0.000` | `0.000 / 2.032` |
| 2 | QK | `2.377` | `80.540` | `1.271 / 1.332` | `3.381 / 0.001 / 0.001` | `0.002 / 2.362` |
| 2 | PV | `1.492` | `93.434` | `0.620 / 0.793` | `1.434 / 3.511 / 0.000` | `0.001 / 2.069` |
| 2 | Combine | `0.000` | `97.325` | `46.118 / 2.611` | `0.181 / 0.181 / 0.181` | `0.019 / 0.006` |
| 4 | QK | `2.297` | `80.402` | `1.778 / 1.885` | `3.266 / 0.001 / 0.001` | `0.003 / 2.369` |
| 4 | PV | `1.490` | `93.069` | `1.096 / 1.259` | `1.432 / 3.620 / 0.001` | `0.002 / 1.991` |
| 4 | Combine | `0.000` | `98.517` | `47.358 / 1.448` | `0.180 / 0.180 / 0.180` | `0.011 / 0.004` |

Wait ratios are means across the same rows. AIC values list
`Cube/MTE1/MTE2/MTE3`; AIV values list `Vector/MTE2/MTE3`.

| P | Kernel | AIC wait % | AIV wait % |
| ---: | --- | ---: | ---: |
| 1 | QK | `99.753 / 99.626 / 0.000 / 0.000` | `99.541 / 0.000 / 99.815` |
| 1 | PV | `99.736 / 99.653 / 99.664 / 0.000` | `99.242 / 0.000 / 99.755` |
| 2 | QK | `99.370 / 99.194 / 0.000 / 0.000` | `98.936 / 0.000 / 99.473` |
| 2 | PV | `99.673 / 99.549 / 99.536 / 0.000` | `98.684 / 0.000 / 99.716` |
| 2 | Combine | `0.000 / 0.000 / 0.000 / 0.000` | `0.000 / 0.000 / 0.000` |
| 4 | QK | `98.855 / 98.569 / 0.000 / 0.000` | `98.013 / 0.000 / 99.054` |
| 4 | PV | `99.411 / 99.206 / 99.147 / 0.000` | `97.468 / 0.000 / 99.503` |
| 4 | Combine | `0.000 / 0.000 / 0.000 / 0.000` | `0.000 / 0.000 / 0.000` |

GM traffic is the sum of `read_main_memory_datas` and
`write_main_memory_datas`. L2 hit rates are recomputed from the summed raw
hit, miss, and victim counters; `n/a` means the profile recorded no accesses.

| P | Kernel | GM read/write (KiB) | AIC L2 read/write hit % | AIV L2 read/write hit % |
| ---: | --- | ---: | ---: | ---: |
| 1 | QK | `19019.750 / 4096.000` | `87.500 / 78.406` | `61.745 / n/a` |
| 1 | PV | `39835.875 / 3925.125` | `99.513 / 84.727` | `68.016 / 77.361` |
| 2 | QK | `19614.750 / 4096.000` | `95.455 / 80.945` | `63.885 / n/a` |
| 2 | PV | `37965.250 / 5782.625` | `99.098 / 81.670` | `65.703 / 81.722` |
| 2 | Combine | `2057.000 / 1032.750` | `90.000 / n/a` | `21.018 / 89.459` |
| 4 | QK | `20786.000 / 4096.000` | `100.000 / 80.475` | `64.670 / n/a` |
| 4 | PV | `32894.375 / 9548.625` | `97.906 / 84.009` | `56.905 / 82.769` |
| 4 | Combine | `4109.000 / 1032.500` | `100.000 / n/a` | `21.790 / 90.724` |

### DSA Decode Workflow

The existing `dsa_decode` workflow order was used:
`lightning_indexer -> sparse_attention`. Query and shared-KV were materialized
once outside timing. Each measured call obtained a new canonical `[2,2,2048]`
int32 Indexer output and passed that exact tensor object directly as the
`indices` argument to the operator-local sparse-attention wrapper. A preflight
before timing verified matching Python identity and NPU data pointer, input and
output shapes, and dtypes. No sparse-attention callable captured stale indices.

Runtime profiling of the exact Indexer case confirmed the two dispatched
kernels: context-sharded score MIX (`16 / 32`) followed by score TopK Vector
(`4`). Source SHA-256 values:

- context-sharded score: `05b1e54ac0faebe9a51a976f1cfc6bc8fa3a5ae87d036adf63d8ede5b129d302`
- score TopK: `0723df3fa0b96717b146d2c1f734017728821c0a165338db836123da649e02fd`
- host dispatcher: `8f572a077bff8eab9326bedaf45062a5d603a2dccbb69a05076935b44633aae4`

Each P retained three explicit warmups, seven measured rounds, five calls per
round, and the same alternating order as the standalone measurement. Each
timed call used a pre-call synchronize outside the clock and one final
`torch.npu.synchronize()` after both components.

| P | Seven workflow round means (ms) | Median (ms) | Speedup vs P=1 |
| ---: | --- | ---: | ---: |
| 1 | `3.490468, 3.494191, 3.495697, 3.492197, 3.491825, 3.489573, 3.495473` | `3.492197` | `1.000x` |
| 2 | `2.956524, 2.951654, 2.951119, 2.949921, 2.954765, 2.971182, 2.961050` | `2.954765` | `1.182x` |
| 4 | `2.649714, 2.645587, 2.652893, 2.647296, 2.647610, 2.647585, 2.650449` | `2.647610` | `1.319x` |

P=4 reduced the measured workflow median by `0.844586 ms` versus P=1 and was
the fastest workflow configuration in this run.

### Final Review Revalidation

The QK pipeline was revalidated after moving the L1-to-L0A query copy before
the existing MTE1 AIC-to-AIV release flag. This closes query-buffer ownership
when a physical core loops over more than one logical task. The 2026-07-28 run
used remote checkout `/root/cannbench-head64-final-review-20260728`, based on
`b7655df016832a0776b5ed8dca9bc33a22eb6248` plus patch SHA-256
`cfaa671128fb8fc3985ce81aa1742f452d279c04ccb29112a61d50a3d82f3ab1`.
The final device and Host source SHA-256 values were:

- Head64 device: `e9b1835fdda477f21f0bca3105bac9d05e6fa1b2422907612265829067f42fa6`
- Host bridge: `bfc42103a18df2af4ba865e46cb3cb8eb2f8b58870d0fc45b092e913bcb43712`
- reduced runner: `0360833d2513c9b5f4e2b291362ea1aeb1392076d98d7636c55dff8c5cc4058d`

The rebuilt `dav-3510` extension passed all 36 reduced and boundary results.
This included P=1/2/4, int64 overflow indices, `S=0`, `C=0`, both invalid
shared-KV widths, and `B=2,Q=9,S=70`. The latter produces 36/72/144 logical
tasks at P=1/2/4 and therefore exercises physical-core reuse in every tuning.
Its P=4 output/LSE maximum absolute errors are recorded by the final runner,
with zero mismatches.

Full realistic P=1/2/4 accuracy also had zero mismatches for all `262144`
output and `512` LSE elements. Using the same wall-time method documented
above, the post-fix medians were:

| P | Median (ms) | Speedup vs P=1 |
| ---: | ---: | ---: |
| 1 | `1.432600` | `1.000x` |
| 2 | `0.882918` | `1.623x` |
| 4 | `0.574588` | `2.493x` |

P=4 remained the fastest configuration and stayed below the `1.33 ms` gate.
The repeated P=4 QK profile measured `136.606 us`, reported launch dimensions
`32 / 64`, and contained actual Cube/Vector work on all `32 AIC / 64 AIV`
rows.

Supported realistic-family cases:

- `realistic_decode::deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` via `family_hd576`
- `realistic_decode::deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512` via `family_hd512`
- `realistic_decode::deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024` via `family_hd512`
- `realistic_prefill::deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` via `family_hd576`
- `realistic_prefill::deepseek_v4_flash_flashmla_prefill_q4096_ctx32768_top512` via `family_hd512`
- `realistic_prefill::deepseek_v4_pro_vllm_prefill_q4096_ctx131072_top1024` via `family_hd512`

## Unimplemented Shape Families

The following case groups do not match the current fast paths and are therefore
not implemented yet.

### 1. Small smoke-only fallback shapes

- `D = 16`, dense MHA decode: `smoke::tiny_decode_top4`
- `D = 16`, dense MHA prefill: `smoke::tiny_prefill_top8`
- `D = 32`, MQA decode: `smoke::tiny_mqa_decode_top8`

### 2. Dense-MHA `D = 64` prefill family

These cases use `KV_H = H`, so they do not match the current MQA/GQA-oriented
`KV_H = 1` fast paths.

- `realistic::nanogpt_prefill_64_top32`
- `realistic::opt_prefill_2048_top512`
- `realistic::gpt2_large_prefill_1024_top256`

### 3. `D = 128` decode family with non-128 query-head layout

This case has `D = 128` and `KV_H = 1`, but `H = 5`, so it does not match the
current `family_hd128` requirement `H = 128`.

- `realistic::llama4_decode_32760_top2048`

## Tensor Shapes

The sparse attention fast path uses the following logical tensor shapes:

- `Q`: `[B, H, Q, Dqk]`
- `K`: `[B, KV_H, C, Dqk]`
- `V`: `[B, KV_H, C, Dv]`
- `indices`: `[B, Q, S]`
- `output`: `[B, H, Q, Dv]`
- `lse`: `[B, H, Q]`

Where:

- `B`: batch size
- `H`: query head count
- `KV_H`: key/value head count
- `Q`: query token count
- `C`: context token count
- `S`: selected sparse token count per query token
- `Dqk`: query/key head dimension
- `Dv`: value/output head dimension

In the common decode case, `Q = 1`. In MQA/GQA style layouts, `H` may be larger than `KV_H`, which means multiple query heads share the same KV head group.

## Computation

For a fixed batch `b`, query head `h`, and query token `q`, the operator first uses
`indices[b, q, :]` to choose the sparse context positions:

```text
K_sparse = gather(K, indices)
V_sparse = gather(V, indices)
```

Logical sparse shapes:

- `K_sparse`: `[B, H, Q, S, Dqk]`
- `V_sparse`: `[B, H, Q, S, Dv]`

Then sparse attention scores are computed only on the selected positions:

```text
scores[b, h, q, s] = dot(Q[b, h, q, :], K_sparse[b, h, q, s, :]) / sqrt(Dqk)
```

This produces:

- `scores`: `[B, H, Q, S]`

If `causal = true`, any selected position whose key token index is greater than the current query token index is masked out before normalization. Invalid sparse indices are also masked out.

The normalized sparse probabilities are:

```text
prob = softmax(scores)
```

with shape:

- `prob`: `[B, H, Q, S]`

The final attention output is the weighted sum over the gathered sparse values:

```text
output[b, h, q, :] = sum_{s in [0, S)} prob[b, h, q, s] * V_sparse[b, h, q, s, :]
```

The operator also returns:

```text
lse[b, h, q] = log(sum(exp(scores[b, h, q, :])))
```

This `lse` term is the per-row log-sum-exp statistic for the sparse attention scores.
