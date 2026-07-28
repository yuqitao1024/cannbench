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
| Decode | `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` | `family_hd576` | Full BF16 decode pass for split-KV P=2/P=4 | All `262144` output and `512` LSE elements passed at `atol=rtol=0.05`. P=2 output/LSE max abs errors were `0.0078125`/`0.0092763901`; P=4 errors were `0.009765625`/`0.0092763901`. |
| Prefill | `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` | `family_hd576` | Reduced-shape pass; full case pending | Corrected `Dqk = 576`, `Dv = 512` path passed at B1/H128/Q4/C256/S64 with output max abs `0.015625` and LSE `0.014242`. Previous reduced-Q result used the wrong 576/576 layout. |

## Split-KV Device Record

This record was measured on 2026-07-28 from source commit
`b7655df016832a0776b5ed8dca9bc33a22eb6248` on `Ascend950PR_9589`, with
CANN 9.2.0 (`/usr/local/Ascend/cann-9.2.0`), `dav-3510`, and BF16 inputs.
The exact realistic decode shape was:

```text
B=2 Q=2 H=128 KV_H=1 context=32768 selected=2048 Dqk=576 Dv=512 causal=true
```

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
`lightning_indexer -> sparse_attention`. Its canonical `[2,2,2048]` int32
Indexer output was bound once outside timing, matching the workflow runner's
prepared-input contract. Each measured workflow call then launched the
Indexer callable followed by the bound sparse-attention callable and used one
`torch.npu.synchronize()` after both components. Warmups, rounds, calls, and
partition ordering matched the single-operator measurement.

| P | Seven workflow round means (ms) | Median (ms) | Speedup vs P=1 |
| ---: | --- | ---: | ---: |
| 1 | `3.536788, 3.578843, 3.580511, 3.539922, 3.535684, 3.542215, 3.539764` | `3.539922` | `1.000x` |
| 2 | `2.977060, 2.990544, 2.993878, 2.982196, 2.983361, 2.978907, 2.983153` | `2.983153` | `1.187x` |
| 4 | `2.662575, 2.663346, 2.665090, 2.666458, 2.668565, 2.670020, 2.662177` | `2.665090` | `1.328x` |

P=4 reduced the measured workflow median by `0.874832 ms` versus P=1 and was
the fastest workflow configuration in this run.

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
