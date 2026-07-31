# lightning_indexer SIMT

This directory contains Ascend SIMT integration for `lightning_indexer`.

Fast-path families:

- `family_32x128`
- `family_64x128`
- `family_4x64`

Unsupported shapes use plugin-local fallback.

## V3.2 prefill benchmark

Measured on Atlas 350 (`dav-3510`, CANN 9.2.0) on July 28, 2026. The target
shape was BF16 `B=1, Q=4096, C=32768, H=64, D=128, K=2048`, with right-aligned
causal context lengths from 28673 through 32768 and seed 7. Each isolated build
used one warmup followed by five `perf_counter_ns` samples, synchronizing the
NPU after every public custom-op call.

The production common path now uses up to 32 mixed tasks, corresponding to all
32 AICs and 64 AIVs under `KERNEL_TYPE_MIX_AIC_1_2`, and launches its SIMT VF
with 1024 threads. Its former 11-task cap was left over from the per-block
mode-4 flag design. The first correction raised that cap to 16 after the
mode-2/flag-0 handshake fix; the 32-task correction reflects the target
device's complete `32 AIC + 64 AIV` inventory.

| Public-op path | Threads | Median | Min | Max |
| --- | ---: | ---: | ---: | ---: |
| 16-task row baseline (`77c19fa`) | 256 | 11226.1922 ms | 11225.7166 ms | 11227.2125 ms |
| 32-task intermediate | 256 | 16673.0931 ms | 16672.9494 ms | 16674.1895 ms |
| 32-task retained path | 1024 | 16672.1256 ms | 16671.5691 ms | 16672.2434 ms |
| Q=2 dual-AIV candidate (`c381b0a`) | 1024 | 12472.9031 ms | 12472.5124 ms | 12472.9921 ms |

The exact-kernel BasicInfo profiles use a different timing boundary:

| Common fused path | Task Duration | Block Dim | Mix Block Dim |
| --- | ---: | ---: | ---: |
| 16 tasks, 256 threads | 11.218775 s | 16 | 32 |
| 32 tasks, 256 threads | 5.5801695 s | 32 | 64 |
| 32 tasks, 1024 threads | 3.124832 s | 32 | 64 |

On this requested kernel-side boundary, the retained 32-task/1024-thread path
is `3.59020x` faster than the 16-task/256-thread baseline, a `72.1464%`
latency reduction. Its exact sampled score sets and three repeated stability
launches passed. The profiler artifacts are
`/tmp/msopprof-v32-prefill-100s-wb1ntr`,
`/tmp/msopprof-v32-prefill-32task-ju04u4`, and
`/tmp/msopprof-v32-prefill-32task-1024t-7DfD7Y`.

The synchronized public-op median does not follow the BasicInfo task duration:
it increases by `48.5110%` from the 16-task baseline. The 32-task launch was
confirmed as `Block Dim = 32` and `Mix Block Dim = 64`; investigation of the
remaining host/kernel timing gap is deferred, and the performance claim above
is deliberately limited to msopprof's selected-kernel boundary.

Against the historical 16-task row baseline, the Q=2 candidate median ratio is
`1.11105`, so it is 11.11% slower. The 32-token dual-AIV score tile increases
the number of in-kernel TopK merge rounds enough to outweigh Query/Key reuse.
The exact Q=2 dispatch is therefore disabled; V3.2 prefill remains on the
common fused path. The candidate device source stays buildable for the later
full-score/separate-TopK comparison. Its 1024-thread VF reports 0 bytes of
stack and 24 registers.

The historical 16-task corrected path completed the full SIMT workflow with indexer latency
11.2342 s, workflow-attention latency 272.6619 s, and combined workflow latency
283.8962 s. Cross-backend comparison against vLLM-Ascend passed: the indexer
minimum recall was `0.973145`, and sampled attention/workflow output and LSE
had zero mismatches at `atol=0.05, rtol=0.05`. The Q=2 workflow timing was not
run because its standalone gate had already failed.

### Full-score plus radix TopK candidate

On July 31, 2026, the exact V3.2 prefill candidate was implemented as a
256 MiB BF16 score workspace (`[1,4096,32768]`) followed by a separate
two-pass radix-select TopK kernel. Both device libraries and the host bridge
compile with Bisheng for `dav-3510`, using only C API, Tensor API, and SIMT
API in the new device sources.

The first implementation failed with device error `507015` only when launched
after the sampled PyTorch `einsum` reference. Device plog and an optimized
debug build mapped the primary AIC exception to `asc_copy_l0c2ub`; the AIV
exceptions were secondary waits on the failed AIC. The Fixpipe call requested
NZ2DN even though the two AIV destinations consume row-major M slices. That
made the copy depend on `CHANNEL_PARA` state left by the preceding Cube
operator. The fix selects NZ2ND and explicitly disables NZ2DN, matching the
working Basic API `CO2Layout::ROW_MAJOR` behavior without adding Basic API
dependencies.

The original `custom -> einsum -> custom` reproduction now passes. The full
gate used one warmup, five synchronized samples, the four sampled rows above,
and three additional stability launches. All sampled TopK score sets matched
and all stability launches passed. Synchronized samples were `299.689408`,
`300.047016`, `299.456405`, `299.444190`, and `299.425726` ms, for a
`299.456405 ms` median. Against the retained common-path public-op median this
is `55.67463x` faster, a `98.2038%` latency reduction.

The exact BF16 V3.2 shape now dispatches to the full-score plus radix TopK
path. Canonical decode and all generic fallbacks are unchanged.

## Parameterized context-sharded decode benchmark

Measured on Ascend 950PR (`dav-3510`, CANN 9.2.0) on July 29, 2026. The BF16
`family_64x128` decode fast path fixes `C=32768, H=64, D=128, K=2048` and
plans runtime `B/Q` as a single mixed device launch. It chooses the largest
shard count in `{16,8,4,2,1}` for which `B * ceil(Q / 2) * shards <= 32`.
The production `B=2,Q=2` S16 launch therefore uses 32 mixed tasks, or 32 AICs
and 64 AIVs. It skips local TopK because every shard has exactly 2048 scores;
four row owners perform the final TopK inside the same kernel.

For the S16 target, seed 29 used five warmups and 20 timed iterations. The
old BasicInfo medians were `578.193481 us` for the score kernel and
`1566.729981 us` for standalone TopK (`2144.923462 us` combined). The new
single kernel measured `1984.271485 us`, a `7.489870%` reduction. Synchronized
wall time dropped from `2.1796345 ms` to `2.034242 ms` (`6.67%`), and the
score sets matched exactly. msopprof reports `Block Dim = 32` and
`Mix Block Dim = 64`; the remote artifacts are
`/tmp/msopprof-li-kernel-NDQdwo`.

The same BF16 score-set gate and synchronized wall comparison passed for every
enabled planner tier:

| Tier | Representative `[B,Q]` | New / old wall median (ms) | Reduction |
| --- | --- | ---: | ---: |
| S16 | `[2,2]` | 2.034242 / 2.1796345 | 6.67% |
| S8 | `[3,2]` | 1.662996 / 26.159979 | 93.64% |
| S4 | `[5,1]` | 2.012016 / 26.156473 | 92.31% |
| S2 | `[9,1]` | 3.363967 / 26.156201 | 87.14% |
| S1 | `[17,1]` | 6.267947 / 26.170409 | 76.05% |

All five tiers remain enabled. The generic fused family kernel is retained for
unsupported family contracts and planner overflow (`B * ceil(Q / 2) > 32`),
not as a per-tier runtime performance switch.

## Realistic bf16 validation snapshot

Status below reflects the current custom-op-supported realistic cases validated
on Atlas 350 through July 24, 2026.

Custom-op realistic cases are limited to:

- `family_4x64` with `top_k <= 2048`
- `family_32x128` with `top_k <= 2048`
- `family_64x128` with `top_k <= 2048`

Current custom-op realistic case set: `6 / 6` DSA workflow cases.

| Dataset | Case ID | Family | Status | Notes |
| --- | --- | --- | --- | --- |
| `realistic_prefill` | `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` | `family_64x128` | Passed | Current 32-AIC/64-AIV path passed sampled score-set and repeated-launch validation; the historical 16-task path passed full workflow conformance against vLLM-Ascend. |
| `realistic_prefill` | `deepseek_v4_flash_flashmla_prefill_q4096_ctx32768_top512` | `family_64x128` | Not completed | Full-shape case was not rerun after the task-queue accuracy fix. |
| `realistic_prefill` | `deepseek_v4_pro_vllm_prefill_q4096_ctx131072_top1024` | `family_64x128` | Not completed | Full-shape run was stopped before completion and was not restarted. |
| `realistic_decode` | `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` | `family_64x128` | Passed | Full Top-k score sets pass exactly in 2.953s. |
| `realistic_decode` | `deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512` | `family_64x128` | Passed | Exact indices and workflow output/LSE pass. |
| `realistic_decode` | `deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024` | `family_64x128` | Passed | Exact indices and workflow output/LSE pass. |

At this snapshot:

- Passed: `4 / 6`
- Sampled pass: `0 / 6`
- Failed: `0 / 6`
- Not completed: `2 / 6`

## Unsupported realistic cases

None in the current 6-case DSA workflow realistic set.
