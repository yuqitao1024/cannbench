# lightning_indexer SIMT

This directory contains Ascend SIMT integration for `lightning_indexer`.

Fast-path families:

- `family_32x128`
- `family_64x128`
- `family_4x64`

Unsupported shapes use plugin-local fallback.

## Context-sharded decode benchmark

Measured on Atlas 350 (`dav-3510`, CANN 9.2.0) on July 27, 2026. The target
shape was BF16 `B=2, Q=2, C=32768, H=64, D=128, K=2048`, with valid context
lengths `[[32767, 32768], [32767, 32768]]` and seed 7. Each isolated build used
5 warmups followed by 30 `perf_counter_ns` samples, synchronizing the NPU after
every public custom-op call.

| Path | Commit | Median | Min | Max |
| --- | --- | ---: | ---: | ---: |
| Corrected fused baseline | `d608c0b` | 46.7846 ms | 46.7764 ms | 47.5276 ms |
| Context-sharded score + TopK | `8bac22a` | 2.1788 ms | 2.1745 ms | 2.2077 ms |

The candidate/baseline median ratio is `0.04657`, or a `21.47x` speedup. The
exact-shape context-sharded dispatch remains enabled. Both SIMT entries use
1024 threads: the score VF reports 0 bytes of stack and 22 registers, while
the TopK VF reports 0 bytes of stack and 24 registers.

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
| `realistic_prefill` | `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` | `family_64x128` | Sampled pass | Full custom-op output completed in 49.604s; three query rows have exact Top-k score sets. |
| `realistic_prefill` | `deepseek_v4_flash_flashmla_prefill_q4096_ctx32768_top512` | `family_64x128` | Not completed | Full-shape case was not rerun after the task-queue accuracy fix. |
| `realistic_prefill` | `deepseek_v4_pro_vllm_prefill_q4096_ctx131072_top1024` | `family_64x128` | Not completed | Full-shape run was stopped before completion and was not restarted. |
| `realistic_decode` | `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` | `family_64x128` | Passed | Full Top-k score sets pass exactly in 2.953s. |
| `realistic_decode` | `deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512` | `family_64x128` | Passed | Exact indices and workflow output/LSE pass. |
| `realistic_decode` | `deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024` | `family_64x128` | Passed | Exact indices and workflow output/LSE pass. |

At this snapshot:

- Passed: `3 / 6`
- Sampled pass: `1 / 6`
- Failed: `0 / 6`
- Not completed: `2 / 6`

## Unsupported realistic cases

None in the current 6-case DSA workflow realistic set.
