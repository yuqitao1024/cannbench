# lightning_indexer SIMT

This directory contains Ascend SIMT integration for `lightning_indexer`.

Fast-path families:

- `family_32x128`
- `family_64x128`
- `family_4x64`

Unsupported shapes use plugin-local fallback.

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
