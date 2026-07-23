# lightning_indexer SIMT

This directory contains Ascend SIMT integration for `lightning_indexer`.

Fast-path families:

- `family_64x128`
- `family_4x64`

Unsupported shapes use plugin-local fallback.

## Realistic bf16 validation snapshot

Status below reflects the current custom-op-supported realistic cases validated on Atlas 350 on July 23, 2026.

Supported realistic cases are limited to:

- `family_4x64` with `top_k <= 2048`
- `family_64x128` with `top_k <= 1024`

Current supported realistic case set: `11` cases.

| Dataset | Case ID | Family | Status | Notes |
| --- | --- | --- | --- | --- |
| `realistic_prefill` | `deepseek_v32_prefill_b1_q128_ctx16384_top2048` | `family_4x64` | Passed | Top-k score set and workflow output/LSE pass; 32 equal-score indices differ. |
| `realistic_prefill` | `deepseek_v32_prefill_b1_q128_ctx32768_top2048` | `family_4x64` | Passed | Top-k score set and workflow output/LSE pass; 15 equal-score indices differ. |
| `realistic_prefill` | `deepseek_v32_prefill_b1_q128_ctx65536_top2048` | `family_4x64` | Passed | Top-k score set and workflow output/LSE pass; 61 equal-score indices differ. |
| `realistic_prefill` | `deepseek_v32_prefill_b1_q128_ctx131072_top2048` | `family_4x64` | Passed | Top-k score set and workflow output/LSE pass; 49 equal-score indices differ. |
| `realistic_prefill` | `deepseek_v32_prefill_b2_q128_ctx65536_top2048` | `family_4x64` | Passed | Top-k score set and workflow output/LSE pass; 61 equal-score indices differ. |
| `realistic_prefill` | `deepseek_128k_prefill_microbatch_top2048` | `family_4x64` | Passed | Top-k score set and workflow output/LSE pass; 49 equal-score indices differ. |
| `realistic_prefill` | `deepseek_v4_flash_flashmla_prefill_q4096_ctx32768_top512` | `family_64x128` | Not completed | Full-shape case was not rerun after the task-queue accuracy fix. |
| `realistic_prefill` | `deepseek_v4_pro_vllm_prefill_q4096_ctx131072_top1024` | `family_64x128` | Not completed | Full-shape run was stopped before completion and was not restarted. |
| `realistic_decode` | `deepseek_128k_decode_top2048` | `family_4x64` | Passed | Exact indices and workflow output/LSE pass. |
| `realistic_decode` | `deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512` | `family_64x128` | Passed | Exact indices and workflow output/LSE pass. |
| `realistic_decode` | `deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024` | `family_64x128` | Passed | Exact indices and workflow output/LSE pass. |

At this snapshot:

- Passed: `9 / 11`
- Failed: `0 / 11`
- Not completed: `2 / 11`

## Unsupported realistic cases

The following realistic cases do not use the current custom-op fast path and remain on the plugin-local fallback path.

| Dataset | Case ID | Reason |
| --- | --- | --- |
| `realistic_prefill` | `glm52_vllm_ascend_prefill_q4096_ctx131072_top2048` | `index_heads=32`, `index_dim=128`; not in `family_4x64` or `family_64x128`. |
| `realistic_prefill` | `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` | `family_64x128`, but `top_k=2048` exceeds current custom-op limit `top_k <= 1024`. |
| `realistic_decode` | `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` | `family_64x128`, but `top_k=2048` exceeds current custom-op limit `top_k <= 1024`. |
| `realistic_decode` | `glm52_vllm_ascend_decode_b3_q3_ctx131072_top2048` | `index_heads=32`, `index_dim=128`; not in `family_4x64` or `family_64x128`. |
