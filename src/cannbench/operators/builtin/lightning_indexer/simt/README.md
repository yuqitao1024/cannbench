# lightning_indexer SIMT

This directory contains Ascend SIMT integration for `lightning_indexer`.

Fast-path families:

- `family_64x128`
- `family_4x64`

Unsupported shapes use plugin-local fallback.

## Realistic bf16 validation snapshot

Status below reflects the current custom-op-supported realistic cases validated on Atlas 350 on July 22, 2026.

Supported realistic cases are limited to:

- `family_4x64` with `top_k <= 2048`
- `family_64x128` with `top_k <= 1024`

Current supported realistic case set: `9` cases.

| Dataset | Case ID | Family | Status | Notes |
| --- | --- | --- | --- | --- |
| `realistic_prefill` | `deepseek_v32_prefill_b1_q128_ctx16384_top2048` | `family_4x64` | Failed | Top-k tail ordering / merge mismatch. |
| `realistic_prefill` | `deepseek_v32_prefill_b1_q128_ctx32768_top2048` | `family_4x64` | Passed |  |
| `realistic_prefill` | `deepseek_v32_prefill_b1_q128_ctx65536_top2048` | `family_4x64` | Passed |  |
| `realistic_prefill` | `deepseek_v32_prefill_b1_q128_ctx131072_top2048` | `family_4x64` | Failed | Top-k tail ordering / merge mismatch. |
| `realistic_prefill` | `deepseek_v32_prefill_b2_q128_ctx65536_top2048` | `family_4x64` | Failed | Top-k tail ordering / merge mismatch. |
| `realistic_prefill` | `deepseek_128k_prefill_microbatch_top2048` | `family_4x64` | Failed | Top-k tail ordering / merge mismatch. |
| `realistic_prefill` | `deepseek_v4_flash_flashmla_prefill_q4096_ctx32768_top512` | `family_64x128` | Not completed | Re-run was stopped after more than 15 minutes; custom op remained in-flight. |
| `realistic_decode` | `deepseek_128k_decode_top2048` | `family_4x64` | Passed |  |
| `realistic_decode` | `deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512` | `family_64x128` | Failed | Large mismatch from rank 0 on batch-16 decode. |

At this snapshot:

- Passed: `3 / 9`
- Failed: `5 / 9`
- Not completed: `1 / 9`

## Unsupported realistic cases

The following realistic cases do not use the current custom-op fast path and remain on the plugin-local fallback path.

| Dataset | Case ID | Reason |
| --- | --- | --- |
| `realistic_prefill` | `glm52_vllm_ascend_prefill_q4096_ctx131072_top2048` | `index_heads=32`, `index_dim=128`; not in `family_4x64` or `family_64x128`. |
| `realistic_prefill` | `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` | `family_64x128`, but `top_k=2048` exceeds current custom-op limit `top_k <= 1024`. |
| `realistic_decode` | `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` | `family_64x128`, but `top_k=2048` exceeds current custom-op limit `top_k <= 1024`. |
| `realistic_decode` | `glm52_vllm_ascend_decode_b3_q3_ctx131072_top2048` | `index_heads=32`, `index_dim=128`; not in `family_4x64` or `family_64x128`. |
