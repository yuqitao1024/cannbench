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

- `family_hd128`: `H = 128`, `KV_H = 1`, `D = 128`, `selected_tokens <= 2048`
- `family_hd256`: `D = 256`, `KV_H = 1`, `selected_tokens <= 2048`
- `family_hd512`: `D = 512`, `KV_H = 1`, `selected_tokens <= 2048`
- `family_hd576`: `D = 576`, `KV_H = 1`, `selected_tokens <= 2048`

Additional `family_hd128` decode restriction:

- `query_tokens = 1`

For the current case set in this repository, that means:

- `61` cases are covered by the implemented fast paths
- `7` cases are not implemented by the current SIMT custom op

## Realistic Case Classification

For the current realistic case sets in this repository:

- `realistic`: `0 / 4` supported
- `realistic_decode`: `5 / 5` DSA workflow cases supported
- `realistic_prefill`: `10 / 10` DSA workflow cases supported

All `15 / 15` current DSA workflow realistic cases route through a sparse-attention
custom-op fast path. The four HD256/HD576 cases require Atlas 350 validation.

Supported realistic-family cases:

- `realistic_decode::deepseek_128k_decode_top2048` via `family_hd128`
- `realistic_decode::deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` via `family_hd576`
- `realistic_decode::deepseek_v4_flash_vllm_decode_b16_q1_ctx32768_top512` via `family_hd512`
- `realistic_decode::deepseek_v4_pro_vllm_decode_b60_q1_ctx131072_top1024` via `family_hd512`
- `realistic_decode::glm52_vllm_ascend_decode_b3_q3_ctx131072_top2048` via `family_hd256`
- `realistic_prefill::deepseek_v32_prefill_b1_q128_ctx16384_top2048` via `family_hd128`
- `realistic_prefill::deepseek_v32_prefill_b1_q128_ctx32768_top2048` via `family_hd128`
- `realistic_prefill::deepseek_v32_prefill_b1_q128_ctx65536_top2048` via `family_hd128`
- `realistic_prefill::deepseek_v32_prefill_b1_q128_ctx131072_top2048` via `family_hd128`
- `realistic_prefill::deepseek_v32_prefill_b2_q128_ctx65536_top2048` via `family_hd128`
- `realistic_prefill::deepseek_128k_prefill_microbatch_top2048` via `family_hd128`
- `realistic_prefill::deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048` via `family_hd576`
- `realistic_prefill::deepseek_v4_flash_flashmla_prefill_q4096_ctx32768_top512` via `family_hd512`
- `realistic_prefill::deepseek_v4_pro_vllm_prefill_q4096_ctx131072_top1024` via `family_hd512`
- `realistic_prefill::glm52_vllm_ascend_prefill_q4096_ctx131072_top2048` via `family_hd256`

## Unimplemented Shape Families

The following case groups do not match the current `family_hd512` / `family_hd128`
fast paths and are therefore not implemented yet.

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

- `Q`: `[B, H, Q, D]`
- `K`: `[B, KV_H, C, D]`
- `V`: `[B, KV_H, C, D]`
- `indices`: `[B, Q, S]`
- `output`: `[B, H, Q, D]`
- `lse`: `[B, H, Q]`

Where:

- `B`: batch size
- `H`: query head count
- `KV_H`: key/value head count
- `Q`: query token count
- `C`: context token count
- `S`: selected sparse token count per query token
- `D`: head dimension

In the common decode case, `Q = 1`. In MQA/GQA style layouts, `H` may be larger than `KV_H`, which means multiple query heads share the same KV head group.

## Computation

For a fixed batch `b`, query head `h`, and query token `q`, the operator first uses
`indices[b, q, :]` to choose the sparse context positions:

```text
K_sparse = gather(K, indices)
V_sparse = gather(V, indices)
```

Logical sparse shapes:

- `K_sparse`: `[B, H, Q, S, D]`
- `V_sparse`: `[B, H, Q, S, D]`

Then sparse attention scores are computed only on the selected positions:

```text
scores[b, h, q, s] = dot(Q[b, h, q, :], K_sparse[b, h, q, s, :]) / sqrt(D)
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
