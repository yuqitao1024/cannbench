# Sparse Attention V3.2 Decode Automatic Head64 Dispatch

Status: approved for implementation on 2026-07-30.

## Goal

Route every supported BF16 `family_hd576` decode input through the latest
Head64 P4 fused ping-pong path without requiring benchmark endpoint tuning.
Keep the existing V3.2 prefill Head64 P1 automatic route unchanged.

## Scope

The automatic decode route applies when all of the following are true:

- `phase == "decode"`;
- `family == "family_hd576"`;
- query and shared KV use BF16;
- query heads are 128 and KV heads are 1;
- query and shared-KV head dimensions are 576;
- value head dimension is 512;
- selected-token count is at most 2048.

Batch size, query-token count, context-token count, and the causal flag remain
dynamic within the existing Head64 plan contract.

## Dispatch Contract

The Python wrapper continues to accept these tuning pairs:

- `(1, 1)`: no explicit Head64 tuning, currently the wrapper default;
- `(64, 1)`, `(64, 2)`, and `(64, 4)`: explicit Head64 tuning.

The tuple is `(head_tile, selected_partitions)`. `(1, 1)` is the existing
generic configuration, not Head64 P1. When the incoming pair is `(1, 1)`, the
Host bridge may upgrade a supported shape using these automatic rules:

- the existing exact V3.2 prefill predicate selects Head64 P1;
- any decode input matching the supported specification above selects Head64
  P4;
- every other input retains the generic implementation.

Explicit Head64 tuning remains unchanged. The rule belongs in the
operator-local Sparse Attention Host bridge, not in CannBench CLI, backend, or
workflow code.

## Execution

The current V3.2 decode case has `B=2`, `Q=2`, two 64-head groups, and four
selected-token partitions. It therefore creates 32 logical tasks, capped at 32
physical AIC tasks, and launches the latest Head64 fused ping-pong kernel plus
the existing Combine kernel.

Prefill continues to use Head64 P1 with direct BF16 output. It does not allocate
partition outputs and does not launch Combine.

## Verification

Add an operator-local regression test before changing production code. The test
must fail while decode defaults remain `(1, 1)` and pass only when matching
decode inputs resolve to `(64, 4)`. It must also assert that prefill still
resolves to `(64, 1)` and unrelated inputs retain `(1, 1)`.

Remote validation is limited to the current DeepSeek V3.2 workflow cases:

- decode must pass the current accuracy check and profile
  `sparse_attention_head64_fused_kernel` plus Combine, without the generic
  `sparse_attention_fused_family_hd512_kernel`;
- prefill must continue to pass and profile the Head64 P1 fused kernel without
  Combine;
- both workflows must be collected through CannBench commands and republished
  under their canonical run names.

Broader decode accuracy and boundary validation is intentionally deferred.
