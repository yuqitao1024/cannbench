# DSA HD256 and HD576 CV Fusion Design

## Goal

Add BF16 custom-op coverage for the four realistic DSA workflows that are not
covered by the current fast paths:

- `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048`
- `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048`
- `glm52_vllm_ascend_prefill_q4096_ctx131072_top2048`
- `glm52_vllm_ascend_decode_b3_q3_ctx131072_top2048`

The workflow remains two operator calls: `lightning_indexer` produces indices,
then `sparse_attention` consumes them and produces output and LSE.

## Scope

- BF16 only for implementation and remote accuracy validation.
- Preserve the operator-level interfaces and workflow manifests.
- Preserve Cube-to-Vector fusion inside each device tile.
- Do not introduce a fallback, a GM score intermediate, or public-layer
  operator-name branching.
- Keep all implementation changes in the two operator packages.
- Reuse existing synchronization sites; do not add new Basic API or CrossCore
  synchronization dependencies.

FP16 and FP32 validation, workflow-level fusion into one operator, and launch
count optimization are outside this change.

## Current Gaps

The FlashMLA HD576 workflows use a `64x128` indexer with `top_k=2048`. The
current UB Top-k implementation can sort 4096 candidates, so a merge of 2048
retained candidates and a 128-token context tile already fits. Only the host
dispatch limit incorrectly rejects this shape.

The GLM HD256 workflows use a `32x128` indexer family, which is not dispatched
today. Sparse attention currently dispatches only HD128 and HD512 even though
the HD512 Cube path already accumulates K in 64-element tiles and its UB score
area already supports 2048 selected tokens.

## Lightning Indexer Design

Generalize the existing `64x128` source into one wide `x128` implementation
whose runtime head count is either 32 or 64. Static L0/L1 buffers retain the
current maximum 64-head capacity. Tensor layouts, MMAD `m`, Fixpipe `mSize`,
weight strides, and the SIMT head-reduction loop use the runtime head count.

The fused tile data flow remains:

```text
GM query/key -> L1 -> L0A/L0B -> Cube MMAD -> L0C -> Fixpipe -> UB
UB scores + GM weights -> SIMT reduction and Top-k merge -> GM running Top-k
```

Both `family_32x128` and `family_64x128` support prefill and decode with
`top_k <= 2048`. The existing `family_4x64` behavior remains unchanged.

The host continues to tile context by 128 and query by the established family
tile size. This change adds shape coverage; it does not attempt to remove the
cross-launch GM running Top-k state.

## Sparse Attention Design

Generalize only the existing BF16 fused HD512 path into a wide-head path with a
runtime `head_dim` in `{256, 512, 576}`. The existing static per-tile buffers
remain based on `K_TILE=64` and `N_TILE=64`; therefore the new dimensions require
4 and 9 K iterations respectively, while HD512 continues to require 8.

The runtime head dimension is used for query/key/value/output strides and the
SIMT output loop. MMAD continues to accumulate each K tile into the same L0C
score tile. After the last K tile, Fixpipe writes scores directly to UB and the
existing SIMT VF performs masking, softmax, LSE, and weighted value reduction.

```text
GM selected K tiles -> UB/TSCM -> L0B
GM query K tile     -> L1/L0A  -> Cube accumulation -> L0C -> UB scores
UB scores + GM V/indices -> SIMT softmax/LSE/PxV -> GM output/LSE
```

Dispatch adds `family_hd256` and `family_hd576`; all wide families require
`kv_heads=1` and `selected_tokens <= 2048`. Prefill and decode are supported,
including decode query lengths 2 and 3. The existing HD128 family and its decode
restriction are unchanged.

The non-BF16 query-pack and split score/postprocess compatibility path is not
extended for the new families.

## Tests

Test-first coverage will establish the new contracts before production edits:

1. Dispatch tests require `32x128` to select `family_32x128`, HD256 to select
   `family_hd256`, and HD576 to select `family_hd576`.
2. Source-layout tests require runtime head counts/dimensions while preserving
   MMAD, L0C-to-UB Fixpipe, `asc_vf_call`, and the absence of a GM score buffer in
   the BF16 fused path.
3. Host validation tests require `top_k <= 2048` and selected tokens `<= 2048`
   for the new families.
4. Existing HD128, HD512, 4x64, and 64x128 tests must remain green.
5. The full repository suite must pass.

## Remote Validation

Build both v1 custom-op packages from an isolated copy on the Atlas 350 node.
Run a reduced BF16 case for each new family first, then run all four realistic
workflow cases without profiling. Each workflow run validates:

- indexer Top-k score-set accuracy and index range;
- sparse attention output against the BF16 reference;
- LSE against the float reference;
- absence of runtime errors, deadlocks, and OOM.

For Q4096 prefill cases, execute the complete custom-op output and validate
representative query rows across all heads if a full reference would be
prohibitively expensive. Decode cases are validated in full.

## Completion Criteria

- All 15 realistic DSA workflow cases select custom-op fast paths.
- The four newly supported BF16 workflows pass remote accuracy validation.
- The previously supported 11 workflows retain their dispatch behavior.
- No framework/public-layer files gain DSA-specific logic.
