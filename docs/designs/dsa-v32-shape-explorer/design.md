# DSA V3.2 Shape Explorer Design

## Status

Approved design. The visual reference is
[`shape-explorer-mockup.html`](./shape-explorer-mockup.html).

This design covers the first delivery for the two DeepSeek V3.2 canonical
workflow cases:

- `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048`
- `deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048`

The data contract and renderer must support later DSA cases without adding
case-specific frontend code.

## Goals

1. Open a case-specific shape explorer from the existing benchmark case table.
2. Show the complete Indexer-to-Sparse-Attention matrix transformation.
3. Annotate every matrix axis with its symbol, actual value, and meaning.
4. Animate a single declarative trace with playback and direct step selection.
5. Show current device task, core, and tile decomposition for canonical decode.
6. Add the complete canonical prefill matrix analysis without claiming device
   tiling or core allocation that is not yet an optimization contract.
7. Keep workflow and implementation rules inside operator packages.

## Non-Goals

- Visualizing all DSA smoke, realistic, and stress cases in the first delivery.
- Inferring device execution from profiler output in the browser.
- Showing rejected or proposed decode optimization designs.
- Showing prefill device tiling or core allocation.
- Drawing tensors at literal pixel ratios.
- Moving DSA-specific formulas or dataset mappings into `serve.py`, common core,
  or React components.

## User Entry And Routing

The existing `shape` cell becomes an interactive link when a case has a shape
trace. It retains the current table density and uses hover, keyboard focus, and
link semantics rather than an extra action button.

Activation opens a new browser tab with a stable deep link:

```text
/shape-explorer?operator=dsa_decode&dataset=realistic&case=<case-id>
```

The benchmark page and its filter state remain unchanged. Cases without a
trace keep a non-interactive shape cell.

## Architecture

```text
CaseTable shape cell
        |
        | stable deep link
        v
Generic shape-trace endpoint
        |
        v
Operator registry -> plugin.build_shape_trace(...)
        |
        +-- workflow/component case metadata
        +-- operator-local algorithm trace generator
        +-- maximum common numeric vN device descriptor
        |
        v
ShapeTrace JSON -> generic React renderer
```

Add a generic `ShapeTrace` contract and an optional
`OperatorPlugin.build_shape_trace` hook. The API endpoint validates parameters,
resolves the selected plugin through the registry, invokes the hook, and
serializes the result. It must not branch on `dsa_decode`, `dsa_prefill`, or any
concrete case ID.

DSA calculation rules stay in the workflow packages:

```text
src/cannbench/operators/builtin/dsa_decode/
src/cannbench/operators/builtin/dsa_prefill/
```

Generic trace types may live beside the plugin contract. They may describe
stages, tensors, axes, operations, and device tasks, but must not encode DSA
formulas or dataset mappings.

## Trace Contract

The response contains facts, not React layout instructions:

```text
identity
  schema_version / operator / dataset / case_id / phase

symbols[]
  id / value / label / meaning

algorithm_stages[]
  id / component / title / formula / scope
  tensors[] / operation / annotations

device_execution?
  status / implementation / version
  kernels[] / tasks[] / tiles[] / local tensors[]
```

Each tensor axis contains:

- `symbol`, such as `H`, `Dqk`, or `S`;
- `value`, such as `128`, `576`, or `2048`;
- `meaning`, such as query heads or selected tokens;
- `role`: `preserved`, `contracted`, `reduced`, or `produced`.

Supported first-delivery operation types are inputs, matmul, elementwise,
reduction, TopK, gather, softmax/LSE, output, device task split, and device tile
loop. The frontend renders these types generically.

## Shape Scaling

Matrix geometry must communicate approximate proportions without becoming
unusable for ratios such as `32768:128`.

Use one monotonic nonlinear scale for every visible axis:

```text
display_length(dimension) = clamp(min_px, scale * dimension^0.3, max_px)
```

The final constants are responsive design tokens. The following invariants are
contractual:

1. A larger dimension never renders shorter than a smaller dimension in the
   same comparison context.
2. The same dimension has the same rendered length within a stage.
3. Matrix orientation is preserved; a wide matrix cannot appear tall or square.
4. Extreme ratios are compressed but remain visually distinct.
5. Exact axis values and important ratios, such as `C / Di = 256x`, are always
   annotated.
6. Vectors render as strips rather than square matrices.
7. Overflow scrolls horizontally; labels remain attached to their axes.

The approved mockup is the visual baseline for these rules.

## Algorithm Trace

Both phases use the same stage vocabulary:

```text
Index inputs
  -> per-head Indexer matmul
  -> ReLU, weight, and head reduction
  -> causal mask and TopK
  -> selected KV gather
  -> QK matmul
  -> Softmax and LSE
  -> PV matmul and output
```

The main animation collapses `(B,Q)` into `R=B*Q`. This makes decode and
prefill directly comparable and avoids confusing logical matrix operations
with physical BHTD/BTHD layout transformations. The Inspector lists canonical
input and output layouts separately.

### Decode Symbols

```text
B=2, Q=2, R=4, Hi=64, Di=128, C=32768,
H=128, Hkv=1, S=2048, Dqk=576, Dv=512
```

### Prefill Symbols

```text
B=1, Q=4096, R=4096, Hi=64, Di=128, C=32768,
H=128, Hkv=1, S=2048, Dqk=576, Dv=512
```

The per-row matrix formulas are identical. Aggregate prefill shapes include:

```text
index_query   [R,Hi,Di]   = [4096,64,128]
head_scores   [R,Hi,C]    = [4096,64,32768]  logical only
index_scores  [R,C]       = [4096,32768]     logical only
indices       [R,S]       = [4096,2048]

query         [R,H,Dqk]   = [4096,128,576]
selected K    [R,S,Dqk]   = [4096,2048,576]  logical view
selected V    [R,S,Dv]    = [4096,2048,512]  logical view
scores / P    [R,H,S]     = [4096,128,2048]
output        [R,H,Dv]    = [4096,128,512]
LSE           [R,H]       = [4096,128]
```

Logical tensors must be labeled as logical views when the implementation does
not materialize them in global memory.

## Current Decode Device Trace

Only the current implementation is shown. Proposed, rejected, and candidate
layouts are excluded.

Select the maximum numeric version present for every workflow component.
Numeric comparison is required, so `v10` sorts after `v2`. After selecting that
maximum common implementation version, require its matching device trace
descriptor. If the descriptor is absent, return `unavailable` rather than
silently showing an older layout.

For the current V3.2 BF16 `v2` implementation, the trace records:

### Lightning Indexer

- Query atom size: 2 query tokens.
- Base tasks: `B * ceil(Q/2) = 2`.
- Context shards: 16, selected as the largest supported count keeping mixed
  tasks at or below 32.
- Mixed tasks: `2 * 16 = 32`.
- Context per shard: `32768 / 16 = 2048`.
- Context tile: 32, giving 64 tiles per shard.
- Tile matrix view: query atom `[2,64,128]` against key tile `[32,128]`,
  producing score tiles `[2,64,32]` before head reduction.
- The canonical shape uses the implemented distributed radix TopK path and
  returns indices `[2,2,2048]`.

### Sparse Attention

- Automatic Head64 routing with two head groups per query row.
- Selected partitions: `P=1` for the current automatic decode route.
- Logical tasks: `R * ceil(H/64) = 4 * 2 = 8`.
- Used AIC count: 8 for this task count and implementation limit.
- The Head64 plan keeps `selected_tile=64`, while the active fused vLLM
  rolling branch executes an outer selected-token tile of 128.
- QK compute tile: `[64,256] x [256,128] -> [64,128]`. The full 576
  feature dimension contracts as `256 + 256 + 64`.
- The active fused value tile is 256, so PV covers `Dv=512` in two
  256-wide output tiles.
- Online softmax/PV produces direct output for `P=1`; no partial-output Combine
  stage is shown.

The version-local descriptor owns constants that cannot be derived from the
case. Operator-local tests compare descriptor assumptions with the current
implementation source contract so source changes cannot silently leave the
visualization stale.

## Prefill Device State

Prefill has no device execution trace in this delivery. The Device tab remains
visible for a consistent layout and shows:

```text
No device trace for prefill
Prefill is not optimized yet. This view intentionally shows only the
algorithm-level matrix flow.
```

No decode tile values may be reused for prefill.

## Interaction

- The explorer loads the deep-linked case and begins at the first matrix stage.
- `Play` advances through the trace and stops at Output.
- `Pause`, `Prev`, `Next`, speed selection, and direct timeline selection use a
  single stage state.
- Switching Decode/Prefill resets playback to the first stage.
- `prefers-reduced-motion` disables automatic animation while preserving manual
  navigation.
- All controls and the source shape link are keyboard accessible.
- The page uses English UI text to match the current site.

## Error Handling

- Missing or malformed query parameters produce an invalid-link state.
- Unknown cases and unsupported traces produce a not-found state.
- Plugins without the optional trace hook do not expose an interactive shape
  cell.
- Algorithm success with device failure keeps the algorithm view usable and
  reports the unavailable device version only in the Device view.
- Invalid trace schemas fail at the API boundary and never reach the renderer.

## Documentation Delivery

The completed parallel V3.2 prefill analysis is
[`dsa-v2-prefill-full-shape-analysis.zh-CN.md`](../../optimization/dsa-v2-prefill-full-shape-analysis.zh-CN.md).
It covers:

- symbols, causal positions, and valid lengths;
- Indexer inputs and logical intermediates;
- per-row and aggregate matrix calculations;
- TopK and the workflow binding;
- shared KV gather, QK, Softmax/LSE, PV, and output;
- memory/materialization caveats;
- an explicit statement that prefill device tile/core decomposition is not yet
  documented.

Cross-link the decode analysis, prefill analysis, this design, and the approved
mockup. Correct stale decode statements only when current source evidence proves
they no longer describe the maximum `vN` implementation.

## Testing And Verification

### Operator-Local Tests

- Validate every canonical V3.2 symbol and derived dimension.
- Validate matrix compatibility and contracted axes for all stages.
- Validate decode and prefill aggregate shapes.
- Validate numeric version ordering, common-version intersection, and missing
  descriptor behavior.
- Validate the current decode device task/tile calculations and source contract.

### Generic Server Tests

- Valid trace response.
- Missing and malformed parameters.
- Unknown operator/case.
- Plugin without a trace hook.
- Trace serialization failure.
- No concrete DSA name branches in request dispatch.

### Frontend Tests

- Shape cell pointer and keyboard activation.
- Stable deep-link construction and new-tab behavior.
- Loading, invalid-link, not-found, and partial device states.
- Decode/Prefill switching and stage reset.
- Playback stops at Output.
- Manual navigation and speed selection.
- Nonlinear sizing is monotonic, consistent, orientation-preserving, and
  bounded.
- Prefill Device tab never renders decode task data.

### Completion Commands

```bash
pytest -q
cd web && npm test
cd web && npm run build
```

Also run targeted searches for concrete DSA names in public dispatch layers and
capture Playwright screenshots at desktop and mobile widths. Compare the final
page against the saved HTML mockup before claiming completion.
