# Operator And Hardware Visualization Design

## Status

Proposed design for adding two interactive visualization pages to the CannBench
frontend:

- an operator-computation walkthrough page
- a hardware-execution walkthrough page

The design keeps the two pages separate on purpose. They explain different
things, have different interaction density, and benefit from different visual
focus.

## Context

CannBench already has a React + Vite frontend under `web/` and a stable visual
language built around a gruvbox-style dark theme. The current palette lives in:

- `web/src/styles.css`

The repository also already models DSA in two useful layers:

- workflow layer:
  - `lightning_indexer -> sparse_attention`
- operator-internal execution layer:
  - `query_pack`
  - `keys_gather_pack`
  - `score`
  - `decode_direct` or `postprocess`

Relevant source references:

- `src/cannbench/operators/builtin/dsa_decode/__init__.py`
- `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`

## Goals

Build two frontend pages that are:

- clear first, flashy second
- interactive
- visually consistent with the current CannBench gruvbox dark look
- practical to implement in the existing React frontend

The pages should help users answer two different questions.

### Goal A: Operator Computation Clarity

For an operator such as DSA decode or prefill, show:

- input tensors and shapes
- the step-by-step compute path
- intermediate tensors and shape changes
- outputs
- optional tile-level or kernel-level drill-down

### Goal B: Hardware Execution Clarity

For a given operator path, show:

- abstract hardware blocks
- data movement between blocks
- compute stages
- parallel overlap
- wait/notify or queue synchronization behavior

## Non-Goals

- Do not force both explanations into one page.
- Do not build a full generic graph editor in the first phase.
- Do not require exact cycle-accurate hardware replay.
- Do not depend on a 3D engine for the initial rollout.

## Main Decision

Use two separate pages with one shared visual system.

Recommended technology stack:

- graph structure and navigation:
  - `React Flow`
- 2D animation overlay:
  - `react-konva` / `Konva`
- standard React panels for explanation, metadata, and code snippets

This combination is chosen because:

- it fits the current React frontend directly
- it is mature and easy to iterate on
- it cleanly separates graph structure from animation overlays
- it can be extended later without replacing the first version

## Why Not One Component

The operator walkthrough page and the hardware-execution page optimize for
different comprehension tasks.

The operator page is mostly about:

- tensor semantics
- transformation order
- shape evolution

The hardware page is mostly about:

- topology
- movement
- overlap
- synchronization

Trying to solve both with one graph usually makes both harder to read.

## Visual Direction

Preserve and extend the existing gruvbox dark theme.

Base tokens come from the current frontend:

- background:
  - `--bg0`
  - `--bg1`
- text:
  - `--text`
- focus and primary highlight:
  - `--accent`
- secondary highlight:
  - `--accent-2`
- success / active compute:
  - `--green`
- wait / stall / warning:
  - `--red`
- structural lines:
  - `--line`

Recommended page treatment:

- deep layered dark background
- glass-like panels matching existing `surface` and `surface-strong`
- subtle glow on active nodes and edges
- restrained motion
- no permanent full-screen animation noise

Motion should be informative:

- pulse to indicate active compute
- directional particles to indicate data movement
- short event burst to indicate notify
- red lock marker to indicate wait

## Page A: Operator Computation Walkthrough

### Route Examples

- `/viz/operator/dsa-decode`
- `/viz/operator/dsa-prefill`
- future:
  - `/viz/operator/softmax-v3`

### Layout

Use a three-zone layout:

1. top bar
2. left graph canvas
3. right detail panel
4. optional bottom tensor timeline

### Top Bar

Controls:

- operator selector
- phase selector
- dataset or case selector
- dtype selector
- play / pause
- expand level:
  - workflow
  - operator
  - tile

### Main Graph

Use `React Flow` as the primary structure.

Recommended node types:

- `input-node`
- `transform-node`
- `compute-node`
- `reduce-node`
- `output-node`

For DSA decode, the first version should show at least:

- `Q`
- `K`
- `V`
- `indices`
- `query_pack`
- `keys_gather_pack`
- `score`
- `decode_direct` or `postprocess`
- `out`
- `lse`

### Right Detail Panel

Clicking a node opens a stable detail view with:

- human-readable summary
- input tensor list
- output tensor list
- shape changes
- dtype
- pseudo-code snippet
- backing kernel or function name
- related source file link or label

### Bottom Tensor Timeline

Show the tensor-state transitions for the selected step, for example:

- `Q: [B, H, Tq, D]`
- `selected_keys: [B, H, Tq, Tk_sel, D]`
- `scores: [B, H, Tq, Tk_sel]`
- `out: [B, H, Tq, D]`

### Interaction Model

- hover edge:
  - highlight the tensor path
- click node:
  - pin details on the right
- press play:
  - animate one step at a time
- expand step:
  - reveal tile-level substeps when available

### Page A Data Model

```ts
type TensorRef = {
  name: string;
  shape: string[];
  dtype: string;
  layout?: string;
};

type OperatorStep = {
  id: string;
  label: string;
  kind: "input" | "transform" | "compute" | "reduce" | "output";
  inputs: TensorRef[];
  outputs: TensorRef[];
  summary: string;
  pseudocode?: string[];
  kernel?: string;
  next: string[];
};
```

## Page B: Hardware Execution Walkthrough

### Route Examples

- `/viz/hardware/dsa-decode`
- `/viz/hardware/sparse-attention-score`
- `/viz/hardware/softmax-v3`

### Layout

Use a split layout:

1. upper hardware topology view
2. lower time-lane or swim-lane view
3. right-side explanation panel

This is the most important clarity decision in the design.

A hardware topology alone is not enough to explain synchronization. The lower
time-lane view is required to make overlap and wait behavior understandable.

### Upper Topology View

Use `React Flow` for the topology graph.

Recommended hardware node kinds:

- `memory`
- `engine`
- `control`
- `sync`

Typical abstract blocks:

- `GM`
- `L2`
- `L1`
- `UB`
- `MTE2`
- `MTE3`
- `VEC / VF`
- `CUBE`
- `AICORE control`
- `event`

### Lower Time-Lane View

Do not use a graph library here.

Use a custom 2D lane renderer with `Konva` for:

- lane `MTE2`
- lane `VEC/VF`
- lane `CUBE`
- lane `MTE3`
- lane `SYNC`

Each interval should show:

- copy
- compute
- wait
- notify
- overlap windows

### Interaction Model

- click topology edge:
  - highlight the related intervals below
- click time interval:
  - highlight the involved topology nodes and edges above
- scrub timeline:
  - animate movement only for the selected time range
- toggle view:
  - data movement
  - compute
  - synchronization

### Animation Rules

Use animation only to clarify state:

- `GM -> UB`:
  - directional particle flow
- `compute active`:
  - local pulse or glow
- `ub2gm`:
  - reverse directional flow
- `wait`:
  - red lock or pause marker
- `notify`:
  - green burst or ring pulse

### Page B Data Model

```ts
type HardwareNode = {
  id: string;
  label: string;
  kind: "memory" | "engine" | "control" | "sync";
};

type HardwareEvent = {
  id: string;
  lane: "mte2" | "vec" | "cube" | "mte3" | "sync";
  start: number;
  end: number;
  label: string;
  type: "copy" | "compute" | "wait" | "notify";
  source?: string;
  target?: string;
};
```

## Component Breakdown

Recommended component groups:

- `web/src/components/viz/OperatorFlowCanvas.tsx`
- `web/src/components/viz/OperatorStepPanel.tsx`
- `web/src/components/viz/TensorTimeline.tsx`
- `web/src/components/viz/HardwareTopologyCanvas.tsx`
- `web/src/components/viz/HardwareSwimlane.tsx`
- `web/src/components/viz/VizLegend.tsx`

Recommended data groups:

- `web/src/data/viz/operatorSteps.ts`
- `web/src/data/viz/hardwareEvents.ts`
- `web/src/data/viz/dsaDecodeView.ts`

## Accessibility And Clarity Rules

These rules are more important than the animation layer.

- always show labels directly on important nodes
- never encode meaning with color only
- use fixed legends for edge and event types
- keep one primary active path at a time
- allow motion reduction
- allow pausing all animations

Recommended accessibility support:

- reduced-motion mode
- keyboard navigation between nodes and intervals
- static text fallback for every interactive step

## Rollout Plan

### Phase 1

Build Page A for `dsa_decode`.

Scope:

- static graph
- click-to-explain details
- shape transitions
- minimal edge highlight animation

### Phase 2

Build Page B for one representative operator path.

Recommended first candidate:

- `softmax v3`
  or
- `sparse_attention score`

Scope:

- static topology
- linked swim-lane timing
- wait / notify visualization

### Phase 3

Add richer animation overlays.

Scope:

- particle movement
- overlap emphasis
- tile-level expansion

## Future Extensions

- case-driven playback from benchmark metadata
- source-code line linking
- kernel timeline import from profiler output
- operator comparison mode:
  - `simt` vs `cann-ops`
  - `v2` vs `v3`

## Summary

Use two separate pages with one shared visual system.

- operator walkthrough page:
  - `React Flow` as the main structure
- hardware execution page:
  - `React Flow` for topology
  - `Konva` for the time-lane and animation overlay

This gives the best balance of:

- clarity
- technical maturity
- frontend fit with the current CannBench stack
- room for a more cinematic presentation later without sacrificing readability
