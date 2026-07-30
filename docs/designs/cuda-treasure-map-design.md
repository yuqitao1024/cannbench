# CUDA Treasure Map Modal Design

Date: 2026-06-27

## Status

Not implemented. The repository does not contain the proposed treasure-map
modal, route component, or dark-theme logo trigger. This document is retained
as a pending frontend design.

## Goal

Add a hidden modal page to the frontend that presents the CUDA operator optimization path as a treasure-map-style route. The modal is a dark-theme-only easter egg and is intended to make the optimization methodology memorable without changing the benchmark console workflow.

This first version focuses only on the CUDA / H800 optimization path. Ascend SIMT will be added later as a second route or branch family on the same map.

## User Interaction

### Trigger behavior

- In `light` theme:
  - Triple-clicking the `CANNBench` logo keeps opening the existing GPU JSON import modal.
- In `dark` theme:
  - Triple-clicking the `CANNBench` logo opens the new `CUDA Optimization Map` modal instead.

### Modal behavior

- The map modal is independent from the main page layout.
- It uses the same modal conventions as existing dialogs:
  - close button
  - click backdrop to close
  - `Esc` to close
- It is display-oriented, not workflow-oriented:
  - no form submission
  - no persistent user state
  - no branch toggling in v1

## Information Architecture

The modal uses one main route with branch hints. Main nodes are grouped by real optimization order rather than raw CUDA guide chapter count.

### Main route nodes

1. `Profile the Truth`
2. `Guard Correctness`
3. `Shape Parallel Work`
4. `Cut Data Motion`
5. `Fix Global Access`
6. `Stage Through Shared`
7. `Tune Launch Geometry`
8. `Polish Instructions`

### Branch hints

- `Cut Data Motion`
  - `Pinned / Async / Streams`
- `Fix Global Access`
  - `L2 persistence`
- `Stage Through Shared`
  - `Bank conflicts`
  - `Async G2S copy`
- `Tune Launch Geometry`
  - `Concurrent kernels`
- `Polish Instructions`
  - `Target GPU build`

## Content Model

Each main node and branch node should be backed by structured metadata, not inline JSX strings.

Suggested fields:

- `id`
- `label`
- `kind`
  - `main`
  - `branch`
- `position`
  - normalized map coordinates
- `summary`
  - one-sentence target
- `details`
  - 2-4 concise optimization bullets
- `guideSections`
  - e.g. `10.2.1`, `10.2.3`
- `relatedOptimizationIds`
  - e.g. `O9`, `O10`, `O12`
- `importance`
  - optional visual weight

This metadata should be derived from `docs/cuda-operator-optimization-best-practices.md`, but copied into frontend-safe structured data for rendering.

## Visual Direction

### Core look

- Theme: dark-only hidden experience
- Tone: “night treasure chart” instead of generic dashboard
- Palette: continue the existing Gruvbox dark family
  - deep brown-black background
  - copper/gold route
  - muted blue-green secondary accents
  - pale warm text for labels/tooltips

### Backdrop

- Not a flat panel
- Add subtle:
  - contour-line or chart-line textures
  - grid or coordinate marks
  - low-noise paper/console blend
- Keep opacity and contrast low enough to avoid fighting the route itself

### Route

- Main path is an intentionally curved route, not a straight timeline
- Main nodes appear as larger route markers / treasure pins
- Branch nodes appear as smaller side markers
- The most important optimization stations should have slightly stronger glow or scale:
  - `Fix Global Access`
  - `Stage Through Shared`

### Tooltip

On hover, show a compact floating panel with:

- node title
- one-sentence goal
- 2-4 optimization bullets
- guide section references

Tooltip should feel like a field note / expedition card rather than a system tooltip.

## Layout

### Modal frame

- Title: `CUDA Operator Treasure Route`
- Subtitle: `H800 optimization path`
- Main map canvas centered and dominant
- Small legend in a corner:
  - main route
  - branch route
  - hover for field notes
- Minimal controls:
  - close only

### Responsiveness

- Desktop:
  - full route visible in one modal view
- Tablet / small laptop:
  - route scales down while preserving hover targets
- Mobile:
  - likely switch to vertically stacked or scrollable map treatment
  - still modal, but reduced density

## Component Design

Suggested frontend pieces:

- `CudaTreasureMapModal`
  - modal shell and title area
- `CudaTreasureMap`
  - renders path, nodes, branch connectors, legend
- `TreasureNode`
  - individual map point
- `TreasureTooltip`
  - hover detail card
- `cudaOptimizationRoute.ts`
  - static route metadata

## State Model

Minimal local state:

- `mapOpen: boolean`
- `hoveredNodeId: string | null`

Logo click logic should branch by theme:

- light: existing triple-click import logic
- dark: triple-click map logic

The safest implementation is to keep one click counter mechanism and choose modal target after the threshold is met.

## Accessibility

- Modal must use proper dialog semantics
- Close control must be keyboard reachable
- `Esc` closes
- Nodes must have accessible names
- Hover-only information should have keyboard focus fallback
  - focus on node shows the same tooltip

## Testing

### Interaction tests

- light theme triple-click opens GPU import modal
- dark theme triple-click opens CUDA treasure map modal
- dark theme does not open GPU import modal
- close button and backdrop close the map modal
- keyboard focus can reach route nodes

### Rendering tests

- route metadata renders expected main node labels
- tooltip content shows for hovered/focused node

### Visual verification

- dark theme map feels visually distinct from existing modals
- map remains readable on common viewport widths

## Non-Goals for v1

- No Ascend SIMT route yet
- No clickable route progression state
- No persistence of visited nodes
- No live linkage to benchmark records
- No code execution or profiling integration from the map

## Future Extensions

- Add Ascend SIMT as a second route with a contrasting color family
- Allow route mode toggle:
  - CUDA
  - Ascend SIMT
  - overlay compare
- Add links from nodes to internal docs
- Add diff or benchmark examples tied to selected optimization nodes
