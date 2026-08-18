# Task 1 Report: Baseline Domain Model

## Implementation

- Added `PerformanceBaselineKind`, `PerformanceBaseline`, and the expanded `MetricOption` union to `web/src/types.ts`.
- Added pure baseline helpers in `web/src/data/performanceBaseline.ts`:
  - `selectPerformanceBaseline` selects the first eligible CUDA series, then the first eligible CANN Ops series.
  - `relativePerformanceValue` computes baseline latency divided by implementation latency and rejects invalid values.
  - `enforceSelectedSeries` removes unavailable selections, selects all available series when none remain, and ensures the baseline is selected.
- Added 13 Vitest cases in `web/src/data/performanceBaseline.test.ts` covering CUDA priority, CANN fallback, no baseline, ratio calculation and invalid inputs, and selection invariants.

## TDD Evidence

### RED

Command:

```bash
cd web
npx vitest run src/data/performanceBaseline.test.ts
```

Result: failed as expected before implementation because Vitest could not resolve `./performanceBaseline` from the new test file. No tests ran.

### GREEN

Command:

```bash
cd web
npx vitest run src/data/performanceBaseline.test.ts
```

Result:

```text
Test Files  1 passed (1)
Tests       13 passed (13)
```

Additional verification:

```bash
cd web
npx vitest run
```

Result: `17` test files passed, `120` tests passed.

```bash
cd web
npm run build
```

Result: TypeScript and Vite production build passed. Vite emitted the existing chunk-size warning for bundles larger than 500 kB.

```bash
git diff --check
```

Result: clean.

## Files Changed

- `web/src/types.ts`
- `web/src/data/performanceBaseline.ts`
- `web/src/data/performanceBaseline.test.ts`
- `.superpowers/sdd/task-1-report.md`

## Self-Review

- Public framework/backend code was not modified.
- The helpers are pure and use existing `ChartSeries`, `ChartPoint`, and `BenchmarkRecord` fields.
- CUDA backend aliases `nvidia` and `gpu` are supported.
- Invalid latency values are excluded from baseline eligibility and relative performance calculations.
- Selection order is deterministic and follows input order.

## Concerns

- The production build retains the repository's pre-existing large-chunk warning; it does not affect correctness or this task's domain model.
