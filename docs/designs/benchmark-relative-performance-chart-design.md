# Benchmark Relative Performance Chart Design

## Status

Approved for implementation.

## Context

The benchmark chart currently plots case latency in microseconds. Cases are
ordered by shape size, but their latency range can span several orders of
magnitude. Large cases therefore determine most of the y-axis range and make
meaningful performance differences on small cases difficult to see.

The published benchmark records already provide the required structured
`latency_ms` data. Relative performance is a presentation concern and must be
derived in the web application. This change does not modify backend output,
published JSON, or the published-data contract.

## Decision

When the current operator and dataset contain an eligible baseline, the chart
plots per-case performance relative to that baseline instead of raw latency.
Baseline selection is automatic and follows this priority:

1. Use the first eligible CUDA series in the view model's stable series order
   when the current dataset contains CUDA data.
2. Otherwise, use the first eligible CANN Ops series in that same order when
   the current dataset contains CANN Ops data.
3. If neither exists, retain the current raw-latency chart.

An eligible candidate has at least one finite, strictly positive latency point
in the current dataset. A CUDA candidate is a series whose records use the
`nvidia` or `gpu` backend. A CANN Ops candidate is a series whose records use
the `cann_ops_library` implementation. The view model already provides a
deterministic series order, so no new run-name or display-name rule is needed.

The selected baseline applies to the entire chart. A case with no point for
that baseline is left blank; it does not fall back to another baseline.

## Relative Performance Semantics

For implementation `i` and case `c`, the plotted value is:

```text
relative_performance(i, c) = baseline_latency(c) / implementation_latency(i, c)
```

This gives the chart the following semantics:

- every valid positive-latency baseline point is `1`;
- values greater than `1` are faster than the baseline;
- values less than `1` are slower than the baseline;
- a value of `2` means twice the baseline performance;
- a value of `0.5` means half the baseline performance, or two times slower.

The y-axis remains linear. Its label identifies the active baseline:

- `Performance vs CUDA (x)`; or
- `Performance vs CANN Ops (x)`.

A visible reference line marks `1x`.

The existing summary remains a geometric mean over cases with valid paired
measurements. It uses the same relative-performance values as the chart. A
geometric mean above `1` is formatted as faster; one below `1` is formatted as
slower; exactly `1` is formatted as equal performance. Ratios and summary
values retain full precision for plotting and geometric-mean calculations;
only their displayed text is rounded to two decimal places. A series with no
valid pairs is omitted from the summary. The summary title identifies the
active baseline.

## Frontend Data Flow

The view model continues to expose raw `ChartSeries` points backed by
`latency_ms`. The frontend derives an active baseline descriptor from all
series available for the current operator and dataset, using the eligibility
and structured-field predicates defined above. Display names are not used for
baseline detection.

The application passes the active baseline to both the series filter and the
chart. The chart builds derived ECharts data without mutating `ChartSeries`:

```text
published latency records
        |
        v
raw ChartSeries for operator + dataset
        |
        +--> choose CUDA or CANN Ops baseline
        |
        +--> derive per-case relative-performance values
        |
        +--> render chart and summary
        |
        +--> render tooltip from the original latency points
```

If no eligible baseline exists, the active baseline is absent and the chart
uses its current microsecond conversion and `latency us` y-axis. It renders no
`1x` reference line, relative tooltip row, or locked series, and the summary
area is hidden entirely.

## Series Filter Behavior

The active baseline is mandatory:

- it is automatically included in the selected series;
- it remains selected when the dataset changes;
- it cannot be toggled off;
- it is visually distinguished with a lock icon and an accessible locked
  state;
- non-baseline series retain their current toggle behavior.

The locked baseline control uses the native disabled state. Its accessible
name identifies it as the locked baseline, while the lock icon is decorative.
The application also enforces the invariant as state changes rather than
relying only on the disabled button. On an operator or dataset transition, it
removes unavailable series, retains ordinary selections that remain available,
removes the old baseline when it is unavailable, and adds the new baseline.
If nothing remains selected, it retains the existing behavior of selecting all
available series. This prevents stale selection state from removing the
baseline during transitions.

The single metric control displays `Relative performance` while a baseline is
active. It displays `Latency` when the chart falls back to raw latency. This is
an automatic mode, not a user-selectable latency/performance toggle.

## Tooltip Behavior

Hovering or clicking a rendered point continues to show its original latency.
The tooltip retains:

- series name;
- case ID;
- latency in microseconds;
- shape;
- dtype.

It also identifies the active baseline and formats the point as baseline,
faster, slower, or equal performance. Latency remains formatted to two decimal
places in microseconds and ratios to two decimal places. The baseline's own
tooltip says `baseline`; a non-baseline point at exactly `1` says equal
performance. Relative-performance plotting must not replace or round the
source latency stored in `ChartSeries`.

## Missing And Invalid Data

Derived values are `null` when either the implementation point or its selected
baseline point is missing. They are also `null` when either latency is not a
finite, strictly positive number. The chart keeps `connectNulls: false`, so
missing ratios produce visible breaks and never become `NaN`, `Infinity`, or
misleading zero values.

The baseline choice is dataset-wide. A missing CUDA point for one case does
not cause that case to use CANN Ops. If the dataset has neither CUDA nor CANN
Ops, all available series remain usable on the raw-latency chart and no series
is locked as a baseline. Points are paired by `ChartPoint.caseId`; each series
is expected to contain at most one point per case under the existing view-model
contract.

## Scope

Expected implementation changes are limited to the web application, primarily:

- baseline selection and mandatory-series state in `web/src/App.tsx`;
- the locked state in `web/src/components/RunFilters.tsx`;
- ratio derivation, labels, summary, and tooltip formatting in
  `web/src/components/BenchmarkChart.tsx`;
- focused frontend types, styles, and tests as required.

No Python backend, operator plugin, benchmark-record schema, published record,
or data-contract change is in scope.

## Verification

Automated tests must cover:

- CUDA takes priority when the dataset contains both CUDA and CANN Ops;
- multiple eligible candidates resolve to the first candidate in stable view-
  model order;
- CANN Ops is selected when CUDA is absent;
- raw latency is retained when both baselines are absent;
- valid baseline points produce `1` and other series use
  `baseline_latency / implementation_latency`;
- missing, non-finite, or non-positive paired values produce `null` and a
  disconnected line;
- y-axis and summary labels identify the active baseline;
- summary values use the geometric mean of valid per-case performance ratios;
- tooltips retain the original latency and show relative performance;
- the active baseline is selected and cannot be toggled off;
- changing dataset selects and locks the new dataset's baseline while
  retaining ordinary selections that remain available;
- relative-only reference lines, summaries, tooltip rows, and locks are absent
  in raw-latency fallback mode.

Completion requires the full web test suite and production build to pass.

## Acceptance Criteria

The change is complete when:

1. Relative differences on small and large cases are plotted on the same
   baseline-normalized scale.
2. Valid CUDA points are `1` whenever the current dataset has CUDA data.
3. Valid CANN Ops points are `1` when the current dataset has no CUDA but has
   CANN Ops.
4. A selected baseline cannot be deselected.
5. Cases without the selected baseline are blank and never use a mixed
   per-case fallback.
6. Datasets without either baseline continue to display raw latency.
7. Point tooltips continue to expose original latency values.
8. Published benchmark data and backend contracts remain unchanged.
