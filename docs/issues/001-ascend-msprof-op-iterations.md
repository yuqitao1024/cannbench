# Issue 001: Ascend msprof op iteration sample count

## Status

Open

## Context

CannBench can execute the target operator multiple times inside one `cannbench operator` process by setting `--iterations N`. For Ascend profiling, the whole operator process is wrapped by `msprof op`.

An initial validation with `softmax/tiny_logits`, `--warmup 1`, and `--iterations 3` produced only one parsed duration sample from `OpBasicInfo.csv`:

```text
Task Duration(us): 2.592000
sample_count: 1
```

The current `msprof op` artifact looked like an operator-level summary instead of a timeline containing one duration row per CannBench iteration.

## Impact

`--iterations N` should not be interpreted as `N` independent kernel timing samples for Ascend `msprof op` until this behavior is investigated further.

## Current Policy

Use `--iterations 1` for current benchmark runs. Treat each `msprof op` invocation as one kernel-side timing sample.

## Follow-Up

- Verify whether `msprof op` can expose one duration per repeated eager-mode operator call.
- Check whether a different Ascend profiling mode, such as timeline/task profiling, exposes per-launch durations.
- If multiple samples are required, add an explicit `--samples` flow that runs multiple independent `msprof op` invocations and aggregates their `Task Duration(us)` values.
