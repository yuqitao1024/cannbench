# Lightning Indexer 32 Mixed-Task Prefill Design

## Context

The target Atlas 350 device exposes 32 AICs and 64 AIVs. The Lightning
Indexer fused kernels use `KERNEL_TYPE_MIX_AIC_1_2`, so one mixed task maps to
one AIC and two AIVs. Before this change, the 16-task cap therefore used only
16 AICs and 32 AIVs; 32 mixed tasks are required to make the full device
available. Source review also found that the existing common fused VFs still
use 256 threads, while the newer decode and candidate VFs already use the
requested 1024.

The existing V3.2 prefill reference is BF16
`B=1, Q=4096, C=32768, H=64, D=128, K=2048`. Its 16-task common fused path was
measured by msopprof at 11.218775 seconds of kernel time after the profiler
timeout was increased to 100 seconds.

## Scope

Change the production common fused task cap from 16 to 32 in both operator-
local implementations:

- `lightning_indexer_fused_family_4x64.asc`
- `lightning_indexer_fused_family_64x128.asc`

Standardize both common fused VFs from 256 to 1024 SIMT threads at the same
time, as explicitly required for all SIMT paths.

The host launch continues to use:

```text
used_core_num = min(total_rows, 32)
```

No public backend, CLI, configuration, result schema, or plugin boundary
changes are required.

## Preserved Behavior

This correction changes the maximum number of row-parallel mixed tasks and
standardizes the SIMT launch width. It preserves:

- `KERNEL_TYPE_MIX_AIC_1_2`
- cross-core synchronization mode 2 and flag 0
- participation of both AIVs in the mode-2 handshake; the first AIV continues
  to own post-processing while the second AIV executes synchronization only
- 1024 SIMT threads per launched VF after replacing the inherited 256-thread
  setting
- the existing row ownership and output layout
- current dtype, shape, and TopK constraints
- the disabled state of the experimental two-Query-atom prefill candidate

The V3.2 context-sharded decode path remains at 16 mixed tasks. Its task count
is algorithmic: two batches each use eight 4096-token Context shards. Each
mixed task contains one AIC that computes both Query rows and two AIVs that
post-process one Query row each. Moving decode to 32 tasks would require a
separate shard-size and reduction design rather than a cap correction.

## Tests And Validation

Operator-local source contract tests will first be changed to require a
32-task cap and 1024 SIMT threads for both common fused families and to reject
the stale 16-task and 256-thread settings. The implementation will then be
updated to satisfy those tests.

Local verification will run the Lightning Indexer SIMT source tests followed
by the full repository test suite. Remote verification on the target device
will:

1. rebuild the custom operator;
2. run the exact V3.2 prefill correctness and repeated stability checks;
3. profile the exact case with the 100-second msopprof binary;
4. confirm `Block Dim = 32` and `Mix Block Dim = 64`;
5. compare kernel time with the 16-task 11.218775-second baseline.

The 32-task change is retained only if correctness and stability pass and the
kernel-side time does not regress. If it regresses, the measured evidence is
recorded and the production cap remains 16.

## Documentation

Update the operator-local SIMT README and parallel-splitting research notes so
they describe the device as 32 AICs plus 64 AIVs, distinguish hardware task
capacity from decode shard geometry, and record the new profile result.
