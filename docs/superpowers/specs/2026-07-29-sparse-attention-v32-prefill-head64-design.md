# Sparse Attention V3.2 Prefill Head64 Design

## Status

Approved design for automatically routing the exact DeepSeek V3.2 BF16
Sparse Attention prefill case through a 32-AIC/64-AIV Head64 fused kernel.

Design date: 2026-07-29.

## Context

The target case is:

```text
case = deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048
phase = prefill
family = family_hd576
dtype = bfloat16
B = 1
H = 128
KV_H = 1
Q = 4096
C = 32768
S = 2048
Dqk = 576
Dv = 512
causal = true
```

The current default BF16 prefill path uses the persistent wide-family fused
kernel. It launches up to 32 mixed tasks, but each logical task computes one
Query head row with `M=1`, only sub-AIV 0 performs sustained work, and the
postprocess VF uses 256 threads. The kernel therefore occupies all AICs without
using the Head64 data reuse, Cube shape, or second AIV introduced by the V3.2
decode work.

The V3.2 decode Head64 design groups 64 Query heads that share one KV head and
uses both AIVs for gather, online softmax, and output updates. Decode has only

```text
B(2) * Q(2) * HeadGroup(2) = 8
```

base tasks, so its 32-core version partitions the selected-token axis four
ways and runs a Combine kernel. Prefill already has

```text
B(1) * Q(4096) * HeadGroup(2) = 8192
```

base tasks. It does not need selected-token partitioning to keep 32 physical
AICs and 64 physical AIVs busy.

## Goal

Automatically optimize the exact V3.2 BF16 prefill case by reusing the Head64
fused QK/online-softmax/PV kernel with one selected partition. The production
path must:

- launch 32 mixed tasks, corresponding to 32 AICs and 64 AIVs;
- let each physical task consume dynamic logical tasks with a stride of 32;
- use 1024 SIMT threads in each AIV VF;
- perform M64 Tensor API QK and PV;
- keep softmax and output accumulation online and on chip;
- write final BF16 output and FP32 LSE directly;
- avoid selected-token partition workspaces and Combine;
- preserve all existing decode tuning behavior and all non-target prefill
  fallbacks.

## Non-Goals

This milestone does not:

- enable Head64 automatically for non-V3.2 prefill shapes;
- add prefill selected-token P2 or P4 modes;
- change the V3.2 decode P1, P2, or P4 execution graph;
- change `family_hd128`, `family_hd256`, or `family_hd512` dispatch;
- introduce cross-physical-core synchronization;
- add ping-pong buffering or overlap beyond the current fused Head64 pipeline;
- change CLI, common backends, workflow plugins, result schemas, or published
  data contracts;
- add a new Basic API header, synchronization primitive, or CrossCore call.

The existing Head64 source still contains transitional mode-2 Basic API
handshakes. This change reuses that already validated intra-MIX-task protocol
without expanding it. Converting the shared decode/prefill source to the C API
synchronization boundary is a separate cleanup because it changes the decode
kernel too.

## Alternatives

### Head64 P1 persistent tasks

This is the selected design. It preserves `M=64`, avoids reduction, and uses
the 8192 natural prefill tasks to keep the device occupied.

### Head64 P4 Split-S

This would create 32768 logical tasks but would not increase physical
occupancy. It would add partition-local output/LSE traffic and a Combine
launch, so it is rejected for prefill.

### Rebuild the current M1 prefill kernel

Packing Head64, dual-AIV work, and Cube PV into the current wide-family source
would duplicate the decode implementation and make the same math harder to
maintain. It is rejected in favor of extending the isolated Head64 kernel.

## Automatic Dispatch

The operator-local Host bridge identifies the automatic route before entering
the generic wide-family branch. It requires all of:

```text
head_tile == 1
selected_partitions == 1
phase == prefill
family == family_hd576
query.dtype == bfloat16
shared_kv.dtype == bfloat16
B == 1
H == 128
KV_H == 1
Q == 4096
C == 32768
S == 2048
Dqk == 576
Dv == 512
```

For this exact predicate, the Host internally constructs a Head64 plan with:

```text
head_tile = 64
selected_partitions = 1
output_mode = direct_bfloat16
```

The environment variables remain operator-local. Explicit
`(head_tile=64, selected_partitions=1)` is accepted for BF16
`family_hd576` prefill reduced-shape validation with dynamic `B/Q/C/S` and
`S <= 2048`. Explicit prefill P2 or P4 is rejected with a clear error. Decode
continues to accept its existing P1/P2/P4 tunings and keeps the current P4
fused-plus-Combine route.

Default non-target prefill inputs retain the current generic fused path. No
operator name or shape branch is added to CLI, core, or shared backends.

## Host Plan And Execution Graph

`SparseAttentionHead64Plan` remains the shared POD boundary. Add an integer
output mode whose supported values distinguish:

- partition-local FP32 output for decode P4 followed by Combine;
- direct final BF16 output for prefill P1.

The prefill plan computes:

```text
head_group_count = H / 64
task_count = B * Q * head_group_count
used_core_num = min(task_count, 32)
```

For the target, `task_count=8192` and `used_core_num=32`. Neither value is
hardcoded in device source. Each AIC and each paired AIV executes:

```text
for logical_task = physical_task_id;
    logical_task < task_count;
    logical_task += used_core_num
```

The logical task decoder maps a P1 task to:

```text
(batch_index, query_token, head_group, partition=0)
```

The automatic Host path allocates only:

- final BF16 output `[B,H,Q,Dv]`;
- final FP32 LSE `[B,H,Q]`;
- the existing Head64 device workspace;
- the plan transfer tensor.

It launches the Head64 fused kernel once and returns tensors already matching
the public dtypes. The outer bridge therefore does not submit an additional
output cast. It allocates no full score/probability tensor, partition-local
output, partial LSE, or Combine output.

The decode P4 Host path continues to allocate FP32 partition output and
partial LSE, launch the same fused source in partition-output mode, and invoke
the existing Combine kernel.

## Device Task Mapping

One logical prefill task owns 64 Query heads for one `(batch, query_token)`.
The two AIVs split those heads symmetrically:

```text
AIV0: local heads 0..31
AIV1: local heads 32..63
```

Both AIVs launch 1024 SIMT threads. The existing Head64 mapping remains:

```text
threads_per_head = 32
local_head = threadIdx.x / 32
lane = threadIdx.x % 32
```

The task owns all `S=2048` selected positions. It iterates 64-position tiles,
so the target executes 32 selected tiles per logical task. No other physical
task contributes to the same output row and no inter-core reduction is
required.

## Fused Data Flow

The fused source reuses the decode Head64 tile pipeline. For each logical task:

```text
pack 64-head Query once
initialize per-AIV online max, sum, and FP32 output
for each selected tile:
    both AIVs gather disjoint 32-token K halves
    AIC computes M64 QK
    both AIVs update 32-head online softmax state
    both AIVs gather disjoint 32-token V halves
    AIC computes M64 PV
    both AIVs update 32-head FP32 online output
normalize and write final output/LSE
```

QK uses:

```text
[64,576] x [576,current_selected]
```

PV uses four value tiles:

```text
[64,current_selected] x [current_selected,128]
```

The causal position for Query token `q` remains right aligned:

```text
absolute_query_position = C - Q + q
```

Negative indices, indices at or beyond `C`, causal-future positions, and the
selected tail are masked before online normalization. An all-invalid row
writes zero output and negative-infinity LSE.

## Direct Output Mode

The current fused Head64 final VF writes partition-local FP32 data by logical
task. Extend its final-store boundary rather than duplicating QK, softmax, or
PV math.

In direct prefill mode, the final VF derives the global head from
`head_group`, sub-AIV index, and local head, then writes:

```text
output[((b * H + h) * Q + q) * Dv + d]  # bfloat16
lse[(b * H + h) * Q + q]                # float32
```

The FP32 online accumulator stays in UB until normalization. Casting only at
the final store is numerically equivalent to the current public BF16 output
conversion while removing the 1 GiB FP32 output intermediate and its separate
conversion launch for the target shape.

In partition-output mode, the existing FP32 task-output and partial-LSE layout
is unchanged so decode P4 Combine remains binary-compatible at the Host
boundary.

## Synchronization

Each physical MIX task is independent. The existing Head64 mode-2 handshakes
coordinate only the one AIC and two AIVs in that task for:

- Query/K readiness before QK;
- score readiness before softmax;
- probability/V readiness before PV;
- PV readiness before online output update.

No global barrier is introduced. Before a physical task advances to its next
logical task, both AIV halves finish final stores and the AIC pipeline drains
to the same initial flag state. The implementation must preserve the existing
physical-core-reuse ordering that decode reduced tests exercise, then stress
it with the much longer prefill loop.

## Error Handling And Fallback

The automatic predicate is exact. If it does not match, dispatch continues to
the existing generic prefill implementation.

Explicit Head64 prefill validates:

- phase and family;
- BF16 query and shared KV;
- `H=128`, `KV_H=1`, `Dqk=576`, and `Dv=512`;
- `selected_partitions=1`;
- `S <= 2048`;
- supported dimensions, strides, and index dtype.

An invalid explicit request raises a descriptive error and does not silently
fall back. Existing empty-input behavior is preserved.

The automatic route is retained only after correctness, stability, and
performance gates pass. If any gate fails during development, the optimized
kernel may remain explicitly testable, but the exact default predicate must
continue to use the generic fallback.

## File Boundaries

All business logic remains in the Sparse Attention operator plugin:

- `simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`: automatic
  predicate, prefill Head64 plan, direct-output allocations, and cast skip;
- `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h`:
  shared output mode and plan fields;
- `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc`:
  P1 direct BF16 final-store mode while preserving P4 partial mode;
- `simt/test/test_sparse_attention_v1_build_shell.py`: source and dispatch
  contracts;
- `simt/test/test_sparse_attention_prefill_reference.py`: local wrapper and
  reference behavior;
- `simt/test/`: operator-local remote accuracy and benchmark runners;
- `simt/README.md` and the operator-local research document: final measured
  results and dispatch decision.

No change is permitted in CLI, common backends, core configuration, or result
schemas for this feature.

## Verification

### Local TDD

Source tests are written first and must fail for the missing behavior. They
cover:

- the exact automatic predicate and internal Head64/P1 plan;
- dynamic `task_count` and `used_core_num=min(task_count,32)`;
- the physical-task stride loop;
- both AIVs performing useful work with 1024-thread VFs;
- direct BF16 output and FP32 LSE offsets;
- absence of prefill partial output, partial LSE, Combine, and output cast;
- rejection of prefill P2/P4;
- unchanged decode P1/P2/P4 routing;
- unchanged generic fallback for non-target prefill shapes;
- no new public-layer hardcoding or Basic API dependency.

After each red/green cycle, run the operator-local suite. Before integration,
run the full repository suite and `git diff --check`.

### Remote Build And Reduced Accuracy

Build the isolated worktree for `dav-3510`. Explicit Head64/P1 prefill reduced
coverage includes:

- `B=1,H=128,Q=4,C=256,S=64,Dqk=576,Dv=512`;
- selected tails such as `S=17` and `S=70`;
- causal masking at early, middle, and last Query positions;
- negative and out-of-range indices;
- an all-invalid row;
- a shape whose logical task count exceeds 32;
- repeated launches and two concurrent streams.

Output and LSE use `atol=0.05` and `rtol=0.05` against the Torch reference.

### Full V3.2 Accuracy

Run the exact automatic case. It must complete with the expected output and
LSE shapes and dtypes. Accuracy sampling covers early, middle, and late Query
tokens, both Head64 groups, invalid/causal boundaries, output dimensions, and
LSE. The same `atol=0.05` and `rtol=0.05` contract applies. Sampling is
performed with bounded-memory row reference calculations rather than
materializing the full `[B,H,Q,S,D]` Torch intermediate.

### Performance Gate

Build the current generic baseline and the candidate from the same
`origin/main` revision on the same device. Use identical inputs, warmups,
iteration counts, stream synchronization, and profiler configuration.

Record:

- synchronized public-op wall time;
- Head64 fused main-kernel duration;
- current generic main-kernel duration;
- Block Dim and Mix Block Dim;
- AIC/AIV rows with actual Cube/Vector work;
- Cube, Vector, Scalar, MTE, and wait utilization;
- GM traffic and L2 hit rates.

The automatic route passes only if:

- output/LSE accuracy and stability pass;
- synchronized wall time is lower than the current generic path;
- the fused main-kernel duration is lower than the current generic main
  kernel;
- the profile reports `Block Dim=32` and `Mix Block Dim=64`;
- all 32 AIC rows perform Cube work and all 64 AIV rows perform Vector work;
- no Combine or output-cast kernel appears in the target execution graph.

## Future Work

After this milestone is validated, separate designs may consider:

- C API replacement for the shared Head64 transitional synchronization;
- ping-pong buffering and gather/QK/softmax/PV overlap;
- broader automatic Head64 prefill dispatch based on dynamic shapes;
- sharing gathered KV across adjacent Head64 groups;
- Head64 optimization for other wide prefill families.
