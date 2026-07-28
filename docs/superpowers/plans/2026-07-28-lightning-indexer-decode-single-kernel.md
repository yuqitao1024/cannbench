# Lightning Indexer Decode Single-Kernel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the V3.2 BF16 `family_64x128` decode score-plus-TopK two-launch path with one dynamic-B/Q mixed kernel that uses 32 AICs and 64 AIVs for `B=2,Q=2`.

**Architecture:** The host computes a power-of-two context shard count from runtime B/Q and launches one `1:2` mixed kernel. Every valid AIV writes its score shard, runs local TopK only when the shard contains more than 2048 scores, joins one mode-0 AIV barrier, and lets the shard-0 AIV perform the final row TopK. Unsupported shapes and any shard tier that fails its performance gate retain the existing fused fallback.

**Tech Stack:** C++/Torch custom op bridge, Ascend CANN 9.2.0 Tensor API, C synchronization API, Ascend SIMT VFs, pytest source contracts, Ascend 950PR NPU correctness tests, msopprof.

## Global Constraints

- Work only in `/root/aiagent/cannbench/.worktrees/lightning-indexer-decode-single-kernel` on `perf/lightning-indexer-decode-single-kernel`.
- Keep all implementation and tests under `src/cannbench/operators/builtin/lightning_indexer/`.
- Do not modify CLI, common backend, core, result, workflow, or published-schema code.
- The fast-path family is BF16, `H=64`, `D=128`, `C=32768`, `K=2048`, with runtime B/Q.
- `mixed_task_count = B * ceil(Q/2) * context_shard_count` must not exceed 32.
- Select `context_shard_count` from `{16,8,4,2,1}`, largest first.
- Use 1024 SIMT threads.
- Local TopK runs only for `shard_size > 2048`; `shard_size == 2048` skips it.
- For more than one shard, execute exactly one all-AIV mode-0 barrier using flag 0.
- Keep mode-2 AIC/AIV synchronization on flag 0 and use public C synchronization calls for new cross-core work.
- Use `__schedmode__(1)` for the barrier kernel.
- Preserve descending BF16 score order and lower global index as the tie breaker.
- Preserve the existing generic fused implementation as fallback until correctness and performance gates pass.

---

### Task 1: Lock the Runtime Planner and Single-Launch Contract

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`

**Interfaces:**
- Produces: host helper `select_context_shard_count(int64_t batch_size, int64_t query_count) -> int32_t`.
- Produces: launcher arguments `batch_size`, `query_count`, `context_shard_count`, `mixed_task_count`.

- [x] **Step 1: Replace the fixed-shape source tests with failing planner and one-launch tests**

Assert the bridge contains the exact shard selection `{16, 8, 4, 2, 1}`, computes `query_atom_count = (query_count + 1) / 2`, rejects `base_task_count > 32`, allocates `[B,Q,32768]` BF16 scores and conditional candidate indices, and calls only the combined launcher in the fast-path body.

- [x] **Step 2: Run the targeted tests and verify RED**

Run:

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py -k 'context_sharded'
```

Expected: failures show the bridge is still fixed to `{2,2,...}` and launches standalone TopK.

- [x] **Step 3: Implement the runtime planner and dynamic bridge allocations**

Keep the fixed family predicate (`BF16/H64/D128/C32768/K2048`) but remove exact B/Q predicates. Return the generic fused path when `B * ceil(Q/2) > 32`. Allocate candidate indices only for shard counts 8, 4, or 2; pass a null pointer for shard counts 16 and 1.

- [x] **Step 4: Run the targeted tests and verify GREEN**

Run the command from Step 2. Expected: all selected tests pass.

- [ ] **Step 5: Commit the bridge contract**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc
git commit -m "perf(lightning-indexer): plan dynamic decode shards"
```

### Task 2: Parameterize Score Generation and C-API Synchronization

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_context_sharded_family_64x128.asc`

**Interfaces:**
- Consumes: runtime `batch_size`, `query_count`, `context_shard_count`, and `mixed_task_count` from Task 1.
- Produces: task mapping `(batch, query_atom, context_shard)` and one valid/padding AIV decision per sub-block.

- [x] **Step 1: Write failing source contracts for dynamic task mapping**

Assert the source derives `query_atom_count`, `base_task_index`, `batch_index`, `atom_index`, `shard_index`, `query_index`, and `shard_size` from launcher arguments. Assert 32 tasks for B2/Q2/S16, an `M=64` MMAD and single-destination Fixpipe for an odd-Q tail atom, both AIVs execute every local mode-2 handshake, and no AIV returns before a required global barrier.

- [x] **Step 2: Write failing C-API boundary contracts**

Assert the context-sharded source includes `c_api/sync/sync.h`, uses `asc_sync_block_arrive/wait`, does not use `CrossCoreSetFlag/WaitFlag`, and declares the mixed kernel with `__schedmode__(1)`.

- [x] **Step 3: Run the source tests and verify RED**

Run:

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py -k 'context_sharded'
```

Expected: failures identify fixed shard constants and Basic API cross-core synchronization.

- [x] **Step 4: Implement dynamic score mapping**

Keep a single 128-row Tensor API copy for complete Query atoms. For odd Q,
load only the valid 64-row tail, set MMAD `M=64`, and disable Fixpipe dual
destination; the padding AIV still performs mode-2 synchronization but skips
score reads and stores. Derive each context range from
`32768 / context_shard_count` and keep both AIVs active for even Query atoms.

- [x] **Step 5: Replace the local mode-2 calls with public C synchronization**

Use:

```cpp
asc_sync_block_arrive(PIPE_V, 0);
asc_sync_block_wait(PIPE_S, 0);
asc_sync_block_arrive(PIPE_FIX, 0);
asc_sync_block_wait(PIPE_S, 0);
```

Preserve existing Tensor API copy/MMAD/Fixpipe operations and their local pipeline ordering.

- [x] **Step 6: Run source tests and verify GREEN**

Run the command from Step 3. Expected: selected tests pass.

- [ ] **Step 7: Commit dynamic score generation**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_context_sharded_family_64x128.asc
git commit -m "perf(lightning-indexer): parameterize decode score shards"
```

### Task 3: Fuse Conditional Local TopK and Final Row TopK

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_context_sharded_family_64x128.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py`
- Delete after the combined path passes tests: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_topk_scores.asc`

**Interfaces:**
- Consumes: BF16 score workspace `[B,Q,32768]` and optional int32 candidate workspace `[B,Q,S,2048]`.
- Produces: final int32 output `[B,Q,2048]` from the same device launch.

- [x] **Step 1: Write failing strict local-TopK source contracts**

Assert `needs_local_topk = shard_size > 2048`, S16 does not invoke local selection, S8/S4/S2 publish 2048 scores and global indices per shard, and S1 writes its local result directly to output.

- [x] **Step 2: Write failing barrier and final-owner contracts**

Assert all launched AIVs execute exactly one `asc_sync_inter_arrive/wait` pair for `S>1`, flag 0 is used, shard-0 owns final TopK, padding AIVs join the barrier, and final candidate counts are derived as `S * min(shard_size, 2048)`.

- [x] **Step 3: Run the source tests and verify RED**

Run the targeted command from Task 2. Expected: failures show no in-kernel TopK or global barrier exists.

- [x] **Step 4: Implement reusable 1024-thread TopK selection in the combined source**

Reuse a 4096-entry dynamic-UB sort buffer. Process candidates in 2048-entry tiles, retain Top2048 after every tile, store BF16 candidate scores plus int32 global indices, and preserve lower-index tie ordering.

- [x] **Step 5: Implement conditional local selection and publication ordering**

Before a local TopK rereads its GM scores, execute:

```cpp
asc_sync_vec();
asc_sync_data_barrier(mem_dsb_t::DSB_DDR);
```

Skip this local call when `shard_size == 2048`.

- [x] **Step 6: Implement the single mode-0 barrier and final row selection**

For S16, the final owner reads all 32768 raw scores with implicit global indices. For S8/S4/S2 it reads retained candidates and explicit indices. For S1, the local selection is the final output and no barrier or second selection occurs.

- [x] **Step 7: Remove the standalone TopK device library and second launcher declaration**

Update `setup.py` and `lightning_indexer.asc` so the fast-path body contains one raw device launch. Keep unrelated family libraries unchanged.

- [x] **Step 8: Run source tests and verify GREEN**

Run:

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
```

Expected: all Lightning Indexer source-contract tests pass.

- [ ] **Step 9: Commit the fused TopK kernel**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/v1 src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
git commit -m "perf(lightning-indexer): fuse decode TopK into score kernel"
```

### Task 4: Validate Dynamic Dispatch and NPU Correctness

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_decode_reference.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_dispatch.py`

**Interfaces:**
- Consumes: registered `aten_dsa_lightning_indexer.lightning_indexer_forward` custom op.
- Produces: correctness coverage for S16/S8/S4/S2/S1, even Q, odd Q, masks, and repeated launches.

- [x] **Step 1: Add parameterized NPU tests for representative shard tiers**

Use small B/Q combinations that select S16, S8, S4, S2, and S1 while keeping C32768/K2048. Compare selected BF16 scores against the existing Torch reference and verify output indices are in range and unique.

- [x] **Step 2: Add odd-Q, asymmetric mask, repeat, and concurrent-stream tests**

Ensure a padding AIV never changes output shape or accesses an invalid row, and stress flag reuse across repeated/concurrent launches.

- [x] **Step 3: Run local tests**

Run:

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test
```

Expected locally: source tests pass; NPU-only tests skip when the custom op is unavailable.

- [x] **Step 4: Synchronize the worktree to an isolated `/tmp` directory on the 20002 machine and build**

Build with CANN 9.2.0 and `NPU_ARCH=dav-3510`. Expected: the combined device library and host extension compile without unresolved launcher symbols.

- [ ] **Step 5: Run NPU correctness and stability tests**

Expected: all supported tiers, odd Q, mask, repeated launch, and concurrent-stream cases pass without deadlock.

- [ ] **Step 6: Commit NPU test coverage or build fixes**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/test src/cannbench/operators/builtin/lightning_indexer/simt/v1
git commit -m "test(lightning-indexer): cover single-kernel decode tiers"
```

### Task 5: Profile, Gate Dispatch, and Run Regression

**Files:**
- Modify if the candidate needs a performance gate: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`
- Modify source-contract expectations if a tier remains fallback: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`
- Update benchmark evidence in: `docs/superpowers/specs/2026-07-28-lightning-indexer-decode-single-kernel-design.md`

**Interfaces:**
- Consumes: old two-kernel and new single-kernel builds on the same Ascend 950PR/CANN 9.2.0 machine.
- Produces: default-dispatch decision per shard tier and recorded kernel/wall timing.

- [x] **Step 1: Refresh the old B2/Q2 baseline**

Capture score kernel, standalone TopK kernel, kernel-side sum, and synchronized operator wall time using identical inputs.

- [x] **Step 2: Profile the combined B2/Q2 kernel**

Verify msopprof captures one Lightning Indexer kernel, 32 AICs/64 AIVs in score generation, no local TopK for S16, and four final row owners.

- [x] **Step 3: Profile representative S8/S4/S2/S1 tiers**

Record correctness and timing. Leave any regressing tier on the generic fused fallback.

- [x] **Step 4: Apply the dispatch gate and rerun targeted tests**

Run:

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test
```

Expected: dispatch tests reflect measured enabled tiers and pass.

- [ ] **Step 5: Run full regression and architecture searches**

Run:

```bash
pytest -q
rg -n "lightning_indexer|dsa_decode" src/cannbench/cli.py src/cannbench/core src/cannbench/backends
git diff --check
```

Expected: 457 or more tests pass with only known skips; targeted public layers contain no new operator-name branches; diff check is clean.

- [ ] **Step 6: Amend benchmark evidence and finalize commits**

Amend documentation changes into the existing design commit when practical; keep implementation commits organized by independently tested behavior. Verify the worktree is clean before branch handoff.
