# Lightning Indexer 32 Mixed-Task Prefill Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the Lightning Indexer production prefill fused path use all 32 AICs and 64 AIVs available on the target device.

**Architecture:** Increase only the operator-local common fused-kernel task cap from 16 to 32 for `family_4x64` and `family_64x128`. Preserve row ownership, mode-2/flag-0 synchronization, the current AIV roles, and 1024 SIMT threads; retain the change only after exact V3.2 prefill correctness, stability, and kernel-time gates pass.

**Tech Stack:** pytest, C++17, Ascend Tensor API and SIMT API, transitional Basic API synchronization, `bisheng --npu-arch=dav-3510`, torch_npu, msopprof 26.1.0 with a 100-second replay timeout.

## Global Constraints

- Work only in `/root/aiagent/cannbench/.worktrees/lightning-indexer-context-split` on `feature/lightning-indexer-context-split`.
- Keep implementation and implementation-level tests under `src/cannbench/operators/builtin/lightning_indexer/`.
- Do not change CLI, core, shared backends, result schemas, or plugin boundaries.
- The target has exactly 32 AICs and 64 AIVs; one `KERNEL_TYPE_MIX_AIC_1_2` task maps to one AIC and two AIVs.
- Keep synchronization mode 2, flag 0, and both AIVs participating in the handshake.
- Keep 1024 SIMT threads per launched VF.
- Do not change the 16-task context-sharded decode kernel.
- Do not enable or modify the experimental two-Query-atom prefill candidate.
- Compare against the exact 16-task msopprof kernel baseline of 11.218775 seconds.
- Retain 32 tasks only if exact V3.2 prefill correctness and stability pass and kernel time does not regress.

## File Map

- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`: require 32 common fused tasks while preserving synchronization contracts.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_4x64.asc`: raise the common fused cap to 32 and use 1024 threads.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc`: raise the common fused cap to 32 and use 1024 threads.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/README.md`: correct hardware occupancy and record the 32-task measurements.
- Modify `src/cannbench/operators/builtin/lightning_indexer/PARALLEL_SPLITTING_RESEARCH.zh-CN.md`: distinguish the 32-task hardware limit from decode's 16-task shard geometry.

---

### Task 1: Establish The 32-Task Source Contract

**Files:**
- Test: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

**Interfaces:**
- Consumes: the two existing common fused source files.
- Produces: a source-level contract requiring `kMaxUsedCoreNum = 32` without changing decode or the candidate.

- [x] **Step 1: Change the source contract to require 32 tasks**

Rename the test and replace its cap assertions with:

```python
def test_lightning_indexer_fused_kernels_use_all_32_aics_and_64_aivs():
    root = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )
    for family in ("4x64", "64x128"):
        source = (root / f"lightning_indexer_fused_family_{family}.asc").read_text(
            encoding="utf-8"
        )
        assert "constexpr int32_t kMaxUsedCoreNum = 32;" in source
        assert "constexpr int32_t kMaxUsedCoreNum = 16;" not in source
        assert "constexpr int32_t kMaxUsedCoreNum = 11;" not in source
        assert "constexpr int32_t kThreadsPerBlock = 1024;" in source
        assert "constexpr int32_t kThreadsPerBlock = 256;" not in source
        assert "constexpr uint8_t kCrossCoreSyncMode = 2;" in source
        assert "constexpr uint16_t kScoreReadyFlag = 0;" in source
```

- [x] **Step 2: Run the focused test and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py \
  -k all_32_aics_and_64_aivs
```

Expected: FAIL because both common fused sources still contain `kMaxUsedCoreNum = 16`.

### Task 2: Raise The Common Prefill Cap And Launch Width

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_4x64.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc`
- Test: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

**Interfaces:**
- Consumes: `total_rows` from the existing fused launcher.
- Produces: `used_core_num = min(total_rows, 32)` and 1024-thread SIMT launches
  for the two common fused families.

- [x] **Step 1: Make the minimal implementation change**

In both fused source files replace:

```cpp
constexpr int32_t kMaxUsedCoreNum = 16;
```

with:

```cpp
constexpr int32_t kMaxUsedCoreNum = 32;
```

Also replace:

```cpp
constexpr int32_t kThreadsPerBlock = 256;
```

with:

```cpp
constexpr int32_t kThreadsPerBlock = 1024;
```

- [x] **Step 2: Run the source suite and verify GREEN**

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
git diff --check
```

Expected: all tests pass and `git diff --check` emits no output.

- [x] **Step 3: Verify excluded kernels remain unchanged**

```bash
git diff -- \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_context_sharded_family_64x128.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_prefill_q2_family_64x128.asc
```

Expected: no output.

### Task 3: Build And Gate The Exact V3.2 Prefill Case Remotely

**Files:**
- Validate: `src/cannbench/operators/builtin/lightning_indexer/simt/test/v32_prefill_benchmark.py`

**Interfaces:**
- Consumes: BF16 `B=1,Q=4096,C=32768,H=64,D=128,K=2048`, seed 7, valid lengths 28673 through 32768.
- Produces: sampled score-set correctness, repeated-launch stability, host timing, and one exact-kernel msopprof record.

- [x] **Step 1: Copy the worktree into a fresh remote directory**

```bash
INDEXER_REMOTE_DIR=$(ssh -p 20002 root@121.41.199.170 \
  'mktemp -d /tmp/cannbench-indexer-prefill-32task-XXXXXX')
case "$INDEXER_REMOTE_DIR" in
  /tmp/cannbench-indexer-prefill-32task-*) ;;
  *) echo "unexpected remote directory: $INDEXER_REMOTE_DIR" >&2; exit 1 ;;
esac
rsync -a --delete -e 'ssh -p 20002' \
  --exclude '.git' --exclude '__pycache__' --exclude '*.so' --exclude 'build' \
  ./ "root@121.41.199.170:$INDEXER_REMOTE_DIR/"
```

Keep `INDEXER_REMOTE_DIR` for all following remote commands; never reuse the earlier candidate directory.

- [x] **Step 2: Build the custom operator remotely**

```bash
source /usr/local/Ascend/cann/set_env.sh
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=$PWD/src
export NPU_ARCH=dav-3510
export ASCEND_VISIBLE_DEVICES=0
cd src/cannbench/operators/builtin/lightning_indexer/simt/v1
/root/miniconda3/bin/python setup.py build_ext --inplace
```

Expected: exit 0 and all Lightning Indexer device libraries are produced.

- [x] **Step 3: Run exact correctness and stability**

From the fresh remote repository root run:

```bash
source /usr/local/Ascend/cann/set_env.sh
export PATH=/root/miniconda3/bin:$PATH
export PYTHONPATH=$PWD/src
export ASCEND_VISIBLE_DEVICES=0
/root/miniconda3/bin/python \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/v32_prefill_benchmark.py \
  --warmups 1 --iters 5 --seed 7 --stability-runs 3
```

Expected: `sampled_score_sets_match` and `stability_match` are both `true`.

- [x] **Step 4: Capture one exact-kernel BasicInfo profile**

```bash
msopprof \
  --output=/tmp/msopprof-v32-prefill-32task \
  --aic-metrics=BasicInfo \
  --launch-count=1 \
  --kernel-name='_ZN12_GLOBAL__N_144lightning_indexer_fused_family_64x128_kernelEPKtS1_S1_PKiPfPiPhiiiiii' \
  /root/miniconda3/bin/python -m cannbench internal-run \
  --backend ascend \
  --prepared-input /root/cannbench-v32-realistic-20260726-220220/cannbench-release/.cannbench-runs/opbench-ascend-950pr-simt-v1-dsa_prefill-realistic-bfloat16/lightning_indexer-realistic_prefill-deepseek_v32_flashmla_prefill_q4096_ctx32768_top2048-bfloat16-seed0/prepared.json \
  --output-dir /tmp/msopprof-v32-prefill-32task-perf \
  --run-name benchmark \
  --implementation simt \
  --implementation-version v1
```

Expected: exit 0, no timeout or `RegisterFuncSymbol` error, `Block Dim = 32`, `Mix Block Dim = 64`, and kernel duration no greater than 11.218775 seconds.

- [x] **Step 5: Apply the performance gate**

Record:

```text
speedup = 11.218775 / kernel_seconds_32task
latency_reduction = (11.218775 - kernel_seconds_32task) / 11.218775 * 100%
```

If kernel time regresses, restore only the Task 1/2 implementation files to their pre-task content with `apply_patch`, document the measurement, and keep the production cap at 16.

### Task 4: Record Evidence And Verify The Repository

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/README.md`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/PARALLEL_SPLITTING_RESEARCH.zh-CN.md`

**Interfaces:**
- Consumes: exact remote correctness, stability, host timing, block dimensions, and kernel duration from Task 3.
- Produces: an operator-local record of the final retained cap and measured comparison.

- [x] **Step 1: Update documentation with exact results**

Replace claims that 16 mixed tasks fill the device with:

```text
32 mixed tasks = 32 AIC + 64 AIV
```

Record the 16-task baseline, 32-task kernel time, speedup, latency reduction, correctness result, stability result, and profiler artifact path. Keep decode documented as `2 batch × 8 shards = 16 mixed tasks`.

- [x] **Step 2: Run final local verification**

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test
pytest -q
git diff --check
rg -n "kMaxUsedCoreNum = (11|16)" \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_4x64.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc
```

Expected: operator-local and full suites pass; whitespace and stale-cap searches emit no output.

- [x] **Step 3: Commit the verified implementation**

```bash
git add \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_4x64.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/README.md \
  src/cannbench/operators/builtin/lightning_indexer/PARALLEL_SPLITTING_RESEARCH.zh-CN.md
git commit -m "perf(lightning-indexer): use all mixed tasks for prefill"
```

Expected: one implementation commit after the committed design and plan documents.
