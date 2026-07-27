# Lightning Indexer Context-Sharded Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a two-launch context-sharded fast path for the DeepSeek V3.2 `lightning_indexer` decode case and enable it only when Ascend 950PR measurements beat the corrected row-based baseline.

**Architecture:** A 16-task mixed kernel treats two Query rows as one M=128 atom and writes BF16 reduced scores to a 256 KiB `[B,Q,C]` workspace. A four-block 1024-thread SIMT kernel performs row-wise TopK; non-target metadata remains on the existing fused path.

**Tech Stack:** pytest, PyTorch/torch_npu custom ops, C++17, Ascend Tensor API and SIMT API, transitional mode-2 AIC/AIV synchronization, `bisheng --npu-arch=dav-3510`.

## Global Constraints

- Keep implementation and tests under `src/cannbench/operators/builtin/lightning_indexer/`.
- Do not change CLI, core, shared backends, or published-data schemas.
- Match only decode, `family_64x128`, BF16, `B=2,Q=2,C=32768,H=64,D=128,K=2048`.
- Use Query atom 2, context shard 4096, 16 logical score tasks, mode 2, and flag 0.
- Both AIVs execute every handshake; AIV0 owns Query 0 and AIV1 owns Query 1.
- New SIMT VF and TopK blocks use `__launch_bounds__(1024)` and 1024 threads. Reduce this only when a remote compiler resource report proves spill at 1024.
- Apply `valid_context_lengths` on device; never read it on the host.
- Preserve current BF16 rounding and score/index tie semantics.
- Do not add alternative shard schemes, local-TopK merge, general shape tuning, or double buffering.
- Build, test, and benchmark on the provided Ascend 950PR endpoint.

## File Map

- Create `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_context_sharded_family_64x128.asc`: Q-atom score kernel.
- Create `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_topk_scores.asc`: BF16 score TopK kernel.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py`: compile new device libraries.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`: two-launch bridge and exact dispatch.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`: source/build contracts.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_decode_reference.py`: NPU correctness and stability.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/README.md`: measured performance decision.

---

### Task 1: Context-Sharded Q-Atom Score Kernel

**Files:**
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_context_sharded_family_64x128.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py`
- Test: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

**Interfaces:**
- Consumes: BF16 `query [2,2,64,128]`, `keys [2,32768,128]`, `weights [2,2,64]`; int32 lengths `[2,2]`.
- Produces: `launch_lightning_indexer_context_sharded_family_64x128_bfloat16(..., bfloat16_t* reduced_scores, aclrtStream)` writing `[2,2,32768]`.

- [ ] **Step 1: Write the failing source test**

```python
def test_context_sharded_family_64x128_uses_q2_atom_and_both_aivs():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_context_sharded_family_64x128.asc"
    ).read_text(encoding="utf-8")
    for expected in (
        "kQueryAtomSize = 2",
        "kContextShardSize = 4096",
        "kContextShardCount = 8",
        "kLogicalTaskCount = 16",
        "kThreadsPerBlock = 1024",
        "__launch_bounds__(kThreadsPerBlock)",
        "params.m = 128",
        "kCrossCoreSyncMode = 2",
        "kScoreReadyFlag = 0",
        "query_in_atom = static_cast<int32_t>(AscendC::GetSubBlockIdx())",
        "context_start = shard_index * kContextShardSize",
        "context_index >= valid_context_lengths[row_index]",
    ):
        assert expected in source
```

Extend the setup test to require the new `.asc` source and
`liblightning_indexer_context_sharded_family_64x128_kernel.so`.

- [ ] **Step 2: Verify RED**

Run:

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py -k context_sharded
```

Expected: FAIL because the source and build entry do not exist.

- [ ] **Step 3: Implement the fixed-shape kernel and launcher**

Use these constants and mapping:

```cpp
constexpr int32_t kQueryAtomSize = 2;
constexpr int32_t kHeadCount = 64;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kContextCount = 32768;
constexpr int32_t kContextTileSize = 32;
constexpr int32_t kContextShardSize = 4096;
constexpr int32_t kContextShardCount = 8;
constexpr int32_t kLogicalTaskCount = 16;
constexpr int32_t kThreadsPerBlock = 1024;
constexpr uint8_t kCrossCoreSyncMode = 2;
constexpr uint16_t kScoreReadyFlag = 0;

const int32_t task_id = static_cast<int32_t>(AscendC::GetBlockIdx());
const int32_t batch_index = task_id / kContextShardCount;
const int32_t shard_index = task_id % kContextShardCount;
const int32_t context_start = shard_index * kContextShardSize;
```

Load the two Query rows as M=128 and run one MMAD per Key tile:

```cpp
MmadParams params;
params.m = 128;
params.n = kContextTileSize;
params.k = kHeadDim;
params.unitFlag = 0;
params.cmatrixInitVal = true;
Mmad(mm.with(params), l0_scores, l0_query_atom, l0_keys);
```

Use `dualDstCtl=1` so each AIV receives one 64-row Query half at UB offset 0.
With N=32, each destination is `64 * 32 * sizeof(float) = 8 KiB`, matching the
C310 mixed-kernel shared window verified on hardware.

Both AIVs handshake before any Query-specific branch. Launch a 1024-thread VF
on each AIV and preserve the current BF16 dot, ReLU, weighted-BF16, float-sum,
final-BF16 order:

```cpp
const int32_t query_in_atom =
    static_cast<int32_t>(AscendC::GetSubBlockIdx());
AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_V>(kScoreReadyFlag);
AscendC::CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_V>(kScoreReadyFlag);
asc_vf_call<lightning_indexer_context_sharded_postprocess_vf>(
    dim3(kThreadsPerBlock, 1, 1),
    shared_scores,
    weights,
    valid_context_lengths,
    reduced_scores,
    batch_index,
    query_in_atom,
    tile_context_start);
```

Export a launcher that always uses `<<<kLogicalTaskCount, 0, stream>>>`. Add its
source/library pair to `KERNEL_LIBRARIES` in `setup.py`.

- [ ] **Step 4: Verify GREEN locally and compile remotely**

Run the selected source tests, then sync to
`/tmp/cannbench-indexer-sync-iCNM54` and run:

```bash
cd src/cannbench/operators/builtin/lightning_indexer/simt/v1
source /usr/local/Ascend/cann/set_env.sh
export PATH=/root/miniconda3/bin:$PATH NPU_ARCH=dav-3510
/root/miniconda3/bin/python setup.py build_ext --inplace
```

Expected: tests PASS and the new device library compiles with exit 0. If the
compiler reports stack use above zero, capture `--cce-res-usage` output before
changing 1024 threads.

- [ ] **Step 5: Commit**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_context_sharded_family_64x128.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
git commit -m "feat(lightning-indexer): add context-sharded score kernel"
```

---

### Task 2: BF16 Score TopK and Two-Launch Bridge

**Files:**
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_topk_scores.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`
- Test: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 1 launcher and BF16 reduced scores `[2,2,32768]`.
- Produces: `launch_lightning_indexer_topk_scores_bfloat16(...)` and a host helper returning int32 `[2,2,2048]`.

- [ ] **Step 1: Write failing TopK and bridge tests**

```python
def test_context_sharded_topk_uses_1024_threads_and_2048_score_tiles():
    source = Path(TOPK_SCORE_SOURCE).read_text(encoding="utf-8")
    for expected in (
        "kThreadsPerBlock = 1024",
        "__launch_bounds__(kThreadsPerBlock)",
        "kTopK = 2048",
        "kScoreTileSize = 2048",
        "kSortCapacity = 4096",
        "kRowCount = 4",
        "index < other_index",
    ):
        assert expected in source


def test_context_sharded_bridge_launches_score_before_topk():
    source = Path(BRIDGE_PATH).read_text(encoding="utf-8")
    body = source.split(
        "lightning_indexer_forward_decode_family_64x128_context_sharded_bfloat16(",
        1,
    )[1].split("\n}\n", 1)[0]
    assert "{2, 2, 32768}" in body
    assert "query.options().dtype(c10::kBFloat16)" in body
    assert body.index("launch_lightning_indexer_context_sharded") < body.index(
        "launch_lightning_indexer_topk_scores_bfloat16"
    )
    assert "{2, 2, 2048}" in body
```

Also assert exact dtype/dimension predicates occur before the old family helper.

- [ ] **Step 2: Verify RED**

Run the build-shell test with `-k 'context_sharded or topk_scores'`. Expected:
FAIL because the TopK source and bridge helper do not exist.

- [ ] **Step 3: Implement the 1024-thread TopK kernel**

```cpp
constexpr int32_t kRowCount = 4;
constexpr int32_t kContextCount = 32768;
constexpr int32_t kTopK = 2048;
constexpr int32_t kScoreTileSize = 2048;
constexpr int32_t kSortCapacity = 4096;
constexpr int32_t kThreadsPerBlock = 1024;
constexpr int32_t kDynamicUbufBytes =
    2 * kSortCapacity * sizeof(uint32_t);
```

Each of four blocks initializes retained candidates, merges 2048 BF16 scores
for 16 rounds, uses score-descending/index-ascending comparison, and writes the
first 2048 indices. Mark the SIMT entry `__launch_bounds__(1024)` and export a
launcher using `<<<kRowCount, kDynamicUbufBytes, stream>>>`.

- [ ] **Step 4: Implement bridge allocation, launch order, and exact dispatch**

```cpp
auto reduced_scores = at::empty(
    {2, 2, 32768}, query.options().dtype(c10::kBFloat16));
auto output = at::empty(
    {2, 2, 2048}, query.options().dtype(c10::kInt));
launch_lightning_indexer_context_sharded_family_64x128_bfloat16(
    query.const_data_ptr<at::BFloat16>(),
    keys.const_data_ptr<at::BFloat16>(),
    weights.const_data_ptr<at::BFloat16>(),
    valid_context_lengths.const_data_ptr<int32_t>(),
    reduced_scores.mutable_data_ptr<at::BFloat16>(), acl_stream);
launch_lightning_indexer_topk_scores_bfloat16(
    reduced_scores.const_data_ptr<at::BFloat16>(),
    output.mutable_data_ptr<int32_t>(), acl_stream);
```

Select only when original dtype and all target dimensions match. Do not inspect
length values on host. Record all input, workspace, and output storage on the
current NPU stream.

- [ ] **Step 5: Verify and commit**

Run the complete build-shell test and `pytest -q`. Expected: no failures. Then:

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_topk_scores.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
git commit -m "feat(lightning-indexer): add context score TopK path"
```

---

### Task 3: Remote NPU Correctness and Stability

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_decode_reference.py`
- Modify if required by failing NPU tests: both new `.asc` files and `lightning_indexer.asc`

**Interfaces:**
- Consumes: registered two-launch custom op from Task 2.
- Produces: automated exact-shape, asymmetric-length, and repeated-launch coverage.

- [ ] **Step 1: Add the exact target-shape test**

Use existing torch/custom-op/NPU availability guards, then create:

```python
query = torch.randn(2, 2, 64, 128, device="npu", dtype=torch.bfloat16)
keys = torch.randn(2, 32768, 128, device="npu", dtype=torch.bfloat16)
weights = torch.rand(2, 2, 64, device="npu", dtype=torch.bfloat16)
valid = torch.tensor(
    [[32767, 32768], [32767, 32768]], device="npu", dtype=torch.int32
)
```

Compute the PyTorch reduced scores, mask
`positions >= valid.unsqueeze(-1)`, and compare gathered custom/reference TopK
scores with `torch.equal`. Assert `[2,2,2048]`, int32, and that Query 0 never
selects context index 32767.

- [ ] **Step 2: Add repeated-launch coverage**

Call the exact custom op 20 times using the same tensors, synchronize after the
loop, and assert every output gathers the same reference score set. This test
catches mode-2 flag counter imbalance and deadlock across launches.

- [ ] **Step 3: Verify local skip and remote PASS**

Run locally:

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_decode_reference.py -k 'context_sharded or repeated'
```

Expected: clean SKIP without torch/NPU. Sync the worktree to
`/tmp/cannbench-indexer-sync-iCNM54`, rebuild in place, print the loaded `_C`
path, and run the same selection remotely. Expected: both tests PASS and the
extension path is inside the temporary directory.

- [ ] **Step 4: Run remote operator regression and commit**

```bash
/root/miniconda3/bin/python -m pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test
```

Expected: no failures. Commit test fixes and any minimal device corrections:

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/test/test_decode_reference.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_context_sharded_family_64x128.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_topk_scores.asc
git commit -m "test(lightning-indexer): validate context-sharded decode"
```

---

### Task 4: Performance Gate and Final Verification

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/README.md`
- Modify if the gate fails: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`

**Interfaces:**
- Consumes: corrected baseline `d608c0b` and Task 3 candidate.
- Produces: measured default-path decision and reproducible operator-local documentation.

- [ ] **Step 1: Prepare isolated baseline and candidate builds**

Create two remote directories with
`mktemp -d /tmp/cannbench-indexer-bench-XXXXXX`. Sync `d608c0b` to baseline and
current HEAD to candidate. Build each extension in place with the same CANN,
Python, `NPU_ARCH=dav-3510`, and `ASCEND_VISIBLE_DEVICES=0` settings.

- [ ] **Step 2: Measure identical inputs**

In separate Python processes, use seed 7 and Task 3 tensors. Run five warmups
and 30 synchronized samples:

```python
for _ in range(5):
    run_once()
torch.npu.synchronize()
samples = []
for _ in range(30):
    started = time.perf_counter_ns()
    run_once()
    torch.npu.synchronize()
    samples.append((time.perf_counter_ns() - started) / 1_000_000)
print(json.dumps({
    "samples_ms": samples,
    "median_ms": statistics.median(samples),
}))
```

Capture compiler resource output for both 1024-thread entries. A nonzero stack
size triggers investigation before any thread-count change.

- [ ] **Step 3: Apply the default-path gate**

If candidate median is lower, keep the Task 2 exact dispatch. If not, keep both
internal kernels but make the predicate false so production calls remain on
the corrected baseline. Rerun bridge source tests after either decision.

- [ ] **Step 4: Document measured results**

In `simt/README.md`, record device, target shape, baseline commit, warmups,
sample count, both medians, candidate/baseline ratio, 1024-thread resource
result, and whether the new path is enabled. Do not present the earlier 52.287
ms cold validation launch as benchmark data.

- [ ] **Step 5: Run final verification**

```bash
pytest -q
git diff --check
rg -n 'lightning_indexer' \
  src/cannbench/cli.py src/cannbench/core src/cannbench/backends
```

Expected: full suite passes, diff check is empty, and the search shows no new
public-layer operator branch. Rebuild the final candidate remotely and rerun
all operator-local tests with no failures.

- [ ] **Step 6: Commit**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/README.md \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc
git commit -m "perf(lightning-indexer): gate context-sharded decode path"
```
