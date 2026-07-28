# Lightning Indexer V3.2 Prefill Row-Parallel Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the obsolete mixed-task cap and add a performance-gated Q=2, dual-AIV fast path for the exact DeepSeek V3.2 BF16 prefill case.

**Architecture:** First establish a 16-mixed-task/32-AIV common-kernel baseline. Then add one fixed-shape operator-local kernel that packs two Query rows into M=128, routes one row to each AIV, and keeps row-local TopK state without a full-score GM workspace; the C++ bridge selects it only for the exact V3.2 prefill metadata.

**Tech Stack:** pytest, PyTorch/torch_npu custom ops, C++17, Ascend Tensor API and SIMT API, transitional mode-2 AIC/AIV synchronization, `bisheng --npu-arch=dav-3510`, Atlas 350 remote validation.

## Global Constraints

- Work only in `/root/aiagent/cannbench/.worktrees/lightning-indexer-context-split` on `feature/lightning-indexer-context-split`.
- Keep implementation and implementation-level tests under `src/cannbench/operators/builtin/lightning_indexer/`.
- Do not change CLI, core, shared backends, result schemas, or workflow plugins.
- Treat `16` as the mixed launch count: `16 AIC + 32 AIV` under `KERNEL_TYPE_MIX_AIC_1_2`.
- Match only BF16 `phase=prefill,family=family_64x128,B=1,Q=4096,C=32768,H=64,D=128,K=2048` for the new path.
- Use Q atom 2, M=128, N=32, mode 2, and flag 0.
- Both AIVs must execute every cross-core set/wait; AIV0 owns the first Query row and AIV1 owns the second.
- New SIMT VFs use `__launch_bounds__(1024)` and 1024 threads unless compiler resource output or measurements prove this regresses.
- Never read `valid_context_lengths` on the host.
- Preserve current BF16 conversion order and score-set correctness semantics.
- Do not add a full-score prefill workspace, context sharding, local-candidate merge, runtime autotuning, or double buffering.
- Do not add new Basic API dependencies; existing transitional synchronization may remain for this function-first slice.
- Compare candidate latency against the corrected 16-task baseline, not the historical 11-task baseline.
- Keep exact dispatch only when standalone latency improves, full `dsa_prefill` correctness passes, and workflow latency does not regress.

## File Map

- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_4x64.asc`: correct the common mixed-task limit.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc`: correct the common mixed-task limit.
- Create `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_prefill_q2_family_64x128.asc`: exact fixed-shape Q=2 mixed kernel.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py`: build the exact prefill kernel as its own device library.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`: exact bridge helper and dispatch.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`: source, build, and dispatch contracts.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_prefill_reference.py`: target-shape sampled correctness and repeated-launch coverage.
- Create `src/cannbench/operators/builtin/lightning_indexer/simt/test/v32_prefill_benchmark.py`: reproducible standalone median and sampled-score gate.
- Modify `src/cannbench/operators/builtin/lightning_indexer/simt/README.md`: corrected baseline, candidate, workflow, and final dispatch decision.

---

### Task 1: Correct the Common Mixed-Task Limit

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_4x64.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc`
- Test: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

**Interfaces:**
- Consumes: existing mode-2/flag-0 mixed-kernel handshake from commit `d608c0b`.
- Produces: a common-path launch limit of 16 mixed tasks for both fused families.

- [ ] **Step 1: Write the failing source contract**

Add:

```python
def test_lightning_indexer_fused_kernels_use_all_32_aivs():
    root = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )
    for family in ("4x64", "64x128"):
        source = (root / f"lightning_indexer_fused_family_{family}.asc").read_text(
            encoding="utf-8"
        )
        assert "constexpr int32_t kMaxUsedCoreNum = 16;" in source
        assert "constexpr int32_t kMaxUsedCoreNum = 11;" not in source
        assert "constexpr uint8_t kCrossCoreSyncMode = 2;" in source
        assert "constexpr uint16_t kScoreReadyFlag = 0;" in source
```

- [ ] **Step 2: Run the test and verify RED**

Run:

```bash
pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py \
  -k all_32_aivs
```

Expected: FAIL because both sources still contain `kMaxUsedCoreNum = 11`.

- [ ] **Step 3: Make the minimal cap correction**

In both fused sources change only:

```cpp
constexpr int32_t kMaxUsedCoreNum = 16;
```

Do not change task mapping, TopK, tile size, flags, or dispatch in this task.

- [ ] **Step 4: Verify locally and remotely**

Run locally:

```bash
pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
git diff --check
```

Expected: PASS and no whitespace errors.

Sync the exact worktree to a new remote directory created with
`mktemp -d /tmp/cannbench-indexer-prefill-baseline-XXXXXX`, then run:

```bash
INDEXER_REMOTE_ROOT=$PWD
cd src/cannbench/operators/builtin/lightning_indexer/simt/v1
source /usr/local/Ascend/cann/set_env.sh
export PATH=/root/miniconda3/bin:$PATH
export NPU_ARCH=dav-3510
/root/miniconda3/bin/python setup.py build_ext --inplace
cd "$INDEXER_REMOTE_ROOT"
/root/miniconda3/bin/python -m pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test
```

Expected: build exit 0 and no operator-local test failures.

- [ ] **Step 5: Commit the corrected baseline**

```bash
git add \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_4x64.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc
git commit -m "fix(lightning-indexer): use all mixed core groups"
```

Record this commit as `CORRECTED_BASELINE_COMMIT` for Task 4.

---

### Task 2: Add the Exact Q=2 Dual-AIV Prefill Kernel

**Files:**
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_prefill_q2_family_64x128.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py`
- Test: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

**Interfaces:**
- Consumes: BF16 Query `[1,4096,64,128]`, Keys `[1,32768,128]`, Weights `[1,4096,64]`, and int32 lengths `[1,4096]`.
- Produces: `launch_lightning_indexer_prefill_q2_family_64x128_bfloat16(const bfloat16_t* query, const bfloat16_t* keys, const bfloat16_t* weights, const int32_t* valid_context_lengths, float* topk_scores, int32_t* topk_indices, aclrtStream stream)`.

- [ ] **Step 1: Write failing build and kernel contracts**

Add tests that require the new setup entry and fixed-shape source:

```python
def test_prefill_q2_family_64x128_builds_a_separate_device_library():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")
    assert "lightning_indexer_prefill_q2_family_64x128.asc" in setup_py
    assert '"liblightning_indexer_prefill_q2_family_64x128_kernel.so"' in setup_py


def test_prefill_q2_family_64x128_uses_16_tasks_and_both_aivs():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_prefill_q2_family_64x128.asc"
    ).read_text(encoding="utf-8")
    for expected in (
        "kQueryAtomSize = 2",
        "kQueryAtomCount = 2048",
        "kLogicalTaskCount = 16",
        "kContextTileSize = 32",
        "kThreadsPerBlock = 1024",
        "__launch_bounds__(kThreadsPerBlock)",
        "params.m = kQueryAtomRows",
        "kCrossCoreSyncMode = 2",
        "kScoreReadyFlag = 0",
        "fixpipe_params.dualDstCtl = 1",
        "AscendC::GetSubBlockIdx()",
        "atom_index += kLogicalTaskCount",
        "query_index = atom_index * kQueryAtomSize + query_in_atom",
    ):
        assert expected in source
```

- [ ] **Step 2: Run the tests and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py \
  -k prefill_q2
```

Expected: FAIL because the source and setup entry do not exist.

- [ ] **Step 3: Add the isolated build entry**

Add to `KERNEL_LIBRARIES`:

```python
"liblightning_indexer_prefill_q2_family_64x128_kernel.so": os.path.join(
    EXTENSIONS_DIR,
    "simt",
    "lightning_indexer_prefill_q2_family_64x128.asc",
),
```

- [ ] **Step 4: Implement the fixed-shape kernel**

Use these constants and task mapping:

```cpp
constexpr int32_t kQueryAtomSize = 2;
constexpr int32_t kQueryCount = 4096;
constexpr int32_t kQueryAtomCount = kQueryCount / kQueryAtomSize;
constexpr int32_t kHeadCount = 64;
constexpr int32_t kHeadDim = 128;
constexpr int32_t kContextCount = 32768;
constexpr int32_t kContextTileSize = 32;
constexpr int32_t kTopK = 2048;
constexpr int32_t kLogicalTaskCount = 16;
constexpr int32_t kThreadsPerBlock = 1024;
constexpr uint8_t kCrossCoreSyncMode = 2;
constexpr uint16_t kScoreReadyFlag = 0;
constexpr int32_t kQueryAtomRows = kQueryAtomSize * kHeadCount;
```

Both AIC and AIV loops use the same atom order:

```cpp
for (int32_t atom_index = task_id; atom_index < kQueryAtomCount;
     atom_index += kLogicalTaskCount) {
  const int32_t query_row_start = atom_index * kQueryAtomSize;
  auto gm_query = MakeTensor(
      MakeMemPtr(
          query + static_cast<int64_t>(query_row_start) * kHeadCount * kHeadDim),
      MakeFrameLayout<NDExtLayoutPtn>(kQueryAtomRows, kHeadDim));
  Copy(copy_gm_to_l1, l1_query, gm_query);
  for (int32_t context_start = 0; context_start < kContextCount;
       context_start += kContextTileSize) {
    auto gm_keys = MakeTensor(
        MakeMemPtr(keys + static_cast<int64_t>(context_start) * kHeadDim),
        MakeFrameLayout<DNExtLayoutPtn>(kHeadDim, kContextTileSize));
    Copy(copy_gm_to_l1, l1_keys, gm_keys);
  }
}
```

The AIC uses M=128, K=128, N=32 and dual-destination Fixpipe:

```cpp
MmadParams params;
params.m = kQueryAtomRows;
params.n = kContextTileSize;
params.k = kHeadDim;
params.unitFlag = 0;
params.cmatrixInitVal = true;

fixpipe_params.nSize = kContextTileSize;
fixpipe_params.mSize = kQueryAtomRows;
fixpipe_params.srcStride = kQueryAtomRows;
fixpipe_params.dstStride = kContextTileSize;
fixpipe_params.dualDstCtl = 1;
fixpipe_params.subBlockId = false;
```

On AIV, derive row ownership only after both sub-AIVs enter the handshake:

```cpp
const int32_t query_in_atom =
    static_cast<int32_t>(AscendC::GetSubBlockIdx());
const int32_t query_index =
    atom_index * kQueryAtomSize + query_in_atom;
AscendC::CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_V>(kScoreReadyFlag);
AscendC::CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_V>(kScoreReadyFlag);
asc_vf_call<lightning_indexer_prefill_q2_postprocess_vf>(
    dim3(kThreadsPerBlock, 1, 1),
    shared_scores,
    weights,
    valid_context_lengths,
    topk_scores,
    topk_indices,
    query_index,
    tile_context_start,
    dynamicStartUB);
```

The VF must use the existing `lightning_indexer_merge_topk_ub` helper and
preserve BF16 dot, ReLU, BF16 weighted multiply, float head accumulation, and
final BF16 rounding before merging. Launch exactly:

```cpp
lightning_indexer_prefill_q2_family_64x128_kernel
    <<<kLogicalTaskCount, kFusedTopkDynamicUbufBytes, stream>>>(
    reinterpret_cast<const uint16_t*>(query),
    reinterpret_cast<const uint16_t*>(keys),
    reinterpret_cast<const uint16_t*>(weights),
    valid_context_lengths,
    topk_scores,
    topk_indices,
    reinterpret_cast<uint8_t*>(topk_scores));
```

- [ ] **Step 5: Verify locally and compile remotely**

Run the selected test, then the whole build-shell file. Sync to a new remote
`/tmp/cannbench-indexer-prefill-candidate-XXXXXX` directory and build with the
same CANN/Python/NPU settings as Task 1.

Expected: source tests PASS; `bisheng` build exits 0. Capture resource usage for
the 1024-thread VF and stop to investigate if stack usage is nonzero.

- [ ] **Step 6: Commit**

```bash
git add \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_prefill_q2_family_64x128.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
git commit -m "feat(lightning-indexer): add dual-AIV prefill kernel"
```

---

### Task 3: Add Exact Dispatch, Correctness, and Benchmark Coverage

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_prefill_reference.py`
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/test/v32_prefill_benchmark.py`

**Interfaces:**
- Consumes: Task 2 launcher.
- Produces: exact host dispatch returning int32 `[1,4096,2048]`, sampled NPU score-set validation, stability coverage, and JSON timing output.

- [ ] **Step 1: Write failing bridge tests**

Require an exact helper and predicates before the generic prefill call:

```python
def test_prefill_q2_bridge_dispatches_only_the_exact_v32_bfloat16_shape():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        'if (phase == "prefill" && family == "family_64x128") {', 1
    )[1].split(
        'if (phase == "prefill" && family == "family_32x128") {', 1
    )[0]
    for predicate in (
        "query.scalar_type() == at::ScalarType::BFloat16",
        "query.size(0) == 1",
        "query.size(1) == 4096",
        "query.size(2) == 64",
        "query.size(3) == 128",
        "keys.size(1) == 32768",
        "keys.size(2) == 128",
        "top_k == 2048",
    ):
        assert predicate in body
    assert body.index(
        "lightning_indexer_forward_prefill_q2_family_64x128_bfloat16("
    ) < body.index("lightning_indexer_forward_prefill_family_64x128_float(")
    assert ".item" not in body
```

- [ ] **Step 2: Verify bridge RED**

Run the exact test. Expected: FAIL because no helper or predicate exists.

- [ ] **Step 3: Implement the bridge helper and dispatch**

Declare the Task 2 launcher and add:

```cpp
at::Tensor lightning_indexer_forward_prefill_q2_family_64x128_bfloat16(
    const at::Tensor& query,
    const at::Tensor& keys,
    const at::Tensor& weights,
    const at::Tensor& valid_context_lengths) {
  auto best_scores = at::full(
      {1, 4096, 2048},
      -std::numeric_limits<float>::infinity(),
      query.options().dtype(c10::kFloat));
  auto output = at::zeros({1, 4096, 2048}, query.options().dtype(c10::kInt));
  const auto npu_stream = c10_npu::getCurrentNPUStream();
  const auto acl_stream = npu_stream.stream(true);
  launch_lightning_indexer_prefill_q2_family_64x128_bfloat16(
      query.const_data_ptr<at::BFloat16>(),
      keys.const_data_ptr<at::BFloat16>(),
      weights.const_data_ptr<at::BFloat16>(),
      valid_context_lengths.const_data_ptr<int32_t>(),
      best_scores.mutable_data_ptr<float>(),
      output.mutable_data_ptr<int32_t>(),
      acl_stream);
  record_tensor_on_stream(query, npu_stream);
  record_tensor_on_stream(keys, npu_stream);
  record_tensor_on_stream(weights, npu_stream);
  record_tensor_on_stream(valid_context_lengths, npu_stream);
  record_tensor_on_stream(best_scores, npu_stream);
  record_tensor_on_stream(output, npu_stream);
  return output;
}
```

Inside the existing `prefill/family_64x128` branch, after family checks and
before the generic helper, select only the eight exact predicates from Step 1.

- [ ] **Step 4: Add exact-shape NPU correctness and stability tests**

Factor a target-input helper in `test_prefill_reference.py`:

```python
def _v32_prefill_target_tensors(torch):
    torch.manual_seed(7)
    device = torch.device("npu")
    query = torch.randn(1, 4096, 64, 128, device=device, dtype=torch.bfloat16)
    keys = torch.randn(1, 32768, 128, device=device, dtype=torch.bfloat16)
    weights = torch.rand(1, 4096, 64, device=device, dtype=torch.bfloat16)
    valid = torch.arange(28673, 32769, device=device, dtype=torch.int32).reshape(
        1, 4096
    )
    return query, keys, weights, valid
```

Run the full custom op once, assert shape/dtype, and validate rows
`(0,1365,2730,4095)` by computing the PyTorch reduced scores only for those
rows. Mask with the corresponding valid length and require gathered custom
scores to equal reference TopK scores. Add a three-launch test that synchronizes
after all launches and repeats the same sampled score checks for every output.

Local execution must cleanly SKIP without an NPU custom op; remote execution
must PASS.

- [ ] **Step 5: Add the reproducible benchmark runner**

The runner accepts `--warmups`, `--iters`, `--seed`, and `--stability-runs`,
uses `_v32_prefill_target_tensors`-equivalent allocation, runs sampled score
checks outside the timed region, and prints:

```json
{
  "shape": [1, 4096, 32768, 64, 128, 2048],
  "warmups": 1,
  "iters": 5,
  "samples_ms": [],
  "median_ms": 0.0,
  "min_ms": 0.0,
  "max_ms": 0.0,
  "sampled_rows": [0, 1365, 2730, 4095],
  "sampled_score_sets_match": true,
  "stability_runs": 3
}
```

Time each public custom-op call with `time.perf_counter_ns()` and
`torch.npu.synchronize()` after every call. Exit nonzero if sampled correctness
or repeated-launch stability fails.

- [ ] **Step 6: Run local and remote coverage**

Run locally:

```bash
pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_prefill_reference.py
```

Expected: source tests PASS and NPU tests SKIP locally. Rebuild remotely and
run the same files. Expected: all available NPU tests PASS, including the exact
full-shape and three-launch tests.

- [ ] **Step 7: Commit**

```bash
git add \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_prefill_reference.py \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/v32_prefill_benchmark.py
git commit -m "test(lightning-indexer): validate V3.2 prefill fast path"
```

---

### Task 4: Apply the Standalone and Workflow Performance Gate

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/README.md`
- Modify if the candidate fails: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`
- Modify if dispatch changes: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`
- Modify: `docs/superpowers/plans/2026-07-28-lightning-indexer-v32-prefill-row-parallel.md`

**Interfaces:**
- Consumes: Task 1 `CORRECTED_BASELINE_COMMIT`, Task 3 candidate, and existing `dsa_conformance` runner/comparator.
- Produces: measured default-path decision, reproducible records, completed checklist, and final verification evidence.

- [ ] **Step 1: Build isolated baseline and candidate trees remotely**

Create two explicit remote directories using:

```bash
mktemp -d /tmp/cannbench-indexer-prefill-baseline-XXXXXX
mktemp -d /tmp/cannbench-indexer-prefill-candidate-XXXXXX
```

Populate the first from `CORRECTED_BASELINE_COMMIT` and the second from current
HEAD. Build each in a separate Python process using identical CANN,
`NPU_ARCH=dav-3510`, and `ASCEND_VISIBLE_DEVICES=0` settings. Print the loaded
`aten_dsa_lightning_indexer._C` path before every benchmark to prove isolation.

- [ ] **Step 2: Measure standalone indexer latency**

In both builds run:

```bash
/root/miniconda3/bin/python \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/v32_prefill_benchmark.py \
  --warmups 1 --iters 5 --seed 7 --stability-runs 3
```

The baseline build needs the benchmark script copied without candidate source
or bridge changes. Expected: both JSON results report sampled correctness and
stability success. Keep the exact dispatch only when candidate median is lower
than corrected-baseline median.

- [ ] **Step 3: Run the full workflow gate**

Generate baseline and candidate artifacts using the same seed:

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m \
  cannbench.operators.builtin.dsa_conformance.runner \
  --backend simt --phase prefill --seed 7 --output /tmp/dsa-prefill-baseline

PYTHONPATH=src /root/miniconda3/bin/python -m \
  cannbench.operators.builtin.dsa_conformance.runner \
  --backend simt --phase prefill --seed 7 --output /tmp/dsa-prefill-candidate
```

Compare artifacts from the candidate environment:

```bash
PYTHONPATH=src /root/miniconda3/bin/python -m \
  cannbench.operators.builtin.dsa_conformance.compare \
  /tmp/dsa-prefill-baseline /tmp/dsa-prefill-candidate \
  --atol 0.05 --rtol 0.05 --min-indexer-recall 0.95
```

Expected: exit 0, `passed=true`, no sampled attention/workflow mismatches, and
candidate `timing_seconds.workflow_total` no greater than baseline. Repeat the
workflow timing three times if the first ratio is within 5% of 1.0 and decide
using the median of those three runs.

- [ ] **Step 4: Apply the dispatch decision**

If standalone and workflow gates pass, retain the exact predicate. If either
performance gate fails, remove the call from the exact predicate so production
prefill uses the corrected common kernel; retain the isolated device source for
the deferred comparison only when it remains buildable and covered by source
tests. In either case, keep the Task 1 16-task correction after its correctness
suite passes.

Rerun bridge tests after the final choice. Do not leave a hardcoded false
predicate.

- [ ] **Step 5: Record results and resource usage**

Add a V3.2 prefill benchmark section to `simt/README.md` containing:

- Atlas 350 / `dav-3510` / CANN version and date
- exact BF16 shape and causal length range
- corrected baseline commit and candidate commit
- one warmup and five synchronized samples
- median/min/max for both paths
- candidate/baseline ratio and speedup
- full workflow baseline/candidate timing and conformance result
- 1024-thread stack/register resource report
- whether exact dispatch remains enabled

- [ ] **Step 6: Run final verification**

Run locally:

```bash
pytest -q
git diff --check
rg -n 'lightning_indexer|deepseek_v32_flashmla_prefill' \
  src/cannbench/cli.py src/cannbench/core src/cannbench/backends
```

Expected: full suite passes, diff check is empty, and no new concrete-operator
branch appears in public layers. Rebuild final HEAD remotely and run:

```bash
/root/miniconda3/bin/python -m pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test
```

Expected: no failures.

- [ ] **Step 7: Backfill the plan and commit the gate**

Mark only actually completed checkboxes in this plan, then commit final source,
tests, benchmark documentation, and the backfilled plan:

```bash
git add \
  src/cannbench/operators/builtin/lightning_indexer/simt/README.md \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
git add -f \
  docs/superpowers/plans/2026-07-28-lightning-indexer-v32-prefill-row-parallel.md
git commit -m "perf(lightning-indexer): gate V3.2 prefill fast path"
```
