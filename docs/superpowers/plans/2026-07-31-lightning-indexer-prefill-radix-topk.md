# Lightning Indexer Prefill Radix TopK Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the canonical V3.2 prefill path's per-128-token full bitonic merge with a 256 MiB BF16 score workspace and a separate radix-select TopK stage.

**Architecture:** An exact-shape operator-local fast path uses 32 persistent `MIX_AIC_1_2` tasks to compute pairs of Query rows and write complete BF16 reduced-score rows. A second pure-Vector launch assigns rows persistently across 32 blocks, finds the Top-2048 BF16 threshold with two 8-bit radix histogram passes, deterministically compacts lower-index ties, and bitonic-sorts only the selected 2048 candidates. Canonical decode and every unsupported prefill shape keep their existing dispatch.

**Tech Stack:** C++17 torch-npu custom op bridge, Ascend C API, Tensor API, SIMT API, Bisheng `dav-3510`, pytest source contracts, operator-local NPU benchmark runner.

## Global Constraints

- Keep every implementation change under `src/cannbench/operators/builtin/lightning_indexer/` except this plan document.
- Do not add concrete operator branches to CLI, core, shared backends, or result modules.
- New device source may use only C API, Tensor API, and SIMT API; do not add Basic API headers, `AscendC::LocalTensor`, Basic synchronization calls, or inter-core synchronization.
- The fast path is exact BF16 `B=1,Q=4096,C=32768,H=64,D=128,K=2048` prefill only.
- Preserve lower context index as the tie breaker and return sorted int32 indices with shape `[1,4096,2048]`.
- Preserve canonical decode dispatch and the generic fused fallback.
- Enable the fast path only after correctness, stability, and synchronized public-op latency gates pass on `dav-3510`.

---

### Task 1: Lock Source, Build, and Dispatch Contracts

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_topk_simt_source.py`

**Interfaces:**
- Consumes: current `setup.py` build map and `lightning_indexer.asc` dispatch source.
- Produces: failing contracts for `lightning_indexer_prefill_full_score_family_64x128.asc`, `lightning_indexer_radix_topk_bfloat16.asc`, and the exact host bridge.

- [ ] **Step 1: Write the failing full-score source contract**

```python
def test_prefill_full_score_path_uses_q2_persistent_tasks_and_allowed_apis():
    source = (_simt_root() / "lightning_indexer_prefill_full_score_family_64x128.asc").read_text()
    for expected in (
        "kQueryAtomSize = 2", "kPersistentTaskCount = 32",
        "kContextTileSize = 32", "kThreadsPerBlock = 1024",
        "atom_index += kPersistentTaskCount", "dual_dst_ctl = 1",
        "asc_sync_block_arrive", "asc_sync_block_wait",
        "reduced_scores[output_base + context_index]",
    ):
        assert expected in source
    for forbidden in (
        "basic_api/", "kernel_operator.h", "AscendC::LocalTensor",
        "SetFlag", "WaitFlag", "PipeBarrier", "CrossCore",
        "lightning_indexer_merge_topk_ub",
    ):
        assert forbidden not in source
```

- [ ] **Step 2: Write the failing radix TopK contract**

```python
def test_prefill_radix_topk_selects_bf16_threshold_then_sorts_only_topk():
    source = (_simt_root() / "lightning_indexer_radix_topk_bfloat16.asc").read_text()
    for expected in (
        "kRadixBits = 8", "kRadixBins = 256", "kRadixPassCount = 2",
        "asc_atomic_add", "threshold_key", "score_key > threshold_key",
        "score_key == threshold_key", "candidate_index < other_index",
        "bitonic_size <= kTopK", "row_index += kPersistentBlockCount",
    ):
        assert expected in source
    assert "kSortCapacity = 4096" not in source
    assert "basic_api/" not in source
```

- [ ] **Step 3: Write failing build and exact-dispatch contracts**

```python
def test_prefill_full_score_and_radix_topk_build_as_separate_libraries():
    setup = _setup_source()
    assert "lightning_indexer_prefill_full_score_family_64x128.asc" in setup
    assert "lightning_indexer_radix_topk_bfloat16.asc" in setup

def test_exact_v32_prefill_dispatches_full_score_then_radix_topk():
    bridge = _bridge_source()
    body = bridge.split("lightning_indexer_forward_prefill_full_score_bfloat16(", 1)[1].split("\n}\n", 1)[0]
    assert "{1, 4096, 32768}" in body
    assert "c10::kBFloat16" in body
    assert body.index("launch_lightning_indexer_prefill_full_score") < body.index("launch_lightning_indexer_radix_topk_bfloat16")
```

- [ ] **Step 4: Run the targeted tests and verify RED**

Run: `pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py src/cannbench/operators/builtin/lightning_indexer/simt/test/test_topk_simt_source.py`

Expected: FAIL because both new device sources and bridge symbols are absent.

- [ ] **Step 5: Commit the red contracts**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/test
git commit -m "test(lightning-indexer): specify prefill radix topk path"
```

### Task 2: Implement BF16 Radix Selection

**Files:**
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_radix_topk_bfloat16.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py`

**Interfaces:**
- Consumes: BF16 `reduced_scores[row_count,32768]` and runtime `row_count`.
- Produces: `launch_lightning_indexer_radix_topk_bfloat16(const bfloat16_t*, int32_t*, int32_t, aclrtStream)`.

- [ ] **Step 1: Add the separate device library to `KERNEL_LIBRARIES`**

```python
"liblightning_indexer_radix_topk_bfloat16_kernel.so": os.path.join(
    EXTENSIONS_DIR, "simt", "lightning_indexer_radix_topk_bfloat16.asc"
),
```

- [ ] **Step 2: Implement monotonic BF16 radix keys and two histogram passes**

```cpp
__SIMT_DEVICE_FUNCTIONS_DECL__ inline uint16_t ordered_bf16_key(uint16_t bits) {
  return (bits & 0x8000U) != 0U
      ? static_cast<uint16_t>(~bits)
      : static_cast<uint16_t>(bits ^ 0x8000U);
}

for (int32_t pass = 0; pass < kRadixPassCount; ++pass) {
  const int32_t shift = (kRadixPassCount - 1 - pass) * kRadixBits;
  // Clear 256 UB counters, histogram keys matching the selected prefix,
  // and let thread 0 choose the descending bucket containing remaining_rank.
}
```

- [ ] **Step 3: Implement deterministic compaction and final TopK sort**

```cpp
// Compact scores strictly above the threshold with a UB atomic counter.
// For equal-threshold scores, scan each 1024-index chunk in index order and
// retain exactly remaining_rank entries so lower context indices win ties.
// Sort candidate_scores[0:2048] by score descending/index ascending.
for (uint32_t bitonic_size = 2U; bitonic_size <= kTopK; bitonic_size <<= 1U) {
  for (uint32_t stride = bitonic_size >> 1U; stride > 0U; stride >>= 1U) {
    // Parallel compare/swap followed by asc_syncthreads().
  }
}
```

- [ ] **Step 4: Run targeted tests and verify the radix contracts are GREEN**

Run: `pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_topk_simt_source.py src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

Expected: radix and build-map assertions pass; full-score/bridge assertions remain red.

- [ ] **Step 5: Commit radix selection**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/v1
git commit -m "feat(lightning-indexer): add bf16 radix topk kernel"
```

### Task 3: Implement Persistent Full-Score Prefill

**Files:**
- Create: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_prefill_full_score_family_64x128.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py`

**Interfaces:**
- Consumes: exact BF16 Query/Key/weight tensors and int32 valid lengths.
- Produces: `launch_lightning_indexer_prefill_full_score_family_64x128_bfloat16(..., bfloat16_t* reduced_scores, aclrtStream)`.

- [ ] **Step 1: Add the full-score device library to `KERNEL_LIBRARIES`**

```python
"liblightning_indexer_prefill_full_score_family_64x128_kernel.so": os.path.join(
    EXTENSIONS_DIR, "simt", "lightning_indexer_prefill_full_score_family_64x128.asc"
),
```

- [ ] **Step 2: Implement 32-task Q=2 score production**

```cpp
constexpr int32_t kQueryAtomSize = 2;
constexpr int32_t kQueryAtomCount = 2048;
constexpr int32_t kPersistentTaskCount = 32;
constexpr int32_t kContextTileSize = 32;

for (int32_t atom_index = task_id; atom_index < kQueryAtomCount;
     atom_index += kPersistentTaskCount) {
  // M=128, K=128, N=32 MMAD. dual_dst_ctl=1 routes one Query row to each AIV.
  // C API mode-2 block synchronization protects the shared 8 KiB score tile.
}
```

- [ ] **Step 3: Implement score postprocess without TopK**

```cpp
const int32_t query_index = atom_index * kQueryAtomSize + query_in_atom;
const int64_t output_base = static_cast<int64_t>(query_index) * kContextCount;
if (context_index >= valid_context_lengths[query_index]) {
  reduced_scores[output_base + context_index] =
      static_cast<bfloat16_t>(-std::numeric_limits<float>::infinity());
} else {
  // Preserve MMAD->BF16, ReLU, BF16 weight multiply, FP32 accumulation,
  // and final BF16 rounding order from the validated decode score path.
}
```

- [ ] **Step 4: Run targeted tests and verify the source contract is GREEN**

Run: `pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

Expected: full-score and build-map assertions pass; bridge assertions remain red.

- [ ] **Step 5: Commit score production**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/v1
git commit -m "feat(lightning-indexer): materialize prefill scores"
```

### Task 4: Wire the Exact Host Fast Path

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`

**Interfaces:**
- Consumes: both new launcher functions.
- Produces: `lightning_indexer_forward_prefill_full_score_bfloat16(...) -> at::Tensor`.

- [ ] **Step 1: Declare the two launchers**

```cpp
extern "C" void launch_lightning_indexer_prefill_full_score_family_64x128_bfloat16(
    const at::BFloat16*, const at::BFloat16*, const at::BFloat16*,
    const int32_t*, at::BFloat16*, aclrtStream);
extern "C" void launch_lightning_indexer_radix_topk_bfloat16(
    const at::BFloat16*, int32_t*, int32_t, aclrtStream);
```

- [ ] **Step 2: Allocate workspace and launch in stream order**

```cpp
auto reduced_scores = at::empty(
    {1, 4096, 32768}, query.options().dtype(c10::kBFloat16));
auto output = at::empty(
    {1, 4096, 2048}, query.options().dtype(c10::kInt));
launch_lightning_indexer_prefill_full_score_family_64x128_bfloat16(...);
launch_lightning_indexer_radix_topk_bfloat16(
    reduced_scores.const_data_ptr<at::BFloat16>(),
    output.mutable_data_ptr<int32_t>(), 4096, acl_stream);
```

- [ ] **Step 3: Gate only the canonical shape before generic prefill dispatch**

```cpp
if (query.scalar_type() == at::ScalarType::BFloat16 &&
    query.size(0) == 1 && query.size(1) == 4096 &&
    keys.size(1) == 32768 && top_k == 2048) {
  return lightning_indexer_forward_prefill_full_score_bfloat16(...);
}
```

- [ ] **Step 4: Run all operator-local source tests and verify GREEN**

Run: `pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test`

Expected: all source, dispatch, and reference tests pass.

- [ ] **Step 5: Commit host dispatch**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt
git commit -m "perf(lightning-indexer): dispatch prefill radix topk"
```

### Task 5: Build, Accuracy, Stability, and Performance Gate

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/v32_prefill_benchmark.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/README.md`

**Interfaces:**
- Consumes: installed custom op and current canonical prefill benchmark inputs.
- Produces: score-set accuracy, repeated-launch stability, public-op timing, and profile evidence.

- [ ] **Step 1: Extend benchmark output with workspace and launch expectations**

```python
result.update(
    score_workspace_bytes=1 * 4096 * 32768 * 2,
    expected_score_kernel_launches=1,
    expected_topk_kernel_launches=1,
)
```

- [ ] **Step 2: Run local source and full repository verification**

Run: `pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test`

Expected: all operator-local tests pass.

Run: `pytest -q`

Expected: all repository tests pass with the existing skips only.

- [ ] **Step 3: Run architecture-boundary searches**

```bash
rg -n "lightning_indexer" src/cannbench/cli.py src/cannbench/core src/cannbench/backends
rg -n "basic_api/|kernel_operator.h|AscendC::LocalTensor|SetFlag|WaitFlag|PipeBarrier|CrossCore" \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_prefill_full_score_family_64x128.asc \
  src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_radix_topk_bfloat16.asc
```

Expected: no public-layer operator hardcoding and no forbidden API usage in either new source.

- [ ] **Step 4: Build and validate on `dav-3510`**

Run in an isolated remote checkout:

```bash
cd src/cannbench/operators/builtin/lightning_indexer/simt/v1
./install.sh
python ../test/v32_prefill_benchmark.py --warmups 1 --iters 5 --stability-runs 3
```

Expected: Bisheng build succeeds, sampled selected-score sets match, and all repeated launches complete.

- [ ] **Step 5: Profile both kernels and enforce the public-op gate**

```bash
msopprof --output="$(mktemp -d /tmp/msopprof-li-prefill-radix-XXXXXX)" \
  --aic-metrics=BasicInfo \
  python ../test/v32_prefill_benchmark.py --warmups 1 --iters 1 --stability-runs 1
```

Expected: one score kernel and one radix TopK kernel are visible; synchronized public-op median is lower than the retained common-path baseline measured on the same checkout and input seed. If it is not lower, remove the exact dispatch gate while retaining the experimental sources and measurements.

- [ ] **Step 6: Record measured results and commit documentation**

```bash
git add src/cannbench/operators/builtin/lightning_indexer/simt/README.md \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/v32_prefill_benchmark.py
git commit -m "docs(lightning-indexer): record prefill radix topk results"
```
