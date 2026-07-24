# DSA HD256 and HD576 CV Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Add BF16 custom-op fast paths for the four realistic HD256 and HD576 DSA workflows while preserving two-stage workflow execution and per-tile CV fusion.

**Architecture:** Generalize the existing lightning-indexer `64x128` kernel to runtime 32/64 index heads, and generalize the existing sparse-attention HD512 BF16 fused kernel to runtime head dimensions 256/512/576. Existing device synchronization sites and Cube-to-UB-to-SIMT data flow are reused rather than duplicated.

**Tech Stack:** Python, pytest, PyTorch custom operators, torch_npu, Ascend C Tensor API, Ascend SIMT VF, CANN/Bisheng, Atlas 350.

## Global Constraints

- Implement and validate BF16 only.
- Keep `dsa_prefill` and `dsa_decode` as two-step workflows.
- Do not add operator-specific logic to public CLI, core, or backend modules.
- Do not add fallback execution or a GM score intermediate.
- Do not add new Basic API or CrossCore synchronization call sites.
- Preserve existing HD128, HD512, 4x64, and 64x128 behavior.
- Do not profile; remote validation is accuracy and runtime stability only.

---

### Task 1: Add Failing Dispatch Contracts

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_dispatch.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_dispatch.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/__init__.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/__init__.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/ops.py`

**Interfaces:**
- Consumes: `_select_simt_family(payload: dict[str, object]) -> str`
- Produces: dispatch contracts for `family_32x128`, `family_hd256`, and `family_hd576`

- [ ] **Step 1: Add the indexer dispatch test**

```python
def test_select_simt_family_prefers_32x128_family():
    payload = {
        "index_heads": 32,
        "index_dim": 128,
        "phase": "prefill",
        "top_k": 2048,
    }
    assert _select_simt_family(payload) == "family_32x128"
```

- [ ] **Step 2: Add sparse-attention dispatch tests**

```python
@pytest.mark.parametrize(
    ("head_dim", "expected"),
    [(256, "family_hd256"), (576, "family_hd576")],
)
def test_select_simt_family_prefers_new_wide_head_families(head_dim, expected):
    payload = {
        "head_dim": head_dim,
        "kv_heads": 1,
        "query_heads": 64 if head_dim == 256 else 128,
        "selected_tokens": 2048,
    }
    assert _select_simt_family(payload) == expected
```

- [ ] **Step 3: Run the tests and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/lightning_indexer/simt/test/test_dispatch.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_dispatch.py
```

Expected: the three new assertions fail because selectors return `fallback`.

- [ ] **Step 4: Extend the operator-local selectors**

```python
if payload["index_heads"] == 32 and payload["index_dim"] == 128:
    return "family_32x128"

if payload["head_dim"] == 256 and payload["kv_heads"] == 1:
    return "family_hd256"
if payload["head_dim"] == 576 and payload["kv_heads"] == 1:
    return "family_hd576"
```

Extend the sparse wrapper's accepted family set without adding a fallback.

- [ ] **Step 5: Run dispatch tests and verify GREEN**

Run the command from Step 3. Expected: all dispatch tests pass.

- [ ] **Step 6: Commit the dispatch contract**

```bash
git add src/cannbench/operators/builtin/lightning_indexer src/cannbench/operators/builtin/sparse_attention
git commit -m "feat(dsa): dispatch HD256 and HD576 families"
```

### Task 2: Generalize Lightning Indexer x128 CV Kernel

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/lightning_indexer.asc`
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/v1/aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_fused_family_64x128.asc`

**Interfaces:**
- Consumes: `family_32x128` and `family_64x128` dispatch strings
- Produces: one x128 launcher accepting head count 32 or 64 and `top_k <= 2048`

- [ ] **Step 1: Add source-layout tests for runtime x128 heads**

```python
def test_lightning_indexer_x128_kernel_supports_runtime_32_or_64_heads():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_fused_family_64x128.asc"
    ).read_text(encoding="utf-8")
    assert "int32_t head_count" in source
    assert "params.m = static_cast<uint16_t>(head_count);" in source
    assert "fixpipe_params.mSize = head_count;" in source
    assert "head_index < head_count" in source

def test_lightning_indexer_x128_bridge_accepts_top2048():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    assert "family_32x128 custom op requires top_k <= 2048" in source
    assert "family_64x128 custom op requires top_k <= 2048" in source
```

- [ ] **Step 2: Run focused tests and verify RED**

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test/test_lightning_indexer_v1_build_shell.py
```

Expected: runtime-head and top-2048 assertions fail.

- [ ] **Step 3: Generalize the existing device source**

Thread `head_count` through the launcher, mixed kernel, AIC function, AIV function, and SIMT VF. Keep maximum static buffers at 64 heads.

```cpp
params.m = static_cast<uint16_t>(head_count);
fixpipe_params.mSize = static_cast<uint32_t>(head_count);
for (int32_t head_index = 0; head_index < head_count; ++head_index) {
  // Existing BF16 relu, weighting, and reduction sequence.
}
```

Do not add synchronization calls. Preserve `MakeMmad`, Fixpipe, LCM, and `asc_vf_call` ordering.

- [ ] **Step 4: Generalize bridge validation and dispatch**

Use one x128 host helper with `head_count`. Add 32x128 shape checks and change 64x128's limit to 2048.

```cpp
TORCH_CHECK(query.size(2) == 32, "family_32x128 requires H == 32");
TORCH_CHECK(query.size(3) == 128, "family_32x128 requires D == 128");
TORCH_CHECK(top_k <= 2048, "family_32x128 custom op requires top_k <= 2048");
```

- [ ] **Step 5: Run focused indexer tests and verify GREEN**

```bash
pytest -q src/cannbench/operators/builtin/lightning_indexer/simt/test
```

Expected: all indexer tests pass.

- [ ] **Step 6: Commit the x128 implementation**

```bash
git add src/cannbench/operators/builtin/lightning_indexer
git commit -m "feat(lightning_indexer): support 32x128 and top2048"
```

### Task 3: Generalize Sparse Attention Wide-Head BF16 CV Kernel

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_score_family_hd512.asc`

**Interfaces:**
- Consumes: BF16 tensors in HD256, HD512, and HD576 families
- Produces: `(output, lse)` from one CV kernel per query tile without a GM score tensor

- [ ] **Step 1: Add wide-head source-layout tests**

```python
def test_sparse_attention_wide_fused_kernel_uses_runtime_head_dim():
    source = _score_source(512)
    fused = _function_body(
        source,
        "__aicore__ inline void sparse_attention_fused_family_hd512_aic(",
        "extern \"C\" void launch_sparse_attention_score_gather_hd512_float(",
    )
    assert "shape.k" in fused
    assert "int32_t head_dim" in fused
    assert "dim_index < head_dim" in fused
    assert "kHeadDim" not in fused
```

Add bridge assertions that all three wide families use the same fused helper, allow selected tokens 2048, and do not allocate `scores` in the BF16 branch.

- [ ] **Step 2: Add reduced numerical reference cases**

```python
("family_hd256", (1, 64, 3, 256), (1, 1, 32, 256), (1, 3, 16)),
("family_hd576", (1, 128, 2, 576), (1, 1, 32, 576), (1, 2, 16)),
```

- [ ] **Step 3: Run focused tests and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py
```

Expected: new family and runtime-head assertions fail.

- [ ] **Step 4: Generalize the fused device section**

Keep legacy compatibility functions fixed to HD512, but make the BF16 fused section use `shape.k`. Thread it through AIC addressing, AIV key/value/output strides, and SIMT postprocess.

```cpp
const int32_t head_dim = shape.k;
const int64_t key_offset = batch_index * context_tokens * head_dim;
for (int32_t dim_index = threadIdx.x;
     dim_index < head_dim;
     dim_index += blockDim.x) {
  // Existing weighted V reduction.
}
```

Change the fused launch function to accept `head_dim` and initialize tiling `k=head_dim`. Preserve `kBaseK=64`, `kBaseN=64`, selected-token UB capacity 2048, and every existing synchronization call site.

- [ ] **Step 5: Generalize the bridge**

Route `family_hd256`, `family_hd512`, and `family_hd576` through one BF16 wide helper. Validate exact head dimension, `kv_heads == 1`, and `selected_tokens <= 2048`. Do not apply the HD128 single-query decode restriction to wide families.

- [ ] **Step 6: Run focused sparse-attention tests and verify GREEN**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
```

Expected: all sparse-attention tests pass.

- [ ] **Step 7: Commit the wide-head implementation**

```bash
git add src/cannbench/operators/builtin/sparse_attention
git commit -m "feat(sparse_attention): fuse HD256 and HD576 families"
```

### Task 4: Update Coverage and Run Local Regression

**Files:**
- Modify: `src/cannbench/operators/builtin/lightning_indexer/simt/README.md`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/README.md`

**Interfaces:**
- Consumes: implemented family limits and realistic manifests
- Produces: documentation declaring 15/15 realistic workflow fast-path coverage

- [ ] **Step 1: Update coverage tables and limits**

```text
lightning_indexer family_32x128: top_k <= 2048
lightning_indexer family_64x128: top_k <= 2048
sparse_attention family_hd256: kv_heads=1, selected_tokens <= 2048
sparse_attention family_hd576: kv_heads=1, selected_tokens <= 2048
```

- [ ] **Step 2: Check architecture boundaries**

```bash
rg -n "dsa_prefill|dsa_decode|family_hd256|family_hd576|family_32x128" \
  src/cannbench/cli.py src/cannbench/core src/cannbench/backends
```

Expected: no new public-layer family or workflow branches.

- [ ] **Step 3: Run the full suite**

```bash
pytest -q
```

Expected: zero failures.

- [ ] **Step 4: Verify diff hygiene and commit**

```bash
git diff --check
git status --short
git add src/cannbench/operators/builtin/lightning_indexer/simt/README.md \
  src/cannbench/operators/builtin/sparse_attention/simt/README.md
git commit -m "docs(dsa): record complete realistic fast-path coverage"
```

### Task 5: Build and Validate on Atlas 350

**Files:**
- No repository files created or modified
- Remote build directory: `/root/cannbench-dsa-hd256-hd576-<commit>`
- Remote result directory: `<remote-workdir>/dsa-results/`

**Interfaces:**
- Consumes: committed BF16 custom-op sources
- Produces: build logs and per-case accuracy results

- [ ] **Step 1: Deploy an exact-commit archive to an isolated remote directory**

```bash
git archive --format=tar.gz --prefix=cannbench-dsa-hd256-hd576/ \
  -o /tmp/cannbench-dsa-hd256-hd576.tar.gz HEAD
scp -P 20002 /tmp/cannbench-dsa-hd256-hd576.tar.gz root@121.41.199.170:/root/
```

- [ ] **Step 2: Build both v1 packages**

Source CANN, set `NPU_ARCH=dav-3510`, select device 0, and run both operator-local install scripts with `/root/miniconda3/bin/python`.

Expected: both editable wheels build and install successfully.

- [ ] **Step 3: Run reduced family checks**

```text
indexer 32x128/top2048
indexer 64x128/top2048
sparse HD256/Q3/top2048
sparse HD576/Q2/top2048
```

Expected: no runtime error or deadlock; indexer score sets, sparse output, and LSE meet `atol=0.05, rtol=0.05`.

- [ ] **Step 4: Run all four realistic workflows**

Run the design document's four case IDs with BF16, `ASCEND_LAUNCH_BLOCKING=1`, accuracy only, and no profiler. Continue after individual failures.

- [ ] **Step 5: Summarize and verify results**

Produce a four-row table with phase, case ID, indexer status, sparse output/LSE status, runtime error, and elapsed time. Verify no process is left running.

- [ ] **Step 6: Run final repository verification**

```bash
pytest -q
git status --short --branch
git log -1 --oneline
```

Expected: the full suite passes and the worktree is clean.

## Plan Self-Review

- Both component operators and all four unsupported workflows are covered.
- Every production behavior is preceded by a failing test.
- Family names, limits, and runtime dimensions are consistent across dispatch, bridge, device source, tests, and documentation.
- No task modifies public framework layers or adds fallback, GM score storage, or synchronization dependencies.
