# Sparse Attention V3.2 Prefill Head64 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Automatically run the exact DeepSeek V3.2 BF16 Sparse Attention prefill case through one Head64/P1 mixed kernel that keeps 32 AICs and 64 AIVs working and writes final BF16 output and FP32 LSE directly.

**Architecture:** Reuse the isolated V3.2 decode Head64 fused QK/online-softmax/PV source with a new direct-output mode. Prefill has 8192 natural `(batch, query_token, head_group64)` tasks, so 32 physical mixed tasks consume them persistently without Split-S or Combine; only the exact target shape is selected automatically, while explicit Head64/P1 remains available for reduced validation.

**Tech Stack:** Ascend CANN 9.2.0, dav-3510 mixed AIC/AIV kernels, Tensor API, SIMT API, existing transitional mode-2 Head64 synchronization, PyTorch/torch-npu custom op, pytest, msopprof.

## Global Constraints

- Keep code and tests under `src/cannbench/operators/builtin/sparse_attention/`; only spec and plan documents live under `docs/superpowers/`.
- Do not change CLI, shared backends, core configuration, workflow plugins, result schemas, or published data contracts.
- Automatic dispatch matches only BF16 `prefill/family_hd576/B1/H128/KV_H1/Q4096/C32768/S2048/Dqk576/Dv512` with default tuning.
- Explicit BF16 Head64/P1 prefill supports dynamic `B/Q/C/S` with `S <= 2048`; explicit prefill P2/P4 fails clearly.
- Preserve every existing decode P1/P2/P4 route and layout.
- Launch at most 32 mixed tasks; one mixed task maps to one AIC and two AIVs.
- Every Head64 SIMT VF launches 1024 threads.
- Add no Basic API header, synchronization primitive, or CrossCore call. Reuse the shared Head64 mode-2 sequence unchanged.
- Write tests before production changes and observe each intended failure.
- Final history contains the amended design/plan commit and one implementation commit; later changes amend those commits.

## File Structure

- `sparse_attention_head64_plan.h`: output-mode enum and shared POD plan.
- `sparse_attention.asc`: exact predicate, plan validation, direct Host branch, and cast skip.
- `sparse_attention_head64_fused_hd576.asc`: shared P4 partial and P1 direct-output modes.
- `test_sparse_attention_v1_build_shell.py`: dispatch, device, launch, and API-boundary contracts.
- `head64_reduced_accuracy.py`: explicit P1 prefill reduced, boundary, reuse, repeat, and concurrent coverage.
- `v32_full_accuracy.py`: existing bounded-memory full-shape validation.
- `v32_prefill_benchmark.py`: same-input synchronized wall-time runner.
- `test_v32_full_accuracy_runner.py`: runner unit contracts.
- `simt/README.md` and `PARALLEL_SPLITTING_RESEARCH.zh-CN.md`: measured evidence and final dispatch decision.

---

### Task 1: Add The Prefill Output Mode And Automatic Plan

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`

**Interfaces:**
- Consumes: `SparseAttentionHead64Plan` and `make_sparse_attention_head64_plan(...)`.
- Produces: `SparseAttentionHead64OutputMode`, `output_mode`, `is_automatic_v32_prefill_head64(...)`, and phase-aware validation.

- [x] **Step 1: Write failing output-mode and plan-validation tests**

```python
def test_sparse_attention_head64_plan_has_partial_and_direct_output_modes():
    source = _head64_plan_source()
    assert "kHead64OutputPartialFloat = 0" in source
    assert "kHead64OutputDirectBfloat16 = 1" in source
    assert "int32_t output_mode;" in source


def test_sparse_attention_head64_plan_accepts_only_p1_for_prefill():
    plan = _function_definition(
        _bridge_source(), "make_sparse_attention_head64_plan("
    )
    assert 'phase == "decode" || phase == "prefill"' in plan
    assert 'phase != "prefill" || selected_partitions == 1' in plan
    assert "head64 prefill requires selected_partitions=1" in plan
    assert "kHead64OutputDirectBfloat16" in plan
```

- [x] **Step 2: Write a failing exact-predicate test**

```python
def test_sparse_attention_v32_prefill_head64_automatic_predicate_is_exact():
    predicate = _function_definition(
        _bridge_source(), "is_automatic_v32_prefill_head64("
    )
    for expected in (
        'phase == "prefill"',
        'family == "family_hd576"',
        "query.scalar_type() == at::ScalarType::BFloat16",
        "shared_kv.scalar_type() == at::ScalarType::BFloat16",
        "query.size(0) == 1",
        "query.size(1) == 128",
        "query.size(2) == 4096",
        "query.size(3) == 576",
        "shared_kv.size(1) == 1",
        "shared_kv.size(2) == 32768",
        "indices.size(2) == 2048",
        "value_head_dim == 512",
    ):
        assert expected in predicate
```

- [x] **Step 3: Verify RED**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  -k 'head64_plan_has_partial or head64_plan_accepts_only_p1 or automatic_predicate_is_exact'
```

Expected: three assertion failures because the new output mode and predicate do not exist.

- [x] **Step 4: Add the shared output mode**

```cpp
enum SparseAttentionHead64OutputMode : int32_t {
  kHead64OutputPartialFloat = 0,
  kHead64OutputDirectBfloat16 = 1,
};
```

Append `int32_t output_mode;` to `SparseAttentionHead64Plan`.

- [x] **Step 5: Implement the exact predicate and phase-aware plan**

Add:

```cpp
bool is_automatic_v32_prefill_head64(
    const at::Tensor& query,
    const at::Tensor& shared_kv,
    const at::Tensor& indices,
    int64_t value_head_dim,
    std::string_view phase,
    std::string_view family);
```

Implement the exact conjunction from the design. Permit `decode` or `prefill` in the plan builder, reject prefill partitions other than one, and set:

```cpp
plan.output_mode = phase == "prefill"
    ? kHead64OutputDirectBfloat16
    : kHead64OutputPartialFloat;
```

Do not change dispatch in this task.

- [x] **Step 6: Verify GREEN**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py -k 'head64_plan or automatic_predicate'
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
git diff --check
```

Expected: all selected and operator-local tests pass; diff check is empty.

- [x] **Step 7: Create the implementation commit**

```bash
git add src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc
git commit -m "perf(sparse-attention): add v32 prefill head64 path"
```

---

### Task 2: Add Direct BF16 Output To The Fused Kernel

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`

**Interfaces:**
- Consumes: Task 1 `plan.output_mode` and `Head64Task`.
- Produces: an extended `launch_sparse_attention_head64_fused_hd576_bf16(...)` supporting unchanged FP32 partials or direct BF16 output/FP32 LSE.

- [x] **Step 1: Write failing direct and partial layout tests**

```python
def test_sparse_attention_head64_fused_direct_mode_writes_public_layout():
    writer = _function_definition(
        _head64_fused_source(), "head64_fused_output_write_vf("
    )
    assert "plan.output_mode == kHead64OutputDirectBfloat16" in writer
    assert "task.batch_index" in writer
    assert "task.query_token" in writer
    assert "task.head_group * kHead64Tile" in writer
    assert "static_cast<bfloat16_t>" in writer
    assert "plan.query_heads" in writer
    assert "plan.query_tokens" in writer


def test_sparse_attention_head64_fused_keeps_decode_partial_layout():
    writer = _function_definition(
        _head64_fused_source(), "head64_fused_output_write_vf("
    )
    assert "kHead64OutputPartialFloat" in writer
    assert "static_cast<int64_t>(logical_task) * kHead64Tile" in writer
    assert "task_output[" in writer
    assert "partial_lse[" in writer
```

- [x] **Step 2: Write failing launcher and API-boundary tests**

```python
def test_sparse_attention_head64_fused_launcher_accepts_direct_outputs():
    launcher = _head64_fused_source().split(
        'extern "C" void launch_sparse_attention_head64_fused_hd576_bf16(', 1
    )[1]
    assert "bfloat16_t* output" in launcher
    assert "float* lse" in launcher
    assert "plan->used_core_num" in launcher


def test_sparse_attention_prefill_head64_adds_no_sync_dependency():
    source = _head64_fused_source()
    assert source.count('#include "basic_api/') == 2
    assert source.count("CrossCoreSetFlag") == 9
    assert source.count("CrossCoreWaitFlag") == 9
```

The exact counts come from `origin/main` commit `1297d3e`; the test freezes them.

- [x] **Step 3: Verify RED**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  -k 'direct_mode_writes_public or keeps_decode_partial or launcher_accepts_direct'
```

Expected: failures show the writer only accepts FP32 logical-task output.

- [x] **Step 4: Extend the kernel and launcher boundary**

Add nullable final pointers after the existing partial pointers:

```cpp
__gm__ bfloat16_t* output,
__gm__ float* lse,
```

Pass the decoded task and full plan into `head64_fused_output_write_vf(...)`. Keep the current partial branch unchanged. For direct output compute:

```cpp
const int32_t global_head =
    task.head_group * kHead64Tile + task_head;
const int64_t final_row =
    (static_cast<int64_t>(task.batch_index) * plan.query_heads + global_head) *
        plan.query_tokens +
    task.query_token;
output[final_row * plan.value_head_dim + dim] =
    running_sum > 0.0F
    ? static_cast<bfloat16_t>(
          running_output_values[local_head * 512 + dim] / running_sum)
    : static_cast<bfloat16_t>(0.0F);
lse[final_row] = running_sum > 0.0F
    ? running_max + logf(running_sum)
    : -std::numeric_limits<float>::infinity();
```

Both AIVs execute the writer. Add no subblock return and do not change the AIC/AIV handshake sequence.

- [x] **Step 5: Update Host declarations and wrappers**

Update the `extern "C"` declaration and `run_sparse_attention_head64_fused_hd576_bf16(...)` with the same pointer order. Decode P4 passes `nullptr, nullptr`; Task 3 supplies final tensors.

- [x] **Step 6: Verify GREEN and amend**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git diff --check
git add src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc
git commit --amend --no-edit
```

Expected: all source tests pass and the implementation commit is amended.

---

### Task 3: Route Automatic Prefill Through One Direct Launch

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`

**Interfaces:**
- Consumes: Tasks 1 and 2 predicate, direct output mode, and launcher.
- Produces: exact default dispatch plus explicit reduced P1 prefill, with no partials, Combine, or cast.

- [x] **Step 1: Write failing route tests**

```python
def test_sparse_attention_v32_prefill_automatically_routes_head64_p1():
    private = _function_definition(
        _bridge_source(), "sparse_attention_forward_privateuse1("
    )
    assert "is_automatic_v32_prefill_head64(" in private
    assert "auto_head64_prefill" in private
    assert "effective_head_tile" in private
    assert "effective_selected_partitions" in private
    assert private.index("auto_head64_prefill") < private.index("is_wide_family")


def test_sparse_attention_prefill_head64_has_no_partials_or_combine():
    body = _function_definition(
        _bridge_source(), "sparse_attention_forward_family_hd576_head64("
    )
    direct = body.split(
        "if (plan.output_mode == kHead64OutputDirectBfloat16)", 1
    )[1].split("if (plan.selected_partitions == 4)", 1)[0]
    assert "run_sparse_attention_head64_fused_hd576_bf16(" in direct
    for forbidden in (
        "task_output",
        "task_lse",
        "run_sparse_attention_head64_combine_hd576_bf16",
        "run_sparse_attention_head64_qk_hd576_bf16",
        "run_sparse_attention_head64_pv_hd576_bf16",
    ):
        assert forbidden not in direct


def test_sparse_attention_head64_skips_matching_dtype_cast():
    private = _function_definition(
        _bridge_source(), "sparse_attention_forward_privateuse1("
    )
    assert "raw_output.scalar_type() == query.scalar_type()" in private
```

- [x] **Step 2: Verify RED**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  -k 'automatically_routes_head64 or has_no_partials_or_combine or skips_matching_dtype'
```

Expected: failures because default prefill still enters the generic wide-family helper.

- [x] **Step 3: Add effective automatic tuning**

```cpp
const bool auto_head64_prefill =
    head_tile == 1 && selected_partitions == 1 &&
    is_automatic_v32_prefill_head64(
        query, shared_kv, indices, value_head_dim, phase, family);
const int64_t effective_head_tile = auto_head64_prefill ? 64 : head_tile;
const int64_t effective_selected_partitions =
    auto_head64_prefill ? 1 : selected_partitions;
```

Use effective values for Head64 eligibility and plan creation. Leave non-target defaults on the generic path.

- [x] **Step 4: Add the direct Host branch**

At the beginning of `sparse_attention_forward_family_hd576_head64(...)`, allocate BF16 `[B,H,Q,Dv]` output, FP32 `[B,H,Q]` LSE, current workspace, and plan tensor when `output_mode` is direct. Call the fused launcher once with null partial pointers and valid final pointers, then return. Do not call staged QK/PV or Combine.

- [x] **Step 5: Skip the redundant cast**

```cpp
auto raw_output = std::get<0>(result);
auto output = raw_output.scalar_type() == query.scalar_type()
    ? raw_output
    : raw_output.to(query.scalar_type());
```

Decode P4 still converts FP32 combined output; direct BF16 prefill does not.

- [x] **Step 6: Verify GREEN and amend**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
pytest -q
git diff --check
rg -n "sparse_attention" src/cannbench/cli.py src/cannbench/core src/cannbench/backends || true
git add src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc
git commit --amend --no-edit
```

Expected: all tests pass, no new public-layer branch exists, and the implementation commit is amended.

---

### Task 4: Add Accuracy And Performance Runners

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/test/v32_prefill_benchmark.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_v32_full_accuracy_runner.py`

**Interfaces:**
- Consumes: explicit P1 and exact automatic routes; deterministic helpers from `v32_full_accuracy.py`.
- Produces: `head64_reduced_accuracy.py --phase prefill` and `v32_prefill_benchmark.py --warmups N --iters N --seed N` JSON evidence.

- [x] **Step 1: Write failing reduced-runner contracts**

```python
def test_sparse_attention_head64_reduced_accuracy_covers_prefill_p1():
    source = Path(__file__).with_name("head64_reduced_accuracy.py").read_text()
    assert 'choices=("decode", "prefill")' in source
    assert 'phase == "prefill"' in source
    assert "partitions = (1,)" in source
    assert "ops._prefill_reference" in source
    assert "torch.npu.Stream()" in source
```

- [x] **Step 2: Write failing benchmark-runner tests**

```python
def test_prefill_benchmark_summary_uses_median_and_preserves_samples():
    result = summarize_samples([4.0, 2.0, 3.0])
    assert result == {
        "samples_ms": [4.0, 2.0, 3.0],
        "median_ms": 3.0,
        "min_ms": 2.0,
        "max_ms": 4.0,
    }
```

Also assert the runner source contains the exact realistic prefill case ID, does not set either tuning environment variable, materializes inputs before warmup, synchronizes every timed call, and validates output/LSE shape and dtype.

- [x] **Step 3: Verify RED**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_v32_full_accuracy_runner.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  -k 'prefill_benchmark or reduced_accuracy_covers_prefill'
```

Expected: import and source-contract failures because the benchmark does not exist and the reduced runner is decode-only.

- [x] **Step 4: Parameterize the reduced runner**

Add `--phase {decode,prefill}`. Decode keeps P1/P2/P4. Prefill runs only P1, calls `_prefill_reference`, and passes `phase="prefill"`. Preserve S17/S64/S70/S128/S2048, invalid, causal, int64 overflow, empty, width-rejection, and `B=2,Q=9` reuse cases. Add eight alternating valid-S70 launches on two NPU streams and compare every result after one final synchronize.

- [x] **Step 5: Add the automatic benchmark**

Create `v32_prefill_benchmark.py` with:

```python
def summarize_samples(samples_ms: list[float]) -> dict[str, object]:
    return {
        "samples_ms": samples_ms,
        "median_ms": statistics.median(samples_ms),
        "min_ms": min(samples_ms),
        "max_ms": max(samples_ms),
    }
```

Use the exact prefill case plus `_fill_deterministic` and `_build_indices` from `v32_full_accuracy.py`. Allocate once, leave tuning environment variables unset, warm up, time only `ops.sparse_attention_forward(...)` plus NPU synchronization, and emit JSON with shape/dtype checks and samples.

- [x] **Step 6: Verify GREEN and amend**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_v32_full_accuracy_runner.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git diff --check
git add src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/v32_prefill_benchmark.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_v32_full_accuracy_runner.py
git commit --amend --no-edit
```

Expected: runner tests pass and the implementation commit is amended.

---

### Task 5: Build, Validate, Profile, Gate, And Document

**Files:**
- Validate: `src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py`
- Validate: `src/cannbench/operators/builtin/sparse_attention/simt/test/v32_full_accuracy.py`
- Validate: `src/cannbench/operators/builtin/sparse_attention/simt/test/v32_prefill_benchmark.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/README.md`
- Modify: `src/cannbench/operators/builtin/sparse_attention/PARALLEL_SPLITTING_RESEARCH.zh-CN.md`
- Modify: `docs/superpowers/plans/2026-07-29-sparse-attention-v32-prefill-head64.md`

**Interfaces:**
- Consumes: complete candidate and `1297d3e` generic baseline.
- Produces: same-device build, reduced/full accuracy, stability, wall-time, profile, dispatch decision, and final evidence.

- [x] **Step 1: Create isolated remote trees**

On `root@121.41.199.170:20002`, create one baseline and one candidate directory with:

```bash
sa_baseline_dir=$(mktemp -d /tmp/cannbench-sa-v32-prefill-baseline-XXXXXX)
sa_candidate_dir=$(mktemp -d /tmp/cannbench-sa-v32-prefill-candidate-XXXXXX)
printf '%s\n%s\n' "$sa_baseline_dir" "$sa_candidate_dir" \
  > /tmp/cannbench-sa-v32-prefill-paths-20260729
```

Copy an `origin/main` archive into the baseline and this worktree into the candidate with `rsync -a --delete -e 'ssh -p 20002'`, excluding `.git`, `.worktrees`, caches, build output, and generated shared libraries. Record both SHAs, remote paths, `bisheng --version`, and `/usr/local/Ascend/cann-9.2.0`.

From the local machine, resolve the two paths and copy the trees:

```bash
mapfile -t sa_remote_dirs < <(
  ssh -p 20002 root@121.41.199.170 \
    'cat /tmp/cannbench-sa-v32-prefill-paths-20260729'
)
sa_baseline_remote=${sa_remote_dirs[0]}
sa_candidate_remote=${sa_remote_dirs[1]}
rsync -a --delete -e 'ssh -p 20002' \
  --exclude=.git --exclude=.worktrees --exclude=.pytest_cache \
  --exclude=build --exclude='*.so' \
  /root/aiagent/cannbench/ \
  "root@121.41.199.170:$sa_baseline_remote/"
rsync -a --delete -e 'ssh -p 20002' \
  --exclude=.git --exclude=.worktrees --exclude=.pytest_cache \
  --exclude=build --exclude='*.so' \
  /root/aiagent/cannbench/.worktrees/sparse-attention-v32-prefill-32core/ \
  "root@121.41.199.170:$sa_candidate_remote/"
```

The first source is the clean local `main` checkout at `1297d3e`; verify that SHA before copying.

- [x] **Step 2: Build both dav-3510 extensions**

For each concrete directory returned in Step 1:

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
while IFS= read -r sa_tree; do
  cd "$sa_tree/src/cannbench/operators/builtin/sparse_attention/simt/v1"
  NPU_ARCH=dav-3510 \
    /root/miniconda3/envs/cannbench-vllm-ascend-py311/bin/python \
    setup.py build_ext --inplace
done < /tmp/cannbench-sa-v32-prefill-paths-20260729
```

Expected: exit 0; the fused Head64 ELF and `_C*.so` are present in both trees.

- [x] **Step 3: Run candidate reduced prefill and decode regression**

With `PYTHONPATH` containing the candidate `src` and built `simt/v1` directory, run:

```bash
/root/miniconda3/envs/cannbench-vllm-ascend-py311/bin/python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py \
  --phase prefill
/root/miniconda3/envs/cannbench-vllm-ascend-py311/bin/python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py \
  --phase decode
```

Expected: all prefill P1 and decode P1/P2/P4 output/LSE results pass `atol=rtol=0.05`; reuse, repeated launches, concurrent streams, and boundaries do not hang.

Final causal-boundary acceptance rerun: reduced prefill passed `14 / 14`.
The `B=1,H=128,Q=4,C=256,S=64` case contained future indices on rows 0/1/2
(`2/2/1` respectively), the end boundary on row 3, and valid-past, negative,
and out-of-range categories on all four rows. Maximum output/LSE errors were
`0.017578125 / 0.020476341247558594`.

- [x] **Step 4: Run full automatic prefill accuracy**

```bash
env -u CANNBENCH_SPARSE_ATTENTION_HEAD_TILE \
    -u CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS \
  /root/miniconda3/envs/cannbench-vllm-ascend-py311/bin/python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/v32_full_accuracy.py \
  --phase prefill --seed 7 --atol 0.05 --rtol 0.05 \
  --output /tmp/sparse-attention-v32-prefill-head64-accuracy.json
```

Expected: rows `0/1365/2730/4095`, all 128 heads per sampled row, zero output/LSE mismatches, and `passed=true`.

Final causal-boundary acceptance rerun: rows `0/1365/2730` each contained one
in-range future index; row `4095` contained the causal end boundary; all four
contained one negative and one out-of-range index. All `262144` output and
`512` LSE samples had zero mismatches. Maximum errors were
`0.0078125 / 0.008250236511230469`.

- [x] **Step 5: Measure baseline and candidate wall time**

Run `v32_prefill_benchmark.py --warmups 1 --iters 3 --seed 7` in each build with tuning variables unset. Synchronize every call. Because the generic baseline can exceed 100 seconds, retain all three samples instead of increasing the count.

Expected gate: candidate median is strictly lower than baseline median and all shape/dtype checks pass.

- [x] **Step 6: Profile the candidate and preserve the unavailable baseline evidence**

Use the modified 100-second msopprof binary under `/home/y00621698/msopprof`, `--aic-metrics=Default`, five profiler warmups, and `--launch-count=1` against `v32_full_accuracy.py --phase prefill --runtime-only`. Store results in distinct `/tmp/msopprof-sa-v32-prefill-head64-*` directories.

Record BasicInfo, arithmetic/pipe/resource metrics, GM traffic, and L2 statistics. Candidate gates:

```text
Block Dim = 32
Mix Block Dim = 64
Cube work rows = 32
Vector work rows = 64
Combine kernels = 0
output cast kernels = 0
```

The candidate selected-kernel duration must be lower than the baseline generic kernel duration.

Outcome: the final candidate produced a valid Default profile at
`334959.437500 us`, launch `32 / 64`, with actual work on all 32 Cube and 64
Vector rows, no Combine, and no output Cast. The original baseline application
exceeded the 100-second AI Core timeout: the analyzer printed
`Get op basic info [Task Duration] failed` and `0.000000 us`, while its CSV
stored `NA` and only truncated post-timeout rows. A 600-second retry was stopped
during warmup by explicit instruction and was not restarted; its preserved
output contains only `pc_start_addr.txt` and `aicore_binary.o` setup dumps,
with no BasicInfo, metric CSV, analyzed duration, or exit marker. The strict
candidate-versus-baseline selected-kernel comparison is therefore unavailable
and explicitly waived, not passed; all failed and aborted artifacts are
preserved.

- [x] **Step 7: Apply the automatic-dispatch decision**

Keep automatic dispatch only when accuracy, stability, wall median, selected-kernel duration, and effective-work gates pass. If any gate fails, remove automatic selection while retaining explicit P1 prefill for diagnosis. Do not add P4 or ping-pong in this plan.

Outcome: retain automatic dispatch under the approved baseline-duration
waiver. Reduced/full accuracy, decode regression, stability, same-input wall
median, valid candidate duration, `32 / 32` Cube work, `64 / 64` Vector work,
zero Combine, and zero output Cast all passed. No invalid baseline profiler
counter supports this decision, and the duration-comparison gap remains
explicit in the final evidence.

- [x] **Step 8: Backfill measured evidence**

Update README and research with source SHAs, remote paths, device/CANN/compiler, input seed, reduced/full accuracy errors, sample distributions, medians, kernel durations, launch dimensions, actual work rows, utilization/wait/GM/L2 metrics, and the retained/rejected dispatch decision. Mark completed plan checkboxes only after evidence exists.

- [x] **Step 9: Run final verification**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
pytest -q
git diff --check
rg -n "sparse_attention" src/cannbench/cli.py src/cannbench/core src/cannbench/backends || true
git status --short
```

Expected: all tests pass, diff check is empty, and no public-layer branch was added.

Outcome after rebasing onto `origin/main`: operator-local tests reported
`178 passed, 13 skipped`; the full repository reported
`457 passed, 2 skipped`; the diff check and public-layer hardcoding search were
empty. The pytest results were collected against the final implementation
blobs after rebase. Later evidence-only documentation autosquashes did not
change the profiled device or Host source hashes, so those results remain
applicable; pytest was not rerun after the last docs-only autosquash. Diff,
history, status, and public-layer checks were rerun afterward.

- [x] **Step 10: Produce the final two-commit history**

Amend the completed plan into the docs commit and all operator/test/runner/result changes into the implementation commit. The final non-merge history is exactly:

```text
docs(sparse-attention): design v32 prefill head64 path
perf(sparse-attention): add v32 prefill head64 path
```

Verify:

```bash
git log --oneline origin/main..HEAD
git rev-list --merges origin/main..HEAD
git diff --check origin/main..HEAD
git status --short
```

Expected: two commits, no merge commit, clean diff, and clean worktree.

Outcome: the branch was rebased non-interactively and without conflict onto
`3d45bf9`; its only upstream sparse-attention edit is a test-mock signature
update, not runtime operator code. The final history contains the two approved
subjects, no merge commit, and no extra tracked worktree change.

Fresh final-review acceptance used the preserved runner-only copy
`/tmp/cannbench-sa-v32-prefill-final-review-ACYDoO`. Reduced prefill again
passed `14 / 14`: the Q4 causal rows had future counts `2/2/1/0`, with
valid-past, negative, and out-of-range values on every row. Full seed-7
prefill again had zero output/LSE mismatches over `262144 / 512` samples; rows
`0/1365/2730` each had one future index and row `4095` used the end boundary.
Device/Host/`_C` hashes remained unchanged. The benchmark runner hash and
default index tensor matched the historical candidate exactly, so wall-time
and profiler were not rerun and their existing evidence remains applicable.

## Plan Self-Review

- Spec coverage: Tasks 1-3 cover exact auto routing, dynamic P1 tasks, direct BF16 output, FP32 LSE, one fused launch, cast removal, fallback, and decode preservation.
- Validation: Task 4 covers reduced/reuse/concurrency and reproducible timing; Task 5 covers remote build, bounded-memory full accuracy, decode regression, msopprof, and dispatch gating.
- API boundary: no shared framework file changes; tests freeze current Basic API include and CrossCore call counts.
- Types: `output_mode` is int32, direct output BF16, partial output and all LSE storage FP32, and Host/device signatures match.
- Scope: no prefill P2/P4, ping-pong, shared gather, other family, or workflow change.
- Commit policy: one amended docs commit plus one amended implementation commit.
