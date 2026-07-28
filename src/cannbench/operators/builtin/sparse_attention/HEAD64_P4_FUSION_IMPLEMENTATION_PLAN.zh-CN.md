# Sparse Attention Head64 P=4 Fusion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to
> implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将 V3.2 decode 的 Head64 P=4 staged `QK -> PV -> Combine` 攨为
`fused QK/online-softmax/PV -> Combine`，保持 32 AIC + 64 AIV 实际工作并删除
P=4 完整 scores/probabilities workspace。

**Architecture:** P=1/P=2 继续走现有 staged device ELF，仅作为历史对照。P=4
使用独立 MIX device ELF；每个
`(batch, query_token, head_group, partition)` task 在片内按 selected tile 完成 QK、
在线 softmax 和 PV，写 partition-local output/LSE 后复用现有 Combine。

**Tech Stack:** C++17 Host bridge、CANN 9.2 `dav-3510`、C API、Tensor API、
SIMT API、PyTorch custom op、pytest、msopprof。

## Global Constraints

- 所有业务逻辑、测试和文档改动必须位于
  `src/cannbench/operators/builtin/sparse_attention/`。
- 不修改 `src/cannbench/cli.py`、`src/cannbench/backends/`、
  `src/cannbench/core/` 或 workflow 包。
- 只优化 `(head_tile=64, selected_partitions=4)`；P=1/P=2 保留 staged 对照。
- fused 主 kernel launch 必须是 32 AIC / 64 AIV；每个 AIV 使用 1024 threads。
- 两个 AIV 都必须执行 gather、softmax、probability pack、output update 和 writeback。
- QK 和 PV 必须继续使用 Tensor API MMAD。
- fused source 必须使用 `CrossCoreSetFlag/CrossCoreWaitFlag` mode 2 完成 AIC/AIV
  握手；这是用户明确要求的任务级 API 例外。
- fused source 只允许现有 CrossCoreFlag 所需的 `basic_api/kernel_common.h` 和
  `basic_api/kernel_operator_block_sync_intf.h`，不得使用其他 Basic API。
- 首版使用单 buffer；本计划不预先加入 ping-pong。
- output/LSE 精度阈值固定为 `atol=0.05, rtol=0.05`。
- 端到端性能门槛为 fused P=4 + Combine 中位延迟不高于 `0.574588 ms`。

---

## File Structure

- `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_common.h`：
  无 Basic API 的 task decode、partition 边界、plan copy 和 warp reduction helper。
- `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc`：
  P=4 fused MIX kernel 与 launcher。
- `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`：
  staged P=1/P=2 和现有 Combine；只抽取无状态 common helper，不改变执行语义。
- `simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`：P=4 Host 路由、
  partial workspace 和 fused launcher 包装。
- `simt/v1/setup.py`：注册独立 fused device ELF。
- `simt/test/test_sparse_attention_v1_build_shell.py`：源码、API、launch 与 workspace 契约。
- `simt/test/head64_reduced_accuracy.py`：P=4 reduced/boundary/物理核复用精度。
- `simt/test/v32_full_accuracy.py`：full realistic decode 精度入口。
- `simt/README.md`：最终设备、精度、延迟和 profiler 记录。

---

### Task 1: Add the Isolated Fused Device Boundary

**Files:**
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_common.h`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/setup.py`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: `SparseAttentionHead64Plan` and `kHead64*` constants.
- Produces: `Head64Task`、`decode_head64_task()`、`head64_partition_begin/end()`、
  `launch_sparse_attention_head64_fused_hd576_bf16(...)` 和
  `libsparse_attention_head64_fused_hd576_kernel.so`。

- [ ] **Step 1: Write failing source-boundary tests**

Add `_head64_fused_source()` and `_head64_common_source()`, then add:

```python
def test_sparse_attention_head64_fused_source_is_registered_separately():
    setup = (Path(__file__).parents[1] / "v1/setup.py").read_text()
    assert "libsparse_attention_head64_fused_hd576_kernel.so" in setup
    assert "sparse_attention_head64_fused_hd576.asc" in setup


def test_sparse_attention_head64_fused_source_uses_required_cross_core_flags():
    source = _head64_fused_source()
    assert '#include "c_api/asc_simd.h"' in source
    basic_headers = {
        line.strip() for line in source.splitlines()
        if line.strip().startswith('#include "basic_api/')
    }
    assert basic_headers == {
        '#include "basic_api/kernel_common.h"',
        '#include "basic_api/kernel_operator_block_sync_intf.h"',
    }
    assert 'kernel_operator.h' not in source
    assert 'CrossCoreSetFlag<2' in source
    assert 'CrossCoreWaitFlag<2' in source
    assert 'AscendC::SetFlag' not in source
    assert 'AscendC::WaitFlag' not in source


def test_sparse_attention_head64_fused_keeps_dual_aiv_1024_contract():
    source = _head64_fused_source()
    assert "KERNEL_TYPE_MIX_AIC_1_2" in source
    assert "__launch_bounds__(1024)" in source
    assert "dim3(1024, 1, 1)" in source
    assert "GetSubBlockIdx()" in source
    assert "GetSubBlockIdx() != 0" not in source
    assert "threadIdx.x / 32" in source
    assert "threadIdx.x % 32" in source
```

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
```

Expected: FAIL because the common header, fused source and ELF registration do not exist.

- [ ] **Step 3: Extract dependency-clean common helpers**

Move the plan copy, task decode, partition bounds and warp reductions to
`sparse_attention_head64_common.h`. The header must expose:

```cpp
struct Head64Task {
  int32_t batch_index;
  int32_t query_token;
  int32_t head_group;
  int32_t partition;
};

__aicore__ inline Head64Task decode_head64_task(
    int32_t task_id, const SparseAttentionHead64Plan& plan);
__aicore__ inline int32_t head64_partition_begin(
    const Head64Task& task, const SparseAttentionHead64Plan& plan);
__aicore__ inline int32_t head64_partition_end(
    const Head64Task& task, const SparseAttentionHead64Plan& plan);
__aicore__ inline void copy_head64_plan(
    SparseAttentionHead64Plan* plan, __gm__ const uint8_t* plan_gm);
```

Include the header from staged and fused sources. Do not move CrossCore calls or staged
kernel state into the common header.

- [ ] **Step 4: Add a compilable fused MIX skeleton**

The skeleton must use CrossCoreFlag mode 2 with flag IDs 8 and 9 and give both AIVs real
SIMT work through a temporary per-task output initialization:

```cpp
constexpr uint8_t kAivToAicReady = 8;
constexpr uint8_t kAicToAivReady = 9;

__global__ __aicore__ void sparse_attention_head64_fused_kernel(...) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if ASCEND_IS_AIC {
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE1>(kAivToAicReady);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE1>(kAicToAivReady);
  } else if ASCEND_IS_AIV {
    const uint32_t subblock_index = AscendC::GetSubBlockIdx();
    asc_vf_call<head64_fused_init_vf>(
        dim3(1024, 1, 1), task_output, partial_lse, plan_gm,
        subblock_index);
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(kAivToAicReady);
    AscendC::CrossCoreWaitFlag<2, PIPE_V>(kAicToAivReady);
  }
}
```

The temporary initialization is replaced in Tasks 2-3; it exists only to compile and
validate the independent API boundary.

- [ ] **Step 5: Run local tests and remote compile smoke**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
git diff --check
```

Sync the worktree to a fresh remote `/tmp/cannbench-head64-p4-fused-*` directory and run:

```bash
NPU_ARCH=dav-3510 pip install -v --no-build-isolation --force-reinstall \
  src/cannbench/operators/builtin/sparse_attention/simt/v1
```

Expected: local tests pass and bisheng links the new device ELF.

- [ ] **Step 6: Commit Task 1**

```bash
git add src/cannbench/operators/builtin/sparse_attention/simt
git commit -m "feat(sparse-attention): add isolated head64 fused kernel"
```

---

### Task 2: Fuse QK and Online Softmax Per Selected Tile

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 1 common helpers and CrossCoreFlag mode-2 synchronization.
- Produces: one tile-local FP32 score buffer, BF16 probability buffer, running max/sum,
  and no GM score/probability arguments.

- [ ] **Step 1: Write failing fused-QK contracts**

```python
def test_sparse_attention_head64_fused_qk_is_m64_tensor_api():
    source = _head64_fused_source()
    assert "Head64QkMmadTrait" in source
    assert "params.m = 64" in source
    assert "params.n = current_selected" in source
    assert "params.k = current_k" in source
    assert "Mmad(" in source


def test_sparse_attention_head64_fused_scores_stay_tile_local():
    source = _head64_fused_source()
    launcher = _function_definition(
        source, "launch_sparse_attention_head64_fused_hd576_bf16(")
    assert "float* scores" not in launcher
    assert "bfloat16_t* probabilities" not in launcher
    assert "l1_scores_buf" in source
    assert "ub_scores_buf" in source
    assert "ub_probabilities_buf" in source
    assert "running_max" in source
    assert "running_sum" in source
```

- [ ] **Step 2: Run focused tests and verify RED**

Run the build-shell test. Expected: FAIL because the skeleton has no QK or softmax.

- [ ] **Step 3: Implement Q pack and M64 QK in the fused task loop**

For every logical task, decode P=4 partition bounds, pack Q once, and iterate its selected
tiles. For each tile, both AIVs pack non-overlapping 32-column K halves into shared L1;
AIC copies Q/K into L0 and executes nine K=64 MMAD segments with:

```cpp
MmadParams params;
params.m = 64;
params.n = current_selected;
params.k = current_k;
params.cmatrixInitVal = k_start == 0;
Mmad(qk_mm.with(params), l0_scores, l0_query, l0_keys);
```

AIC Fixpipe writes the 64x64 FP32 score tile to the two AIV-owned L1 halves rather than GM.
The AIC must copy the current Q segment into L0A before releasing L1 ownership to AIV.

- [ ] **Step 4: Implement dual-AIV online softmax and probability pack**

Map 1024 threads as `local_head=threadIdx.x/32`, `lane=threadIdx.x%32`. Each AIV owns
32 rows. For every tile apply causal/invalid masks and update:

```cpp
new_max = max(running_max, tile_max);
old_scale = exp(running_max - new_max);
probability = exp(score - new_max);
running_sum = old_scale * running_sum + warp_sum(probability);
running_max = new_max;
```

Write BF16 probabilities to the AIV's non-overlapping L1 half for immediate PV. Preserve
`old_scale` per row for Task 3 output rescaling. Empty partitions produce max `-inf`, sum
zero and no out-of-bounds reads.

- [ ] **Step 5: Run tests and remote compile smoke**

Run the Task 1 local suite and remote build command. Expected: PASS and no compiler UB/L1
allocation error.

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): fuse head64 qk and softmax"
```

---

### Task 3: Add Cube PV and Partition-Local Output

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 2 tile-local probabilities, `old_scale`, running max/sum.
- Produces: FP32 `task_output[task_count,64,512]` and
  `partial_lse[task_count,64]` compatible with the existing Combine.

- [ ] **Step 1: Write failing fused-PV contracts**

```python
def test_sparse_attention_head64_fused_pv_is_m64_tensor_api():
    source = _head64_fused_source()
    assert "Head64PvMmadTrait" in source
    assert "pv_params.m = 64" in source
    assert "pv_params.n = current_value" in source
    assert "pv_params.k = current_selected" in source
    assert "running_output" in source
    assert "old_scale" in source


def test_sparse_attention_head64_fused_writes_combine_compatible_partials():
    source = _head64_fused_source()
    assert "task_output" in source
    assert "partial_lse" in source
    assert "running_max + logf(running_sum)" in source
    assert "running_output / running_sum" in source
    assert "-std::numeric_limits<float>::infinity()" in source
```

- [ ] **Step 2: Run focused tests and verify RED**

Run the build-shell test. Expected: FAIL because the fused source does not perform PV.

- [ ] **Step 3: Gather V and execute M64N128 Cube PV**

For each score tile and each 128-dimension value tile, the two AIVs gather 32 selected
rows into non-overlapping L1 halves. AIC consumes tile-local BF16 probabilities and V:

```cpp
MmadParams pv_params;
pv_params.m = 64;
pv_params.n = current_value;
pv_params.k = current_selected;
pv_params.cmatrixInitVal = true;
Mmad(pv_mm.with(pv_params), l0_pv, l0_probability, l0_values);
```

Fixpipe returns each 32x128 FP32 half to its owning AIV. No GM probability tensor is used.

- [ ] **Step 4: Update and normalize running output**

Each AIV maintains its 32x512 FP32 running output in UB. Before adding the current PV tile:

```cpp
running_output[row][dim] =
    old_scale[row] * running_output[row][dim] + pv_tile[row][dim];
```

After the partition loop, write `running_output/running_sum` and
`running_max + logf(running_sum)`. For an all-invalid row, write zero output and `-inf` LSE.

- [ ] **Step 5: Run tests and remote compile smoke**

Run the local sparse-attention tests and remote build. Expected: PASS, and compiler resource
reports fit the single-buffer design. If 128-wide PV staging exceeds UB, reduce only
`kHead64ValueTile` to 64 and repeat; do not change Head64/P=4 semantics.

- [ ] **Step 6: Commit Task 3**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_fused_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): fuse head64 cube pv"
```

---

### Task 4: Route P=4 Through Fused Kernel and Remove Full Workspaces

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 3 fused launcher and existing Combine launcher.
- Produces: P=4 `fused -> Combine`; P=1/P=2 unchanged staged route.

- [ ] **Step 1: Write failing Host route/workspace tests**

```python
def test_sparse_attention_head64_p4_routes_fused_then_combine():
    bridge = _bridge_source()
    body = _function_definition(
        bridge, "sparse_attention_forward_family_hd576_head64(")
    assert "if (plan.selected_partitions == 4)" in body
    fused = body.index("run_sparse_attention_head64_fused_hd576_bf16(")
    combine = body.index("run_sparse_attention_head64_combine_hd576_bf16(")
    assert fused < combine


def test_sparse_attention_head64_p4_has_no_full_score_probability_workspace():
    bridge = _bridge_source()
    body = _function_definition(
        bridge, "sparse_attention_forward_family_hd576_head64(")
    p4 = body.split("if (plan.selected_partitions == 4)", 1)[1].split(
        "auto task_scores", 1
    )[0]
    assert "task_scores" not in p4
    assert "task_probabilities" not in p4
    assert "run_sparse_attention_head64_qk_hd576_bf16" not in p4
    assert "run_sparse_attention_head64_pv_hd576_bf16" not in p4
```

- [ ] **Step 2: Run focused tests and verify RED**

Run the build-shell test. Expected: FAIL because all P values use staged QK/PV.

- [ ] **Step 3: Add the Host fused launcher wrapper**

Declare and wrap:

```cpp
extern "C" void launch_sparse_attention_head64_fused_hd576_bf16(
    const at::BFloat16* query,
    const at::BFloat16* shared_kv,
    const int32_t* indices,
    float* task_output,
    float* partial_lse,
    uint8_t* workspace,
    uint8_t* plan_gm,
    bool causal,
    const SparseAttentionHead64Plan* plan,
    aclrtStream stream);
```

Use `OpCommand::RunOpApiV2` with profile name
`aten_dsa_sparse_attention::sparse_attention_head64_fused_hd576_bf16`.

- [ ] **Step 4: Add the P=4 fast branch before staged allocations**

Allocate only the KFC workspace, plan tensor, partition-local `task_output/task_lse`, final
output/LSE; launch fused then Combine and return. Keep the existing staged body below for
P=1/P=2:

```cpp
if (plan.selected_partitions == 4) {
  auto workspace = at::empty({16 * 1024 * 1024}, byte_options);
  auto plan_tensor = at::empty({4096}, byte_options);
  auto task_output = at::empty(
      {plan.task_count, kHead64Tile, plan.value_head_dim}, float_options);
  auto task_lse = at::empty(
      {plan.task_count, kHead64Tile}, float_options);
  auto output = at::empty(
      {plan.batch_size, plan.query_heads, plan.query_tokens,
       plan.value_head_dim}, float_options);
  auto lse = at::empty(
      {plan.batch_size, plan.query_heads, plan.query_tokens}, float_options);
  run_sparse_attention_head64_fused_hd576_bf16(...);
  run_sparse_attention_head64_combine_hd576_bf16(...);
  return {output, lse};
}
```

- [ ] **Step 5: Run focused and full local tests**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
pytest -q
git diff --check
```

Expected: no new failures and no public-layer sparse-attention branch.

- [ ] **Step 6: Commit Task 4**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "perf(sparse-attention): route p4 through fused head64"
```

---

### Task 5: Remote Accuracy, Profile, and Documentation

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/README.md`
- Modify only if required by a failing contract:
  `src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py`
  and `src/cannbench/operators/builtin/sparse_attention/simt/test/v32_full_accuracy.py`

**Interfaces:**
- Consumes: committed P=4 fused route.
- Produces: reproducible `dav-3510` accuracy, latency and msopprof evidence.

- [ ] **Step 1: Build in a fresh remote `/tmp` directory**

Use port 20002 and a new directory; record local commit, remote directory, CANN version and
source SHA-256. Install with `NPU_ARCH=dav-3510` and confirm the fused ELF is present.

- [ ] **Step 2: Run reduced and boundary accuracy**

Run P=4 for `S=17/64/70/128/2048`, valid/causal/invalid/all-invalid, int64 overflow,
`S=0`, `C=0`, invalid shared-KV widths, and `B=2,Q=9,S=70` physical-core reuse.

Expected: every output/LSE element passes `atol=rtol=0.05`; repeated runs do not hang.

- [ ] **Step 3: Run full realistic decode accuracy**

Run `deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048` with P=4.

Expected: all `262144` output and `512` LSE elements pass, with max errors recorded.

- [ ] **Step 4: Measure wall time against staged P=4**

Use three warmups, seven measured rounds, five calls per round, and synchronize after each
call. Retain the existing `0.574588 ms` staged median as baseline.

Expected gate: fused P=4 + Combine median `<= 0.574588 ms`.

- [ ] **Step 5: Profile exact fused and Combine kernels**

Run msopprof exact-kernel replay with five profiler warmups and launch count one. Record:

- fused and Combine kernel-side duration;
- launch dimensions;
- finite AIC/AIV rows and rows with Cube/Vector work;
- Cube/Vector/Scalar/MTE utilization, wait ratios, GM traffic and L2 hit rate.

Expected: fused launch `32 / 64`, Cube work on all 32 AIC rows, Vector work on all 64 AIV
rows. Compare fused + Combine duration against staged QK + PV + Combine.

- [ ] **Step 6: Decide single buffer versus ping-pong**

If the performance gate passes, keep single buffer. If it fails and profiler attributes the
gap to serialized gather/compute waits, add a separate follow-up design for ping-pong; do not
silently expand this implementation plan.

- [ ] **Step 7: Update README with measured facts and run final verification**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
pytest -q
git diff --check
git status --short
```

Document exact commands, device/CANN, commit, accuracy errors, wall-time distribution,
kernel durations and utilization. Do not claim performance success unless the gate passes.

- [ ] **Step 8: Commit Task 5**

```bash
git add src/cannbench/operators/builtin/sparse_attention/simt/README.md
git commit -m "docs(sparse-attention): record fused p4 validation"
```

---

## Plan Self-Review

- Spec coverage: isolated API boundary, P=4-only fusion, 32/64 work, 1024 threads, workspace
  removal, reduced/full accuracy, physical reuse, wall time and profiler are each covered.
- Scope: no P=1/P=2 fusion, default change, public-layer change, prefill or ping-pong work.
- Rollback: Tasks 1-3 do not affect Host routing; until Task 4, all runtime calls remain on
  the validated staged implementation.
- Type consistency: fused launcher writes the same FP32 partial output/LSE layout consumed
  by the existing Combine launcher.
