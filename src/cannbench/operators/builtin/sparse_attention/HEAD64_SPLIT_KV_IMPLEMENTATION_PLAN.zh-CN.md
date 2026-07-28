# Head64 Split-KV Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 Head64 staged sparse attention 上实现 `selected_partitions=2/4`，通过 partition-local QK/PV 与独立 SIMT combine 将 realistic decode 的主 kernel task 数扩展到 16/32，并用精度和性能实测决定是否保留优化。

**Architecture:** Host 继续由算子本地 tuning 构造 `SparseAttentionHead64Plan`，device task 按 `[B,Q,head_group,partition]` 映射。QK 和 PV 复用现有 M64 主体，但只遍历当前 partition；PV 输出局部归一化 FP32 output 和局部 LSE，独立 1024-thread SIMT combine 按 log-sum-exp 权重生成最终 output/LSE。`P=1` 不启动 combine。

**Tech Stack:** Python、PyTorch custom op、C++ Host bridge、Ascend C API、Tensor API、SIMT API、pytest、CANN profiler。

## Global Constraints

- 所有业务逻辑和测试必须保留在 `src/cannbench/operators/builtin/sparse_attention/`。
- 不修改 `src/cannbench/cli.py`、`src/cannbench/core/` 或公共 backend。
- 默认 tuning 保持 `(head_tile=1, selected_partitions=1)`。
- 只新增 `(64,2)` 和 `(64,4)`；其他组合报错，不静默回退。
- SIMT launch 使用 1024 threads；不能套用 CUDA 256-thread 配置。
- 不新增 Basic API include、Basic API 同步或 AIC 间 CrossCore flag；现有 MIX task 内遗留协作只做复用。
- 保留现有首个完整 PV tile 重算 workaround 及其问题记录，本计划不同时定位该 CANN 问题。
- 设备精度阈值保持 `atol=0.05, rtol=0.05`。
- `P=4` 只有在 realistic decode 端到端中位延迟低于方案 A 当前约 `1.33 ms` 时才通过主性能门槛。

---

## File Structure

- `simt/v1/aten_dsa_sparse_attention/ops.py`：接受算子本地 Split-KV tuning。
- `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h`：共享 partition 容量和 task 数字段。
- `simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`：Host 校验、workspace、launch 顺序与 `P=1/P>1` 输出路径。
- `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`：partition-aware QK/PV 和 SIMT combine。
- `simt/test/test_sparse_attention_decode_reference.py`：Python tuning 行为测试。
- `simt/test/test_sparse_attention_v1_build_shell.py`：Host/device 源码契约测试。
- `simt/test/head64_reduced_accuracy.py`：远程 `P=2/4` 边界精度入口。
- `simt/README.md`：只记录真实设备精度、延迟和 profiler 结论。

---

### Task 1: Split-KV Tuning 与 Host Plan ABI

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/ops.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: `_resolve_tuning() -> tuple[int, int]`；现有 `SparseAttentionHead64Plan`。
- Produces: `(64,2)/(64,4)` tuning；`selected_partition_tile_capacity` 和 partition-inner `task_count`。

- [ ] **Step 1: Write failing wrapper tuning tests**

将单一 Head64 测试参数化，并把非法 partition 从 `2` 改为 `3`：

```python
@pytest.mark.parametrize("selected_partitions", [1, 2, 4])
def test_sparse_attention_forward_passes_head64_tuning(
    monkeypatch, selected_partitions
):
    captured = {}

    def fake_custom_op(*args):
        captured["tuning"] = args[-2:]
        return "custom"

    monkeypatch.setenv("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", "64")
    monkeypatch.setenv(
        "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS",
        str(selected_partitions),
    )
    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op)
    result = ops.sparse_attention_forward(
        object(), object(), object(),
        value_head_dim=512,
        phase="decode",
        family="family_hd576",
        causal=True,
    )
    assert result == "custom"
    assert captured["tuning"] == (64, selected_partitions)
```

- [ ] **Step 2: Write failing Host plan source tests**

更新源码契约，要求 Host 支持 `P=1/2/4`，按 selected tile 对齐 partition capacity：

```python
def test_sparse_attention_head64_host_supports_split_kv_route():
    bridge = _bridge_source()
    plan = _head64_plan_source()
    assert "is_supported_head64_partitions" in bridge
    assert "selected_partitions == 2" in bridge
    assert "selected_partitions == 4" in bridge
    assert "selected_partition_tile_capacity" in plan
    assert "selected_tile_count" in bridge
    assert "partition_tile_capacity" in bridge
```

- [ ] **Step 3: Run focused tests and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
```

Expected: FAIL because `(64,2)/(64,4)` and the aligned partition fields do not exist.

- [ ] **Step 4: Implement minimal tuning and plan changes**

Python 支持集合改为：

```python
_SUPPORTED_TUNING = {(1, 1), (64, 1), (64, 2), (64, 4)}
```

Host 增加显式 helper：

```cpp
bool is_supported_head64_partitions(int64_t partitions) {
  return partitions == 1 || partitions == 2 || partitions == 4;
}

const int64_t selected_tile_count =
    (indices.size(2) + kHead64SelectedTile - 1) / kHead64SelectedTile;
const int64_t partition_tile_capacity =
    (selected_tile_count + selected_partitions - 1) / selected_partitions;
```

删除 `selected_partitions == 1` 的 phase-one 限制，改成 helper 校验。`use_head64`
改为 `head_tile == 64 && is_supported_head64_partitions(selected_partitions)`。
plan 末字段改为：

```cpp
int32_t selected_partition_tile_capacity;
```

其值为 `partition_tile_capacity`；token capacity 在 Host/device 通过
`selected_partition_tile_capacity * selected_tile` 推导。

- [ ] **Step 5: Run focused tests and verify GREEN**

执行 Step 3 命令。Expected: PASS。

- [ ] **Step 6: Commit Task 1**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/ops.py \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): add split-kv launch plan"
```

---

### Task 2: Partition-Aware QK

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 1 的 `selected_partition_tile_capacity` 和 partition-inner `task_count`。
- Produces: `Head64Task.partition`、`head64_partition_begin/end()`、partition-local scores。

- [ ] **Step 1: Write failing partition mapping and QK layout tests**

```python
def test_sparse_attention_head64_task_mapping_keeps_partition_innermost():
    source = _head64_source()
    assert "int32_t partition;" in source
    assert "task_id % plan.selected_partitions" in source
    assert "task_id /= plan.selected_partitions" in source
    assert "head64_partition_begin" in source
    assert "head64_partition_end" in source


def test_sparse_attention_head64_qk_uses_partition_local_scores():
    source = _head64_source()
    bridge = _bridge_source()
    assert "plan.selected_partition_tile_capacity * plan.selected_tile" in bridge
    assert "partition_token_capacity" in source
    assert "partition_begin + local_selected_start" in source
    assert "logical_task) * kHead64Tile * partition_token_capacity" in " ".join(
        source.split()
    )
```

- [ ] **Step 2: Run tests and verify RED**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
```

Expected: FAIL because device task 没有 partition，scores 仍按完整 `S` stride。

- [ ] **Step 3: Implement task decode and partition helpers**

```cpp
struct Head64Task {
  int32_t batch_index;
  int32_t query_token;
  int32_t head_group;
  int32_t partition;
};

__aicore__ inline Head64Task decode_head64_task(
    int32_t task_id,
    const SparseAttentionHead64Plan& plan) {
  const int32_t partition = task_id % plan.selected_partitions;
  task_id /= plan.selected_partitions;
  const int32_t head_group = task_id % plan.head_group_count;
  task_id /= plan.head_group_count;
  return {
      task_id / plan.query_tokens,
      task_id % plan.query_tokens,
      head_group,
      partition,
  };
}

__aicore__ inline int32_t head64_partition_begin(
    const Head64Task& task,
    const SparseAttentionHead64Plan& plan) {
  return task.partition * plan.selected_partition_tile_capacity *
      plan.selected_tile;
}

__aicore__ inline int32_t head64_partition_end(
    const Head64Task& task,
    const SparseAttentionHead64Plan& plan) {
  const int32_t begin = head64_partition_begin(task, plan);
  const int32_t capacity =
      plan.selected_partition_tile_capacity * plan.selected_tile;
  const int32_t end = begin + capacity;
  return end < plan.selected_tokens ? end : plan.selected_tokens;
}
```

- [ ] **Step 4: Restrict QK AIV/AIC loops to the partition**

为每个 task 推导：

```cpp
const int32_t partition_begin = head64_partition_begin(task, plan);
const int32_t partition_end = head64_partition_end(task, plan);
const int32_t partition_length =
    partition_end > partition_begin ? partition_end - partition_begin : 0;
const int32_t partition_token_capacity =
    plan.selected_partition_tile_capacity * plan.selected_tile;
```

QK loop 使用 `local_selected_start` 遍历 `[0, partition_length)`；indices gather 的
绝对位置使用 `partition_begin + local_selected_start`；scores GM tile 使用
`local_selected_start`，row stride 使用 `partition_token_capacity`。Host scores shape 改为：

```cpp
{plan.task_count,
 kHead64Tile,
 plan.selected_partition_tile_capacity * plan.selected_tile}
```

空 partition 直接跳过循环，不读 query 之外的 indices，也不写 scores。
QK 的 AIC 和两个 AIV 必须对空 partition 采取同一控制流：要么都跳过 task 内握手，
要么都完成现有 query-ready 握手；不能只让一侧提前返回。

- [ ] **Step 5: Verify GREEN and preserve P=1 source contracts**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git diff --check
```

Expected: PASS；M64、1024 threads 和双 AIV测试继续通过。

- [ ] **Step 6: Commit Task 2**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): shard head64 qk by selected tokens"
```

---

### Task 3: Partition-Local Softmax 与 PV

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 2 的 local score stride 和绝对 partition bounds。
- Produces: task-major `partial_output` 与 `partial_lse`；`P=1` 保持最终 lse layout。

- [ ] **Step 1: Write failing PV partition tests**

```python
def test_sparse_attention_head64_pv_uses_partition_bounds_and_local_stride():
    source = _head64_source()
    assert "partition_begin" in _function_body(
        source, "sparse_attention_head64_pv_aiv(",
        "__aicore__ inline void sparse_attention_head64_pv_aic("
    )
    assert "partition_length" in source
    assert "partition_token_capacity" in source
    assert "partial_lse" in source
    assert "task.partition" in source
```

- [ ] **Step 2: Run the PV test and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py::test_sparse_attention_head64_pv_uses_partition_bounds_and_local_stride
```

Expected: FAIL because softmax/PV 仍遍历完整 selected tokens。

- [ ] **Step 3: Make softmax and probability packing partition-local**

`head64_online_softmax_vf` 和 `head64_probability_pack_vf` 接收：

```cpp
int32_t partition_begin,
int32_t partition_length,
int32_t partition_token_capacity
```

score row 使用 local offset，indices 使用绝对 offset：

```cpp
const int64_t score_row =
    (static_cast<int64_t>(logical_task) * kHead64Tile + task_head) *
    partition_token_capacity;
const int64_t indices_row =
    (static_cast<int64_t>(batch_index) * query_tokens + query_token) *
    selected_tokens + partition_begin;
```

所有 selected loop 上界改为 `partition_length`。空 partition 写 `partial_lse=-inf`，
probabilities 不读 scores 且保持为零。

- [ ] **Step 4: Make value gather and Cube PV partition-local**

`head64_value_pack_vf` 的 selected index 使用：

```cpp
const int32_t absolute_selected =
    partition_begin + local_selected_start + selected_offset;
```

AIC/AIV 的 PV loop、probability tile count、`cmatrixInitVal` 和末 tile notify 都改成
基于 `partition_length`。空 partition 必须将整个 task output 写零；非空 partition
继续保留 `head64_pv_value_tile_index()` 的首 tile 重算。

partial LSE 对 `P>1` 写：

```cpp
partial_lse[static_cast<int64_t>(logical_task) * kHead64Tile + task_head]
```

`P=1` 继续写原有 `[B,H,Q]` offset，避免启动 combine 或额外 reorder。

空 partition 必须在 PV 每个逻辑 task 的第一次 CrossCore 操作之前由 AIC 和两个 AIV
同时识别：AIV 写 `partial_lse=-inf`，AIC 将对应 task output 全部写零，随后三侧都
跳到下一个逻辑 task。不能进入半套 probability/value ready 握手。

- [ ] **Step 5: Allocate partition-local probability and partial tensors**

Host shape：

```cpp
auto task_probabilities = at::empty(
    {plan.task_count,
     plan.selected_partition_tile_capacity,
     kHead64Tile,
     kHead64SelectedTile},
    query.options().dtype(c10::kBFloat16));
auto task_output = at::empty(
    {plan.task_count, kHead64Tile, plan.value_head_dim},
    query.options().dtype(c10::kFloat));
auto task_lse = plan.selected_partitions == 1
    ? at::empty({plan.batch_size, plan.query_heads, plan.query_tokens},
                query.options().dtype(c10::kFloat))
    : at::empty({plan.task_count, kHead64Tile},
                query.options().dtype(c10::kFloat));
```

- [ ] **Step 6: Verify focused and full local tests**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py::test_sparse_attention_head64_pv_uses_partition_bounds_and_local_stride
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
git diff --check
```

Expected: focused PASS；全算子测试 PASS。

- [ ] **Step 7: Commit Task 3**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): add partition-local head64 pv"
```

---

### Task 4: 1024-Thread SIMT Combine 与 Host Route

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 3 的 `[task_count,64,Dv]` partial output 和 `[task_count,64]` partial LSE。
- Produces: `launch_sparse_attention_head64_combine_hd576_bf16(const float*, const float*, float*, float*, uint8_t*, const SparseAttentionHead64Plan*, aclrtStream)`；最终 `[B,H,Q,Dv]` output/LSE。

- [ ] **Step 1: Write failing combine source tests**

```python
def test_sparse_attention_head64_combine_uses_1024_thread_dual_aiv():
    source = _head64_source()
    body = _function_body(
        source,
        "head64_combine_vf(",
        "__global__ __aicore__ void sparse_attention_head64_combine_kernel("
    )
    assert "__launch_bounds__(1024)" in source
    assert "dim3(1024, 1, 1)" in source
    assert "GetSubBlockIdx()" in source
    assert "partial_lse" in body
    assert "partial_output" in body
    assert "__expf(partial_lse_value - global_lse)" in body
    assert "isfinite" in body
    assert "output[output_row + dim] = 0.0F" in body


def test_sparse_attention_head64_host_skips_combine_for_p1():
    body = _function_body(
        _bridge_source(),
        "sparse_attention_forward_family_hd576_head64(",
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_privateuse1("
    )
    p1 = body.index("if (plan.selected_partitions == 1)")
    combine = body.index("run_sparse_attention_head64_combine_hd576_bf16(")
    assert p1 < combine
```

- [ ] **Step 2: Run combine tests and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py::test_sparse_attention_head64_combine_uses_1024_thread_dual_aiv \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py::test_sparse_attention_head64_host_skips_combine_for_p1
```

Expected: FAIL because combine kernel/launcher 不存在。

- [ ] **Step 3: Implement the combine VF**

每个 MIX block 对应一个 base task；AIC 分支直接返回，两个 AIV 各负责 32 heads。每个
SIMT lane 独立计算 2/4 个 LSE 的稳定 log-sum-exp，然后按 `dim += 32` 合并输出：

```cpp
const uint32_t local_head = threadIdx.x / 32;
const uint32_t lane = threadIdx.x % 32;
int32_t remainder = base_task;
const int32_t head_group = remainder % head_group_count;
remainder /= head_group_count;
const int32_t query_token = remainder % query_tokens;
const int32_t batch_index = remainder / query_tokens;
const int32_t task_head =
    sub_block_index * 32 + static_cast<int32_t>(local_head);
const int32_t global_head = head_group * kHead64Tile + task_head;
const int64_t lse_offset =
    (static_cast<int64_t>(batch_index) * query_heads + global_head) *
        query_tokens +
    query_token;
const int64_t output_row = lse_offset * value_head_dim;

float global_max = -std::numeric_limits<float>::infinity();
for (int32_t p = 0; p < selected_partitions; ++p) {
  const float value = partial_lse[(base_task * selected_partitions + p) *
                                  kHead64Tile + task_head];
  global_max = global_max < value ? value : global_max;
}

if (!isfinite(global_max)) {
  if (lane == 0) {
    lse[lse_offset] = -std::numeric_limits<float>::infinity();
  }
  for (int32_t dim = lane; dim < value_head_dim; dim += 32) {
    output[output_row + dim] = 0.0F;
  }
  return;
}

float global_sum = 0.0F;
for (int32_t p = 0; p < selected_partitions; ++p) {
  const float value = partial_lse[(base_task * selected_partitions + p) *
                                  kHead64Tile + task_head];
  if (isfinite(value)) {
    global_sum += __expf(value - global_max);
  }
}
const float global_lse = global_max + logf(global_sum);
if (lane == 0) {
  lse[lse_offset] = global_lse;
}
```

每个输出元素：

```cpp
for (int32_t dim = static_cast<int32_t>(lane);
     dim < value_head_dim;
     dim += 32) {
  float combined = 0.0F;
  for (int32_t p = 0; p < selected_partitions; ++p) {
    const int64_t partial_row =
        (static_cast<int64_t>(base_task) * selected_partitions + p) *
            kHead64Tile +
        task_head;
    const float partial_lse_value = partial_lse[partial_row];
    if (isfinite(partial_lse_value)) {
      combined += __expf(partial_lse_value - global_lse) *
          partial_output[partial_row * value_head_dim + dim];
    }
  }
  output[output_row + dim] = combined;
}
```

- [ ] **Step 4: Add kernel launcher and Host wrapper**

launcher 签名：

```cpp
extern "C" void launch_sparse_attention_head64_combine_hd576_bf16(
    const float* partial_output,
    const float* partial_lse,
    float* output,
    float* lse,
    uint8_t* plan_gm,
    const SparseAttentionHead64Plan* plan,
    aclrtStream stream);
```

launch block 数为：

```cpp
plan->task_count / plan->selected_partitions
```

kernel 保持 `KERNEL_TYPE_MIX_AIC_1_2`。AIV 侧通过
`GetBlockIdx() / GetTaskRatio()` 得到 `base_task`，读取 `GetSubBlockIdx()` 后调用：

```cpp
asc_vf_call<head64_combine_vf>(
    dim3(1024, 1, 1),
    partial_output,
    partial_lse,
    output,
    lse,
    plan.query_heads,
    plan.query_tokens,
    plan.head_group_count,
    plan.value_head_dim,
    plan.selected_partitions,
    base_task,
    sub_block_index);
```

不增加 CrossCore flag。

- [ ] **Step 5: Route P=1 and P>1 outputs**

Host 在 PV 后：

```cpp
if (plan.selected_partitions == 1) {
  auto output = task_output
      .view({plan.batch_size, plan.query_tokens, plan.query_heads,
             plan.value_head_dim})
      .permute({0, 2, 1, 3})
      .contiguous();
  return {output, task_lse};
}

auto output = at::empty(
    {plan.batch_size, plan.query_heads, plan.query_tokens, plan.value_head_dim},
    query.options().dtype(c10::kFloat));
auto lse = at::empty(
    {plan.batch_size, plan.query_heads, plan.query_tokens},
    query.options().dtype(c10::kFloat));
run_sparse_attention_head64_combine_hd576_bf16(
    task_output, task_lse, output, lse, plan_tensor, plan, acl_stream);
return {output, lse};
```

- [ ] **Step 6: Verify GREEN and API boundary**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
rg -n "CrossCore(Set|Wait)Flag" \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc
git diff --check
```

Expected: tests PASS；CrossCore 匹配只来自 Task 4 前已存在的 QK/PV MIX 协作，combine
函数体内无 CrossCore；没有新增 Basic API include。

- [ ] **Step 7: Commit Task 4**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): combine head64 split-kv outputs"
```

---

### Task 5: Remote Build 与 Split-KV 精度

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Tasks 1-4 的完整 `(64,2)/(64,4)` 路径。
- Produces: 可重复的边界精度结果和 realistic 全 shape 精度记录。

- [ ] **Step 1: Write a failing Split-KV accuracy-runner contract test**

```python
def test_sparse_attention_head64_reduced_accuracy_covers_split_kv():
    source = Path(__file__).with_name("head64_reduced_accuracy.py").read_text(
        encoding="utf-8"
    )
    assert "PARTITIONS = (2, 4)" in source
    assert '"valid_s128"' in source
    assert '"valid_s2048"' in source
    assert 'result["selected_partitions"]' in source
    assert 'str(selected_partitions)' in source
```

- [ ] **Step 2: Run the contract test and verify RED**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py::test_sparse_attention_head64_reduced_accuracy_covers_split_kv
```

Expected: FAIL because runner 仍固定为 `selected_partitions=1`。

- [ ] **Step 3: Extend the reduced accuracy runner before remote build**

增加 `S=128` 和 `S=2048`，并让主函数遍历 partitions：

```python
PARTITIONS = (2, 4)

CASES = (
    {"name": "all_invalid_s17", "context": 256, "selected": 17, "mode": "invalid"},
    {"name": "valid_s64", "context": 256, "selected": 64, "mode": "valid"},
    {"name": "tail_s70", "context": 256, "selected": 70, "mode": "valid"},
    {"name": "invalid_causal_s70", "context": 256, "selected": 70, "mode": "mixed"},
    {"name": "valid_s128", "context": 256, "selected": 128, "mode": "valid"},
    {"name": "valid_s2048", "context": 32768, "selected": 2048, "mode": "valid"},
)
```

主循环改为：

```python
results = []
os.environ["CANNBENCH_SPARSE_ATTENTION_HEAD_TILE"] = "64"
for selected_partitions in PARTITIONS:
    os.environ["CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS"] = str(
        selected_partitions
    )
    for case in CASES:
        result = _run_case(torch, ops, case)
        result["selected_partitions"] = selected_partitions
        results.append(result)
passed = all(result["passed"] for result in results)
```

- [ ] **Step 4: Run local tests and verify GREEN**

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
pytest -q
git diff --check
```

Expected: complete local suite PASS。

- [ ] **Step 5: Sync exact commit to the remote validation directory**

使用此前已验证的远程环境：

```text
host: root@121.41.199.170:20002
CANN: /usr/local/Ascend/cann-9.2.0
NPU_ARCH: dav-3510
```

先比较本地和远程目标 source hash，再同步 operator-local changed files。不能覆盖远程
目录中来源不明的改动；若目标不干净，创建新的 `/root/cannbench-head64-splitkv-<commit>`。

- [ ] **Step 6: Build on remote and capture the first compiler error completely**

```bash
source /usr/local/Ascend/cann-9.2.0/set_env.sh
export NPU_ARCH=dav-3510
python setup.py build_ext --inplace
```

Expected: build exit 0。若失败，按 systematic-debugging 从第一条设备编译错误定位，
不得跳过失败继续跑精度。

- [ ] **Step 7: Run reduced and realistic accuracy**

```bash
python src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py
```

再对 realistic `B=2,Q=2,H=128,C=32768,S=2048` 验证全部 output/LSE 元素。
Expected: `P=2/4` 的 output/LSE 都通过 `atol=0.05,rtol=0.05`；all-invalid output 全零、
LSE 全负无穷且无 NaN。

- [ ] **Step 8: Commit reproducible accuracy coverage**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "test(sparse-attention): cover head64 split-kv accuracy"
```

---

### Task 6: P=1/2/4 性能、Profiler 与结果记录

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/README.md`

**Interfaces:**
- Consumes: Task 5 的 exact remote build 和正确性通过记录。
- Produces: 同输入的 `P=1/2/4` wall-time、kernel profiler 和是否通过门槛的结论。

- [ ] **Step 1: Run alternating wall-time benchmark**

固定 realistic shape：

```text
B=2 Q=2 H=128 KV_H=1 context=32768 selected=2048 Dqk=576 Dv=512 causal=true
```

每个 partition 配置使用 3 warmups、7 rounds、5 calls/round，并按 `P=1,2,4,4,2,1`
交替次序减少频率漂移。每次调用后使用同一 `torch.npu.synchronize()` 边界。

- [ ] **Step 2: Evaluate the primary performance gate**

记录三个中位数：

```text
P=1 median
P=2 median and speedup vs P=1
P=4 median and speedup vs P=1
```

判定：

```text
P=4 < P=1 and P=4 < 1.33 ms  -> 32-task 主配置通过
otherwise                    -> P=4 未通过
```

若 P=2 单独最快，明确记录 P=2 是实测最优配置，同时保留 P=4 未达标结论。

- [ ] **Step 3: Profile P=1/2/4**

每个配置至少提取：

```text
QK duration
PV duration
combine duration
Block Dim / Mix Block Dim
effective AIC/AIV count
Cube / Vector / Scalar / MTE utilization
AIC/AIV wait ratio
GM/L2 traffic and L2 hit rate
```

必须验证 `P=4` 的 QK/PV profiler 显示 `32 AIC / 64 AIV` 对应的 MIX launch；combine
单独报告自己的 8 AIC/16 AIV MIX launch，不能把 Host blockDim 当成有效利用率结论。

- [ ] **Step 4: Measure full workflow**

用相同 dataset/case 测完整 `dsa_decode` workflow，分别设置 P=1/2/4。记录 sparse
attention 单算子收益是否转化为 workflow 总延迟收益。

- [ ] **Step 5: Write only measured facts to the operator README**

记录：commit、remote device/CANN、精度最大误差、P=1/2/4 中位延迟、combine 开销、
实际 AIC/AIV、workflow 延迟和性能门槛结论。未测字段不能填写估计值。

- [ ] **Step 6: Final verification and commit**

```bash
pytest -q
git diff --check
git status --short --branch
git add src/cannbench/operators/builtin/sparse_attention/simt/README.md
git commit -m "docs(sparse-attention): record split-kv validation"
```

Expected: full suite PASS；只提交算子本地真实测量记录；worktree clean。

## Plan Self-Review

- Task 1 覆盖 tuning、Host 校验、动态 16/32 task 和 tile-aligned partition capacity。
- Task 2 只改变 QK task/bounds/stride，保留 M64 Cube 和双 AIV pack。
- Task 3 只改变 softmax/PV 的局部范围和 partial 输出，保留首 PV tile workaround。
- Task 4 独立增加数值稳定 combine，明确 all-invalid 防 NaN 和 `P=1` fast path。
- Task 5 覆盖 `S=17/64/70/128/2048`、valid/causal/invalid/all-invalid 和 P=2/4。
- Task 6 对 P=1/2/4 做同条件端到端与 profiler 对照，并分别报告主门槛和实测最优值。
- 所有实现与测试文件都位于 sparse-attention operator package，没有公共层硬编码。
