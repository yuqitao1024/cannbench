# Sparse Attention Head64 Implementation Plan

> 状态说明（2026-07-28）：本计划的 staged QK 与 staged softmax/PV 检查点已交付；
> Task 5 的单 persistent MIX kernel 融合及“移除完整 scores workspace”尚未实现，已
> 延期。当前实现与后续优化以 `HEAD64_SPLIT_KV_DESIGN.zh-CN.md` 和
> `HEAD64_SPLIT_KV_IMPLEMENTATION_PLAN.zh-CN.md` 为准，不能把 Task 5 描述当作
> 当前交付状态。

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 为 `family_hd576` BF16 decode 实现实验性的 Head64 fused Sparse Attention，使 64 个 Query heads 共享 K/V gather，并使用 1024-thread 双 AIV、Cube QK、online softmax 和 Cube PV。

**Architecture:** Host 根据 `head_tile` 与 `selected_partitions` 构造算子本地 launch plan；方案 A 将每个 `(batch, query_token, head_group_64)` 映射为一个独立 MIX task。实现按 QK workspace、softmax+PV workspace、单 kernel 融合三个设备检查点推进，最后只保留 legacy 和 fused Head64 路径。

**Tech Stack:** Python 3.11+、PyTorch custom op、C++17 Host bridge、CANN `dav-3510`、C API、Tensor API、SIMT API、`CrossCoreSetFlag/CrossCoreWaitFlag` mode 2/mode 4、pytest。

## Global Constraints

- 所有实现与测试改动必须位于 `src/cannbench/operators/builtin/sparse_attention/`。
- 不修改 `src/cannbench/cli.py`、`src/cannbench/backends/`、`src/cannbench/core/` 或 DSA workflow 包。
- 默认配置必须保持 `head_tile=1, selected_partitions=1`，继续执行 legacy kernel。
- 第一阶段唯一新增配置为 `head_tile=64, selected_partitions=1`。
- Head64 只支持 `phase=decode`、`family=family_hd576`、BF16、`H=128`、`KV_H=1`、`Dqk=576`、`Dv=512`、`S<=2048`。
- 每个 AIV 必须使用 `__launch_bounds__(1024)`，Host 必须使用 `dim3(1024, 1, 1)` 发起 SIMT VF。
- 每个 AIV 处理 32 个 Head，每个 Head 固定由 32 个 SIMT threads 协作。
- QK 的 AIC/AIV 阶段握手使用 `CrossCoreSetFlag/CrossCoreWaitFlag` mode 2；
  staged PV 的双 AIV ready 使用 mode 4，反方向继续使用 mode 2。所有 set/wait
  必须成对。
- 当前交付未调用 Mutex；单个 AIC/AIV 内部异步流水通过 pipe ordering 管理 buffer
  复用顺序。
- 新 kernel 只允许额外包含 `basic_api/kernel_common.h` 和 `basic_api/kernel_operator_block_sync_intf.h`；不得使用其他 Basic API header。
- 新 kernel 不得使用 `SetFlag`、`WaitFlag`、`PipeBarrier`、`SyncAll`、GM flag 自旋或跨逻辑 task 数据共享。
- 目标编译架构固定为 `NPU_ARCH=dav-3510`。
- output/LSE 精度标准固定为 `atol=0.05, rtol=0.05`。
- 单 buffer fused Head64 必须先完成精度与 profiler 验证；ping-pong 优化不属于本计划。

---

## File Map

| 文件 | 职责 |
| --- | --- |
| `simt/v1/aten_dsa_sparse_attention/ops.py` | 解析算子本地实验参数，并把整数参数传给 custom op |
| `simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc` | 校验 Head64 shape、构造 Host plan、路由 legacy/Head64 |
| `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h` | Host/device 共享的 POD launch plan 和固定 tile 常量 |
| `simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc` | Head64 QK、softmax、PV 和最终 fused MIX kernel |
| `simt/v1/setup.py` | 构建并链接独立 Head64 device ELF |
| `simt/test/test_sparse_attention_decode_reference.py` | Python wrapper 参数解析与默认行为 |
| `simt/test/test_sparse_attention_v1_build_shell.py` | Host/device 源码边界、任务映射、线程结构和同步契约 |
| `simt/test/head64_reduced_accuracy.py` | dav-3510 reduced shape 精度与边界场景脚本 |
| `simt/test/v32_full_accuracy.py` | full realistic decode 精度脚本及 tuning 元数据 |
| `simt/README.md` | 最终精度、延迟和 profiler 结论 |

## Task 1: Add Operator-Local Tuning Controls

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/ops.py:1-61`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc:1130-1286`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py:1-65`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: 现有 `sparse_attention_forward(query, shared_kv, indices, *, value_head_dim, phase, family, causal)` Python API。
- Produces: `_resolve_tuning() -> tuple[int, int]`；custom op 新增带默认值的 `head_tile: int` 和 `selected_partitions: int`。

- [ ] **Step 1: Write failing Python wrapper tests**

在 `test_sparse_attention_decode_reference.py` 增加：

```python
def test_sparse_attention_forward_passes_default_legacy_tuning(monkeypatch):
    captured = {}

    def fake_custom_op(
        query,
        shared_kv,
        indices,
        value_head_dim,
        phase,
        family,
        causal,
        head_tile,
        selected_partitions,
    ):
        del query, shared_kv, indices, value_head_dim, phase, family, causal
        captured["tuning"] = (head_tile, selected_partitions)
        return "custom"

    monkeypatch.delenv("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", raising=False)
    monkeypatch.delenv(
        "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS", raising=False
    )
    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op)

    result = ops.sparse_attention_forward(
        object(),
        object(),
        object(),
        value_head_dim=512,
        phase="decode",
        family="family_hd576",
        causal=True,
    )

    assert result == "custom"
    assert captured["tuning"] == (1, 1)


def test_sparse_attention_forward_passes_head64_tuning(monkeypatch):
    captured = {}

    def fake_custom_op(*args):
        captured["tuning"] = args[-2:]
        return "custom"

    monkeypatch.setenv("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", "64")
    monkeypatch.setenv("CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS", "1")
    monkeypatch.setattr(ops, "_load_registered_op", lambda: fake_custom_op)

    result = ops.sparse_attention_forward(
        object(),
        object(),
        object(),
        value_head_dim=512,
        phase="decode",
        family="family_hd576",
        causal=True,
    )

    assert result == "custom"
    assert captured["tuning"] == (64, 1)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [
        ("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", "abc", "must be an integer"),
        ("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", "0", "must be positive"),
        (
            "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS",
            "2",
            "unsupported sparse_attention tuning",
        ),
    ],
)
def test_sparse_attention_forward_rejects_invalid_tuning(
    monkeypatch, name, value, message
):
    monkeypatch.delenv("CANNBENCH_SPARSE_ATTENTION_HEAD_TILE", raising=False)
    monkeypatch.delenv(
        "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS", raising=False
    )
    monkeypatch.setenv(name, value)

    with pytest.raises(RuntimeError, match=message):
        ops.sparse_attention_forward(
            object(),
            object(),
            object(),
            value_head_dim=512,
            phase="decode",
            family="family_hd576",
            causal=True,
        )
```

- [ ] **Step 2: Run the wrapper tests and confirm red state**

Run:

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py
```

Expected: the new tests fail because the wrapper still passes seven positional custom-op arguments and does not parse the environment variables.

- [ ] **Step 3: Implement the Python tuning resolver**

在 `ops.py` 中增加：

```python
import os

_HEAD_TILE_ENV = "CANNBENCH_SPARSE_ATTENTION_HEAD_TILE"
_SELECTED_PARTITIONS_ENV = "CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS"
_SUPPORTED_TUNING = {(1, 1), (64, 1)}


def _read_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise RuntimeError(f"{name} must be an integer, got {raw!r}") from exc
    if value <= 0:
        raise RuntimeError(f"{name} must be positive, got {value}")
    return value


def _resolve_tuning() -> tuple[int, int]:
    tuning = (
        _read_positive_int(_HEAD_TILE_ENV, 1),
        _read_positive_int(_SELECTED_PARTITIONS_ENV, 1),
    )
    if tuning not in _SUPPORTED_TUNING:
        raise RuntimeError(
            "unsupported sparse_attention tuning: "
            f"head_tile={tuning[0]}, selected_partitions={tuning[1]}"
        )
    return tuning
```

在调用 custom op 前解析一次，并追加两个 positional arguments：

```python
    head_tile, selected_partitions = _resolve_tuning()
    result = custom_op(
        query,
        shared_kv,
        indices,
        value_head_dim,
        phase,
        family,
        causal,
        head_tile,
        selected_partitions,
    )
```

- [ ] **Step 4: Extend the internal Torch schema without changing old direct callers**

将 C++ 函数签名扩展为：

```cpp
std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_privateuse1(
    const at::Tensor& query,
    const at::Tensor& shared_kv,
    const at::Tensor& indices,
    int64_t value_head_dim,
    std::string_view phase,
    std::string_view family,
    bool causal,
    int64_t head_tile,
    int64_t selected_partitions) {
```

在 Head64 route 尚未接入前，只允许 legacy：

```cpp
  TORCH_CHECK(
      head_tile == 1 && selected_partitions == 1,
      "head64 sparse_attention kernel is not installed yet");
```

schema 使用默认值保持现有七参数 direct calls 可用：

```cpp
m.def(
    "sparse_attention_forward(Tensor query, Tensor shared_kv, Tensor indices, "
    "int value_head_dim, str phase, str family, bool causal, "
    "int head_tile=1, int selected_partitions=1) -> (Tensor, Tensor)");
```

- [ ] **Step 5: Add a source-contract test for schema defaults**

在 `test_sparse_attention_v1_build_shell.py` 增加：

```python
def test_sparse_attention_custom_op_schema_keeps_legacy_tuning_defaults():
    source = _bridge_source()

    assert "int64_t head_tile" in source
    assert "int64_t selected_partitions" in source
    assert "int head_tile=1, int selected_partitions=1" in source
```

如果文件没有 `_bridge_source()`，在顶部增加：

```python
def _bridge_source() -> str:
    return (
        Path(__file__).parents[1]
        / "v1"
        / "aten_dsa_sparse_attention"
        / "csrc"
        / "sparse_attention.asc"
    ).read_text(encoding="utf-8")
```

- [ ] **Step 6: Run focused and full tests**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
pytest -q
```

Expected: focused tests pass; full suite reports no new failure.

- [ ] **Step 7: Commit the tuning control plane**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/ops.py \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): add head tile tuning controls"
```

## Task 2: Add the Shared Launch Plan and Compile Skeleton

**Files:**
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/setup.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 1 的 `head_tile` 与 `selected_partitions`。
- Produces: `SparseAttentionHead64Plan`、`make_sparse_attention_head64_plan(...)`、独立 device ELF `libsparse_attention_head64_hd576_kernel.so`。

- [ ] **Step 1: Write failing file-layout and API-boundary tests**

```python
def test_sparse_attention_head64_sources_are_registered():
    project = Path(__file__).parents[1] / "v1" / "aten_dsa_sparse_attention"
    setup_source = (Path(__file__).parents[1] / "v1" / "setup.py").read_text()

    assert (project / "csrc/simt/sparse_attention_head64_plan.h").is_file()
    assert (project / "csrc/simt/sparse_attention_head64_hd576.asc").is_file()
    assert "libsparse_attention_head64_hd576_kernel.so" in setup_source
    assert "sparse_attention_head64_hd576.asc" in setup_source


def test_sparse_attention_head64_source_uses_only_allowed_basic_api():
    source = _head64_source()
    basic_headers = {
        line.strip()
        for line in source.splitlines()
        if line.strip().startswith('#include "basic_api/')
    }

    assert basic_headers == {
        '#include "basic_api/kernel_common.h"',
        '#include "basic_api/kernel_operator_block_sync_intf.h"',
    }
    assert '#include "kernel_operator.h"' not in source
    assert "AscendC::SetFlag" not in source
    assert "AscendC::WaitFlag" not in source
    assert "AscendC::PipeBarrier" not in source
    assert "AscendC::SyncAll" not in source


def test_sparse_attention_head64_uses_1024_thread_dual_aiv_contract():
    source = _head64_source()

    assert "__launch_bounds__(1024)" in source
    assert "dim3(1024, 1, 1)" in source
    assert "GetSubBlockIdx()" in source
    assert "GetSubBlockIdx() != 0" not in source
    assert "threadIdx.x / 32" in source
    assert "threadIdx.x % 32" in source
```

同时增加：

```python
def _head64_source() -> str:
    return (
        Path(__file__).parents[1]
        / "v1/aten_dsa_sparse_attention/csrc/simt"
        / "sparse_attention_head64_hd576.asc"
    ).read_text(encoding="utf-8")
```

- [ ] **Step 2: Run the source tests and confirm red state**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
```

Expected: tests fail because the plan header, Head64 source and setup entry do not exist.

- [ ] **Step 3: Create the shared POD plan**

`sparse_attention_head64_plan.h` 必须定义：

```cpp
#pragma once

#include <cstdint>

namespace aten_dsa_sparse_attention {

constexpr int32_t kHead64Tile = 64;
constexpr int32_t kHead64SelectedTile = 64;
constexpr int32_t kHead64QkTile = 64;
constexpr int32_t kHead64ValueTile = 128;
constexpr int32_t kHead64Threads = 1024;
constexpr int32_t kHead64ThreadsPerHead = 32;
constexpr int32_t kHead64PhysicalAicLimit = 32;

struct SparseAttentionHead64Plan {
  int32_t used_core_num;
  int32_t task_count;
  int32_t batch_size;
  int32_t query_heads;
  int32_t query_tokens;
  int32_t context_tokens;
  int32_t selected_tokens;
  int32_t qk_head_dim;
  int32_t value_head_dim;
  int32_t head_tile;
  int32_t head_group_count;
  int32_t selected_tile;
  int32_t selected_partitions;
  int32_t selected_partition_size;
};

}  // namespace aten_dsa_sparse_attention
```

- [ ] **Step 4: Add the Host plan builder and strict Head64 validation**

在 `sparse_attention.asc` 包含 plan header，并实现：

```cpp
SparseAttentionHead64Plan make_sparse_attention_head64_plan(
    const at::Tensor& query,
    const at::Tensor& shared_kv,
    const at::Tensor& indices,
    int64_t value_head_dim,
    std::string_view phase,
    std::string_view family,
    int64_t head_tile,
    int64_t selected_partitions) {
  TORCH_CHECK(phase == "decode", "head64 requires phase=decode");
  TORCH_CHECK(family == "family_hd576", "head64 requires family_hd576");
  TORCH_CHECK(
      query.scalar_type() == at::ScalarType::BFloat16,
      "head64 requires bfloat16 query");
  TORCH_CHECK(
      shared_kv.scalar_type() == at::ScalarType::BFloat16,
      "head64 requires bfloat16 shared_kv");
  TORCH_CHECK(query.size(1) == 128, "head64 requires query_heads=128");
  TORCH_CHECK(shared_kv.size(1) == 1, "head64 requires kv_heads=1");
  TORCH_CHECK(query.size(3) == 576, "head64 requires qk_head_dim=576");
  TORCH_CHECK(value_head_dim == 512, "head64 requires value_head_dim=512");
  TORCH_CHECK(head_tile == 64, "head64 requires head_tile=64");
  TORCH_CHECK(
      selected_partitions == 1,
      "head64 phase one requires selected_partitions=1");
  TORCH_CHECK(indices.size(2) <= 2048, "head64 requires selected_tokens<=2048");

  const int64_t head_group_count = query.size(1) / head_tile;
  const int64_t task_count =
      query.size(0) * query.size(2) * head_group_count * selected_partitions;
  return {
      static_cast<int32_t>(std::min<int64_t>(task_count, 32)),
      static_cast<int32_t>(task_count),
      static_cast<int32_t>(query.size(0)),
      static_cast<int32_t>(query.size(1)),
      static_cast<int32_t>(query.size(2)),
      static_cast<int32_t>(shared_kv.size(2)),
      static_cast<int32_t>(indices.size(2)),
      static_cast<int32_t>(query.size(3)),
      static_cast<int32_t>(value_head_dim),
      static_cast<int32_t>(head_tile),
      static_cast<int32_t>(head_group_count),
      kHead64SelectedTile,
      static_cast<int32_t>(selected_partitions),
      static_cast<int32_t>(
          (indices.size(2) + selected_partitions - 1) / selected_partitions),
  };
}
```

legacy 判断必须位于 Head64 validation 之前：

```cpp
const bool use_head64 = head_tile == 64 && selected_partitions == 1;
```

不允许对显式 Head64 请求静默回退。

- [ ] **Step 5: Create the minimal compilable mixed-kernel source**

新 device source 先只提供 ABI、plan copy、task decode 和同步编译探针。核心结构为：

```cpp
#include <cstdint>

#include "acl/acl.h"
#include "basic_api/kernel_common.h"
#include "basic_api/kernel_operator_block_sync_intf.h"
#include "c_api/asc_simd.h"
#include "simt_api/asc_bf16.h"
#include "simt_api/asc_simt.h"
#include "tensor_api/tensor.h"

#include "sparse_attention_head64_plan.h"

namespace {
using namespace AscendC::Te;
using aten_dsa_sparse_attention::SparseAttentionHead64Plan;

constexpr uint8_t kAivToAicReady = 8;
constexpr uint8_t kAicToAivReady = 9;

__simt_vf__ __aicore__ __launch_bounds__(1024) inline void
head64_probe_vf(__gm__ int32_t* probe) {
  const uint32_t local_head = threadIdx.x / 32;
  const uint32_t lane = threadIdx.x % 32;
  if (lane == 0) {
    probe[AscendC::GetSubBlockIdx() * 32 + local_head] = 1;
  }
}

__global__ __aicore__ void sparse_attention_head64_probe_kernel(
    __gm__ int32_t* probe,
    __gm__ uint8_t* plan_gm) {
  KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
  if ASCEND_IS_AIC {
    AscendC::CrossCoreWaitFlag<2, PIPE_MTE1>(kAivToAicReady);
    AscendC::CrossCoreSetFlag<2, PIPE_M>(kAicToAivReady);
  } else if ASCEND_IS_AIV {
    AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(kAivToAicReady);
    AscendC::CrossCoreWaitFlag<2, PIPE_V>(kAicToAivReady);
    asc_vf_call<head64_probe_vf>(dim3(1024, 1, 1), probe);
  }
}
}  // namespace
```

探针只用于确认 ABI 与线程结构；Task 3 接入真实 QK 后删除 probe output。

- [ ] **Step 6: Register the device ELF**

在 `setup.py` 的 `KERNEL_LIBRARIES` 增加：

```python
"libsparse_attention_head64_hd576_kernel.so": os.path.join(
    EXTENSIONS_DIR,
    "simt",
    "sparse_attention_head64_hd576.asc",
),
```

- [ ] **Step 7: Run local source tests**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git diff --check
```

Expected: source tests pass and the new source contains exactly the two permitted Basic API headers.

- [ ] **Step 8: Compile on dav-3510 before adding math**

在目标机隔离目录中运行：

```bash
export NPU_ARCH=dav-3510
export PATH=/root/miniconda3/bin:$PATH
src/cannbench/operators/builtin/sparse_attention/simt/v1/install.sh
```

Expected: `libsparse_attention_head64_hd576_kernel.so` 与 Python extension 均成功生成并安装。若编译器要求调整 CrossCore template 的 pipe 参数，只按 SDK 中
`cross_core_set_wait_flag` 官方样例修正，不增加第三个 Basic API header。

- [ ] **Step 9: Commit the launch plan and compile skeleton**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_plan.h \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/setup.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): add head64 launch skeleton"
```

## Task 3: Implement Head64 QK With a Temporary Score Workspace

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Create: `src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: `SparseAttentionHead64Plan` and Task 2 device ELF。
- Produces: `launch_sparse_attention_head64_qk_hd576_bf16(...)`；task-major FP32 scores `[task_count, 64, S]`；end-to-end staged Head64 result。

- [ ] **Step 1: Write failing source-contract tests for M=64 QK**

```python
def test_sparse_attention_head64_qk_uses_m64_tensor_api():
    source = _head64_source()

    assert "launch_sparse_attention_head64_qk_hd576_bf16" in source
    assert "sparse_attention_head64_qk_kernel" in source
    assert "MmadParams" in source
    assert "kHead64Tile" in source
    assert "params.m = 64" in source
    assert "params.n = current_selected" in source
    assert "params.k = current_k" in source
    assert "MakeMmad(" in source


def test_sparse_attention_head64_qk_maps_dynamic_tasks():
    source = _head64_source()

    assert "logical_task += plan.used_core_num" in source
    assert "plan.task_count" in source
    assert "plan.head_group_count" in source
    assert "plan.query_tokens" in source
    assert "plan.selected_tokens" in source


def test_sparse_attention_head64_qk_uses_both_aiv_for_query_and_key_pack():
    source = _head64_source()

    assert "const uint32_t sub_block = AscendC::GetSubBlockIdx();" in source
    assert "const uint32_t head_begin = sub_block * 32;" in source
    assert "const uint32_t selected_begin = sub_block * 32;" in source
    assert "CrossCoreSetFlag<2" in source
    assert "CrossCoreWaitFlag<2" in source
```

- [ ] **Step 2: Run the source tests and confirm red state**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
```

Expected: new QK tests fail because the probe kernel has no Q/K pack or MMAD.

- [ ] **Step 3: Implement task decode and AIV packing**

对每个 `logical_task` 使用：

```cpp
int32_t remainder = logical_task;
const int32_t head_group = remainder % plan.head_group_count;
remainder /= plan.head_group_count;
const int32_t query_token = remainder % plan.query_tokens;
const int32_t batch = remainder / plan.query_tokens;
const int32_t head_base = head_group * kHead64Tile;
```

每个 AIV 的 `1024` threads 使用：

```cpp
const uint32_t local_head = threadIdx.x / 32;
const uint32_t lane = threadIdx.x % 32;
const uint32_t head = head_base + sub_block * 32 + local_head;
```

在 task 开始时把 32 行 strided Query 搬到共享 L1 中对应半区。每个 selected tile 中，
AIV0 gather positions `0..31`，AIV1 gather positions `32..63`，并按 Tensor API QK
要求写成 ZN K tile。Invalid index 写零；mask 语义留给 Task 4 softmax。

- [ ] **Step 4: Implement AIC M=64 QK and task-major score store**

AIC 对每个 selected tile 和 9 个 K tiles 执行：

```cpp
MmadParams params;
params.m = 64;
params.n = current_selected;
params.k = current_k;
params.cmatrixInitVal = k_index == 0;
Mmad(mm.with(params), l0_scores, l0_query, l0_keys);
```

score workspace 地址固定为：

```cpp
const int64_t score_base =
    static_cast<int64_t>(logical_task) * 64 * plan.selected_tokens;
```

最终布局是 `[B,Q,head_group,64,S]`，物理上等价于
`[task_count,64,S]`。这是开发期布局，不是最终 fused 输出。

- [ ] **Step 5: Add the staged Host route**

Host 为 Head64 分配：

```cpp
auto task_scores = at::empty(
    {plan.task_count, 64, plan.selected_tokens},
    query.options().dtype(c10::kFloat));
```

QK 后暂时转换为 legacy score layout：

```cpp
auto legacy_scores = task_scores
    .view({plan.batch_size, plan.query_tokens, plan.query_heads, plan.selected_tokens})
    .permute({0, 2, 1, 3})
    .contiguous();
```

调用现有 `run_sparse_attention_family_hd512_decode_direct_tile(...)` 完成 softmax/PV。
该 permute/contiguous 仅用于检查点三，Task 5 必须删除。

- [ ] **Step 6: Add the reduced-shape accuracy script**

`head64_reduced_accuracy.py` 提供四个确定性场景：

```python
CASES = (
    {"name": "valid_s64", "context": 256, "selected": 64, "mode": "valid"},
    {"name": "tail_s70", "context": 256, "selected": 70, "mode": "valid"},
    {"name": "invalid_causal_s70", "context": 256, "selected": 70, "mode": "mixed"},
    {"name": "all_invalid_s17", "context": 256, "selected": 17, "mode": "invalid"},
)
```

每个 case 使用 `B=1,H=128,Q=1,Dqk=576,Dv=512`，调用
`ops._decode_reference` 与 `ops.sparse_attention_forward`，并断言：

```python
torch.allclose(actual_out.float(), expected_out.float(), atol=0.05, rtol=0.05)
torch.allclose(
    actual_lse.float(),
    expected_lse.float(),
    atol=0.05,
    rtol=0.05,
    equal_nan=True,
)
```

全无效行额外断言 output 全零、LSE 全为负无穷且无 NaN。脚本输出每个 case 的
`max_abs_error` 和 pass/fail JSON，并在任一 case 失败时返回 1。

- [ ] **Step 7: Run local tests, target build and reduced accuracy**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_decode_reference.py
pytest -q
```

目标机：

```bash
export NPU_ARCH=dav-3510
export CANNBENCH_SPARSE_ATTENTION_HEAD_TILE=64
export CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS=1
src/cannbench/operators/builtin/sparse_attention/simt/v1/install.sh
python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py
```

Expected: build succeeds; all four reduced cases pass; no deadlock or runtime timeout.

- [ ] **Step 8: Commit the QK checkpoint**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): add staged head64 qk"
```

## Task 4: Add 1024-Thread Online Softmax and Cube PV

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py`

**Interfaces:**
- Consumes: task-major FP32 scores `[task_count,64,S]` from Task 3。
- Produces: `launch_sparse_attention_head64_pv_hd576_bf16(...)`，直接写标准 output `[B,H,Q,512]` 与 LSE `[B,H,Q]`。

- [ ] **Step 1: Write failing source tests for the SIMT mapping and PV MMAD**

```python
def test_sparse_attention_head64_softmax_uses_1024_threads():
    source = _head64_source()

    assert "head64_online_softmax_vf" in source
    assert "__launch_bounds__(1024)" in source
    assert "const uint32_t local_head = threadIdx.x / 32;" in source
    assert "const uint32_t lane = threadIdx.x % 32;" in source
    assert "selected_index += 32" in source
    assert "dim_index += 32" in source


def test_sparse_attention_head64_pv_uses_cube_m64_n128():
    source = _head64_source()

    assert "launch_sparse_attention_head64_pv_hd576_bf16" in source
    assert "sparse_attention_head64_pv_kernel" in source
    assert "pv_params.m = 64" in source
    assert "pv_params.n = current_value" in source
    assert "pv_params.k = current_selected" in source
    assert "kHead64ValueTile" in source


def test_sparse_attention_head64_crosscore_handshakes_are_paired():
    source = _head64_source()

    assert "constexpr uint8_t kAivToAicReady = 8;" in source
    assert "constexpr uint8_t kAicToAivReady = 9;" in source
    assert source.count("CrossCoreSetFlag<2") >= 2
    assert "CrossCoreWaitFlag<2" in source
    assert "CrossCoreSetFlag<0" not in source
    assert "CrossCoreSetFlag<1" not in source
```

- [ ] **Step 2: Run tests and confirm red state**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
```

Expected: softmax/PV tests fail because the staged path still uses legacy postprocess.

- [ ] **Step 3: Implement per-row online-softmax state**

每个 AIV 的 1024 threads 采用：

```cpp
const uint32_t local_head = threadIdx.x / 32;
const uint32_t lane = threadIdx.x % 32;
```

每个 Head 的 32 lanes 对 `S_tile=64` 各处理两个 selected positions。实现以下状态：

```cpp
float tile_max = lane_local_max;
tile_max = group_reduce_max_32(tile_max);
const float new_max = max(running_max[local_head], tile_max);
const float old_scale = expf(running_max[local_head] - new_max);
float lane_sum = 0.0F;
for (int32_t selected_index = lane;
     selected_index < current_selected;
     selected_index += 32) {
  const bool valid = valid_selected_position(...);
  const float probability =
      valid ? expf(score * score_scale - new_max) : 0.0F;
  probability_tile[...] = static_cast<bfloat16_t>(probability);
  lane_sum += probability;
}
const float tile_sum = group_reduce_sum_32(lane_sum);
running_sum[local_head] =
    old_scale * running_sum[local_head] + tile_sum;
running_max[local_head] = new_max;
```

`valid_selected_position` 必须同时判断：

```text
0 <= context_index < context_tokens
context_index <= context_tokens - query_tokens + query_token  (causal=true)
selected_index < selected_tokens
```

- [ ] **Step 4: Implement V gather and M=64 Cube PV**

两个 AIV 分别 gather selected tile 的前后 32 行 V，并按 128-dimension output tile
写入共享 L1。AIC 对四个 Value tiles 执行：

```cpp
MmadParams pv_params;
pv_params.m = 64;
pv_params.n = current_value;
pv_params.k = current_selected;
pv_params.cmatrixInitVal = true;
Mmad(pv_mm.with(pv_params), l0_pv, l0_probability, l0_values);
```

Fixpipe 将 Head `0..31` 发给 AIV0，将 Head `32..63` 发给 AIV1。每个 AIV 更新：

```cpp
for (int32_t dim_index = lane;
     dim_index < plan.value_head_dim;
     dim_index += 32) {
  output_accumulator[local_head][dim_index] =
      output_accumulator[local_head][dim_index] * old_scale + pv_tile_value;
}
```

selected 循环结束后：

```cpp
if (running_sum[local_head] == 0.0F) {
  output[...] = 0.0F;
  if (lane == 0) {
    lse[...] = -std::numeric_limits<float>::infinity();
  }
} else {
  output[...] = output_accumulator[...] / running_sum[local_head];
  if (lane == 0) {
    lse[...] = running_max[local_head] + logf(running_sum[local_head]);
  }
}
```

- [ ] **Step 5: Route staged scores directly to the new PV kernel**

删除 Task 3 的 `permute(...).contiguous()` 与 legacy postprocess 调用。Host 保留
`task_scores`，依次 launch Head64 QK 和 Head64 PV，输出标准 output/LSE。

- [ ] **Step 6: Run local, reduced and full accuracy checks**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test
pytest -q
```

目标机：

```bash
export CANNBENCH_SPARSE_ATTENTION_HEAD_TILE=64
export CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS=1
python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py
python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/v32_full_accuracy.py \
  --phase decode \
  --atol 0.05 \
  --rtol 0.05 \
  --output /root/cannbench-head64-stage-pv.json
```

Expected: four reduced cases pass; full V3.2 decode output/LSE pass; no NaN or deadlock.

- [ ] **Step 7: Commit the staged softmax/PV checkpoint**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py
git commit -m "feat(sparse-attention): add head64 online softmax and pv"
```

## Task 5: Fuse QK, Softmax and PV Into One MIX Kernel（延期）

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py`

**Interfaces:**
- Consumes: Task 3 QK helpers、Task 4 softmax/PV helpers、`SparseAttentionHead64Plan`。
- Produces: `launch_sparse_attention_head64_fused_hd576_bf16(...)`；单个 persistent MIX launch；无完整 scores workspace。

- [ ] **Step 1: Write failing fusion and cleanup tests**

```python
def test_sparse_attention_head64_final_path_is_one_fused_launch():
    source = _head64_source()
    bridge = _bridge_source()

    assert "launch_sparse_attention_head64_fused_hd576_bf16" in source
    assert source.count("sparse_attention_head64_fused_kernel<<<") == 1
    assert "launch_sparse_attention_head64_qk_hd576_bf16" not in bridge
    assert "launch_sparse_attention_head64_pv_hd576_bf16" not in bridge


def test_sparse_attention_head64_final_path_has_no_full_score_workspace():
    bridge = _bridge_source()
    head64_body = _function_body(
        bridge,
        "sparse_attention_forward_family_hd576_head64(",
        "std::tuple<at::Tensor, at::Tensor> sparse_attention_forward_privateuse1(",
    )

    assert "task_scores" not in head64_body
    assert ".permute(" not in head64_body
    assert "run_sparse_attention_family_hd512_decode_direct_tile" not in head64_body


def test_sparse_attention_head64_fused_kernel_keeps_1024_thread_contract():
    source = _head64_source()
    fused = _function_body(
        source,
        "sparse_attention_head64_fused_kernel(",
        'extern "C" void launch_sparse_attention_head64_fused_hd576_bf16(',
    )

    assert "KERNEL_TYPE_MIX_AIC_1_2" in fused
    assert "asc_vf_call<head64_fused_vf>(dim3(1024, 1, 1)" in fused
    assert "CrossCoreSetFlag<2" in fused
    assert "CrossCoreWaitFlag<2" in fused
```

- [ ] **Step 2: Run tests and confirm red state**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
```

Expected: tests fail because Host still launches staged QK and PV kernels and allocates scores.

- [ ] **Step 3: Merge the selected-tile loop**

最终 kernel 对每个 logical task 执行以下严格顺序：

```text
pack Q once
for selected_tile in S:
    gather K tile
    QK M=64
    Fixpipe scores to AIV0/AIV1
    online softmax and probability pack
    for value_tile in Dv:
        gather V tile
        PV M=64
        Fixpipe PV to AIV0/AIV1
        update online output
normalize output and write LSE
```

使用 mode 2 CrossCore flags：

```cpp
constexpr uint8_t kAivToAicReady = 8;
constexpr uint8_t kAicToAivReady = 9;
```

mode 2 的 flag ID 合法范围是 `0..15`。高级 Matmul 使用 `0..7`，`SyncAll` 保留
`11..14`，所以本 kernel 不使用这些区间，也不使用会截断低 4 bit 的 `16` 以上 ID。
`kAivToAicReady` 和 `kAicToAivReady` 分别只承载单一方向；每个阶段按严格顺序完成
set/wait 配对后才能在下一阶段复用，任一方向不得同时存在两个未消费事件。不得使用
mode 0/1 全核同步。

- [ ] **Step 4: Add the final Host helper and remove stage workspaces**

Host helper 固定签名：

```cpp
std::tuple<at::Tensor, at::Tensor>
sparse_attention_forward_family_hd576_head64(
    const at::Tensor& query,
    const at::Tensor& shared_kv,
    const at::Tensor& indices,
    int64_t value_head_dim,
    bool causal,
    const SparseAttentionHead64Plan& plan);
```

它只分配：

```cpp
auto output = at::empty(
    {plan.batch_size, plan.query_heads, plan.query_tokens, plan.value_head_dim},
    query.options().dtype(c10::kFloat));
auto lse = at::empty(
    {plan.batch_size, plan.query_heads, plan.query_tokens},
    query.options().dtype(c10::kFloat));
auto plan_tensor = at::empty({4096}, query.options().dtype(c10::kByte));
```

除 KFC workspace 外，不分配与 `B*H*Q*S` 成比例的完整 score/probability workspace。

- [ ] **Step 5: Remove the staged symbols and dead helpers**

从 Host/device 删除：

```text
launch_sparse_attention_head64_qk_hd576_bf16
launch_sparse_attention_head64_pv_hd576_bf16
sparse_attention_head64_qk_kernel
sparse_attention_head64_pv_kernel
task_scores
legacy score-layout conversion
probe kernel and probe output
```

保留可复用的 QK MMAD、mask、online-softmax、V gather、PV MMAD 和 output normalize
device helper。

- [ ] **Step 6: Run local and target correctness verification**

```bash
pytest -q \
  src/cannbench/operators/builtin/sparse_attention/simt/test
pytest -q
git diff --check
```

目标机重新安装后运行：

```bash
export CANNBENCH_SPARSE_ATTENTION_HEAD_TILE=64
export CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS=1
python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py
python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/v32_full_accuracy.py \
  --phase decode \
  --atol 0.05 \
  --rtol 0.05 \
  --output /root/cannbench-head64-fused.json
```

Expected: reduced and full output/LSE pass; kernel does not hang; full route launches one Head64 MIX kernel.

- [ ] **Step 7: Commit the fused Head64 kernel**

```bash
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/simt/sparse_attention_head64_hd576.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/v1/aten_dsa_sparse_attention/csrc/sparse_attention.asc \
  src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v1_build_shell.py
git commit -m "feat(sparse-attention): fuse head64 decode pipeline"
```

## Task 6: Run Full Accuracy, Performance and Profiler Acceptance

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/v32_full_accuracy.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/README.md`

**Interfaces:**
- Consumes: committed fused Head64 implementation from Task 5。
- Produces: exact-commit dav-3510 build、reduced/full accuracy JSON、legacy/Head64 benchmark artifacts、profiler conclusions。

- [ ] **Step 1: Record tuning in the full-accuracy JSON**

在 `v32_full_accuracy.py` 的 result 中增加：

```python
head_tile, selected_partitions = ops._resolve_tuning()
common_result = {
    "case_id": case.case_id,
    "phase": case.phase,
    "head_tile": head_tile,
    "selected_partitions": selected_partitions,
    "query_shape": list(query.shape),
    "shared_kv_shape": list(shared_kv.shape),
    "indices_shape": list(indices.shape),
    "kernel_wall_seconds": kernel_wall_seconds,
}
```

增加单元测试断言默认 result helper 使用 `(1,1)`，Head64 环境使用 `(64,1)`。

- [ ] **Step 2: Deploy the exact commit to the existing dav-3510 host**

本地：

```bash
feature_commit=$(git rev-parse --short HEAD)
archive=/tmp/cannbench-sparse-head64-${feature_commit}.tar.gz
remote_dir=/root/cannbench-sparse-head64-${feature_commit}
git archive --format=tar.gz \
  --prefix=cannbench-sparse-head64-${feature_commit}/ \
  -o "${archive}" HEAD
scp -P 20002 "${archive}" root@121.41.199.170:/root/
ssh -p 20002 root@121.41.199.170 bash -s -- "${feature_commit}" <<'REMOTE'
set -euo pipefail
feature_commit=$1
remote_dir=/root/cannbench-sparse-head64-${feature_commit}
mkdir -p "${remote_dir}"
tar -xzf "/root/cannbench-sparse-head64-${feature_commit}.tar.gz" \
  -C "${remote_dir}" \
  --strip-components=1
REMOTE
```

本地把 `feature_commit` 作为位置参数传给远端脚本；远端目录不是 Git checkout，不得
在远端调用 `git rev-parse`。新目录不得删除或覆盖其他开发目录。

- [ ] **Step 3: Build and install the exact source**

从本地启动远端构建，并再次把同一个 commit 作为位置参数传入：

```bash
ssh -p 20002 root@121.41.199.170 bash -s -- "${feature_commit}" <<'REMOTE'
set -euo pipefail
feature_commit=$1
remote_dir=/root/cannbench-sparse-head64-${feature_commit}
source /usr/local/Ascend/cann/set_env.sh
export NPU_ARCH=dav-3510
export ASCEND_RT_VISIBLE_DEVICES=0
export PATH=/root/miniconda3/bin:$PATH
cd "${remote_dir}"
python -m pip install -e . --no-build-isolation --no-deps
src/cannbench/operators/builtin/sparse_attention/simt/v1/install.sh
REMOTE
```

Expected: Python package、Head64 ELF 和 extension 安装成功。

- [ ] **Step 4: Run reduced and full Head64 accuracy**

```bash
export CANNBENCH_SPARSE_ATTENTION_HEAD_TILE=64
export CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS=1
export ASCEND_LAUNCH_BLOCKING=1
python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/head64_reduced_accuracy.py \
  > /root/head64-reduced-${feature_commit}.json
python \
  src/cannbench/operators/builtin/sparse_attention/simt/test/v32_full_accuracy.py \
  --phase decode \
  --atol 0.05 \
  --rtol 0.05 \
  --output /root/head64-full-${feature_commit}.json
```

Expected: every reduced case and all full `B/Q/H` rows pass output/LSE validation。

- [ ] **Step 5: Run equal-condition legacy and Head64 benchmarks**

Legacy：

```bash
export CANNBENCH_SPARSE_ATTENTION_HEAD_TILE=1
export CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS=1
cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v1 \
  --op sparse_attention \
  --dataset realistic_decode \
  --case-id deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048 \
  --dtype bfloat16 \
  --seed 7 \
  --output-dir /root/cannbench-head64-legacy-${feature_commit}
```

Head64：

```bash
export CANNBENCH_SPARSE_ATTENTION_HEAD_TILE=64
export CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS=1
cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v1 \
  --op sparse_attention \
  --dataset realistic_decode \
  --case-id deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048 \
  --dtype bfloat16 \
  --seed 7 \
  --output-dir /root/cannbench-head64-fused-${feature_commit}
```

两个命令除 tuning 与 output directory 外必须完全一致。验收要求 Head64 端到端 Sparse
Attention 延迟低于 legacy。

- [ ] **Step 6: Inspect profiler evidence**

从 Head64 profile artifacts 提取并记录：

```text
Block Dim / Mix Block Dim
有效 AIC 数
AIV0 和 AIV1 duration
Cube MMAD instruction count
Cube utilization
Vector / Scalar / MTE utilization
AIC/AIV wait ratio
GM/L2 traffic and L2 hit rate
```

必须验证：

```text
logical task count = 8
两个 AIV 均有持续工作
SIMT launch threads = 1024
profile 中同时存在 QK 与 PV MMAD
```

如果 Head64 未快于 legacy，保持默认 `1,1`，在 README 写明实测瓶颈；不在本计划中
临时加入 Split-KV 或 ping-pong。

- [ ] **Step 7: Update operator-local documentation with measured facts**

在 `simt/README.md` 增加 `Head64 Experimental Path`，只写入本次 exact commit 的真实
数据：

```text
commit
device / NPU_ARCH
reduced accuracy status
full accuracy max error and mismatch count
legacy latency
Head64 latency
speedup = legacy_latency / head64_latency
effective AIC count
AIV0/AIV1 duration
Cube/MTE/wait conclusions
default remains legacy or switches in a later change
```

不要修改 published benchmark schema，也不要在本任务中切换默认配置。

- [ ] **Step 8: Run final verification and commit results**

```bash
pytest -q
git diff --check
git status --short --branch
git add \
  src/cannbench/operators/builtin/sparse_attention/simt/test/v32_full_accuracy.py \
  src/cannbench/operators/builtin/sparse_attention/simt/README.md
git commit -m "docs(sparse-attention): record head64 validation"
```

Expected: full suite passes; only intended operator-local files are committed; worktree is clean.

## Plan Self-Review

- Task 1 covers legacy-default tuning, explicit Head64 opt-in and invalid parameter errors.
- Task 2 covers dynamic Host task count, shared plan ABI, allowed headers, CrossCore mode 2 and the 1024-thread structure.
- Task 3 independently validates Head64 Query/K gather and M=64 QK before changing softmax/PV.
- Task 4 independently validates invalid/causal masking, online-softmax, both AIVs and Cube PV.
- Task 5 removes all development score workspaces and leaves one fused Head64 launch.
- Task 6 covers reduced/full accuracy, equal-condition benchmark, profiler evidence and failure-to-beat-legacy behavior.
- Function names and plan fields are consistent across all tasks.
- No task modifies a public framework layer or implements Split-KV, Head16, prefill or ping-pong.
