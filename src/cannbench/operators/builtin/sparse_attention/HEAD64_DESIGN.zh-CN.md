# Sparse Attention Head64 渐进式融合设计

本文定义 CannBench Ascend SIMT Sparse Attention 第一阶段 Head64 优化的实现边界、
数据流、Host 参数、设备侧任务映射、开发检查点和验收标准。

本文是已确认的实现设计，不替代
[`PARALLEL_SPLITTING_RESEARCH.zh-CN.md`](PARALLEL_SPLITTING_RESEARCH.zh-CN.md)
中的外部方案调研。调研文档保留 A/B/C 三组长期对照；本文只实现其中优先级最高、
也是后续 Split-KV 基础的方案 A。

设计日期：2026-07-27。

## 1. 目标

第一阶段为 DeepSeek V3.2 realistic decode 增加一个实验性的 `head64` 路径：

```text
B = 2
H = 128
KV_H = 1
Q = 2
C = 32768
S = 2048
Dqk = 576
Dv = 512
dtype = bfloat16
```

该路径将 64 个共享同一 KV head 的 Query heads 组织为一个矩阵计算任务：

```text
QK: [64, 576] x [576, S_tile]
PV: [64, S_tile] x [S_tile, 512]
```

需要实现以下结果：

1. QK 的 `M` 从 legacy 的 1 提高到 64。
2. 同一个 Head64 task 内的 64 个 Query heads 复用一次 K/V gather。
3. MIX task 的两个 AIV 都参与稳定、对称的有效工作。
4. QK 和 PV 都使用 Cube/Tensor API MMAD。
5. softmax 使用在线归一化，最终直接输出 attention output 和 LSE。
6. legacy 路径继续作为默认值；`head64` 只能通过算子本地实验参数显式启用。
7. launch plan 从一开始保留 selected-token partition 字段，为后续方案 B 复用。

## 2. 非目标

第一阶段不做以下工作：

- 不实现沿 selected tokens 的 Split-KV。
- 不实现 partial output workspace 和跨 partition combine。
- 不实现 `head_tile=16` 对照。
- 不让两个独立 AIC task 跨核共享 K/V staging。
- 不优化 `family_hd128`、`family_hd256` 或普通 `family_hd512`。
- 不优化 prefill。
- 不修改公共 CLI、Backend、Workflow 或全局配置。
- 不在 benchmark 前将 `head64` 设为默认。
- 除 kernel-local Mutex 所需的 `basic_api/kernel_common.h` 外，不新增 C++ Basic
  API、`SetFlag`、`WaitFlag`、`PipeBarrier` 或跨核 flag 依赖。

## 3. 实现策略

采用渐进式融合，而不是一次完成整个 fused kernel。

最终目标是单个 Head64 MIX kernel：

```text
双 AIV gather K/V tile
        -> AIC QK
        -> 双 AIV online softmax
        -> AIC PV
        -> 双 AIV online output update
```

开发中先保留阶段性 workspace，使 QK、softmax 和 PV 可以分别验证。每个检查点通过
精度和运行稳定性验证后再融合。最终代码只保留 legacy 和完整 `head64` 两条路径，
阶段性调试实现保留在 Git 历史中，不作为长期公开变体。

这一选择兼顾两点：

- 每一步都能定位错误属于任务映射、QK、softmax 还是 PV。
- 最终路径仍然消除完整 scores workspace，不把多 kernel 方案当作性能终点。

## 4. 算子本地参数

### 4.1 参数形式

使用两个与具体方案名称无关的 Host 参数：

```text
CANNBENCH_SPARSE_ATTENTION_HEAD_TILE
CANNBENCH_SPARSE_ATTENTION_SELECTED_PARTITIONS
```

参数映射为：

| 配置 | `HEAD_TILE` | `SELECTED_PARTITIONS` | 含义 |
| --- | ---: | ---: | --- |
| 默认 | 1 | 1 | legacy |
| 第一阶段 | 64 | 1 | 方案 A：Head64，不拆 S |
| 后续候选 | 64 | 4 | 方案 B：Head64 + Split-KV |
| 后续候选 | 16 | 1 | 方案 C：Head16，不拆 S |

本阶段只接受 `(1,1)` 和 `(64,1)`。显式设置其他组合时返回清晰错误，不静默回退，
避免 profiler 误测 legacy。

### 4.2 Torch custom op 边界

算子本地 Python wrapper 负责读取、解析和校验环境变量，并将以下带默认值的参数传给
custom op：

```text
head_tile = 1
selected_partitions = 1
```

C++ Host bridge 根据 shape 和这两个参数生成 `SparseAttentionLaunchPlan`。原有调用者
未传参数时保持 legacy 行为。该变更只发生在
`src/cannbench/operators/builtin/sparse_attention/` 内部。

### 4.3 实验路径准入条件

`head64` 第一阶段只接受：

```text
phase == decode
family == family_hd576
query.dtype == bfloat16
shared_kv.dtype == bfloat16
H == 128
KV_H == 1
Dqk == 576
Dv == 512
selected_partitions == 1
```

`B`、`Q`、`C` 和 `S` 由 Host plan 动态读取。`S` 支持不足 64 tokens 的尾块，
但仍受现有 wide-head contract 的 `S <= 2048` 限制。

显式启用 `head64` 后如果 shape 不满足准入条件，Host 必须报错并打印不满足的条件，
不能自动调用 legacy。

## 5. Host launch plan

### 5.1 结构

Host 和 device 共享一个只包含整数和偏移的 POD plan，至少包括：

```text
used_core_num
task_count
head_tile
head_group_count
selected_tile
selected_partitions
selected_partition_size
qk_head_dim
value_head_dim
batch_size
query_heads
query_tokens
context_tokens
selected_tokens
```

第一阶段固定：

```text
head_tile = 64
selected_tile = 64
selected_partitions = 1
```

其余循环次数、尾块大小和 task 映射全部由 plan 与输入 shape 推导。device kernel 不
根据“方案 A/B/C”字符串分支。

### 5.2 Task 映射

```text
head_group_count = H / head_tile
task_count = B * Q * head_group_count * selected_partitions
used_core_num = min(task_count, physical_aic_limit)
```

第一阶段 `selected_partitions=1`，task ID 映射为：

```text
head_group = task_id % head_group_count
query_token = (task_id / head_group_count) % Q
batch = task_id / (head_group_count * Q)
```

V3.2 full case：

```text
head_group_count = 128 / 64 = 2
task_count = 2 * 2 * 2 = 8
```

每个 task 完整处理对应 `(batch, query_token, head_group)` 的 `S` 个 selected
tokens。不同 task 不共享临时状态，也不需要跨 AIC 同步。

## 6. Device 数据流

### 6.1 Tile

第一阶段使用：

```text
M tile = 64 heads
S tile = 64 selected tokens
K tile = 64 dimensions
V output tile = 128 dimensions
```

一次 selected tile 的 QK 需要 9 个完整 K tile：前 8 个覆盖 512 个 NoPE 维度，
第 9 个覆盖 64 个 RoPE 维度，总有效 `Dqk` 为 576。

PV 将 `Dv=512` 拆为 4 个 128-dimension output tiles。

### 6.2 双 AIV 分工

两个 AIV 对称处理 Head 维：

```text
AIV0: local heads 0..31
AIV1: local heads 32..63
```

每个 AIV 维护自己的：

- 32 行 online max。
- 32 行 online exp sum。
- `32 x 512` FP32 online output accumulator。
- 当前 `32 x 64` score tile。
- 当前 `32 x 64` BF16 probability tile。
- 当前 PV output tile 的 FP32 staging。

K/V gather 也由两个 AIV 分担 selected-token 子区间。两个 AIV 向共享片内布局的互不
重叠区域写入数据，因此不需要相互原子更新。

### 6.3 AIC QK

AIC 对每个 selected tile 执行：

```text
[64, 576] x [576, current_selected]
```

Q 在 Head64 task 开始时加载并复用；K 按 `K tile=64` gather、搬运和 MMAD。
FP32 score 通过 Fixpipe 按 Head 行分给两个 AIV，每个 AIV 接收 32 行。

Invalid index 对应的 K 元素可以填零，但最终有效性由 AIV softmax mask 决定，不能
让无效位置以零分数参与归一化。

### 6.4 Online softmax

对 selected tile `t`，每个 AIV 按行计算局部统计：

```text
tile_max = max(scores_t)
new_max = max(running_max, tile_max)
old_scale = exp(running_max - new_max)
prob_t = exp(scores_t - new_max)
new_sum = old_scale * running_sum + sum(prob_t)
```

然后：

```text
running_output *= old_scale
running_output += prob_t @ V_t
running_max = new_max
running_sum = new_sum
```

最终：

```text
output = running_output / running_sum
lse = running_max + log(running_sum)
```

Invalid index、超过 causal 上界的 index 和 selected tail 之外的位置都按负无穷处理。
整行没有有效元素时输出全零，LSE 为负无穷。

### 6.5 AIC PV

AIV 将当前 64 行 probability 打包为 BF16 矩阵，AIC 执行：

```text
[64, current_selected] x [current_selected, 128]
```

四次 MMAD 覆盖 512 个 Value dimensions。Fixpipe 再按 Head 行把 FP32 PV tile 分给
两个 AIV。AIV 按 online-softmax 的 `old_scale` 更新自己的 FP32 output accumulator。

## 7. 片内存储预算

以下是第一版单 buffer 的目标预算，不包含编译器运行时保留区：

### 7.1 每个 AIV

| 数据 | 近似大小 |
| --- | ---: |
| `32 x 512` FP32 output accumulator | 64 KiB |
| `32 x 64` FP32 scores | 8 KiB |
| `32 x 64` BF16 probabilities | 4 KiB |
| `32 x 128` FP32 PV staging | 16 KiB |
| K/V gather 子 tile | 不超过 12 KiB |

第一版避免同时双缓冲上述大对象。目标环境编译会验证实际 UB 分配；如果超限，优先将
PV output tile 从 128 降为 64，而不是改变 Head64 task 语义。

### 7.2 AIC

QK 和 PV 不同时占用同一组 L0 资源。主要 tile 为：

| 阶段 | 数据 | 近似大小 |
| --- | --- | ---: |
| QK | `64 x 64` BF16 Q | 8 KiB |
| QK | `64 x 64` BF16 K | 8 KiB |
| QK | `64 x 64` FP32 score | 16 KiB |
| PV | `64 x 64` BF16 probability | 8 KiB |
| PV | `64 x 128` BF16 V | 16 KiB |
| PV | `64 x 128` FP32 output | 32 KiB |

Q、K、probability 和 V 在 L1 中按阶段复用。最终布局以 `dav-3510` 编译器报告为准。

## 8. 同步边界

新 Head64 kernel 只允许 kernel-local `AscendC::Mutex::Lock/Unlock` 协调同一个 MIX
task 内的 AIC/AIV流水。允许包含 `basic_api/kernel_common.h` 以取得 Mutex 定义；
不得从该 header 使用其他 Basic API。

逻辑阶段为：

```text
K_READY
SCORE_READY
P_READY
PV_READY
```

第一版使用单 buffer 顺序状态机：

```text
AIV gather -> AIC QK -> AIV softmax -> AIC PV -> AIV output update
```

正确性稳定后，最终融合检查点才引入 ping-pong buffer，使下一 selected tile 的
gather 与当前 tile 的 QK/softmax/PV 重叠。

禁止：

- AIC 之间共享状态。
- 使用 GM flag 自旋同步。
- 新增 `CrossCoreSetFlag/CrossCoreWaitFlag`。
- 新增 `SetFlag/WaitFlag/PipeBarrier`。
- 引入 `basic_api/kernel_common.h` 之外的 Basic API header 或
  `kernel_operator.h`。

## 9. 文件边界

实现只修改 Sparse Attention 算子包：

```text
src/cannbench/operators/builtin/sparse_attention/
  simt/v1/aten_dsa_sparse_attention/
    ops.py
    csrc/sparse_attention.asc
    csrc/simt/sparse_attention_head64_plan.h
    csrc/simt/sparse_attention_head64_hd576.asc
    setup.py
  simt/test/
    test_sparse_attention_dispatch.py
    test_sparse_attention_v1_build_shell.py
    v32_full_accuracy.py
```

职责：

- `ops.py`：读取算子本地实验参数，保留 legacy 默认值。
- `sparse_attention.asc`：校验参数、构造 Host plan、选择 legacy/head64 launch。
- `sparse_attention_head64_plan.h`：Host/device 共享 POD plan 和整数常量。
- `sparse_attention_head64_hd576.asc`：Head64 device kernel 和设备侧数学 helper。
- `setup.py`：独立注册 Head64 device ELF。
- 算子本地测试：验证参数、源码边界、编译、精度和运行稳定性。

不修改：

```text
src/cannbench/cli.py
src/cannbench/backends/
src/cannbench/core/
src/cannbench/operators/builtin/dsa_decode/
```

## 10. 开发检查点

### 10.1 检查点一：控制面与编译骨架

交付：

- 参数解析和默认 legacy 行为。
- Host launch plan。
- `8` 个逻辑 task 的映射测试。
- 新 device ELF 构建入口。
- 双 AIV和 kernel-local Mutex 的最小编译验证。
- 新源码 API 边界测试。

### 10.2 检查点二：Head64 QK

交付：

- 双 AIV 分担 K gather。
- AIC 执行 `M=64` QK。
- 临时 scores workspace 使用现有布局 `[B,H,Q,S]`。
- 复用 legacy postprocess 得到 output/LSE。
- reduced shape 端到端精度通过。

此阶段隔离验证 task mapping、K gather、Q 布局、MMAD 参数和 Fixpipe score 布局。

### 10.3 检查点三：双 AIV softmax 和 Cube PV

交付：

- 两个 AIV各处理 32 行 online-softmax。
- probability BF16 pack。
- AIC `M=64` PV。
- FP32 online output accumulator。
- 全无效行、causal mask 和 selected tail 精度通过。

此阶段可以继续使用完整 scores workspace，以便把 PV/softmax 错误与 QK 错误分开。

### 10.4 检查点四：最终融合与性能验收

交付：

- QK、online-softmax 和 PV 合入单个 MIX kernel。
- 删除 Head64 路径的完整 scores workspace。
- 根据单 buffer profiler 决定是否启用 ping-pong。
- reduced shape 和 full realistic decode 精度通过。
- 与 legacy 做等条件延迟和 profiler 对照。

最终代码不保留只用于检查点二、三的调试入口。

## 11. 错误处理

Host 在 launch 前拒绝：

- 环境变量不是合法整数。
- 本阶段未支持的 `(head_tile, selected_partitions)`。
- 显式 `head64` 与 phase/family/dtype/shape 不匹配。
- `S > 2048`。
- tensor 维度或 stride 不满足现有 custom-op contract。

Device 处理：

- 负 index。
- `index >= context_tokens`。
- causal 上界之外的 index。
- 不足 64 个 selected tokens 的尾块。
- 整行没有有效 token。

默认参数未设置时始终走 legacy，不改变现有用户行为。

## 12. 验证

### 12.1 本地测试

每个检查点至少运行：

```bash
pytest -q src/cannbench/operators/builtin/sparse_attention/simt/test
pytest -q
```

源码契约测试需要确认：

- 新 kernel 的 Basic API header 只能有提供 Mutex 的
  `basic_api/kernel_common.h`。
- 新 kernel 从该 header 只能调用 `AscendC::Mutex::Lock/Unlock`。
- 新 kernel 不调用禁止的同步 API。
- 两个 AIV均有有效分支，不存在 `subBlockIdx != 0` 直接退出。
- QK 和 PV 使用 Tensor API MMAD。
- Host 根据 shape 计算 task count，不写死 V3.2 的 `8`。
- 默认参数仍选择 legacy。

### 12.2 目标环境编译

目标环境使用：

```text
NPU_ARCH=dav-3510
```

当前本地 worktree 没有 `bisheng` 和 NPU，因此 device 编译、真机精度与 profiler
必须沿用仓库现有的远端隔离部署流程。

### 12.3 Reduced shape 精度

至少覆盖：

- `B=1,H=128,Q=1,C=256,S=64,Dqk=576,Dv=512`。
- `S` 不足 64 的尾块。
- causal mask。
- 负 index 和越界 index。
- 整行 index 都无效。

output 和 LSE 使用现有 contract：

```text
atol = 0.05
rtol = 0.05
```

### 12.4 Full realistic decode

运行：

```text
realistic_decode::deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048
```

要求全部 `B/Q/H` 行通过 output/LSE 校验，而不是只抽查一个 Head group。

### 12.5 性能与 profiler

legacy 和 `head64` 必须使用相同输入、warmup、重复次数和同步方式。至少记录：

- 端到端 Sparse Attention 延迟。
- 实际有效 AIC/AIV 数。
- 两个 AIV 的耗时和利用率。
- Cube MMAD 指令数。
- QK 与 PV 的 Cube 利用率。
- MTE、Scalar、Vector 利用率。
- AIC/AIV等待比例。
- GM/L2 流量和 L2 hit rate。

首轮性能门槛是 `head64` 延迟低于 legacy，不预设固定加速倍数。如果未达到门槛，
保持实验态，记录瓶颈并决定继续优化方案 A 还是进入 Split-KV；不得切换默认路径。

## 13. 后续演进

方案 A 稳定后，方案 B 复用同一 launch plan 和 Head64 kernel 主体：

```text
selected_partitions > 1
task = (batch, query_token, head_group, selected_partition)
```

主 kernel 改为输出局部 max、sum 和 weighted output，独立 combine kernel 完成跨
partition 合并。方案 B 不应复制 QK、softmax 或 PV 数学实现。

`head_tile=16` 只作为后续满核对照。是否支持它由方案 A/B profiler 决定，本阶段
不为它增加 device 分支。
