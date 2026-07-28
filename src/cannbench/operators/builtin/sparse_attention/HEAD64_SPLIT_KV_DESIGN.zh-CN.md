# Head64 Split-KV 优化设计

日期：2026-07-28

## 1. 目标

在现有 Head64 staged sparse attention 路径上增加 Split-KV，使 DeepSeek V3.2
realistic decode case 从 8 个逻辑 AIC task 扩展到 32 个：

```text
B = 2
Q = 2
H = 128
head_tile = 64
selected_tokens = 2048
selected_partitions = 4

head_group_count = H / head_tile = 2
task_count = B * Q * head_group_count * selected_partitions = 32
```

本优化保持以下行为不变：

- 默认 tuning 仍为 `(head_tile=1, selected_partitions=1)`。
- `(64,1)` 继续作为方案 A，不额外启动 combine kernel。
- 新路径只在显式配置 `(64,2)` 或 `(64,4)` 时启用。
- 不改变公共 backend、CLI 或 published data contract。

性能准入条件是 `(64,4)` 在 realistic decode 上端到端延迟低于方案 A 当前约
`1.33 ms` 的基线。未达到门槛时保留实验路径，不替换方案 A。

## 2. 方案选择

采用“分区复用 staged QK/PV，再独立 combine”的实现：

```text
partition-aware QK
  -> partition-local softmax + PV
  -> partial_output + partial_lse
  -> SIMT combine
  -> output + lse
```

不在第一版同时进行 QK/PV 融合。这样只增加 partition 边界、局部输出布局和 combine，
能够继续复用当前 Head64 的 M64 Cube QK、1024-thread softmax 与 M64N128 Cube PV。

第一版不增加 AIC 之间的同步。现有 MIX task 内部的 AIC/AIV 协作保持不变；combine
是独立 kernel，不依赖不同 partition task 的执行顺序。新代码不引入 Basic API 或
新的跨核 flag 依赖。

## 3. Host launch plan

### 3.1 支持的 tuning

算子本地 wrapper 接受：

```text
(1, 1)   legacy
(64, 1)  Head64 方案 A
(64, 2)  Head64 Split-KV 16-task 对照
(64, 4)  Head64 Split-KV 32-task 主配置
```

其他组合直接报 `unsupported sparse_attention tuning`，不能静默回退。

### 3.2 Task 映射

partition 是最内层维度：

```text
partition = task_id % selected_partitions
remainder = task_id / selected_partitions
head_group = remainder % head_group_count
query_token = (remainder / head_group_count) % query_tokens
batch = remainder / (head_group_count * query_tokens)
```

当 `selected_partitions=1` 时，该映射与当前方案 A 等价。

### 3.3 Partition 边界

按 `selected_tile=64` 的完整 tile 分配 partition：

```text
selected_tile_count = ceil_div(selected_tokens, selected_tile)
partition_tile_capacity = ceil_div(selected_tile_count, selected_partitions)
partition_begin = partition * partition_tile_capacity * selected_tile
partition_end = min(selected_tokens,
                    partition_begin + partition_tile_capacity * selected_tile)
partition_length = max(0, partition_end - partition_begin)
```

因此 realistic case 每个 partition 恰好处理 512 tokens。非整除 shape 的最后一个
partition 可以包含尾 tile；当 `selected_partitions` 大于有效 tile 数时，后续
partition 为空。空 partition 必须产生零 output 和负无穷 lse，不能读越界。

## 4. Workspace 与布局

QK scores 和 probabilities 改为 partition-local stride：

```text
task_scores:
  [task_count, 64, partition_token_capacity], FP32

task_probabilities:
  [task_count, partition_tile_capacity, 64, 64], BF16

partial_output:
  [B, Q, head_group_count, selected_partitions, 64, Dv], FP32

partial_lse:
  [B, Q, head_group_count, selected_partitions, 64], FP32
```

对 `S=2048,P=4`，scores 和 probabilities 的有效总元素数与方案 A 基本不变；新增
的主要流量是 4 份 FP32 partial output。`P=1` 直接将 task output 重排为最终 output，
不分配或启动 combine 专用结果路径。

## 5. Device 数据流

### 5.1 QK

QK kernel 继续以 M64K64 Cube tile 计算。每个 task 只循环自己的
`[partition_begin, partition_end)`，scores 使用 partition-local selected offset；
indices 和 shared KV gather 使用加上 `partition_begin` 的绝对 selected offset。

当 `task_count` 超过 32、物理 AIC/AIV 核开始循环复用时，AIC 必须先把当前 query
tile 从共享 L1 拷入 L0A，再通过已有的 MTE1 CrossCore flag 允许 AIV 生成 key。
该所有权顺序防止 AIV 提前进入下一个逻辑 task 并覆盖当前 task 的 L1 query。

### 5.2 Softmax 与 PV

softmax 只在当前 partition 的有效 token 上计算。每个 partition、每个 head 输出：

- 局部归一化概率；
- `partial_lse = log(sum(exp(score)))`；
- 由局部概率计算的 `partial_output = softmax(local_scores) @ V_local`。

PV 保持当前 M64N128 Cube 主体以及已有的首个完整 PV tile 重算 workaround。本阶段
不同时定位该 CANN 事务问题，避免把无关设备状态问题引入 Split-KV 验证。

### 5.3 Combine

combine 对一个 `(batch, query_token, head_group)` 的所有 partition 做稳定归约。对每个
head：

```text
global_lse = logsumexp(partial_lse[p])
weight[p] = exp(partial_lse[p] - global_lse)
output = sum(weight[p] * partial_output[p])
```

若所有 `partial_lse` 都是负无穷：

```text
global_lse = -inf
output = 0
```

该分支必须在计算 `partial_lse - global_lse` 前处理，避免 `-inf - -inf` 产生 NaN。
combine 使用 1024-thread SIMT，并让一个 base task 覆盖一个 64-head group；两个 AIV
各处理 32 heads。它只读取已经落到 GM 的 partial，不使用跨 task 同步。

## 6. Host 输出路径

Host 根据 partition 数选择结果处理：

- `P=1`：沿用当前 task output reshape/permute，不 launch combine。
- `P=2/4`：分配 partial output/lse，QK 和 PV 完成后 launch combine，直接写最终
  `[B,H,Q,Dv]` output 与 `[B,H,Q]` lse。

launch 顺序在同一个 NPU stream 中保持 QK、PV、combine 的依赖，不增加 host 同步。

## 7. 错误处理与实现边界

Split-KV 继续沿用 Head64 准入条件：decode、family_hd576、BF16、`H=128`、
`KV_H=1`、query/shared-KV 的 `Dqk=576`、`Dv=512` 和 `S<=2048`。显式启用后
shape 不满足条件时返回清晰错误，不调用 legacy。`S=0` 或 `C=0` 返回零 output 和
负无穷 LSE；int64 indices 在窄化为 int32 前钳位到 int32 范围，使溢出值保持无效。

所有新逻辑保留在：

```text
src/cannbench/operators/builtin/sparse_attention/
```

不在 CLI、core 或公共 backend 中增加 sparse-attention 分支。

## 8. 验证与性能门槛

### 8.1 Host 与源码测试

采用测试先行，至少覆盖：

- wrapper 接受 `(64,2)` 和 `(64,4)`，继续拒绝其他组合；
- task decode 的 partition 最内层映射；
- partition begin/end、尾 tile 和空 partition；
- QK/PV 使用 partition-local stride 和绝对 indices offset；
- `P=1` 不启动 combine；
- `P=2/4` 启动 combine；
- combine 的 all-invalid 防 NaN 分支；
- 不新增公共层 operator-name 分支和不允许的 API 依赖。

### 8.2 远程设备精度

分别验证 `P=1`、`P=2` 和 `P=4`：

- `S=17/64/70/128/2048`；
- valid、causal、invalid index、all-invalid；
- realistic case 的全部 `B/Q/H` 行；
- 至少一个 `task_count>32` 的物理核循环复用 case；
- `S=0/C=0`、shared-KV 宽度拒绝和 int64 index 溢出契约；
- output 和 lse 延用当前 `atol=0.05, rtol=0.05`。

### 8.3 性能

在同一输入、warmup、轮次与同步方式下对比 `P=1/2/4`，记录：

- 端到端 Sparse Attention 延迟；
- QK、PV、combine 分阶段延迟；
- 实际 AIC/AIV 数与利用率；
- partial workspace 和 GM/L2 流量；
- 完整 `dsa_decode` workflow 延迟。

只有 `P=4` 端到端中位延迟低于方案 A 当前约 `1.33 ms` 时，方案 B 才通过性能门槛。
如果 `P=4` 未通过该门槛，则不能把 32-task 主配置标记为性能通过。若 `P=2` 独立快于
方案 A，则可以保留为实测最优的 Split-KV 配置，但必须分别报告 `P=4` 未达标，不能
因为 32 AIC 占满而隐藏回退结果。
