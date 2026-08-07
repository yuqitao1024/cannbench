# DSA V2 Sparse Attention vLLM 风格三槽流水设计

## 1. 目标与边界

本设计针对 DeepSeek V3.2 BF16 decode canonical shape：

```text
B=2, Q=2, Hq=128, Hkv=1
context=32768, selected=2048
Dqk=576, Dv=512, causal=true
```

目标是将 SIMT V2 的数据切分和跨 tile 调度改成与 vLLM-Ascend MLA 路径一致的
核心结构：

- Query head tile 固定为 `64`；
- selected token tile 固定为 `128`；
- 两个 AIV subblock 执行相同阶段，各处理 32 个 Query heads 和 64 行 KV；
- 使用三个 rolling slot，交错执行 tile `t` 的 gather/QK、tile `t-1` 的
  softmax/PV 和 tile `t-2` 的 output update；
- selected 维不再拆成 4 个独立 task，canonical decode 使用 P=1 并直接写最终
  output/LSE，移除 4 MiB partial output 和 Combine kernel。

API 边界不变：Vector 算术使用 SIMT API，Cube 算术使用 Tensor API；同步继续使用
当前 CrossCore flag，核内搬运继续使用当前 C API/Tensor API。不得为了复制
vLLM-Ascend 的实现而新增 Basic API 数据搬运、LocalTensor 或 Fixpipe 依赖。

## 2. 对称 AIV 数据切分

每个 AIC 对应两个 AIV subblock。两端执行完全相同的业务函数，只由
`subblock_index` 决定数据范围：

```text
                 subblock 0              subblock 1
Query heads      [0, 32)                 [32, 64)
selected rows    [0, 64)                 [64, 128)
QK score rows    heads [0, 32)           heads [32, 64)
PV/output rows   heads [0, 32)           heads [32, 64)
```

每个 AIV 在每个调度 round 中按同样顺序尝试三个阶段：

```text
gather tile t -> softmax tile t-1 -> output update tile t-2
```

两端不交换 UB 数据。AIV0/AIV1 各自从物理 UB 地址 0 建立相同布局的局部视图，
分别向共享 L1 slot 的前半/后半写入 KV，并分别消费 AIC 下发的 32-head QK/PV
结果。由于 AIC 的一次 ready 必须广播给两个对称 AIV，同步 mode 使用 `2`。

## 3. 三槽时序

一个 Head64 task 包含 `2048 / 128 = 16` 个 selected tile。调度增加两个 drain
round，共执行 18 个 round：

```text
round r     AIV MTE/SIMT                    AIC Tensor
----------------------------------------------------------------
0           gather(0)                       QK(0)
1           gather(1), softmax(0)           QK(1), PV(0)
2           gather(2), softmax(1), update(0) QK(2), PV(1)
...
15          gather(15), softmax(14), update(13) QK(15), PV(14)
16          softmax(15), update(14)         PV(15)
17          update(15), final store         drain
```

slot 由 `tile_index % 3` 决定。复用 slot 前必须同时满足：旧 tile 的 QK 已经消费
KV，且旧 tile 的 PV 已经消费该 slot 前 512 维。填充和排空使用 tile index
有效性判断，不复制单独的首轮/末轮实现。

## 4. L1 布局与容量

L1 由顶层 mixed kernel 申请并传给 AIC/AIV。canonical 布局如下，单位为 byte：

```text
[0,       73728)   packed Query: 64 x 576 x BF16
[73728,  221184)   KV slot 0:   128 x 576 x BF16
[221184, 368640)   KV slot 1:   128 x 576 x BF16
[368640, 516096)   KV slot 2:   128 x 576 x BF16
[516096, 524288)   synchronization/reserved headroom
```

总有效载荷为 `516096 B`，未超过 512 KiB L1，余量为 `8192 B`。因此不允许另行
申请完整 score 或 probability L1 区域。

每个 KV slot 的前 512 维在 QK 后继续作为 PV 的 Value；末尾 64 维区域大小为
`128 x 64 x 2 = 16384 B`，在 QK 完成后复用为 `64 x 128` BF16 probability。
这两个逻辑视图大小完全相同，且生命周期不重叠。

## 5. UB 布局

UB workspace 由顶层 kernel 的动态 UB launch 参数申请。AIV 侧不声明静态
`__ubuf__` 数组，而是把物理地址 `0` 转成 `__ubuf__ uint8_t*`，再按布局计算
offset。两个 AIV 使用同一套相对 offset，但物理 UB 属于各自 subblock。

每个 AIV 的持久区包括 32-head running max、sum、old scale 和 FP32 running
output；阶段临时区由 gather、softmax 和 output update 复用。最终布局为：

```text
[0,      128)     running max
[128,    256)     running sum
[256,    640)     三个 old-scale slot
[640,    66176)   running output，32 x 512 FP32
[66176,  66432)   当前 AIV 的 64 个 KV row offset
[66432,  103296)  Query pack 或 KV gather staging
[103296, 119680)  QK scores，32 x 128 FP32 row-major
[119680, 127872)  probabilities，32 x 128 BF16 NZ
[127872, 160640)  PV result，32 x 256 FP32 row-major
```

动态 UB 总量为 `160640 B`，小于 `216 KiB`。最终常量和源码注释列出每个
`[begin, end)` byte range，并用 `static_assert` 检查容量。

AIC 不持有 AIV UB 指针，也不分别计算两个 subblock 的物理地址。QK/PV 结果通过
`asc_copy_l0c2ub(..., dual_dst_ctl=1)` 直接从 L0C 下发；AIC 只提供物理地址 0 上的
同一个相对 offset，硬件沿 M 维把 64-head 结果拆到两个 AIV 的本地 UB。两个 AIV
随后都按相同布局消费各自的 32-head half。

## 6. 同步规则

CrossCore flag 保持现有 API，模板 mode 统一为 `2`：

- AIC -> 两个 AIV：发布 Query、QK score 和 PV result ready；
- 两个 AIV -> AIC：发布 KV half、probability half 和 output update complete；
- 每个 rolling slot 使用独立事件编号，避免复用中的 ABA 冲突；
- AIC 只有在两个 AIV 都完成对应阶段后才复用 L1 slot；
- AIV 只有在 AIC 确认消费完成后才覆盖本地阶段临时区。

核内 `asc_sync_notify/asc_sync_wait` 继续只解决同一核心流水线之间的可见性，不替代
CrossCore 握手。

## 7. 验证与验收

实现按以下层次验收：

1. 源码契约测试：selected128、三槽、双 AIV 对称、mode 2、顶层 L1/UB、地址 0
   和布局注释均存在；
2. 3510 编译成功，L1/UB/栈/寄存器资源未越界；
3. seed 7 和 seed 19 的 output/LSE 精度通过；
4. BasicInfo 比较完整 Sparse Attention 和 workflow，包含 kernel launch 数变化；
5. InstrTimeline 证明稳态 round 中 gather/QK、softmax/PV、update 存在重叠，而不是
   仅由单阶段时长下降造成表面收益。

设计收益以相同设备、shape、二进制和计时边界下的 A/B 数据为准。若三槽结构
通过精度但没有收益，应保留 profile 证据，并继续从 gather 指令形态、8-AIC
利用率和 L0C 到 AIV UB 路径归因，不用单 kernel 局部时间代替端到端结论。

## 8. 实现中的正确性问题

### 8.1 KV staging 复用需要 MTE3 到 V 依赖

一个 AIV 用同一段 UB staging 依次生成 Query 和三个 K-dimension chunk。UB 写入
L1 后，下一次 SIMT VF 覆盖 staging 前必须等待 MTE3 完成。因此 gather loop 在
每次 UB 到 L1 copy 后增加：

```text
asc_sync_notify(PIPE_MTE3, PIPE_V, EVENT_ID0)
asc_sync_wait(PIPE_MTE3, PIPE_V, EVENT_ID0)
```

这个依赖只约束同一 AIV 内的 staging 生命周期，不承担跨 AIC/AIV 同步。

### 8.2 mixed launch 必须展开 task ratio

dav-3510 上 AIV 的 `GetBlockIdx()` 包含 subblock 维。AIV 需要除以
`GetTaskRatio()` 得到 paired AIC id，同时 host launch grid 必须乘 task ratio：

```cpp
constexpr uint32_t kHead64FusedTaskRatio = 2;

const uint32_t block_id =
    AscendC::GetBlockIdx() / AscendC::GetTaskRatio();

grid = plan->used_core_num * kHead64FusedTaskRatio;
```

只 launch `used_core_num` 会覆盖一半 AIV subblocks。该错误曾表现为一半任务正常、
另一半输出接近 `1/128`，不是 softmax 或 UB 容量问题。

### 8.3 device program cache 会污染 A/B 结果

同名 device program 在当前 CANN 9.2 / dav-3510 环境中可能跨 ELF 和新进程复用。
仅替换 `.so`、修改 global kernel 名甚至回退到历史通过 ELF，都不足以证明新代码
实际执行；导出的 host launcher 名也必须唯一。最终入口使用：

```text
sparse_attention_head64_fused_mix12_restored_kernel
launch_sparse_attention_head64_fused_hd576_bf16_v2_rolling_restored
```

精度证据均来自 fresh process，并核对了 Python import 路径、`_C.so` 路径、
`$ORIGIN` kernel ELF 路径和 SHA-256。

## 9. 精度结果

canonical BF16 decode 使用 CPU FP32/BLAS oracle，容差为 `atol=0.05`、
`rtol=0.05`。最终 restored 版本结果如下：

```text
host _C.so SHA-256:
9d012d1951615c73dd5f691d59c2fe550b85c1834070e906bd58fb0f3ed3134a

kernel ELF SHA-256:
96a4ec5f792e71042ef148d975cde2b700274494de748437aa55e15db632bd77
```

| Case | Output mismatch | Output max abs | LSE mismatch | LSE max abs |
| --- | ---: | ---: | ---: | ---: |
| all ones | 0 / 262144 | 0 | 0 / 512 | 0 |
| seed 7 | 0 / 262144 | 0.00390625 | 0 / 512 | 9.536743e-7 |
| seed 19 | 0 / 262144 | 0.00390625 | 0 / 512 | 9.536743e-7 |

远端原始记录位于：

```text
/root/cannbench-dsa-v2-vllm-rolling-20260806-215750/evidence/
  rolling-final-20260807/restored-unique-launcher-ones.json
  rolling-final-20260807/restored-unique-launcher-seed7.json
  rolling-final-20260807/restored-unique-launcher-seed19.json
```

## 10. 性能结果

Profiler 只使用 application replay：

```text
--replay-mode=application --warm-up=0 --launch-count=1 --kill=on
```

默认 kernel replay 会触发 `RegisterFuncSymbol 107000` 和 stream timeout
`507046`，其结果无效。两轮 BasicInfo 数据如下，单位为微秒：

| Run | P=1 rolling | main P=4 fused | main Combine | main total |
| --- | ---: | ---: | ---: | ---: |
| r1 | 250.322006 | 106.681999 | 9.537000 | 116.218999 |
| r2 | 211.544006 | 85.836998 | 9.432000 | 95.268998 |
| mean | 230.933006 | - | - | 105.743999 |

P=1 rolling 平均约为 main P=4 完整 Sparse Attention 的 `2.18x`，即使移除了
Combine 和 4 MiB partial output，仍显著回退。因此该候选不应作为性能优化合入
main；特性分支保留实现和证据，供后续改变 gather 算法时继续使用。

## 11. Timeline 归因

PipeTimeline 汇总：

```text
Vector union   167.8376 us
Cube            27.4521 us
Vector & Cube    26.2885 us
Cube overlap       95.8%
```

Cube 与 Vector 并非完全串行，Cube 已有较高重叠；关键路径仍是 Vector。按
InstrTimeline 估算，每个 selected128 tile：

```text
三个 gather VF chunk       约 2.69 + 1.88 + 0.80 = 5.37 us
softmax VF                通常约 2.6 us，部分轮次膨胀到 7.7-10.4 us
两次 output update VF     各约 0.91 us
```

本地未跟踪原始 profile 位于：

```text
artifacts/dsa-v2-vllm-rolling-pipeline-20260807/
  candidate-mix12-pipetimeline/
  candidate-mix12-instrtimeline/
```

## 12. 与 vLLM-Ascend 的剩余差异

当前候选只复刻了 P=1、selected128、双 AIV 对称和三槽调度，没有获得 vLLM
路径最关键的两项数据搬运优势：

1. 相邻两个 Head64 group 仍各自从原始 KV 做离散 gather，没有跨 group 共享
   一次 gather；同一 KV tile 仍被重复读取。
2. 当前每个 AIV 用 SIMT lanes 对 64 行、576 维数据做离散 GM load；vLLM 使用
   DataCopy 风格的聚合搬运，指令数和地址生成方式不同。

此外，三槽产生了 16 轮 CrossCore wait 和多个 VF boundary。当前 profile 说明仅靠
调度重排不足以抵消这些开销；后续优化应先改变原始 gather 的复用范围或搬运指令
形态，再复用本分支的 rolling 框架。

InstrTimeline 后曾尝试 E0：为三个 slot 缓存完整 128-row offset，让 softmax 不再
读取 GM indices。该实验把 UB 增到 `161920 B`，但破坏精度，已完整撤销。定位过程
还证明当前设备缓存以 host launcher 为关键标识之一；后续每个真实设备 A/B 都应
使用唯一 launcher，并保留二进制 hash 和导入路径。

## 13. 跨 Head64 组共享 KV gather

### 13.1 重复边界

P=1 rolling 的 task 编号以 `head_group` 为最内层维度，因此相邻两个 task：

```text
logical task 2p      heads [0, 64)
logical task 2p + 1  heads [64, 128)
```

对应同一个 `(batch, query_token)`，并读取完全相同的 indices 和 KV 行。现有实现中
两个 Head64 task 各由两个 AIV gather 64 行，导致每个 selected128 tile 被完整
gather 两次。canonical case 有 4 个 `(batch, query_token)` pair 和 16 个 tile，
完整 tile gather 数量为 `4 x 16 x 2 = 128`；去重后应为 `4 x 16 = 64`。

### 13.2 四 AIV quarter 映射

保留 8 个 AIC，不把两个 Head64 的 Cube 计算串到一个 AIC。相邻两个 AIC 的四个
AIV 共同产生一份 selected128 KV tile：

```text
pair_id = logical_task / 2
quarter = head_group * 2 + subblock_index

quarter 0  Head64-0 AIV0  selected rows [0, 32)
quarter 1  Head64-0 AIV1  selected rows [32, 64)
quarter 2  Head64-1 AIV0  selected rows [64, 96)
quarter 3  Head64-1 AIV1  selected rows [96, 128)
```

每个 producer 仍使用 SIMT VF 从原始 KV 离散 gather 到本地 UB，但只处理 32 行，
再用 C API 将该 quarter 写入共享 GM slot 的 ZN 布局。四个 quarter ready 后，每个
Head64 task 的 AIV0/AIV1 分别用 C API 从共享 GM 读取 `[0, 64)` 和 `[64, 128)`，
写入本 AIC 的 L1 slot。这样 Cube 侧仍只通过现有 mode 2 CrossCore flag 与直属
两个 AIV 交互，不增加跨 AIC 的 Basic API 同步。

### 13.3 共享 GM 布局

沿用 host 已申请的 16 MiB byte workspace。前 512 B 保存计数器，数据区按
`pair -> rolling slot` 排列：

```text
[0, 512)          ready/consumed counters，4 pairs x 3 slots x 2 uint32
[512, 147968)     pair 0 slot 0，128 x 576 BF16
[147968, 295424)  pair 0 slot 1，128 x 576 BF16
[295424, 442880)  pair 0 slot 2，128 x 576 BF16
...               pair 1..3 使用相同的三槽步长
```

数据总量为：

```text
512 + 4 x 3 x 128 x 576 x 2 = 1769984 B
```

小于 16 MiB。共享 GM slot 与本地 L1 slot 使用相同的 `K-block x selected-block`
ZN 物理顺序；quarter 写和 half 读均使用二维 stride copy，不增加 transpose VF。

### 13.4 三槽计数协议

新同步只使用 SIMT API 的 GM atomic 和 thread fence。每个 `(pair, slot)` 有
`ready_count` 与 `consumed_count` 两个单调计数器，host 在 fused kernel launch 前
用 `aclrtMemsetAsync` 清零 512 B counter 区。tile 对某个 slot 的代际为
`generation = tile_index / 3`：

```text
producer 等待 consumed_count == 4 * generation
4 个 producer 写完各自 quarter 后 ready_count += 1
consumer 等待 ready_count == 4 * (generation + 1)
4 个 consumer 完成本地 GM -> UB -> L1 后 consumed_count += 1
```

发布计数前必须保证数据写完成；观察 ready 后才允许读取共享 slot；发布 consumed
前必须保证本地 L1 写完成。计数器不回绕、不重置 slot，因此避免同一 flag ID 在
rolling 复用中的 ABA 问题。现有 AIC/AIV 内部 CrossCore mode 2 协议保持不变。

### 13.5 验证门槛

首先用源码契约测试固定 pair/quarter 映射、三槽 workspace、host memset、SIMT
atomic/fence 和二维 stride copy；随后在 dav-3510 fresh process 中验证 all ones、
seed 7、seed 19。精度通过后才采 BasicInfo、PipeTimeline 和 InstrTimeline。

性能验收比较完整同步边界，包含 host memset。主要判据是 P=1 rolling fused kernel
相对 `230.933 us` 是否下降，以及完整 Sparse Attention 是否接近或优于 main 的
`105.744 us`；同时核对 InstrTimeline 中离散 gather VF 总次数是否从 128 个完整
tile 等价工作量下降到 64 个。
