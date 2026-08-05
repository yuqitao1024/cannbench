# DSA V2 Sparse Attention 与 vLLM-Ascend 实现对比

## 1. 结论

当前 DeepSeek V3.2 decode shape 下，SIMT V2 与 vLLM-Ascend 的 workflow
性能差距集中在 Sparse Attention，而不是 Lightning Indexer。

vLLM-Ascend 最值得借鉴的不是某一条 MMAD 或转置指令，而是以下完整数据流：

1. 以 `64` 个 Query heads 和 `128` 个 selected tokens 为基本计算块。
2. 一次 gather 完整的 `576` 维 KV，QK 使用全部 `576` 维，PV 复用前
   `512` 维。
3. 使用三个 rolling slot，把不同 selected tile 的 KV gather、QK、online
   softmax、PV 和 output update 交错执行。
4. `G=128` 时让相邻两个 Head64 AIC group 共享一次原始 KV gather。
5. 不沿 selected 维拆分计算任务，直接写最终 output/LSE，避免大块 partial
   output 和 Combine。

当前 SIMT V2 已经实现 Head64、Cube QK、Cube PV、selected256 外层 tile、
128-row 计算 subtile、KV row offset 复用和 L1 到 L0B 随路转置。后续不应继续把
主要精力放在孤立的 MMAD 或 transpose 微调上。现有 InstrTimeline 中 Cube 路径约
`6.9 us`，Vector critical path 约 `84-86 us`，说明主要剩余空间在离散 gather、
跨阶段流水、metadata 复用和 partial/combine 边界。

本文中的源码结构为已核对事实；性能收益方向仍是假设，必须通过同一设备、同一
shape、同一计时边界下的独立 A/B 验证。

## 2. 对比边界

对比 shape：

```text
device: Ascend 950PR
phase: decode
dtype: BF16
B=2, Q=2, Hq=128, Hkv=1
context=32768, selected=2048
Dqk=576, Dv=512
causal=true, return LSE=true
```

CannBench 的 vLLM-Ascend adapter 使用 TND Query/KV、token-wise sparse
indices，并设置 `return_softmax_lse=True`。因此这里的优势不是来自省略 LSE、
使用 FP8 或换用不同 phase。

相关 adapter：

- `src/cannbench/operators/builtin/sparse_attention/external.py`

## 3. 时间归因

当前采集和发布数据如下，单位为微秒：

| 边界 | SIMT V2 | vLLM-Ascend | SIMT 相对差值 |
| --- | ---: | ---: | ---: |
| Lightning Indexer | 92.576 | 97.674 | -5.098 |
| Sparse Attention | 116.271 | 74.161 | +42.110 |
| Published workflow | 208.847 | 169.797 | +39.050 |

注意：vLLM-Ascend 的 component profile 和 published workflow 来自不同采集，
`97.674 + 74.161 = 171.835 us`，不能用它们重新计算 published workflow。
但两组数据都支持同一结论：SIMT Indexer 已略快，workflow 剩余差距来自 Sparse
Attention。

SIMT Sparse Attention 的最近拆分为：

```text
fused main kernel 约 104.010 us
Combine/other     约  12.261 us
total             约 116.271 us
```

详细指标 profile 归档位于本地未跟踪目录：

```text
artifacts/v32-decode-profile-20260805/
  v32-decode-profile-all-metrics-instr-20260805.tar.zst
```

归档信息：

```text
size: 149284193 bytes
sha256: 7a0da566bfb34655c381076bd84731140ed06dc1cc70109e8c0c8d52dfe89f3a
```

Published workflow 记录：

- `published/opbench-ascend-950pr-simt-v2-dsa_decode-realistic-bfloat16/`
- `published/opbench-ascend-950pr-vllm-ascend-dsa_decode-realistic-bfloat16/`

## 4. vLLM-Ascend 的整体实现

本节核对的 vLLM-Ascend 源码版本为：

```text
a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca
```

关键源码：

- [sparse_flash_attention_tiling.cpp](https://github.com/vllm-project/vllm-ascend/blob/a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca/csrc/attention/sparse_flash_attention/op_host/sparse_flash_attention_tiling.cpp)
- [sparse_flash_attention_kernel_mla.h](https://github.com/vllm-project/vllm-ascend/blob/a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca/csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_kernel_mla.h)
- [sparse_flash_attention_service_cube_mla.h](https://github.com/vllm-project/vllm-ascend/blob/a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca/csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_cube_mla.h)
- [sparse_flash_attention_service_vector_mla.h](https://github.com/vllm-project/vllm-ascend/blob/a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca/csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h)

### 4.1 基本 tile 和有效核数

Arch35 MLA 路径使用：

```text
M tile = 64 query heads
S tile = 128 selected tokens
K      = 512 NoPE + 64 RoPE = 576
```

当前 shape 中：

```text
base tasks = B * Q * ceil(H / 64)
           = 2 * 2 * 2
           = 8 AIC
```

vLLM-Ascend 不为了占满 32 个 AIC 而沿 selected 维继续拆分。每个有效 AIC
顺序处理 `2048 / 128 = 16` 个 selected tile，以较长的单核流水摊薄固定开销。

### 4.2 三槽 rolling pipeline

Kernel 保存三个 `RunInfo`，在同一轮调度中交错执行：

```text
tile t:     Vec0 gather KV + BMM1 QK
tile t-1:   Vec1 online softmax + BMM2 PV
tile t-2:   Vec2 online output update/final store
```

这不是先完成一个 tile 的全部阶段再处理下一个 tile。三个 slot 让 MTE、Vector
和 Cube 更长时间保持并行，是 vLLM-Ascend 与当前 SIMT V2 最重要的调度差异。

### 4.3 KV gather 和跨 Head64 group 复用

Vec0 通过双 UB slot，每次最多聚合 16 行 KV。单行的 512 维 NoPE 和 64 维
RoPE 使用 DataCopy 类接口搬运，而不是让 SIMT lane 对 GM 做完整的逐元素离散
load。

`G=128` 时，相邻两个 AIC 分别计算 heads `0..63` 和 `64..127`。对应的四个
AIV 合作 gather 一个 128-row KV tile，将聚合结果写到两个 AIC 共享的 GM
workspace。两个 AIC 再分别把连续 workspace 搬到 L1。

这个设计没有消灭所有 GM 流量，而是把昂贵的重复随机读取转换为：

```text
一次原始 KV 离散 gather
+ 一次连续 workspace 写入
+ 两个 Head64 group 的连续 workspace 读取
```

对当前 shape 做源代码级流量估算：

```text
当前 SIMT 原始 KV 读取：
  Key:   4(BQ) * 2(groups) * 2048 * 576 * 2 bytes = 18 MiB
  Value: 4(BQ) * 2(groups) * 2048 * 512 * 2 bytes = 16 MiB
  total:                                                    34 MiB

vLLM 原始离散 KV 读取：
  4(BQ) * 2048 * 576 * 2 bytes = 9 MiB
```

vLLM 仍有 workspace 的连续读写，因此不能只根据总字节数预测性能。真正差异是
原始随机读取量、搬运指令形态，以及 gather 与 Cube/Vector 计算能否重叠。

### 4.4 QK、softmax、PV 和写出

- BMM1 计算 `[64, 576] x [576, 128]`。
- BMM1 的 L0C 结果通过 Fixpipe 直接送到两个 AIV 的 UB。
- Vec1 做 online softmax，并把 BF16 probability 直接写到 BMM2 使用的 L1。
- BMM2 计算 `[64, 128] x [128, 512]`。
- BMM2 的 L0C 结果同样通过 Fixpipe 直接送 UB。
- Vec2 在线更新 running output，最后一次迭代直接写最终 output/LSE。

因此 vLLM-Ascend 没有完整 scores/probabilities workspace，也没有跨 selected
partition 的 partial output 和 Combine。

## 5. 当前 SIMT V2 的实现与差异

当前 canonical V2 decode 路径定义：

```text
head tile              = 64
selected outer tile    = 256
selected compute tile  = 128
QK tile                = 256
Value tile             = 256
selected partitions    = 4
logical tasks          = 32
```

源码：

- `src/cannbench/operators/builtin/sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- `src/cannbench/operators/builtin/sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/sparse_attention.asc`

当前实现已经具备以下优化，后续分析不能再以旧 V1 的逐 head 实现作为比较对象：

- Head64 Cube QK 和 Cube PV。
- selected256 outer tile 和 128-row compute subtile。
- QK/PV 更大的 K/N tile。
- 32 个 active warps。
- 首次 Key pack 生成 UB `kv_row_offsets`，后续 Key 和 Value pack 复用。
- 后续 Key pack 使用 BF16 paired load/store。
- PV 路径使用 L1 到 L0B 的随路 ND/ZN 转换。

### 5.1 P=4 partial output 和 Combine

decode 自动选择 `selected_partitions=4`：

```text
task_count = 2 * 2 * 2 * 4 = 32
```

Fused kernel 为每个 partition 写 FP32 partial output：

```text
32 tasks * 64 heads * 512 dims * 4 bytes = 4 MiB
partial LSE = 32 * 64 * 4 bytes           = 8 KiB
```

Combine 随后读取约 4 MiB partial output，做稳定 log-sum-exp 合并，并写约
1 MiB FP32 final output。Combine launch 本身约 `12 us`，但 P=4 同时把有效 AIC
任务数从 8 增加到 32，因此不能断言改为 P=1 一定净赚 12 us。

### 5.2 当前流水仍以阶段握手为主

当前路径在一个 selected outer tile 内已经有 Key/Value gather slot，但主要顺序仍是：

```text
完成 QK gather/MMAD
-> 搬出 scores
-> softmax
-> 完成 Value gather/PV
-> output update
-> 下一个 outer tile
```

各阶段存在 AIC/AIV ready/wait 握手。它没有像 vLLM-Ascend 一样，用三个
selected tile slot 同时承载 `gather/QK`、`softmax/PV` 和 `output update`。

### 5.3 metadata 只被部分复用

Canonical Key pack 会读取 indices 并生成 UB `kv_row_offsets`，后续 Key 和 Value
pack 已经复用该 offset。但是 softmax 仍在 max 和 exp 两遍循环中直接读取 GM
indices，并且该读取位于逐 head 循环内。

因此“缓存 indices/offset”不是一个全新的优化，而是现有优化尚未覆盖 softmax
validity/causal mask 的剩余边界。

### 5.4 KV 数据仍为 QK/PV 分别 gather

虽然 Key 和 Value 使用同一份 `shared_kv`，当前实现仍按不同目标布局分别 gather：

- QK 把 KV pack 成 `K x selected` 的 ZN operand。
- PV 再把前 512 维 pack 成 `selected x Dv` operand。

vLLM-Ascend 则先得到完整的 128x576 KV tile，再由同一份 tile 同时服务 QK 和
PV。要借鉴这一点，需要同时解决 L1 生命周期和 QK/PV 不同消费布局，不能简单地
删除一次 gather 调用。

### 5.5 L0C 结果路径

当前 QK/PV 结果路径为：

```text
L0C -> L1 -> 两个 AIV UB
```

vLLM-Ascend 使用带 dual-destination 的 Fixpipe 直接执行：

```text
L0C -> 两个 AIV UB
```

这可能减少一次中转和同步，但 vLLM 源码使用 Basic API。CannBench 的新增设计
必须遵守 `C API + Tensor API + SIMT API` 边界，所以只能借鉴数据流，不能直接
复制 `LocalTensor`、`Fixpipe`、`SetFlag/WaitFlag` 或新的跨核 Basic API 用法。

## 6. 建议的验证顺序

### 6.1 E0：让 softmax 复用已有 metadata

保持 P=4、Head64、selected256 和现有 launch geometry 不变，仅让 softmax 使用
UB 中已有的 row offset、validity 或 causal mask，避免逐 head 的两遍 GM indices
读取。

该实验改动边界最小，适合先验证 Vector critical path 是否对 metadata load 敏感。

### 6.2 E1：单 Head64 group 内复用完整 KV tile

以 selected128 为受控 tile，尝试一次 gather 完整 576 维 KV，并让同一个 Head64
group 的 QK 和 PV 都消费该 tile。

需要先完成 L1/UB/L0 容量 worksheet，并明确：

- KV tile 的生产布局；
- QK `K x S` 和 PV `S x Dv` 的转换位置；
- probability 和 running output 的同时存活范围；
- 转换成本是否仍在 changed kernel boundary 内。

### 6.3 E2：P=1 + 三槽 rolling pipeline + direct output

目标执行图：

```text
8 Head64 tasks
16 selected128 tiles per task
3-slot gather/QK -> softmax/PV -> output update
final BF16 output/LSE direct store
```

这是最接近 vLLM-Ascend 的核心实验。P=1 单独测试可用于量化并行度损失，但不能
代表该方案的最终性能；vLLM 的 8-AIC 路径依赖长流水来保持执行单元繁忙。

### 6.4 E3：跨两个 Head64 group 共享 KV gather

只有在 E1/E2 仍显示 gather 是主要瓶颈时，再实现跨 group 共享。

vLLM-Ascend 使用共享 GM workspace 和跨核同步。当前仓库不允许为新设计引入
Basic API 跨核 flags，因此候选实现必须先证明可由允许的 API 表达。若需要额外
gather kernel，则必须把约 `5 us` launch 开销、workspace 写读和完整 operator
时间一起计入，不能只比较 attention main kernel。

### 6.5 E4：确认允许 API 下的 direct L0C-to-UB 能力

单独确认 C API 或 Tensor API 是否支持 vLLM 所需的：

- L0C 到 UB；
- row-major/NZ 目标布局；
- 双 AIV destination 或等价分发。

如果接口不满足，就保留当前 `L0C -> L1 -> UB` 路径，不应为了复制 vLLM 源码
重新引入 Basic API。

## 7. 验收要求

每个实验只改变一个主要因素，并保留当前 P=4 实现作为对照。至少记录：

1. 相同 BF16 decode shape 和 output/LSE 语义。
2. 生产 `-O3` ELF、加载路径和二进制 SHA-256。
3. accuracy、资源使用和 Stack/register 信息。
4. 两组交替、clean-process BasicInfo 数据。
5. fused main、Combine、完整 Sparse Attention 和 workflow 时间。
6. Default/InstrTimeline 只用于归因，不与 BasicInfo 时间直接混比。
7. raw profile 路径、kernel launch 数和排除的 materialization kernel。

只有 correctness、重复性能和同栈 baseline 同时通过，才能认定某个 vLLM-Ascend
方向已在 SIMT V2 上产生收益。
