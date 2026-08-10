# DSA V2 Sparse Attention 六项差距实验

## 1. 目标

本文逐项验证 vLLM-Ascend 对比图中的六类实现差距。实验初始基线为
`main@7dbe8a2`，已经保留逻辑 PV512 handoff；E2 完成后分支 rebase 到
`origin/main@22ba333`，该 upstream 已包含同一 lane-local softmax 优化。canonical
case 固定为：

```text
Ascend 950PR, CANN 9.2, BF16
B=2, Q=2, Hq=128, Hkv=1
context=32768, selected=2048, Dqk=576, Dv=512
causal=true, return_lse=true
```

所有改动仅作用于 V2 canonical decode rolling kernel，不修改 operator plugin、
公共 backend、CLI、数据集或发布 schema。

## 2. 当前完成度

| 图中项目 | 当前状态 | 本轮处理 |
| --- | --- | --- |
| Softmax metadata | E1 精度通过但两轮性能回退，拒绝 | 保持 softmax 读取 GM indices |
| PV/Vec2 粒度 | 已完成 | 保持两个 PV256 MMAD，一次 PV512 handoff/VF update |
| Query movement | E3 已完成并保留 | AIC 直接 GM-to-L1 ND2NZ |
| Vector 实现质量 | E2 已完成并进入 upstream | lane-local score/validity 寄存器复用 |
| Cube 内部流水 | E4 设备运行时失败，拒绝 | 保持 QK256 单 staging |
| Sync/buffer 管理 | 部分完成 | E5：PV free 从 PIPE_V 直接发布 |

E1 至 E5 按顺序执行。每个候选只从上一个已接受 checkpoint 开始；候选失败时，
恢复源码到该 checkpoint，再开始下一项。这样最终代码只包含逐项可归因且组合后仍
通过性能门槛的改动。

## 3. E1：Gather-produced Validity Mask

四个 gather producer 已分别读取 selected128 tile 的 32 个 indices。每个 producer
在读取 token 时同步构造一个 `uint32_t` validity mask，并把四个 mask 写入该
`pair/slot/generation` 对应的 GM metadata。现有 gather barrier 和 slot-ready 协议
保证 softmax 读取前四个 mask 都已发布。

softmax 的每个 warp 只读取四个 mask，并以 lane bit 判断自己负责的四个 selected
位置；max 和 exp 两遍复用相同的寄存器 mask。该方案不增加 UB、不增加
`__syncthreads()`，并移除 softmax 中逐元素的 indices load、范围比较和 causal 比较。

metadata 必须位于现有 16 MiB operator workspace 内，与三个 rolling KV slot 分开
计算 offset。mask 的 slot/generation 生命周期与 KV slot 相同，不引入新同步协议。

### 3.1 实验结果：拒绝

第一次实现由 gather 直接执行 4-byte GM 标量 store。远端编译成功，但 seed 7 的
output mismatch 为 `247845 / 262144`，LSE mismatch 为 `512 / 512`，且 LSE 出现
`inf/-inf`。证据表明该标量 store 没有被现有 MTE3 slot-ready 边界正确发布，softmax
观察到全零 mask。

修正实验把每个 mask 放入复用的 KV gather UB，并使用 32-byte 对齐的 Tensor API
`DataCopy` 发布；每个 slot 的四个 mask 占 128 bytes。该版本在 Ascend 950PR、
CANN 9.2 上通过两个独立进程的完整 output/LSE 精度：

| seed | output mismatch | LSE mismatch | max output abs error | max LSE abs error |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0 / 262144 | 0 / 512 | 0.01806640625 | 0.02170562744140625 |
| 19 | 0 / 262144 | 0 / 512 | 0.0185546875 | 0.025426864624023438 |

但相对同一 E2 lane-cache checkpoint，连续两轮 `BasicInfo` 均回退：

| 轮次 | E2 baseline | E1 aligned mask | 变化 |
| ---: | ---: | ---: | ---: |
| 1 | 81.133 us | 83.099 us | +2.42% |
| 2 | 81.343 us | 82.612 us | +1.56% |

两轮目标 kernel 的 Block/Mix Block Dim 均为 `16/32`，当前/额定频率均为
`1650/1650 MHz`，failure count 为 0。E1 不满足“两轮均不劣化”门槛，因此撤销
mask metadata 源码和契约测试，只保留本节证据。

对齐版本 provenance：

```text
source SHA256:     bd174c6473dd611894e72703eb54ec08eea450e298872731358fb800e5ea1a27
_C.so SHA256:      64406638e6fe50ccb0465a2633f42fbb0113eb04513912359a0a5a87ab41f820
kernel ELF SHA256: 8867f972504ec4c5598ebc2169cb05a7cbdf8990c139145642c69baaffb71327
```

证据路径：

```text
/tmp/cannbench-dsa-v2-gap-pWg42i/six-gap-e1-aligned-build.log
/tmp/cannbench-dsa-v2-gap-pWg42i/six-gap-e1-aligned-accuracy-seed7.json
/tmp/cannbench-dsa-v2-gap-pWg42i/six-gap-e1-aligned-accuracy-seed19.json
/tmp/cannbench-dsa-six-gap-e1-aligned-r1-20260810/
/tmp/cannbench-dsa-six-gap-e1-aligned-r2-20260810/
```

## 4. E2：Lane-local Score Cache

在 PV512 checkpoint 上，每个 softmax lane 在 max pass 中读取并缩放自己负责的最多
四个 score，将其保存在固定大小的局部标量中。exp pass 复用这些标量，不再第二次
读取 UB score 或重复乘以 scale。

不使用动态数组索引，不手工展开整个 head 循环。编译后必须检查 Stack 和寄存器
元数据；若发生 spill、资源异常或性能回退，完整撤销 E2，回到 PV512 checkpoint。

E2 的 seed 7/19 output/LSE mismatch 均为 0，两轮 `BasicInfo` 从 PV512 的
`90.723/91.131 us` 降至 `81.133/81.343 us`，两轮均值提升约 10.66%。本 worktree
原提交在 rebase 时因 patch 已由 upstream `22ba333` 包含而自动去重；最终源码仍保留
该优化。完整 provenance 和 artifacts 见
`dsa-v2-sparse-attention-remaining-gap-experiments.zh-CN.md` 第 8.6 节。

## 5. E3：AIC Direct Query GM-to-L1

canonical Query 的 GM 布局为 `[B, H, Q, 576]`。AIC 从对应
`(batch, head_group, query_token)` 起始地址执行 ND2NZ：

```text
nValue   = 64
dValue   = 576
srcDValue = Q * 576
```

直接生成当前 `64 x 576` L1 NZ Query，删除 canonical rolling AIV 的 Query pack VF、
UB-to-L1 copy、Query-ready CrossCore handoff。通用路径保持不变。

本候选复用源码已经存在的 GM-to-L1 ND2NZ 数据搬运设施，不增加新的 include 或新的
API 家族。若工具链不支持该 source stride，或者实际布局不正确，候选在编译/精度门槛
处终止并撤销。

### 5.1 实验结果：保留

AIC 使用 Query GM 起始地址和 `srcDValue=query_tokens * 576` 直接生成
`64 x 576 BF16` NZ L1 Query。canonical rolling AIV 不再执行 Query pack VF、
UB-to-L1 copy 或 Query-ready CrossCore handoff；通用路径未修改。

远端 clean build 成功，seed 7/19 都覆盖 negative、out-of-range、causal-future 和
完整 16 tiles，output/LSE mismatch 均为 0：

| seed | output mismatch | LSE mismatch | max output abs error | max LSE abs error |
| ---: | ---: | ---: | ---: | ---: |
| 7 | 0 / 262144 | 0 / 512 | 0.0078125 | 0.009283065795898438 |
| 19 | 0 / 262144 | 0 / 512 | 0.009765625 | 0.009229660034179688 |

两轮独立 CannBench framework `BasicInfo` 均优于 rebase 后的 upstream lane-cache
checkpoint：

| 轮次 | upstream lane-cache | E3 direct Query | 变化 |
| ---: | ---: | ---: | ---: |
| 1 | 81.133 us | 76.711 us | -5.45% |
| 2 | 81.343 us | 76.975 us | -5.37% |

两轮目标 kernel 的 Block/Mix Block Dim 均为 `16/32`，当前/额定频率均为
`1650/1650 MHz`，failure count 为 0。E3 满足精度和两轮性能门槛，予以保留并作为
E4 的新 checkpoint。

E3 provenance：

```text
source SHA256:     9184b4c05bfcc244c5d77f56d183f0ef1d172a53bd10b3256fd0b3ed9f13a818
_C.so SHA256:      64406638e6fe50ccb0465a2633f42fbb0113eb04513912359a0a5a87ab41f820
kernel ELF SHA256: b9a5c761112db013ce095389bb9d420ced26d24258ed02e41d9eb6d4f5211be4
```

证据路径：

```text
/tmp/cannbench-dsa-v2-gap-pWg42i/six-gap-e3-direct-query-build.log
/tmp/cannbench-dsa-v2-gap-pWg42i/six-gap-e3-direct-query-accuracy-seed7.json
/tmp/cannbench-dsa-v2-gap-pWg42i/six-gap-e3-direct-query-accuracy-seed19.json
/tmp/cannbench-dsa-six-gap-e3-direct-query-r1-20260810/
/tmp/cannbench-dsa-six-gap-e3-direct-query-r2-20260810/
```

## 6. E4：QK128 Double Staging

当前 QK576 使用一个 QK256 L0A/L0B staging，按 `256 + 256 + 64` 串行复用。
E4 将 canonical rolling QK staging 改为两个 QK128 slot，按
`128 + 128 + 128 + 128 + 64` 执行。两个 slot 分别持有 L0A Query 与 L0B Key，
使 MTE1 可以准备下一段，而 Matrix pipe 消费当前段。

L0A 总量为 `2 x 64 x 128 x BF16 = 32 KiB`，L0B 总量为
`2 x 128 x 128 x BF16 = 64 KiB`。L0C score 保持 `64 x 128 FP32`，不改变
softmax 数据布局。每个 slot 有独立 MTE1-to-M 和 M-to-MTE1 event，首段初始化
L0C，后续段累加。

该候选可能因 MMAD 次数从 3 增至 5 而回退，因此只在真实 profile 中接受，不依据
理论 overlap 或资源可行性保留。

### 6.1 实验结果：拒绝

E4 使用两个独立 L0A/L0B slot 和两个 MTE1-to-M/M-to-MTE1 event，L0 总容量与
QK256 单 staging 相同。operator-local 源码契约测试 `32 passed`，远端 clean build
成功，但 canonical seed 7 在第一次 `torch.npu.synchronize()` 返回设备错误
`507015`，尚未进入 output/LSE 比较，因此没有采集性能数据。

运行日志只确认 kernel 触发异步设备异常，没有提供可归因到单条 event 的 trap 地址。
同一加载路径上的 E3 binary 在此前 seed 7/19 均通过，所以将失败边界限定在 E4 新增
的双 staging/event 协议；不在该候选上继续叠加同步修补。按“设备运行异常直接拒绝”
门槛，撤销 E4 源码和契约测试，恢复 QK256 `256+256+64` 单 staging。

E4 provenance：

```text
source SHA256:     80a9dfaff8843b4cebda400584b37916320ddb7fa68ed0b1fca64ab4ee18d7ff
_C.so SHA256:      64406638e6fe50ccb0465a2633f42fbb0113eb04513912359a0a5a87ab41f820
kernel ELF SHA256: 03cd175a37729bc0f49cdb25eac2c3eb49beac992096b65d872f654d58aa5b1e
```

证据路径：

```text
/tmp/cannbench-dsa-v2-gap-pWg42i/six-gap-e4-qk128-double-stage-build.log
/root/ascend/log/run/plog/plog-595235_20260810221533579.log
/root/ascend/log/run/device-0/device-595235_20260810221541079.log
```

## 7. E5：PV-free 同步收敛

当前 AIV 完成 output update VF 后，先执行 V-to-MTE3 本地 event，再从 PIPE_MTE3
发布 PV-free CrossCore flag。E5 改为从 PIPE_V 直接发布 PV-free，使 flag 自身承担
“VF 已完成读取 PV UB”的排序边界，删除该处本地 notify/wait。

Score-free、probability-ready、Query/KV 搬运以及 rolling slot 复用边界不变。
若 `PIPE_V` CrossCore 发布不受当前工具链支持、出现 hang/精度问题或没有性能收益，
恢复原协议。

## 8. 验证和保留门槛

每个候选依次执行：

1. operator-local 源码契约测试先红后绿；
2. 生产参数远端干净编译，并记录 `_C.so`、kernel ELF 路径和 SHA-256；
3. all-ones、seed 7、seed 19 完整 output/LSE 精度；
4. 两组 fresh-process CannBench framework BasicInfo A/B，每组
   `launch-count=10`，显式参数只含 `BasicInfo` 和 launch count；
5. 审计目标 kernel、launch 数、Block/Mix Block Dim、频率和 failure count；
6. 必要时另采 InstrTimeline 解释结果，但不与 BasicInfo latency 混比。

接受条件是精度完整通过，并且两组 candidate latency 都不高于同组 baseline。
收益落在抖动范围但两组均不劣化时可以保留；任一组明确回退则撤销源码，只把负面
结果和 raw artifact 路径写回本文。

所有 E1-E5 完成后，对最终叠加版本重新执行两组 A/B。若组合版本相对
`main@7dbe8a2` 回退，则按最近一次接受改动的逆序剥离，直到恢复不劣化。

## 9. 发布边界

只有最终叠加版本通过 operator-local 测试、完整 `pytest -q`、两轮 Sparse Attention
和 DSA decode workflow 性能门槛后，才更新现有 canonical published record。run id、
schema 和前端字段保持不变，workflow 数值使用同一次验证中可审计的 retained
Lightning Indexer 与 Sparse Attention component 数据。
