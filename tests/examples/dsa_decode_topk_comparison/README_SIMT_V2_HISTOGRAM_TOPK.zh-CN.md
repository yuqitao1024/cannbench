# SIMT v2 基于直方图的 Decode TopK

本文说明 `dsa_decode_topk_comparison` 中 SIMT v2 如何在 DSA v3.2 decode
固定 shape 上使用两级 radix 直方图完成 TopK，并记录跨 AIV 聚合、缓存可见性和
row-global GM atomic 实验的边界。目标读者是需要定位正确性、同步或性能问题的算子
开发同事。

## 1. 固定问题与输出契约

当前 example 只处理一个固定 case：

```text
scores: BF16 [4, 32768]
topk: 2048
context shards per row: 16
score elements per shard: 2048
physical AIV blocks: 4 * 16 = 64
threads per shard block: 1024
```

输出是每行 2,048 个 index。example 校验以下语义：

- 每个 index 都在 `[0, 32768)` 内；
- index 不重复；
- 所有已选 score 都大于等于第 2,048 大的 score；
- 大于阈值的元素必须全部选中；
- 等于阈值的元素只补足到 2,048 个。

输出 index 不要求按 score 排序；当阈值处有并列值时，也不要求选中固定的一组并列
index。因此这里实现的是 TopK score set，不是 sorted TopK。

## 2. 算法总览

完整流程如下：

```text
BF16 score
  -> 转换为保持数值顺序的 16-bit key
  -> 统计 key 高 8 位的 256-bin 直方图
  -> inclusive suffix，定位阈值所在的高位桶
  -> 只对目标高位桶统计低 8 位的 256-bin 直方图
  -> inclusive suffix，定位完整 16-bit 阈值
  -> 输出所有 key > threshold 的 index
  -> 从 key == threshold 的 index 中补足到 2048 个
```

该路径不对 32,768 个 score 做全排序。它以多次线性扫描和两个固定大小的 256-bin
统计替代 `O(N log N)` 排序。

## 3. BF16 sortable key

BF16 原始 bit pattern 不能直接按无符号整数比较，因为负数区间的编码顺序与数值顺序
相反。SIMT v2 先执行：

```cpp
key = (bits & 0x8000U) != 0U ? ~bits : bits ^ 0x8000U;
```

在当前输入约束下，转换后满足：

```text
score_a > score_b  <=>  key_a > key_b
```

随后将 key 拆成两个 byte：

```text
high = key >> 8
low  = key & 0xff
```

每个 byte 的取值范围都是 `[0, 255]`，所以每一级只需要 256 个 bin。`256-bin`
表示 256 个计数桶，不表示只能处理 256 个元素。

## 4. Inclusive suffix 如何定位第 K 大

对一个普通直方图 `histogram[b]`，inclusive suffix 定义为：

```text
suffix[b] = histogram[b] + histogram[b + 1] + ... + histogram[255]
```

因此：

```text
suffix[b]     = key >= b 的元素数量
suffix[b + 1] = key >  b 的元素数量
```

第 K 大元素所在的桶满足：

```text
suffix[b] >= K
suffix[b + 1] < K
```

高位阶段用 `K=2048` 找到 `selected_high`。如果严格高于该高位桶的元素数为
`high_greater`，低位阶段只需要在目标高位桶内找第：

```text
remaining_rank = 2048 - high_greater
```

大的低位 byte。最终阈值为：

```text
threshold_key = (selected_high << 8) | selected_low
```

本 example 的确定性输入实测每行阈值 score 为 `112.0`：

```text
strictly greater: 2008
threshold-equal needed: 2048 - 2008 = 40
```

所以 compact 必须输出全部 2,008 个大于阈值的 index，再输出 40 个等于阈值的
index。

## 5. Retained SIMT v2 的跨核聚合

本节描述 row-global 实验之前的 retained baseline，即 commit
`2e0e393b491fcf931aa348d262a40ee988fe3a4f` 所使用的数据流。

### 5.1 每个 shard 在 UB 统计

每个 block 先用一次 MTE GM-to-UB copy 搬入自己的 2,048 个 BF16 score。1024 个
SIMT 线程各处理两个元素，并通过 UB atomic add 更新一份 block-local
`uint32_t histogram[256]`。

前 256 个线程随后对 256 个 bin 做 inclusive suffix。suffix 先在 32-thread warp
内通过 `asc_shfl_down` 完成，再通过 8 个 warp boundary count 补上更高 warp 的
计数。

### 5.2 每个 shard 使用独立 GM slice

baseline 不让 16 个 shard 竞争同一份 GM histogram，而是使用：

```text
high_histogram[row][shard][256]
low_histogram [row][shard][256]
```

单份 high 或 low workspace 大小为：

```text
4 rows * 16 shards * 256 bins * 4 B = 64 KiB
```

各 shard 写互不重叠的 1 KiB GM slice。写完后先发布 cache，再进入全核屏障。

### 5.3 Row reducer 聚合 16 份 suffix

每行由一个 reducer block 负责。它通过 MTE 将该行的：

```text
16 shards * 256 bins * 4 B = 16 KiB
```

搬入本核 UB。256 个 reducer 线程各负责一个 bin：

```text
row_suffix[b] = sum(shard_suffix[s][b], s=0..15)
```

suffix 求和是线性的，所以 16 份 shard suffix 的逐 bin 和正好等于整行 suffix。
高位和低位各执行一次这种 producer/reducer 过程。

### 5.4 Per-shard offset 与 compact

完整阈值确定后，low reducer 从每份 shard histogram 中得到该 shard 的
`greater_count` 和 `equal_count`，再对 16 个 shard 做 exclusive prefix sum：

```text
greater_count:   10, 7, 12, 5
greater_offset:   0,10, 17,29
```

每个 shard 因而拥有互不重叠的输出区间。block 内部使用 ballot、popcount 和 UB
atomic reservation 为 warp 分配连续 slot；跨 block 不需要 GM atomic。

## 6. MTE、GM store、DCCI 与 SyncAll

MTE 是搬运引擎，不是存储空间。baseline 的主要路径是：

| 数据边界 | 实现 |
| --- | --- |
| score GM -> shard UB | MTE `asc_copy_gm2ub_align` |
| score UB -> local histogram UB | SIMT UB atomic add |
| local suffix -> per-shard GM histogram | SIMT 普通 GM store |
| per-shard GM histogram -> reducer UB | MTE `asc_copy_gm2ub_align` |
| reducer state/offset -> GM | SIMT 普通 GM store |
| compact index -> GM output | SIMT 普通 GM store |

SIMT 普通 GM store 可能先进入本核 DCache。下游消费者可能运行在其他 AIV，或者
通过 MTE 读取这段 GM，因此 baseline 使用：

```text
asc_syncthreads()
-> thread 0 执行 DCCI
-> 所有物理 block 执行 SyncAll
-> 下游读取
```

两者职责不同：

- DCCI 发布当前核 cache 中的 GM 写；
- `SyncAll` 让所有物理 block 在阶段边界汇合。

不能仅凭 `SyncAll` 推断其他核或 MTE 已能看到尚未发布的普通 GM store。反过来，
DCCI 也不能代替所有 block 必须到达的控制流屏障。

## 7. 为什么 baseline 的 L2 read hit 较高

baseline 将 high/low per-shard histogram 写到 GM 后，很快又由 row reducer 读回。
这些中间数据通常仍驻留在 L2，因此 profiler 会看到较高的 L2 read hit。

本次 `Default` profile 的加权原始计数为：

```text
vLLM-Ascend: 2 / 2052 = 0.097%
SIMT v2:     1072 / 3452 = 31.05%
```

按 128-byte cache line 估算，两级 per-shard histogram 的读回量为：

```text
4 rows * 2 stages * 16 shards * 256 bins * 4 B
= 128 KiB
= 1024 cache lines
```

这与 SIMT v2 的 1,072 次 read hit 很接近。高命中主要来自额外的 GM 中间数据
往返，不表示 score 读取更高效，也不等价于 kernel 更快。

## 8. 已拒绝的 row-global GM atomic 实验

本次实验候选曾将 workspace 改为：

```text
high_histogram[row][256]
low_histogram [row][256]
```

每个 shard 仍在 UB 形成自己的 suffix，但前 256 个线程直接对 row-global GM bin
执行 atomic add。一个 row/bin 地址最多接收 16 次 atomic update。这样 reducer 每级
只需搬运 1 KiB，且不再执行 16-shard 求和。

该实验还必须解决两个附带问题：

1. Row-global histogram 在每次调用前必须清零。当前实验由 host 构造零数据并在
   measured kernel launch 前 H2D copy，因此初始化不计入 kernel 时间。
2. 删除 per-shard histogram 后也失去了 per-shard compact offset。候选改用每行
   `greater/equal` 两个 GM counter，并让非空 warp 通过 GM atomic reservation 分配
   输出 slot。

### 8.1 正确性与 DCCI 结论

Row-global high/low histogram 的 GM atomic update 在现有四个阶段屏障下可被 reducer
的 MTE 读取，histogram producer 本身不需要为 atomic update 增加 DCCI。删除 high
producer DCCI 后，阈值和 compact 均正确。

但 low producer 仍必须保留一次 DCCI。原因不是发布 histogram atomic，而是 16 个
shard 在 low histogram 阶段都读取过 `state.selected_high`；low reducer 随后会在同一
GM cache line 写入最终 threshold 和 total-greater。删除这次 DCCI 后，compact 可继续
读到旧 state，曾观测到 row 0 compact greater 为 25,078，row 1-3 为 32,768，而正确值
应为 2,008。恢复 low-stage DCCI 后，direct oracle 四行全部通过。

因此不能把“GM atomic 不需要 producer DCCI”扩展成“该阶段不需要任何 DCCI”。要按
每一条普通 GM cache line 的生产者、旧读者和新写者分别分析可见性。

### 8.2 默认 profiler warmup 的 workspace 累积

Host 只在正常 launch 前做一次 zero H2D；`msopprof` 默认还会用相同参数重放 kernel
5 次。若只有一份 row-global histogram 和 compact counter，warmup 会跨 replay 累积，
direct 虽通过，profile 最终 oracle 会失败。

设备实验为 profiling harness 预清零了 16 个 slot。每个 physical block 在 high VF
中只做一次 replay-counter atomic，并把返回的 slot 保存在 UB word 264；low 和 compact
VF 复用该 UB word。Mixed outer kernel 直接读取 VF 写入的 UB 标量、直接读取 GM
counter、或先 MTE counter 再读取 UB，三种交接方式均在当前 CANN 9.2.0 / 950PR 环境
复现 `aclrtSynchronizeStream error=507035`，不得据此泛化为其他版本的限制。

最终实验的 reducers 固定读取 slot 0，因为 profiler 的 5 次 warmup 与正式采样重放
完全相同的输入；各 producer 和 compact 使用当前 replay slot。这个处理只保证该
standalone example 的同输入 profiler replay，不支持同进程内换输入后重复调用。

### 8.3 性能结果与结论

设备为 Ascend 950PR，CANN 9.2.0，Bisheng clang 15.0.5，`dav-3510`，`msopprof`
26.2.0。20 个正式 profile 采用 10 轮交替顺序，全部使用默认 warmup 5 次、
`--aic-metrics=Default --launch-count=1`、无 `--kernel-name`，且每份只有一个目标 row、
64 blocks、current/rated `1650/1650 MHz`。四行 oracle 全部通过。

| 实现 | median | min | max | 10 轮胜场 |
| --- | ---: | ---: | ---: | ---: |
| retained per-shard baseline | 14.3465 us | 13.916 us | 14.596 us | 10 |
| row-global GM atomic | 503.2505 us | 501.844 us | 505.174 us | 0 |

Candidate median 比 baseline 慢 `35.078x`，即回退 `3407.83%`；paired delta median 为
`+488.9085 us`。`Default` 会对大量 GM atomic 做动态插桩，因此这里的 503 us 不能
外推成无插桩端到端延迟，但它就是本实验预先规定的 retention 边界。在该边界下结果
明确失败，row-global candidate 已撤回，example 保留 commit `2e0e393` 的 per-shard
实现。不要在测量条件没有变化时重试相同方案。

原始 evidence 位于远端：

```text
/tmp/dsa-topk-row-global-histogram-exp-Ab2vG1
```

完整 archive 已保存在远端和本节点：

```text
/tmp/dsa-topk-row-global-histogram-evidence-20260818.tar.gz
SHA-256: 474b0a87a6cefef0f6a557b84aeea1883a9e27a13a9e41bb41e24c2a68d869ff
```

关键 SHA-256：

```text
baseline simt_v2_topk.asc: d0f5821eeb739d00c2ab783c0bc19a401a1500e3b5918746b0c8d07ded13a0ae
baseline executable:          3f5c8a011861fdcb21c4b1ea3c5245a8d46ecf51ff15b4806d0b6a8f5f2da233
candidate simt_v2_topk.asc: c4e388ebf2dad413416f714152e27a5ba0afdef54a27d94b2588b082eb309e64
candidate executable:        99ade10ad74277678d5b8d19cbae4d49d8df01d305af16d17cb3defadf380f32
formal summary.json:         47a052e555abe88254832f25689ec7fab49a25eaae27222286ebcf327d7639b3
formal samples.csv:          d97fccce0b10b2f45e1b7fa61f381e7297f4ef5f729537ebb18d1c388820b08f
```

## 9. 定位问题时的检查顺序

当 TopK 输出错误或性能异常时，按以下顺序检查：

1. 确认输入 shape、BF16 key 转换和 `topk=2048` 未变化。
2. 检查当前实现的 histogram workspace 生命周期；row-global 实验才要求每次调用前
   清零，retained per-shard 路径会覆盖自己的 GM slice。
3. 回读 `selected_high`、`remaining_rank`、`threshold_key` 和
   `total_greater_count`，先区分阈值错误与 compact 错误。
4. 检查 histogram producer 的写入方式；普通 GM store 与 GM atomic 的 cache
   发布要求不能互换推断。
5. 确认所有 64 个 block 以相同顺序到达四个 `SyncAll`。
6. 检查 compact counter 或 per-shard offset 是否产生重复/空洞/越界 slot。
7. 性能对比必须使用同一输入、64 blocks、默认 profiler warmup、
   `--aic-metrics=Default --launch-count=1`、无 `--kernel-name`，并要求
   current/rated `1650/1650 MHz`。

## 10. 相关源码

- `simt_v2_topk.asc`：SIMT v2 standalone kernel。
- `host_common.h`：固定输入、workspace 物化和 score-set oracle。
- `test_dsa_decode_topk_comparison_source.py`：源码结构与 profiling contract。
- `SPEC.md`：已保留和已拒绝实验的测量结论及 evidence 路径。
