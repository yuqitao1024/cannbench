# Consumer-native Layout 案例规格

## 教学目标

本案例对应候选文档 3.10“直接生成消费者布局”。它比较 producer 先生成 row-major 中间结果再显式转换，与 producer 直接生成 consumer 物理布局。为避免依赖版本不确定的 ND2NZ 接口，两个路径都使用纯 SIMT 标量地址映射；教学因素是中间布局与完整转换边界，而不是某个特定 copy-format API。

## 输入、输出与布局

- 逻辑输入：`[4096, 256]`，`float32`，row-major。
- producer 语义：每个逻辑元素执行 `input + 1.0`。
- consumer 需要 16x16 blocked layout，物理 shape 为 `[256, 16, 16, 16]`，轴顺序是 `[row_block, col_block, row_inner, col_inner]`。
- 物理 offset：`(((row / 16) * 16 + col / 16) * 16 + row % 16) * 16 + col % 16`。
- 固定随机种子为 `20260812`。host oracle 独立计算每个 logical element 的 blocked offset，并逐项精确检查完整物理 output。

输入量化为 `0.125` 的整数倍，producer 只加精确可表示的 `1.0`，因此正确性不依赖浮点容差。

## A/B 场景

### SCENARIO_NUM=0：row-major + 显式 packing

`row_major_producer_kernel` 写一份 `[4096, 256]` row-major GM workspace；随后同一 stream 上的 `explicit_blocked_pack_kernel` 读取 workspace 并写 consumer layout。完整生产边界包含两个 kernel、4 MiB row-major GM workspace，以及一次中间 GM write 和 read。

### SCENARIO_NUM=1：直接生成 consumer layout

`consumer_native_layout_kernel` 在 producer 计算时直接求 blocked offset 并写最终 output。完整生产边界只有一个 kernel，不分配 row-major workspace。

## Launch、所有权与计时

- 每个 kernel 使用 64 个 block、每 block 512 个 SIMT 线程；每个逻辑元素由 grid-stride 中唯一 thread 写一次。
- Case 0 每次 application call 预期两个 launch，Case 1 预期一个 launch。
- msopprof 的 Case 0 完整生产边界为两个 selected kernel 的 `Task Duration` 逐 call 求和；Case 1 为一个 selected kernel。该 device-work sum 不包含两个 launch 之间的 host gap，也不是端到端 wall time。
- H2D、D2H、ACL 初始化、内存分配、输入生成和 host oracle 都排除在性能表外。Case 0 workspace 生命周期属于实现合同，但 allocation 本身不计时。
- 解析器必须核对 selected kernel family count，保留 raw rows，并报告 expected launches、captured calls 与完整边界均值。脚本不覆盖 profiler 默认 warmup。

## 预期瓶颈与验收

- Case 0 预期瓶颈是中间 GM workspace 的写回/重读和第二次 kernel launch。
- Case 1 删除这些工作，但 blocked store 地址顺序可能影响合并访问，收益不是源码可证明事实。
- 不得把候选文档的历史百分比复制为本 standalone 案例结果；必须在目标 Ascend 950PR/950DT、固定 CANN/Bisheng 环境下正确性通过后实测。
- `.asc` 仅使用 C API、Tensor API/SIMD 头文件和 SIMT API，禁止 Basic API；host/device 使用纯函数、POD、`malloc/free` 和 raw ACL handles。
