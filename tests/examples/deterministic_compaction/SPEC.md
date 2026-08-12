# Deterministic Compaction 规格

## 目标

该独立案例演示如何在不改变稳定输出语义的前提下，消除候选压缩中的 per-item 重复前缀扫描。两个场景接受完全相同的固定 `int32` 输入和选择谓词，输出均为按原始 input index 升序排列的命中值。

## 语义合同

对输入 `x[0:N)` 和谓词 `selected(x[i])`，输出计数为所有命中项数量。若第 `i` 项命中，其唯一输出 rank 等于区间 `[0, i)` 内命中项数量。任何基于全局 atomic 返回顺序的 rank 分配都不满足本规格。

输出容量为 `N`。有效区间为 `[0, selected_count)`；无效 tail `[selected_count, N)` 必须保留 host 预填的 `OUTPUT_SENTINEL`。

## 场景 0：稳定 atomic scan

每个命中项执行一次 atomic increment，仅用于最终 selected count。该项随后重新扫描所有更小的 input index 并计算稳定 rank，再写入唯一输出位置。该算法保持顺序，但具有二次量级的 predicate/输入访问，是待优化基线。

## 场景 1：packed prefix

单个 512-thread block 依次处理固定大小 chunk。每个 warp 用 ballot 形成选择 mask，`popcount(mask)` 形成 packed count，`popcount(mask & lanemask_lt)` 形成 lane offset。每个 warp 的 count 写入 UB，前缀和形成 warp offset；前序 chunk 的总命中数形成 running offset。

输出 rank 为 `running_offset + warp_offset + lane_offset`。chunk、warp、lane 都与 input index 的升序分段一致，因此写入时序不影响最终顺序。tail lane 不读取输入、不参与 mask。

## 实现边界

- 工程只包含一个 `.asc` translation unit，由 `SCENARIO_NUM=0/1` 选择 launch。
- host 侧使用 C 风格函数、POD `DeviceBuffers`、`malloc/free` 和 `aclrtMalloc/aclrtFree`。
- 不使用 STL 容器、自定义资源封装或面向对象资源生命周期。
- device 源仅依赖 C API 和 SIMT API 能力；不使用 Basic API、`kernel_operator.h`、`LocalTensor`、事件 flag/barrier 或跨核同步。
- 两个场景各启动一个 Vector kernel，block dim 为 1，SIMT thread count 为 512。
- 场景 0 在 kernel 内重置 atomic counter，使默认 profiler warmup/replay 保持幂等。

## 验证与测量

固定输入长度 4099，覆盖多个完整 chunk 和一个 tail chunk。host oracle 验证 selected count、逐 rank 值与未写 tail。任何 profile 前先独立运行可执行文件并检查 `Verification PASSED`。

脚本对两个场景分别配置与构建，并为每次 `msopprof` 创建唯一目录且不设置 kernel name filter。当前每个调用的相关 kernel launch count 为 1。若实现改为多 kernel，设备边界按每次调用聚合全部相关 kernel，并额外测量覆盖 launch gap 的同步端到端区间。

静态 source-contract 只证明源码布局和禁止项，不证明 ASC 编译、设备正确性或性能。提交中不记录未经设备实测的加速比。
