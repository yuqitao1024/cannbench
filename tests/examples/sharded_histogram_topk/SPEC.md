# Sharded Histogram Top-K 实验规格

## 目标

在 Ascend 950 上比较两个语义完全一致的 256-bin histogram/threshold 实现，隔离长 context 的工作分解收益与额外归约 launch 的代价。

## 固定语义

- 输入为固定 seed `0x5a17c9e3` 生成的 `16 * 1024 * 1024` 个 `uint8` 分数。
- 输出包含全部 256 个 bin、bin 总数、`TOP_K=4096` 的降序阈值，以及阈值 bin 中需要保留的 `threshold_tail`。
- `threshold_tail = TOP_K - count(score > threshold)`；其范围必须为 `[1, histogram[threshold]]`。
- host oracle 必须逐 bin 精确比较，并独立检查总数、threshold 和 threshold tail。
- 两个 Case 必须使用完全相同的输入、输出和 oracle，不得改变 dtype、shape、Top-K 或计时边界。

## A/B 实现

- `SCENARIO_NUM=0`：1 个 block 扫描完整 context，在一个 kernel 内生成 histogram 和 threshold；每次调用固定 1 次 kernel launch。
- `SCENARIO_NUM=1`：64 个 block 分别扫描互不重叠的 context shard 并写出 `64 x 256` partial histogram；随后 1 个 block 按 bin 归约并生成 threshold；每次调用固定 2 次 kernel launch。
- Case 1 的两次 launch 是一次算子调用的两个阶段，不得当作两个性能样本取平均。

## API 与代码边界

- host 采用函数、POD struct、`malloc/calloc/free` 和裸 ACL；不得添加 C++ 容器、自定义 RAII 或 class。
- device 只允许 C API、Tensor API 和 SIMT API。
- 不得包含 `kernel_operator.h` 或 `basic_api/*`，不得使用 `AscendC::LocalTensor`、`SetFlag/WaitFlag`、`PipeBarrier` 或 CrossCore 同步。
- 实现必须集中在一个 `.asc` 翻译单元内，通过 CMake 的 `SCENARIO_NUM` 编译期选择 A/B。

## 测量合同

- 目标设备为 Ascend 950 系列，架构参数为 `dav-3510`；CANN、驱动、固件和频率由实测者记录。
- 每次 profile 前必须先 direct run 并得到 `Verification PASSED`。
- Case 0 期望捕获 `baseline_histogram_threshold_kernel` 一行。
- Case 1 期望捕获 `shard_histogram_kernel` 与 `reduce_histogram_threshold_kernel` 各一行；先求同一次调用的 `stage_sum_us`，再对多次独立调用求中位数与范围。
- `stage_sum_us` 是 selected device rows 之和，不包含两次 launch 间隙。若用于最终结论，还必须补充覆盖完整 launch 序列的 ACL Event 或同步 wall interval。
- 每次采集必须使用唯一 run 目录并保留历史 raw；不得为方便重跑而删除整个 profile root。
- 没有设备原始数据时，性能表只能标记“待实测”，不得填入推测数值或声称加速。
