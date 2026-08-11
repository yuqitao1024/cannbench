# Row Width Dispatch 实验规格

## 目标

在 Ascend 950 上对比单一通用路径、按行为区间 dispatch、以及 exact width 过拟合反例。实验只冻结可审计的 source/launch 合同；阈值与 geometry 是否有效必须由设备数据决定。

## 输入输出语义

- 输入是固定 seed `20260812` 生成的正 `uint32`，逻辑 shape 为 `[64, width]`，`width` 支持 `1..8192`。
- 物理 row stride 固定为 8192，逻辑 tail 使用 sentinel；kernel 不得写 `[width, 8192)`。
- 每行先计算 exact `row sum`，再逐元素输出 `normalize = input * 1000000 / row_sum`，整数除法向下取整。
- 三种策略使用相同输入、输出、host oracle、64 blocks 和每行一个 block 的所有权；每次调用 1 次 kernel launch。
- host oracle 必须逐行核对 sum、逐元素核对 normalize，并检查每行 tail sentinel。

## 三种策略

- `SCENARIO_NUM=0`：所有 width 使用 `generic_row_normalize_kernel` 和固定 512-thread geometry，作为单一通用路径。
- `SCENARIO_NUM=1`：`width<=256` 使用 256 threads，`width<=4096` 使用 512 threads，`width>4096` 使用 1024 threads；三个 kernel 名和 geometry 必须可从 raw profile 审计。
- `SCENARIO_NUM=2`：exact width 1024 使用 256 threads、固定四元素循环且没有 width guard；所有邻近 width 使用 1024 threads 的 fallback，并固定扫描到 padded width 8192。该 fallback 是刻意的过拟合反例，用于暴露单点特化的泛化和维护问题，不代表推荐实现。

## API 与实现边界

- host 只使用函数、POD struct、`malloc/free` 和裸 ACL；不得使用 class、STL 或自定义 RAII。
- device 只使用 C API、Tensor API 与 SIMT API；不得包含 `basic_api/*`、`kernel_operator.h`，不得使用 LocalTensor、Set/WaitFlag、PipeBarrier、CrossCore 或 Mutex。
- 归约只在 block-local UB 中进行，禁止跨核同步。

## 正确性与 profile

- correctness 必须覆盖 small/medium boundary 两侧：255/256/257、4095/4096/4097；并覆盖 exact 邻近 shape 1023/1024/1025、最小 width 1 和最大 width 8192。
- 同一组 boundary 与邻近 shape 均需 profile。每次 profile 的程序调用只 launch 打印出的一个目标 kernel。
- `scripts/run.sh` 必须完成一个场景的全部 correctness 后才开始 profile；msopprof 使用默认参数，不得显式设置 profiler warmup。
- raw profiler 目录必须按 scenario/width/run id 保留；解析出的 kernel rows 仅用于归属与 launch 审计。
- 没有目标设备数据时，性能表写“待实测”，不得根据线程数、padding 或 source loop 次数宣称加速或回退。
