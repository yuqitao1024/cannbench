# 跨消费者 Shared Gather 案例规格

## 教学目标

本案例对应 Ascend 950 最佳实践候选“跨消费者复用离散 Gather”。它只比较同一 worker 为两个 consumer 重复读取同一离散地址，与读取一次后在 owner thread 内复用；不把 kernel fusion、workspace、layout 转换、跨流水 handoff 或同步变化混入 A/B。

## 数据与语义

- source table：`[65536, 128]`，`float32`，row-major 连续布局。
- selected indices：`[8192]`，`uint32`，每项范围 `[0, 65536)`；允许重复选择同一 source row。
- consumer A：完整输出 `[8192, 128]`，`output_a[row, feature] = source[index[row], feature] + 1.0`。
- consumer B：完整输出 `[8192, 128]`，`output_b[row, feature] = source[index[row], feature] - 1.0`。
- 两个 scenario 使用固定随机种子 `20260812` 生成完全相同的 source 和 selected indices。
- host oracle 使用同一索引语义计算两个完整输出，device 结果必须逐元素精确相等。

输入值量化为 `0.25` 的整数倍，两个 consumer 只加减精确可表示的 `1.0`，因此精确比较不会混入容差选择。

## 场景

### SCENARIO_NUM=0：两个 consumer 独立 gather

每个输出元素由一个 SIMT thread owner 处理。consumer A 和 consumer B 分别通过独立函数读取同一个 source address，再各自执行变换。读取指针使用 `volatile GM`，这是为了让编译器保留基线要求的两次物理 load，而不是把它们做公共子表达式消除。

### SCENARIO_NUM=1：一次 gather，多 consumer 复用

相同 owner thread 调用 producer 读取一次 source value，把 thread-local 标量同时传给 consumer A/B。复用不跨线程和 core，不需要 UB、GM workspace、flag、barrier、DCCI 或 CrossCore 协议。

## Launch 与计时边界

- 两个场景均使用 64 个 block、每 block 512 个 SIMT 线程，grid-stride 覆盖 `8192 * 128` 个输出元素。
- 每次程序调用只 launch 1 个被选中的 kernel：`shared_gather_independent_kernel` 或 `shared_gather_reuse_kernel`。
- msopprof 的目标 kernel `Task Duration` 是性能表边界，不包含 ACL 初始化、host/device 分配、固定输入生成、host oracle、H2D、D2H 和 host 正确性检查。
- 脚本不覆盖 profiler 默认 warmup 行为。目标机报告必须记录实际 Ascend 950 型号、CANN/Bisheng、driver/firmware、原始目录和完整 kernel 行数。

## 预期瓶颈与验收

- Case 0 预期瓶颈是两个 consumer 重复产生的不规则 GM load 和地址访问。
- Case 1 预期减少 source load 指令与读取流量；剩余瓶颈可能是离散读取延迟、两个完整输出的 GM store 或 launch 固定成本。
- 上述是待测假设，必须先通过 host oracle 并复核 raw profile，不能直接引用历史 workflow 百分比作为本案例结果。
- `.asc` 仅使用 C API、Tensor API/SIMD 头文件和 SIMT API；host/device 采用纯函数、POD、`malloc/free` 和裸 ACL handles。
