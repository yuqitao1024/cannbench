# Top-K 算法选择案例规格

## 教学目标

本案例只改变 Top-K 算法，不在两个 scenario 之间改变 shape、dtype、输入、输出、launch geometry 或计时边界。目标是先比较重复全排序与流式有界选择的工作量，再讨论寄存器、访存、占用率等微架构问题。

## 数据合同

- 输入：`[32, 1024]`，`float32`，连续 ND 布局。
- 输出值：`[32, 32]`，`float32`。
- 输出索引：`[32, 32]`，`uint32`，索引范围 `[0, 1024)`。
- 语义：逐行 Top-32，值降序；值相等时原始索引升序。输入只生成有限、非 NaN、非无穷值。
- 输入生成：host 使用 `RANDOM_SEED=20260811` 的固定随机种子和 xorshift32；值量化为 1024 种离散值，确保 tie 规则被实际覆盖。
- 正确性基准：host oracle 为每行构造 `(value, index)` POD 数组并用 `qsort` 按完整语义排序，device 的值和索引必须逐项精确相等。

## A/B 场景

### SCENARIO_NUM=0：重复 padded bitonic merge

当前 Top-32 每次只接收 16 个新候选，却把 32 个旧候选、16 个新候选和 16 个 sentinel padding 拼成 64 项，执行完整 64 项 bitonic 网络，再保留前 32 项。每行执行 256 次网络，每次 672 次 compare-exchange，其中 25% 槽位是人为 padding。

工作量模型为 `O((N / C) * P * log^2(P))`，这里 `N=1024`、`C=16`、`P=64`。

### SCENARIO_NUM=1：流式有界 Top-K

扫描时只维护容量 32 的 worst-root 二叉 heap。未满时插入；已满后，只有优于 root 的候选才替换并下沉。扫描结束后只对 32 个有效候选做一次降序整理，全程没有 padding。

扫描工作量为 `O(N log K)`，最后排序只作用于 `K=32` 项。

## Launch 与性能边界

- 两个场景均 launch 1 个 block、每个 block 512 个 SIMT 线程。
- 每个线程通过 grid-stride 处理行；本 shape 下每个线程处理 2 行。
- 每次进程执行只 launch 所选 scenario 的 1 个 Top-K kernel。
- msopprof 的 `Task Duration` 是本案例性能表的 device-kernel 边界，不包含 host oracle、ACL 初始化、host/device 分配、H2D、D2H 和 host 正确性检查。
- 两种二进制由同一源文件、同一工具链和同一 launch geometry 构建，仅 `SCENARIO_NUM` 不同。

## 预期瓶颈与判定

- Case 0 预期瓶颈是重复 64 项 compare-exchange 网络及 padding 带来的无效指令工作。
- Case 1 预期瓶颈转向候选读取、heap 的数据依赖和线程局部数组资源占用。
- 上述均是假设，不是实测结论。必须在 Ascend 950PR/950DT 上先通过 oracle，再保留 raw profile 并复核 kernel 行，才能填写性能数据。

## 验收条件

- 两个 `SCENARIO_NUM` 均能独立编译、运行并打印 `Verification PASSED`。
- `.asc` 不使用 Basic API、C++ 资源类、STL 容器或 RAII；host/device 均为纯函数、POD 和显式 `malloc/free`、`aclrtMalloc/aclrtFree`。
- profile 输出按 scenario 和 run id 隔离；raw 目录不被解析步骤覆盖，解析产物可追溯到 raw CSV。
