# 确定性候选压缩实践

本案例比较两个语义完全相同的候选压缩实现：从固定输入中选择满足谓词的元素，并严格按照原始输入 index 升序写入输出。atomic 的完成顺序不能作为输出顺序，否则只是一个不确定性压缩，不能与稳定压缩等价比较。

## 场景

通过 CMake 的 `SCENARIO_NUM` 选择实现：

- `0`：per-item atomic + 重复扫描。每个命中项用一次 `asc_atomic_add` 统计总数，但输出 rank 通过重新扫描 `[0, input_index)` 得到。atomic 返回值不参与输出寻址，因此顺序仍由输入 index 决定。该路径会重复读取并重复计算谓词。
- `1`：packed count/prefix offset。每个 512 项 chunk 用 warp ballot 打包命中位，得到 warp count、warp prefix 和 chunk running offset；lane offset 来自 ballot mask 的低位计数。每个输入只做一次选择判定，输出 rank 按 chunk、warp、lane 顺序递增，也就是按输入 index 递增。

当前固定输入包含 4099 个互异值，长度不是线程数的整数倍。host oracle 使用相同谓词顺序扫描输入，并分别验证：

- selected count；
- 所有有效输出的稳定顺序；
- `[selected_count, 4099)` tail 保持 `OUTPUT_SENTINEL`，从而覆盖最后一个不完整 chunk 的越界写风险。

场景 0 在 kernel 内由 thread 0 清零 atomic counter，并用 block barrier 后再开始统计。这样
msopprof 默认 warmup 重放 kernel 时不会把前一次计数累加到下一次；该初始化属于被测 kernel
边界，两个场景的结果都不依赖 profiler 只执行一次。

## 构建与验证

环境需要已安装支持 `dav-3510` 的 ASC 编译器、ACL runtime 和可用的 Ascend 950/950PR 设备。

```bash
cmake -S . -B build/scenario_0 -DSCENARIO_NUM=0
cmake --build build/scenario_0 -j
./build/scenario_0/deterministic_compaction

cmake -S . -B build/scenario_1 -DSCENARIO_NUM=1
cmake --build build/scenario_1 -j
./build/scenario_1/deterministic_compaction
```

成功时程序打印 `Verification PASSED`。完整的独立构建、验证和 profile 可执行：

```bash
./scripts/run.sh
```

可通过 `BUILD_ROOT` 和 `PROFILE_ROOT` 指定产物目录。脚本为每个场景创建唯一 profile 目录，不复用旧的 `OPPROF_*` 数据。

## 公平比较与聚合边界

当前每次程序调用 1 个相关 kernel：场景 0 是 `stable_atomic_scan_compaction_kernel`，场景 1 是 `packed_prefix_compaction_kernel`，所以当前 launch count 都是 1。输入 H2D、输出初始化、结果 D2H 和 host oracle 在两个场景中相同，但不属于 kernel 的 Task Duration。

若后续某个实现拆成多 kernel，一次压缩调用的设备工作边界必须对所有相关 kernel 的 Task Duration 求和，再跨调用统计分布；不能只挑最快的一行，也不能把同一次调用的多个 stage 当成独立样本求平均。`msopprof` replay 可能产生多次应用调用，必须按 timeline/调用分组核对 observed launch count。

Task Duration 之和只表示选中设备工作的边界，不含 host dispatch 和串行 launch gap。涉及多 kernel 或 launch count 变化时，还必须在相同同步点采集覆盖完整调用的 NPU event 或同步 wall interval，单独报告端到端边界。

## Ascend 950 性能记录

2026-08-12 在 20002 节点完成一次设备复核：Ascend 950 系列 `dav-3510`、CANN 9.2.0、
Bisheng/Clang 15.0.5、driver `7.0.t9.0.B791`、1650 MHz。msopprof 使用默认 warmup，
两个 direct run 和 profiler 内运行均输出 `Verification PASSED`。

| Scenario | Kernel | Task Duration (us) | 原始结果目录 | 正确性 |
|:---:|:---|---:|:---|:---:|
| 0 | `stable_atomic_scan_compaction_kernel` | 1620.112061 | `profiles/scenario_0_NK9KST` | PASS |
| 1 | `packed_prefix_compaction_kernel` | 18.013000 | `profiles/scenario_1_mFOw64` | PASS |

本次固定输入下 Scenario 1 相对 Scenario 0 降低 `98.89%`，约 `89.94x`。这是单次 raw
结果，正式发布仍需记录多 run 中位数和离散程度；不能外推到其他选择率、长度或多 block
实现。
