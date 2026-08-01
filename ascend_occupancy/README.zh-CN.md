# Ascend 950 Occupancy

`ascend_occupancy` 是面向 Ascend 950 `dav-3510` SIMT kernel 的独立子工程，提供：

- C99 兼容的资源可行性分析 API；
- 基于 `--cce-res-usage` 的寄存器和 Stack 构建期提取；
- Block 和 `launch_bounds` 候选生成；
- ACL Event 性能 runner 与 JSON/CSV 聚合结果；
- register spill、launch geometry 和 loop unroll 三组对比样例。

它不预测真实性能最优线程数。静态 API 只判断已知资源约束并生成值得实测的候选，性能结论来自目标设备 benchmark。

## 构建

仅构建 host API 和测试：

```bash
cmake -S ascend_occupancy -B build/ascend_occupancy
cmake --build build/ascend_occupancy --parallel
ctest --test-dir build/ascend_occupancy --output-on-failure
```

在 950 节点构建性能样例：

```bash
source /path/to/Ascend/cann-9.1.0/set_env.sh
cmake -S ascend_occupancy -B build/ascend_occupancy \
  -DASCEND_CANN_PACKAGE_PATH=/path/to/Ascend/cann-9.1.0 \
  -DASC_OCCUPANCY_BUILD_PERF_TESTS=ON \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/ascend_occupancy --parallel
```

配置阶段会为每个变体执行一次 ASC probe 编译，从 `--cce-res-usage` 输出生成 host 资源头；正式构建再编译和链接完整的含 `main` `.asc` 可执行文件。probe 与正式目标共享 source、include、宏、编译选项、架构和 SIMT 模式。

## 性能样例

构建会产生五个独立进程：

| 对比 | executable | 固定工作量 |
| --- | --- | --- |
| register spill | `register_spill_lb1024_occupancy_bench` | 48 blocks、512 threads、每线程 16 次 sincos |
| register spill | `register_spill_lb512_occupancy_bench` | 同上，仅 `launch_bounds` 不同 |
| launch geometry | `launch_geometry_lb2048_occupancy_bench` | 16384 个 gather 元素，扫描五组 Grid/Block |
| loop unroll | `loop_unroll_loop_occupancy_bench` | 64 blocks、2048 threads、每线程 4 次加法 |
| loop unroll | `loop_unroll_manual_occupancy_bench` | 同上，仅循环写法不同 |

运行全部变体：

```bash
ASC_OCCUPANCY_RESULTS_DIR=occupancy-results \
  ./build/ascend_occupancy/run-all-occupancy-benchmarks.sh \
  --warmup 10 \
  --iterations 100 \
  --environment-id host/device/cann-version \
  --profile-path /absolute/path/to/raw/profiles
```

脚本会继续执行所有独立 executable，并生成：

- 每个变体的原始 JSON/CSV；
- `occupancy-summary.json`；
- `occupancy-summary.csv`。

JSON 保留每个 ACL Event 样本、中位数、最小/最大值、正确性、Grid/Block 和编译资源。`is_best_candidate` 会跨同一 `benchmark + environment_id` 重新计算；当最低中位数候选与其他候选的样本范围重叠时，它们会共同标记，表示不能据此宣称稳定胜出。

单独执行一个 executable 时，未传 `--json/--csv` 也会在当前目录生成默认结果文件。`--candidate-index N` 可只运行第 N 个候选，供 profiler 精确选择 Grid/Block。

## Profiler

详细指标应与低开销 ACL Event timing 分开采集。例如只 profile `launch_geometry` 的第 2 个候选：

```bash
msopprof \
  --output=profiles/launch-geometry \
  --aic-metrics=Default \
  --launch-count=1 \
  ./build/ascend_occupancy/perf_tests/launch_geometry/launch_geometry_lb2048_occupancy_bench \
  --candidate-index 2 \
  --warmup 1 \
  --iterations 3
```

带详细指标的 profiler 时延有额外开销，不应与 ACL Event 结果直接混用。

## 边界

- 第一版只支持 Ascend 950 `dav-3510`。
- `Stack size > 0` 只表示 Bisheng 报告了 spill 风险，不代表实测 GM 流量或固定性能损失。
- 一核一 Block 下，理论 Warp occupancy 不是性能模型。
- runner 不推断参数语义、Grid 映射或正确性标准；这些由各 kernel adapter 明确定义。
- 生成资源常量只供 host 使用，不得参与 device 控制流或模板实例化。
