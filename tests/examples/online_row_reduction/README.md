# 在线稳定行归约实践

本案例在完全相同的 float32 输入与完整 softmax 输出语义下，对比三种稳定 row reduction 扫描结构。所有场景都先从输入减去稳定 max 或按等价在线公式更新统计量，不使用不稳定的直接 `exp(x)` 求和。

## 场景

| 场景 | 统计结构 | 每行输入访问 | 实测延迟 |
| --- | --- | --- | --- |
| `SCENARIO_NUM=0` | max scan、exp/sum scan、normalize scan | 3 遍 | 见设备结果 |
| `SCENARIO_NUM=1` | 单遍在线稳定 `(m,s)` 更新，随后 normalize | 2 遍 | 见设备结果 |
| `SCENARIO_NUM=2` | 每个 4096 tile 在线得到 `(m,s)`，稳定合并 tile pair，随后 normalize | 2 遍 | 见设备结果 |

在线更新为：新值不超过当前 max 时累加 `exp(x-m)`；出现新 max 时使用 `s <- s*exp(m-x)+1`。tile pair 使用

```text
m = max(m1, m2)
s = s1 * exp(m1 - m) + s2 * exp(m2 - m)
```

合并，因此普通值、极值和 tile tail 都保持数值稳定。

## 正确性 case

程序每次只运行一个 case，命令行参数为 `0|1|2`：

- `0 ordinary`：`64 x 1024`，有限范围普通值；
- `1 extreme`：`8 x 4096`，每行显式包含 `1000`、`999`、`-1000` 和 `998.5`；
- `2 tail`：`4 x 65537`，宽行最后一个 tile 只有 1 个元素。

host 使用 double host oracle：先以 double 求 row max，再以 double 求指数和及全部概率。device float32 输出逐元素检查：

```text
abs(actual - oracle) <= absolute tolerance + relative tolerance * abs(oracle)
absolute tolerance = 2e-6
relative tolerance = 3e-3
```

此外检查每行概率和误差不超过 `3e-3`，并验证输出后的 64 个 sentinel guard 未被改写。程序也校验统计扫描数、normalize 扫描数、输入元素访问数、tile pair 数和 tail 长度；这些是逻辑工作量，不是性能数据。

## 构建与运行

```bash
cmake -S . -B build/scenario_0 -DSCENARIO_NUM=0
cmake --build build/scenario_0 -j
./build/scenario_0/online_row_reduction 0
./build/scenario_0/online_row_reduction 1
./build/scenario_0/online_row_reduction 2
```

将 `SCENARIO_NUM` 改为 1 或 2 可构建其余算法。完整 correctness-first 采集：

```bash
./scripts/run.sh
```

脚本对每个 scenario/case 先单独验证，再写入唯一 profile 目录。完整 raw profiler 输出、stdout 和环境摘要均保留；脚本不显式设置 warmup，使用当前 msopprof 默认行为。

## 完整调用与归因边界

三个场景都固定为 `1 block`、`512` 个 SIMT threads、每线程独立处理一行；每个进程只运行一个 case，并产生 `1 次相关 kernel launch`。因此当前完整 softmax 调用边界就是该 kernel 的完整 Task Duration，输入 H2D、输出 D2H 和 double oracle 在边界外。

不同 case 的 shape 和工作量不同，不得把三条 kernel row 求和后当作一个 softmax 样本。msopprof replay 产生的重复行也必须按应用调用区分。若后续场景采用不同并行度、辅助 kernel 或 launch count，必须同时报告完整调用的所有 stage 之和及相同同步点的端到端区间，不能把差异全部归因于扫描融合。

## Ascend 950 性能结果

2026-08-12 在 20002 节点、CANN 9.2.0、`dav-3510`、1650 MHz 完成单次采集，9 组
direct/profiler run 全部通过 double oracle、行和与 guard 校验。

| Case | Shape | 三扫描 (us) | 在线统计 (us) | Tiled online (us) | 在线统计相对基线 |
|:---:|:---|---:|---:|---:|---:|
| 0 | `64 x 1024` | 340.984985 | 296.438995 | 297.520996 | -13.06% |
| 1 | `8 x 4096` extreme | 833.766968 | 711.765015 | 714.135986 | -14.63% |
| 2 | `4 x 65537` tail | 15296.893555 | 12981.391602 | 13026.042969 | -15.14% |

raw 位于 `profiles/scenario_<N>_case_<N>_*`。Tiled online 在三组 shape 上均仅比非 tiled
在线统计慢约 `0.34%-0.40%`，但状态容量不依赖整行宽度。以上为单次结果，仍需多 run
离散度；源码扫描计数不能代替实际 profile。
