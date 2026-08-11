# 关键路径收益上界实践

本案例演示在编码候选优化前，先计算它对并发 workflow 的物理收益上界。两条 ACL stream 表示两个可并发 physical lanes：lane A 是 3-stage 长链，lane B 是待评估候选。并发边界的近似关键路径是：

```text
critical = max(lane_A_sum, lane_B_sum)
```

两条 lane 的 duration 不能相加成 workflow latency。`lane_A_sum + lane_B_sum` 仅作为明确标记为 forbidden 的反例字段输出。

## 场景与语义

- `SCENARIO_NUM=0`：lane B baseline candidate，每元素执行 32 次 `+1`。
- `SCENARIO_NUM=1`：counterfactual，每元素执行 8 次 `+4`，代表 declared ideal speedup `4x` 的教学模型。

两者 lane B 完整输出都严格等于 `input+32`。lane A 在三个顺序 stage 中分别执行 64 次 `+1`，最终严格等于 `input+192`。固定 float32 输入为整数范围，host 检查两条 lane 的全部输出、64 个 guard、launch count、物理循环数和等价增量。

## 计时边界

每条 stream 各有同 stream ACL event start/end pair，得到 lane A 与 lane B 的设备 event interval。host 从首次 enqueue 前开始计时，等待两个 end event 后结束，得到 synchronized wall interval，也就是包含 host dispatch、并发执行和 join 等待的完整 join 边界。

ACL event 的两个独立 interval 本身不能证明 lanes 完全重叠；必须结合 msopprof raw timeline 核对 stream overlap 和 launch offset。event max 是教学估算，synchronized wall interval 是当次完整调用观测，两者都不能用 lane duration 相加替代。

## 理论上界与 retention gate

场景 0 的 parser 对 lane A 三个 stage 的 Task Duration 求和，对 lane B 一个 stage 求和，并验证 launch count 恰为 `3+1`。然后使用：

```text
baseline_critical = max(lane_A_sum, lane_B_sum)
ideal_lane_B = lane_B_sum / declared_ideal_speedup
ideal_critical = max(lane_A_sum, ideal_lane_B)
upper_bound = (baseline_critical - ideal_critical) / baseline_critical
```

本案例预声明 `declared ideal speedup=4x`、`retention gate=5%`。它们是实验假设和决策阈值，不是实测收益。若 upper bound 低于 gate，应在编码前淘汰候选；达到 gate 只表示值得实验，不表示优化已成立。

## 构建与采集

```bash
cmake -S . -B build/scenario_0 -DSCENARIO_NUM=0
cmake --build build/scenario_0 -j
./build/scenario_0/critical_path_upper_bound
```

完整流程：

```bash
./scripts/run.sh
```

脚本对两个场景分别 correctness 后 profile，为每次采集创建唯一 raw 目录，不显式设置 warmup。`analyze_profile.py` 遇到 replay、额外 launch 或缺失 kernel 时拒绝分析，不会把多个调用静默混为一个样本。

场景 1 的实际完整 join 边界与场景 0 在 `comparison.json` 中并列，但单次 wall change 只标记为 observation；需要重复独立进程并报告分布后才能做 retention 判断。

## Ascend 950 设备验证

2026-08-12 在 20002 节点完成一次设备复核：目标架构 `dav-3510`、CANN 9.2.0、
Bisheng/Clang 15.0.5、driver `7.0.t9.0.B791`、1650 MHz。节点未提供可直接调用的
`npu-smi`，因此具体 950 子型号和 firmware build 未单独采集。两个场景的完整输出、guard、
counter oracle 均通过。脚本没有显式设置 warmup；msopprof 日志显示工具默认 warmup 为 5 次。

低开销 direct run 的完整 join 边界如下。它包含 host enqueue、设备执行和两个 stream 的 join，
但这里只各采集一次，因此不能把约 1% 的变化解释为稳定性能收益。

| Scenario | Lane A event (ms) | Lane B event (ms) | Join wall (ms) | Gate decision |
|:---:|---:|---:|---:|:---:|
| 0, baseline | 1.205399 | 0.028630 | 1.376100 | reject before coding |
| 1, counterfactual | 1.216932 | 0.016258 | 1.389057 | reject before coding |

msopprof `OpBasicInfo` 用于 kernel attribution。Lane A 的三个 stage 必须求和，两个并发 lane
之间必须取 `max`：

| Scenario | Lane A stages (us) | Lane A sum (us) | Lane B (us) | `max(A, B)` (us) | 4x ideal upper bound |
|:---:|:---|---:|---:|---:|---:|
| 0 | 40.050999 / 39.952000 / 39.973999 | 119.976998 | 24.084000 | 119.976998 | 0.00% |
| 1 | 39.964001 / 39.859001 / 40.054001 | 119.877003 | 12.196000 | 119.877003 | 0.00% |

虽然 counterfactual 将 lane B 的 kernel row 从 24.084 us 降到 12.196 us，lane A 始终约为
120 us 的关键路径；即使按预声明的 lane B 理想 4x 加速计算，完整边界的理论收益上界仍为
`0.00%`，低于 5% retention gate。因此本案例的正确结论是编码前拒绝，而不是保留一个局部
kernel 有收益但 workflow 无收益的优化项。

msopprof 包裹运行中的 ACL event 和 wall 值被逐 kernel 插桩放大到秒级，只保留为
profiler 扰动证据，不作为正常调用延迟。原始结果位于远端：

- 场景 0：`/root/cannbench-p0-examples-lMCYpb/critical_path_upper_bound/profiles/scenario_0_fQT61d/raw`
- 场景 1：`/root/cannbench-p0-examples-lMCYpb/critical_path_upper_bound/profiles/scenario_1_MzdT1C/raw`
- 对比：`/root/cannbench-p0-examples-lMCYpb/critical_path_upper_bound/profiles/comparison_apWxiK/comparison.json`

本次是单次设备复核，未形成重复采样分布；跨 950 子型号、软件版本或频率迁移时必须重新验证。
