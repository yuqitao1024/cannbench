# 历史 Workaround 复验与撤销实践

## 概述

历史 correctness workaround 往往是合理的紧急降级，但根因修复后若不重新验证，它会永久限制性能。本案例展示严格顺序：重现旧失败条件，证明 scratch ownership 根因，修复所有权，再恢复多 block 并验证边界、tail 和重复 launch。

| Case | 编译选项 | 正确路径 |
|:---:|:---|:---|
| 0 | `SCENARIO_NUM=0` | 保留单 block workaround，`scratch[threadIdx.x]` 在一个 block 内唯一 |
| 1 | `SCENARIO_NUM=1` | 根因修复为全局 worker slot，恢复 64-block 并行 |

本案例没有默认错误场景。unsafe removal 仅在源码注释和 SPEC 中定义为“64 blocks + 旧 512-slot scratch ownership”，host 模型确认它会发生跨 block slot 碰撞，device 不执行该路径。

## 规格

| 项目 | 规格 |
|:---|:---|
| 输入 | `[4099, 257]`，`float32` |
| 输出 | `[4099]`，逐行顺序 FP32 sum，完整输出检查 |
| 输入生成 | 固定随机种子 `20260812` |
| 正确性 | C 风格 host oracle，精确逐行比较 |
| Case 0 | 1 block x 512 SIMT threads，3 次重复 launch |
| Case 1 | 64 blocks x 512 SIMT threads，3 次重复 launch |
| 目标 | Ascend 950PR / Ascend 950DT，`dav-3510` |

奇数 row count 和 width 同时覆盖调度边界、row tail 与 reduction tail。多 block 路径给每个 physical worker 唯一 scratch slot，不使用 Basic API、flag、barrier 或跨核同步。

## 为什么不能只删除 Workaround

只把旧 kernel 的 block dim 从 1 改为 64，会让每个 block 的相同 `threadIdx.x` 重叠写 scratch。这不是性能 candidate，而是已知 unsafe removal。正确流程必须保留原始失败条件证据，确认根因修复，再分别验证单 block 基线、多 block、tail 和同一进程重复执行。

## 预期瓶颈

- Case 0：单 block 限制可用并行度，每个线程通过 grid-stride 处理约 8 到 9 行。
- Case 1：恢复多 block 后每个 active worker 处理一行，但 scratch footprint 增大，launch/调度成本也可能影响结果。

这些是预期瓶颈，不是实测结论。

## 运行

```bash
cmake -S . -B build/scenario_0 -DSCENARIO_NUM=0 -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/scenario_0 -j
./build/scenario_0/workaround_revalidation

cmake -S . -B build/scenario_1 -DSCENARIO_NUM=1 -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/scenario_1 -j
./build/scenario_1/workaround_revalidation
```

输出必须同时包含 `Ownership model PASSED` 和 `Verification PASSED`。完整采集运行：

```bash
./scripts/run.sh
```

脚本先做 correctness，再运行 msopprof；raw 保存在 `profiles/scenario_N/raw/<run-id>/`，结构化解析行、三行 `Task Duration` 总和与 `launch_manifest.txt` 保存在对应 parsed 目录。parser 要求目标 kernel 恰好出现 3 行；缺失、额外 replay 或混入旧 CSV 时直接失败。脚本不自行设置 profiler warmup。

## 性能边界与实测结果

每个 application call 内有 3 个相同 kernel launch，性能边界为这 3 行 `Task Duration` 的总和，不含 launch gap、host oracle、ACL 初始化、分配与 copy。

2026-08-12 在 20002 节点完成一次设备复核：目标架构 `dav-3510`、CANN 9.2.0、
Bisheng/Clang 15.0.5、driver `7.0.t9.0.B791`、1650 MHz。节点未提供可直接调用的
`npu-smi`，因此具体 950 子型号和 firmware build 未单独采集。脚本没有显式设置 warmup；
msopprof 使用工具默认 warmup。两个场景的 ownership model、完整输出和三次重复 launch 均通过。

| Scenario | Kernel | Block dim | 3 个 Task Duration rows (us) | 3-launch sum (us) | 正确性 |
|:---:|:---|---:|:---|---:|:---:|
| 0 | single-block workaround | 1 | 810.958008 / 810.657043 / 810.033997 | 2431.649048 | PASS |
| 1 | fixed multi-block | 64 | 103.209999 / 103.227997 / 103.222000 | 309.659996 | PASS |

在本案例定义的三 kernel-row device-work 边界内，修复 ownership 后恢复 64-block 并行，
总时延从 2431.649048 us 降到 309.659996 us，降低 `87.27%`，加速 `7.85x`。host ownership
模型同时确认 unsafe removal 有 32256 次 slot collision，修复后为 0；因此收益来自根因修复后
恢复并行，而不是直接删除 correctness workaround。

原始结果位于远端：

- 场景 0：`/root/cannbench-p0-examples-lMCYpb/workaround_revalidation/profiles/scenario_0/raw/20260812-063112`
- 场景 1：`/root/cannbench-p0-examples-lMCYpb/workaround_revalidation/profiles/scenario_1/raw/20260812-063112`
- 解析行、`aggregate.csv` 和 launch manifest：对应 `parsed/20260812-063112/` 目录

这些数值是单次 profiler 采集，不包含三次 launch 之间的 gap、host dispatch 或同步 wall interval，
也没有重复独立进程的离散度。跨 950 子型号、软件版本或完整调用边界的结论必须重新采集。
