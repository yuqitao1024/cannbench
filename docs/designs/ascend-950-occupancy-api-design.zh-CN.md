# Ascend 950 Occupancy API 与性能验证设计

## 1. 文档状态

- 日期：2026-08-01
- 目标设备：Ascend 950PR，`dav-3510`
- 开发分支：`feat/ascend-950-occupancy-api`
- 独立 worktree：`/root/aiagent/cannbench/.worktrees/ascend-950-occupancy-api`
- 原始材料：仓库根目录 `occupancy.md` 的内容已归档到本文，原文件已删除
- 当前阶段：第一版 API、CMake probe、性能 runner 和三个样例已实现并完成 950 实机验证

本文整合原始 occupancy 文档、当前讨论形成的 API 取舍、20002 节点实测结果，以及对 Ascend 950 是否需要 occupancy 能力的重新判断。

## 2. 决策摘要

### 2.1 是否有必要在 Ascend 950 上做 occupancy

有必要，但目标不应是机械复制 CUDA Occupancy API，也不应声称能静态算出性能最优线程数。

在 Ascend 950 上真正有价值的是以下能力：

1. 在构建期自动提取 kernel 的寄存器数和 Stack size。
2. 检查某个运行时 `block_threads` 是否满足设备上限、`launch_bounds` 和已知 UB 容量约束。
3. 报告当前二进制是否发生寄存器 spill，以及理论块内 Warp 占用率。
4. 在发生 spill 时生成需要重新编译验证的 `launch_bounds` 候选档位。
5. 为线程数、Grid 数和源码调优写法生成一组小规模实测候选，而不是伪造一个“必然最快”的结果。
6. 在 CI 或性能回归中发现寄存器、Stack size 和 launch 配置的变化。

如果只实现一个 Ascend 版本的 `MaxPotentialBlockSize`，它在“一核同时只驻留一个 Block”的 950 上通常会退化为“返回最大合法 Block Size”，价值有限，而且容易误导调用方把理论 occupancy 当成实际性能。

### 2.2 第一版的定位

第一版完整交付定位为：

> Ascend 950 SIMT kernel 的构建期资源提取、运行时资源可行性分析与性能候选生成工具。

第一版采用 Ascend 原生 API，不提供 CUDA 符号兼容层，不把代码放入任何算子目录，也不修改 CannBench 公共后端或 CLI。

第一版包含后文实施阶段一和阶段二：资源分析 API、CMake probe、Block/`launch_bounds` 候选、可执行 benchmark 和三个对比样例。阶段划分表示实现顺序，不表示拆成两个对外版本。

### 2.3 核心设计决策

- 使用顶层独立目录 `ascend_occupancy/`。
- API 使用 C ABI，内部使用 C++ 实现并调用 ACL Runtime。
- 使用 CMake `try_compile` 对 kernel 做一次 probe 编译。
- probe 编译加入 `--cce-res-usage`，从 Bisheng 输出中解析寄存器数和 Stack size。
- CMake 生成只供 host 侧使用的资源常量头文件，再进行正式编译。
- API 区分“最大资源可行线程数”和“实测性能最优线程数”。
- 正常构建只生成 benchmark 可执行文件，不自动占用设备运行。
- 用户在目标 950 节点执行 benchmark，runner 扫描少量候选并输出 JSON/CSV。

## 3. CUDA Occupancy 实际解决什么问题

CUDA Occupancy API 也不会读取访存模式、动态循环次数、指令延迟或 Grid 调度开销。它只使用编译产物元数据和设备资源上限。

典型输入包括：

- `cudaFuncGetAttributes` 提供的每线程寄存器数、静态 Shared Memory、Local Memory 和最大线程数；
- `cudaGetDeviceProperties` 提供的每个 SM 的寄存器、Shared Memory、Warp、线程和 Block 上限；
- 调用方提供的 `blockSize` 和动态 Shared Memory 大小；
- VariableSMem 版本中由调用方提供的 `blockSize -> dynamicSmem` 回调。

CUDA 对每个候选 Block Size 计算：

```text
active_blocks = min(
  blocks_by_threads,
  blocks_by_registers,
  blocks_by_shared_memory,
  hardware_max_blocks
)

active_threads = active_blocks * block_size
occupancy = active_threads / max_threads_per_sm
```

`cudaOccupancyMaxPotentialBlockSize` 枚举候选 Block Size，选择理论活跃线程数或 Warp 数最大的候选：

```text
block_size = argmax(active_blocks(block_size) * block_size)
min_grid_size = sm_count * active_blocks(block_size)
```

这里的 `min_grid_size` 仅表示填满设备所需的最少 Block 数，不是根据业务输入计算的最佳 Grid。CUDA Occupancy API 也不承诺返回的 Block Size 是性能最优值。

实际访存合并、Cache 命中率、指令流水线停顿、动态循环次数、分支发散、Achieved Occupancy 和 Kernel Launch 开销，需要通过 Nsight Compute、Nsight Systems、CUPTI、CUDA Events 或实际 benchmark 获得。

## 4. Ascend 950 与 CUDA 的关键差异

### 4.1 驻留模型

当前 950 SIMT 模型下，一个 Vector Core 同一时刻只驻留并执行一个 Thread Block。多个 Block 通过全局调度分发到不同 Vector Core；当 Block 数超过物理核数时，后续 Block 排队执行。

因此每核活跃 Block 数退化为：

```text
resident_blocks_per_vector_core =
  launchable ? 1 : 0
```

理论块内 Warp 占用率为：

```text
active_warps = ceil(block_threads / warp_size)
theoretical_warp_occupancy = active_warps / max_warps_per_vector_core
```

这只是块内理论活跃 Warp 比例，不是性能预测。

### 4.2 寄存器分配模型

Ascend 950 上，`__launch_bounds__(N)` 指定的最大线程数决定编译器给每个线程的寄存器配额。当前文档和 Bisheng 实测对应关系为：

| `launch_bounds` | 每线程寄存器上限 |
| ---: | ---: |
| 1-256 | 127 |
| 257-512 | 64 |
| 513-1024 | 32 |
| 1025-2048 | 16 |

未显式配置 `__launch_bounds__` 时，当前工具链默认按 1024 线程档处理，即每线程最多使用 32 个寄存器。

当 kernel 需要的寄存器超过当前档位预算时，Bisheng 会把部分中间数据放入位于 GM 的 Stack。`--cce-res-usage` 输出中的 `Stack size > 0` 是当前最直接的 spill 证据。

### 4.3 为什么 950 的 MaxPotentialBlockSize 会退化

由于每核只能驻留一个 Block，对所有合法 Block Size 都有：

```text
active_blocks_per_core = 1
active_warps = ceil(block_size / warp_size)
```

如果只最大化理论 Warp 占用率，目标会随 `block_size` 基本单调增大，最终偏向最大合法 Block Size。但实测性能还受以下因素影响：

- 同时使用多少个 Vector Core；
- 单个 Block 的 Warp 排队；
- 每线程循环次数和工作量；
- Grid 调度固定开销；
- GM/DCache/UB 访问模式；
- 指令数量、展开和寄存器生命周期；
- 尾块和负载均衡。

所以最大合法 Block Size 可以静态计算，性能最优 Block Size 不能只由 occupancy 计算。

### 4.4 硬件模型的证据来源

一核一 Block、寄存器档位、默认 1024 档和 Stack spill 解释来自 Ascend C 开源开发套件中的 950 SIMT 优化文档：

- 上游仓库：`git@gitcode.com:cann/asc-devkit.git`
- 本次查阅 revision：`5ab3fa9366d917c283925e109a7db2810898672e`
- `docs/guide/算子实践参考/SIMT算子性能优化/执行配置/合理配置线程数避免寄存器溢出.md`
- `docs/guide/算子实践参考/SIMT算子性能优化/执行配置/线程块数量配置优化.md`

本次 20002 编译重新验证了 1024 档的 `32 registers + 32 B Stack` 和 512 档的 `48 registers + 0 B Stack`。后续可行性 probe 通过 `aclrtGetDeviceInfo` 实测得到 64 个 Vector Core、32 线程 Warp、每 Vector Core 和每 Block 最大 2048 线程，以及每 Vector Core `221184 B` UB。尚未通过边界 kernel 独立验证全部四档寄存器上限，因此实现时仍需把架构表集中、版本化，并用 256/257、512/513、1024/1025 边界 probe 补证据。

## 5. 20002 节点实测证据

### 5.1 环境

| 项目 | 值 |
| --- | --- |
| 主机 | `tools2` |
| 日期 | `2026-08-01` |
| 仓库 revision | `3068b6f6b7add15625708e30a4a367a24c6f4aa4` |
| 目标架构 | `dav-3510` |
| 设备 | Ascend 950PR；ACL Runtime 查询并由 profile 的 Block Dim 验证为 64 个 Vector Core |
| CANN | `/home/l00848653/Ascend/cann-9.1.0` |
| Bisheng | clang 15.0.5，构建标识 `clang-5c68a1cb1231` |
| msopprof | `26.1.0-05185f7d50b2abcabb2132dbe01c1ef3a4629aa0` |
| msopscommon | `1026376e17f8ef704f2ef31fa81455c1cbc62726` |
| 驱动/固件 | `7.0.t9.0.B791` |
| 采集频率 | 当前频率和额定频率均为 1650 MHz |
| 原始产物 | `/tmp/occupancy-research-20260801-WPBmhs` |
| 可行性 probe 产物 | `/tmp/occupancy-feasibility-20260801T153037+0800` |

### 5.2 测量边界

- 所有样例均使用 CMake 构建和包含 `main` 的 `.asc` 文件。
- 正确性直接运行先通过，再进行性能采集。
- 性能边界是单个 device kernel 的 `Task Duration(us)`。
- 命令使用 `msopprof --aic-metrics=Default --launch-count=1`。
- 每个 case 使用独立 profile 目录。
- traversal 和 Grid 扫描各采集 3 个独立 round。
- sincos spill/no-spill 对比采集 16 个独立 round，并交替调整执行顺序。
- 保留所有样本，不用最佳值代替分布。

采集命令的固定形式为：

```bash
msopprof \
  --output=<case-specific-output> \
  --aic-metrics=Default \
  --launch-count=1 \
  ./build/<executable> <case>
```

### 5.3 寄存器 spill 对比

固定 workload：48 个 Block，每 Block 实际启动 512 个线程，每线程计算 16 组 `sincosf`。

两个 kernel 的计算逻辑完全相同，仅 `launch_bounds` 不同：

| Kernel | `launch_bounds` | Used registers | Stack size |
| --- | ---: | ---: | ---: |
| `sincos_thread_1024` | 默认 1024 | 32 | 32 B |
| `sincos_thread_512` | 512 | 48 | 0 B |

Bisheng 原始输出：

```text
[BISHENG] Function properties for _Z18sincos_thread_1024PfS_S_m_simt_entry: Stack size: 32 bytes, Used register number: 32
[BISHENG] Function properties for _Z17sincos_thread_512PfS_S_m_simt_entry: Stack size: 0 bytes, Used register number: 48
```

16 轮结果：

| Kernel | 最小值 | 中位数 | 最大值 | 说明 |
| --- | ---: | ---: | ---: | --- |
| 1024 档，有 spill | 97.141 us | 98.219 us | 179.928 us | 存在未归因节点长尾 |
| 512 档，无 spill | 95.503 us | 98.253 us | 179.056 us | 存在未归因节点长尾 |

完整样本如下，单位均为 us：

| Round | 1024 档，有 spill | 512 档，无 spill |
| ---: | ---: | ---: |
| 1 | 97.626 | 98.156 |
| 2 | 103.088 | 98.351 |
| 3 | 97.586 | 97.695 |
| 4 | 100.489 | 97.694 |
| 5 | 179.928 | 179.056 |
| 6 | 98.192 | 98.677 |
| 7 | 97.506 | 168.412 |
| 8 | 98.055 | 96.360 |
| 9 | 97.563 | 177.957 |
| 10 | 97.639 | 99.378 |
| 11 | 99.699 | 96.671 |
| 12 | 99.164 | 97.464 |
| 13 | 97.141 | 95.503 |
| 14 | 100.119 | 177.603 |
| 15 | 98.246 | 174.687 |
| 16 | 169.684 | 97.629 |

结论：

- `launch_bounds(512)` 确实把 Stack size 从 32 B 降为 0，这是确定的编译资源收益。
- 本次 16 轮 Task Duration 中位数基本持平，差异约 0.04%，不能宣称消除 spill 后性能必然提升。
- 两组都有接近 170-180 us 的长尾，均保留在统计中；频率记录保持 1650 MHz，长尾不能简单归因于降频。
- 当前证据说明“spill 是风险信号”，但是否值得降低线程档位仍需结合实际 kernel 实测。
- 原始材料引用的历史数据为 102.47 us 降至 96.22 us，约 6.1%；它可以作为另一次工具链或运行环境下的观测，不能覆盖本次复测结果。

### 5.4 小 shape Grid/Block 扫描

固定 16384 个元素，使用相同 gather-strided kernel，只改变 Grid 和每 Block 线程数：

| Grid Blocks | Block Threads | Round 1 | Round 2 | Round 3 | 中位数 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 2048 | 11.080 us | 11.002 us | 10.990 us | 11.002 us |
| 8 | 2048 | 6.726 us | 6.775 us | 6.756 us | 6.756 us |
| 16 | 1024 | 4.648 us | 4.720 us | 4.657 us | 4.657 us |
| 32 | 512 | 3.816 us | 3.851 us | 3.824 us | **3.824 us** |
| 64 | 256 | 4.081 us | 4.039 us | 4.091 us | 4.081 us |

结论：

- 最优配置是 `32 blocks x 512 threads`。
- 相比使用全部 64 个 Vector Core 的 `64 x 256`，中位数降低约 6.3%。
- `2048 threads` 是合法配置，但 `4 x 2048` 和 `8 x 2048` 都明显更慢。
- 最大线程数、最大核数和真实性能最优值是三个不同概念。
- API 可以给出合法范围和候选，不能只返回最大线程数并称其为最优。

### 5.5 Loop 与手动展开

固定 workload：64 个 Block，每 Block 2048 个线程，每线程处理 4 个 float 加法。对比循环与手动展开：

| 执行方式 | 代码形态 | 三轮结果 | 中位数 |
| --- | --- | --- | ---: |
| direct SIMT | loop | 6.063 / 5.831 / 6.073 us | **6.063 us** |
| direct SIMT | manual unroll | 7.617 / 7.799 / 7.617 us | 7.617 us |
| hybrid | loop | 5.311 / 4.959 / 5.262 us | **5.262 us** |
| hybrid | manual unroll | 7.766 / 7.998 / 8.125 us | 7.998 us |

结论：

- direct SIMT 下 loop 比手动展开的中位数低约 20.4%。
- hybrid 下 loop 比手动展开的中位数低约 34.2%。
- 源码级“减少循环指令”不能直接推导出性能收益；展开会改变寄存器生命周期、指令调度和生成代码。
- 这组数据支持“候选必须实测”，不支持把固定优化技巧写入 occupancy 公式。

### 5.6 第一版实现的端到端复验

第一版代码在 20002 节点重新完成 configure、五个变体的 probe、正式编译、正确性、ACL Event timing 和单 kernel profile。原始产物保存在：

```text
/tmp/ascend-occupancy-task4b-yTSCdv8L
```

`asys info` 记录为 `Ascend 950PR_9589 V100`、架构 3510、32 个 AI Core、64 个 AI Vector 和 128 GiB HBM；runtime、ge-compiler、Bisheng、OAM tools 均为 9.1.0。五个 executable 的所有候选先以 3 个样本运行，九条记录均通过正确性；随后每个候选使用 10 次 warmup 和 100 个 ACL Event 样本：

| Benchmark | Variant | Registers | Stack | Grid x Block | 中位数 | 范围 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| register spill | LB1024 | 32 | 40 B | 48 x 512 | 9.045 us | 8.618-14.233 us |
| register spill | LB512 | 50 | 0 B | 48 x 512 | 9.067 us | 8.612-12.424 us |
| launch geometry | LB2048 | 12 | 0 B | 4 x 2048 | 4.3575 us | 4.095-10.134 us |
| launch geometry | LB2048 | 12 | 0 B | 8 x 2048 | 4.060 us | 3.669-6.429 us |
| launch geometry | LB2048 | 12 | 0 B | 16 x 1024 | **3.967 us** | 3.582-6.858 us |
| launch geometry | LB2048 | 12 | 0 B | 32 x 512 | 4.0025 us | 3.649-7.000 us |
| launch geometry | LB2048 | 12 | 0 B | 64 x 256 | 4.069 us | 3.600-5.917 us |
| loop unroll | loop | 16 | 24 B | 64 x 2048 | **6.4515 us** | 5.964-10.353 us |
| loop unroll | manual | 16 | 40 B | 64 x 2048 | 8.2765 us | 7.102-13.885 us |

本轮结果再次说明：消除 spill 后中位数没有自动变快；最大 Block 不是 launch geometry 的最低中位数配置；手动展开增加了 Stack 并使中位数变慢约 28.3%。所有组的 min/max 范围都有重叠，因此聚合结果保守地报告并列候选，不宣称稳定胜出。

`msopprof --aic-metrics=Default --launch-count=1` 分别捕获五个目标 kernel。Task Duration 为 geometry `16 x 1024` 的 2.365 us、spill LB1024/LB512 的 10.675/10.344 us、loop/manual 的 6.562/7.664 us；Block Dim 分别为 16、48、48、64、64，采集频率均为 1650 MHz。带详细指标的单次 Task Duration 只用于 launch 和 kernel 归属核对，不与低开销 ACL Event 中位数直接混用。

## 6. 950 Occupancy 能解决的问题

### 6.1 构建期资源可见性

当前 ACL Runtime 没有证据表明可以仅凭 kernel 指针可靠取得 Bisheng 的 `Used register number`、`Stack size` 和源代码 `launch_bounds`。CMake probe 可以把原本只在编译日志中可见的信息变为生成常量，使 host 代码和 CI 可以消费。

### 6.2 Launch 可行性检查

给定设备属性、编译资源数据、`block_threads` 和动态 UB 大小，API 可以确定：

- `block_threads` 是否超过 `launch_bounds`；
- `block_threads` 是否超过 Vector Core 最大线程数；
- 动态 UB 是否已经单独超过每核 UB 容量；
- 当前 launch 是否满足所有已知约束；
- 每核理论活跃 Warp 数和块内 occupancy；
- 当前 kernel 是否已有 Stack spill；
- 当前寄存器档位还剩多少 headroom。

`--cce-res-usage` 当前只明确提供寄存器数和 Stack size。如果无法取得 kernel 的静态 UB 用量，API 必须把 UB 检查标为不完整，不能把“动态 UB 单独未超限”表述为完整 launch 可行性证明。

### 6.3 资源回归检测

生成数据可以进入测试或 CI，例如：

- Stack size 从 0 变为非 0 时失败；
- 寄存器数跨越预期阈值时告警；
- `launch_bounds` 与实际 launch 不一致时失败；
- 工具链升级导致 Bisheng 输出格式变化时在配置阶段显式失败；
- probe 资源快照相对受控基线发生变化时由资源回归测试告警或失败。

probe 与正式编译的自动一致性比较需要捕获正式编译输出，列为后续增强。第一版通过共享唯一编译输入定义、禁止生成常量进入 device 代码以及保留两次编译日志降低漂移风险，但不把尚未自动比较的结果描述成强一致性保证。

### 6.4 候选生成

API 可以根据 950 的寄存器分档和设备上限生成少量候选：

- 当前 `launch_bounds` 档位；
- 如果有 spill，加入更低、寄存器预算更高的档位；
- 如果无 spill 且寄存器有余量，可加入更高线程档位作为吞吐候选；
- Block Threads 按 Warp 对齐并受 `launch_bounds` 和设备上限约束；
- benchmark helper 再根据数据量、物理核数和每线程工作量生成 Grid 候选。

候选只表示值得实测，不表示预先知道性能顺序。

## 7. 950 Occupancy 不能解决的问题

第一版明确不解决：

- 静态预测哪个 Block Size 一定最快；
- 自动理解任意 kernel 的参数、输入 shape 和正确性标准；
- 自动推导每线程动态循环次数；
- 静态获取真实访存合并率、Cache 命中率和带宽；
- 静态获取动态分支发散和流水线停顿；
- 用 occupancy 替代 msopprof 或真实 device timing；
- 自动消除 spill；
- 为所有 Ascend SoC 硬编码同一套寄存器档位；
- 在第一版提供 CUDA API 符号兼容。

## 8. 原生 API 设计

### 8.1 API 风格

- 对外提供 C ABI，避免要求使用方采用特定 C++ ABI。
- 使用 plain struct 和显式状态码。
- 不接受 kernel 指针，因为当前无法从指针取得完整资源描述。
- 接受 CMake 生成的 `AscKernelResourceUsage`。
- 名称使用 `ascOccupancy*`，不复用 CUDA 函数名。

### 8.2 数据结构

```cpp
typedef enum AscOccupancyStatus {
    ASC_OCCUPANCY_SUCCESS = 0,
    ASC_OCCUPANCY_INVALID_ARGUMENT,
    ASC_OCCUPANCY_UNSUPPORTED_DEVICE,
    ASC_OCCUPANCY_RESOURCE_DATA_MISSING,
    ASC_OCCUPANCY_RESOURCE_DATA_INCONSISTENT,
    ASC_OCCUPANCY_INSUFFICIENT_CAPACITY,
} AscOccupancyStatus;

typedef struct AscOccupancyDeviceProperties {
    uint32_t abi_version;
    uint32_t struct_size;
    int32_t device_id;
    uint32_t vector_core_count;
    uint32_t warp_size;
    uint32_t max_threads_per_vector_core;
    uint64_t ub_bytes_per_vector_core;
} AscOccupancyDeviceProperties;

typedef struct AscKernelResourceUsage {
    uint32_t abi_version;
    uint32_t struct_size;
    const char* kernel_symbol;
    uint32_t launch_bounds;
    uint32_t used_registers_per_thread;
    uint32_t stack_size_bytes;
    uint64_t static_ub_bytes;
    bool static_ub_bytes_known;
} AscKernelResourceUsage;

typedef uint32_t AscOccupancyConstraintFlags;

#define ASC_OCCUPANCY_CONSTRAINT_THREADS      (1U << 0)
#define ASC_OCCUPANCY_CONSTRAINT_LAUNCH_BOUND (1U << 1)
#define ASC_OCCUPANCY_CONSTRAINT_UB           (1U << 2)

typedef struct AscOccupancyAnalysis {
    uint32_t abi_version;
    uint32_t struct_size;
    bool launchable_under_known_constraints;
    bool has_register_spill;
    bool ub_capacity_check_complete;
    uint32_t resident_blocks_per_vector_core;
    uint32_t active_warps_per_vector_core;
    uint32_t max_warps_per_vector_core;
    uint32_t register_limit_per_thread;
    uint32_t register_headroom;
    uint64_t known_ub_headroom_bytes;
    double theoretical_warp_occupancy;
    AscOccupancyConstraintFlags violated_constraints;
} AscOccupancyAnalysis;

typedef struct AscLaunchBoundsCandidate {
    uint32_t launch_bounds;
    uint32_t register_limit_per_thread;
    bool requires_recompile;
    bool requires_benchmark;
} AscLaunchBoundsCandidate;
```

`stack_size_bytes` 保留 Bisheng `--cce-res-usage` 输出中的原始字段语义。第一版只用 `stack_size_bytes > 0` 判断编译器报告了 Stack spill 风险；它不是实测的 GM spill 流量，也不能用于估算访存次数或性能损失。

`known_ub_headroom_bytes` 只扣除调用方明确提供的静态 UB 和本次动态 UB；当 `static_ub_bytes_known=false` 时，它不能被解释为完整 UB 余量，此时 `ub_capacity_check_complete` 同时为 `false`。

### 8.3 核心函数

```cpp
AscOccupancyStatus ascOccupancyGetDeviceProperties(
    int32_t device_id,
    AscOccupancyDeviceProperties* properties);

AscOccupancyStatus ascOccupancyAnalyzeKernel(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    uint32_t block_threads,
    uint64_t dynamic_ub_bytes,
    AscOccupancyAnalysis* analysis);

AscOccupancyStatus ascOccupancyEnumerateLaunchBounds(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    AscLaunchBoundsCandidate* candidates,
    size_t* candidate_count);

const char* ascOccupancyStatusString(AscOccupancyStatus status);
```

`violated_constraints` 是位集合，因为线程数、`launch_bounds` 和 UB 可能同时越界。Stack spill 作为独立风险字段报告，不把它错误地当成 launch 阻断条件。`kernel_symbol` 指向生成头文件中的静态字符串，生命周期覆盖进程；API 不保存调用方传入的临时字符串。

`ascOccupancyGetDeviceProperties` 只使用已在目标 CANN 版本上验证过的 ACL 查询项。CANN 9.1 已验证 `ACL_DEV_ATTR_NPU_ARCH`、`VECTOR_CORE_NUM`、`WARP_SIZE`、`MAX_THREAD_PER_VECTOR_CORE`、`UBUF_PER_VECTOR_CORE` 和 `MAX_THREADS_PER_BLOCK` 均可通过 `aclrtGetDeviceInfo` 查询。设备描述表只保留一核一 Block 和寄存器档位等无法由这些属性表达的架构语义；实现不得用猜测值填充。无法识别设备或属性缺失时返回显式错误。

公共头文件包含 `<stdbool.h>`、`<stddef.h>` 和 `<stdint.h>`，并用标准 `#ifdef __cplusplus extern "C"` guard。所有可扩展 public struct 带 `abi_version` 和 `struct_size`。生成资源头可以使用 C++17 `inline constexpr`，但底层 API 头本身保持 C 兼容。

### 8.4 可行值、候选值和最优值

没有 workload 信息时，核心 API 不提供 `preferred_block_threads`。它只返回最大资源可行值和候选集合：

```cpp
typedef struct AscOccupancyCandidates {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t max_block_threads_under_known_constraints;
    uint32_t candidate_block_threads[4];
    size_t candidate_count;
    bool benchmark_required_for_optimum;
} AscOccupancyCandidates;

AscOccupancyStatus ascOccupancyEnumerateBlockCandidates(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    AscOccupancyCandidates* candidates);
```

- `max_block_threads_under_known_constraints`：由当前已知资源约束确定，不需要 benchmark；如果静态 UB 未知，不能把它描述成完整资源可行上限。
- `candidate_block_threads`：按 Warp 对齐、寄存器档位和最大可行值生成的少量资源候选。
- 性能最优值只由 benchmark 结果产生，不由静态 API 伪造。
- benchmark helper 拿到 `work_items`、每线程工作量和 Grid 策略后，再把 Block 候选扩展为完整的 Grid/Block 候选矩阵。

### 8.5 不提供 MaxPotentialBlockSize

第一版不提供名为 `ascOccupancyMaxPotentialBlockSize` 的接口，原因是该名称容易被理解为性能最优预测。若后续生态确实需要类似接口，应明确命名为 `ascOccupancyMaxFeasibleBlockSize`，并在文档中说明它只返回资源上最大可行值。

## 9. CMake 双阶段资源提取

### 9.1 构建流程

```text
kernel.asc
  -> try_compile probe，加入 --cce-res-usage
  -> CMake 捕获 stdout/stderr
  -> 严格解析 kernel symbol、Stack size、register count
  -> 生成 occupancy_resources.h
  -> 正式目标包含生成头文件
  -> 正式编译继续输出 --cce-res-usage
```

使用配置期 project-signature `try_compile`，由独立 probe 子项目显式设置 ASC 编译器、架构、include path、宏和编译选项，不需要手工拼接 Bisheng 命令。CANN 9.1 实测表明 source-file signature 的 `CMAKE_ASC_FLAGS` 不会可靠传播 `--cce-res-usage`，因此不采用该路径。

probe 使用：

```cmake
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
```

这样只要求完成 `.asc` 编译，不依赖 probe 可执行文件运行。

CANN 9.1 的 ASC CMake 模块还需要为 probe 显式传入 `CMAKE_AR=/usr/bin/ar`、`CMAKE_RANLIB=/usr/bin/ranlib` 和 `CMAKE_ASC_COMPILER_AR=/usr/bin/ar`；否则编译器虽然输出资源行，静态库归档阶段仍可能生成空命令并导致 probe 失败。

### 9.2 CMake 输入契约

调用方必须显式提供 probe 的完整身份，不能要求 parser 猜测源码语义：

```cmake
asc_occupancy_add_kernel_variant(
  NAME <stable-name>
  SOURCE <kernel.asc>
  KERNEL_SYMBOL_REGEX <unique-simt-entry-regex>
  LAUNCH_BOUNDS <256|512|1024|2048>
  STATIC_UB_BYTES <optional-known-value>
  INCLUDE_DIRECTORIES <...>
  COMPILE_DEFINITIONS <...>
  COMPILE_OPTIONS <...>
)
```

- `LAUNCH_BOUNDS` 是调用方和源码共同声明的构建输入，不从 Bisheng 输出反推。
- `STATIC_UB_BYTES` 是可选的调用方已知值；未提供时生成资源记录中的 `static_ub_bytes_known=false`，API 只能报告已知约束下的可行性。
- 源码通过 `ASC_OCCUPANCY_LAUNCH_BOUNDS` 宏把该值用于 `__launch_bounds__`。
- probe translation unit 默认只允许一个目标 SIMT entry；多 kernel 文件必须提供能唯一匹配的 regex，否则配置失败。
- 架构、include、definitions 和 options 由该函数集中保存，并同时应用到 probe 与正式目标。
- 资源头记录稳定 target name、Bisheng symbol、`launch_bounds`、寄存器数、Stack size、架构和编译输入摘要。

改变 `launch_bounds` 不是运行时行为。CMake 为每个声明的档位生成独立 kernel 变体和独立 benchmark executable，例如：

```text
example_lb256_occupancy_bench
example_lb512_occupancy_bench
example_lb1024_occupancy_bench
example_lb2048_occupancy_bench
```

每个变体在独立二进制中使用相同逻辑和输入契约，避免符号冲突。聚合脚本运行这些可执行文件并合并结果；runner 不会试图在运行时修改已编译的 `launch_bounds`。

### 9.3 解析规则

严格匹配 Bisheng 当前输出：

```text
Function properties for <symbol>: Stack size: <N> bytes, Used register number: <M>
```

配置阶段在以下情况直接失败：

- probe 编译失败；
- 没有匹配到目标 kernel；
- 同一目标匹配到多个结果；
- Stack size 或寄存器数不是非负整数；
- `launch_bounds` 不在设备支持范围；
- Bisheng 输出格式变化；
- probe 使用的架构、宏或关键编译选项与正式目标不同。

完整输出保存在构建目录中的 `occupancy-probe.log`，不能只保留解析结果。

### 9.4 生成常量

生成头文件示例：

```cpp
#pragma once

#include <ascend_occupancy/asc_occupancy.h>

inline constexpr AscKernelResourceUsage kExampleKernelResources{
    ASC_OCCUPANCY_ABI_VERSION,
    sizeof(AscKernelResourceUsage),
    "_Z..._simt_entry",
    512,
    48,
    0,
    0,
    false,
};
```

生成常量只允许在 host 侧用于资源分析和结果输出，不允许进入 device kernel 的控制流、数组大小、展开次数或模板选择。否则第二次编译可能因为注入值改变 kernel 资源使用量，形成自反馈。

### 9.5 增量构建与一致性

- kernel 源码、架构、`launch_bounds` 和影响 device 代码的宏加入 `CMAKE_CONFIGURE_DEPENDS`。
- 上述输入变化时重新执行 probe。
- probe 和正式目标必须共享同一份编译选项集合。
- 正式编译继续打开 `--cce-res-usage`。
- 第一版在构建日志中保留正式输出，并在实机验收时由测试流程人工核对生成值；不宣称已经自动比较。
- 若要自动核对正式编译输出，可增加编译输出 capture wrapper；它属于构建可靠性增强，不应改变 API 语义。

## 10. 自动性能测试流程

### 10.1 为什么只能半自动生成

CMake 可以自动生成资源常量、候选列表和通用 benchmark runner，但不能从 kernel symbol 推断：

- 参数列表和参数语义；
- 输入/输出内存大小；
- 测试数据初始化；
- Grid 与业务 shape 的映射；
- 输出正确性标准；
- 每个候选是否完成相同工作。

因此用户需要提供一个很薄的 adapter：

```cpp
prepare_inputs();
reset_iteration_state();
launch_kernel(grid_blocks, block_threads, stream);
validate_outputs();
work_items();
enumerate_grid_blocks(block_threads, output, capacity);
```

核心 API 只生成 Block 候选。adapter 的 `enumerate_grid_blocks` 负责结合业务 shape 和每线程工作量，把一个 Block 候选映射为一个或多个完成相同语义工作的 Grid 候选；runner 不猜测 Grid，也不会默认使用 `ceil(work_items / block_threads)`。

### 10.2 生成内容

CMake 自动完成：

1. 生成 kernel 资源头文件。
2. 生成候选线程数和 `launch_bounds` 清单。
3. 根据声明的 `launch_bounds` 档位生成独立 kernel 变体。
4. 把通用 `benchmark_main.asc.in` 与 kernel adapter 组合为包含 `main` 的 `.asc`。
5. 构建每个 `<kernel>_lb<N>_occupancy_bench` 可执行文件。
6. 生成聚合运行脚本、运行说明和可选 msopprof 命令。

正常构建不自动运行设备测试，避免在无 NPU 的构建机失败，也避免普通编译占用共享设备。

### 10.3 用户执行

```bash
./build/<kernel>_occupancy_bench \
  --warmup 10 \
  --iterations 100
```

runner 对每个候选：

1. 使用相同输入和相同工作量；
2. 先进行正确性校验；
3. 统一 warmup；
4. 重复计时；
5. 输出每个样本、中位数和范围；
6. 生成 JSON/CSV；
7. 标记实测最佳候选。

需要 DCache、流水线和指令等详细指标时，用户再用 `msopprof` 对单个候选运行。详细 profiler 数据不与低开销 timing 数据混为一谈。

### 10.4 Benchmark 协议与失败处理

- 每个 kernel 变体使用独立进程，避免旧模块、失败状态和前一变体资源影响下一变体。
- 一个变体进程内可以复用输入和分配，但每个 measured iteration 前必须调用 adapter 的 `reset_iteration_state()`；无状态 kernel 可提供空实现。
- 低开销计时使用同一 stream 上的 ACL event，计时边界只覆盖 kernel launch 到完成；初始化、H2D、结果 D2H 和完整校验不计入 kernel latency。
- warmup 完成后再创建 measured samples，且每个候选使用相同 warmup 和 iteration 数。
- adapter 定义精确比较或 dtype 对应的绝对/相对容差，并保证所有候选执行相同语义工作。
- launch/API 失败时记录 ACL 错误码和错误文本，该候选判无效；进程返回非零，聚合器继续收集其他独立变体。
- 正确性失败时记录首个 mismatch、最大绝对/相对误差并排除该候选，不参与性能排序。
- 设备不可用、分配失败或结果文件不可写属于整次运行失败，不生成“最佳候选”。
- “实测最佳”定义为所有正确候选中 latency 中位数最低者；同时报告样本范围。若第一名与第二名的差异小于两者范围重叠所揭示的抖动，不宣称稳定胜出，只报告并列候选。

JSON/CSV 至少记录：环境标识、kernel/变体、`launch_bounds`、编译资源、Grid/Block、warmup、iterations、每个 latency 样本、正确性状态、错误信息、中位数、最小/最大值和原始 profile 路径。

## 11. 目录设计

实现不放入任何算子目录，也不复用或接入 CannBench 现有 benchmark 目录。API、构建探测、性能对比代码和测试全部收敛在一个顶层独立子工程 `ascend_occupancy/` 中：

```text
ascend_occupancy/
  CMakeLists.txt
  README.zh-CN.md
  cmake/
    AscOccupancyBenchmark.cmake
    AscOccupancyProbe.cmake
    OccupancyProbeProject.cmake.in
    ParseBishengResourceUsage.cmake
    benchmark_main.asc.in
  include/
    ascend_occupancy/
      asc_occupancy.h
  src/
    asc_occupancy.cpp
  perf_tests/
    common/
      acl_event_runtime.h
      aggregate_results.py
      benchmark_runner.h
    register_spill/
      CMakeLists.txt
      register_spill_adapter.h
    launch_geometry/
      CMakeLists.txt
      launch_geometry_adapter.h
    loop_unroll/
      CMakeLists.txt
      loop_unroll_adapter.h
  test/
    CMakeLists.txt
    aggregate_results_test.py
    benchmark_runner_test.cpp
    perf_test_contract_test.cmake
    resource_model_test.cpp
```

三个仓库自带 benchmark 都使用 CMake；`benchmark_main.asc.in` 与 adapter 在配置期组合为包含 `main` 的 `.asc`：

- `register_spill`：对比 1024 档 spill 与 512 档无 spill；
- `launch_geometry`：扫描 Grid/Block Threads；
- `loop_unroll`：第一版对比 direct SIMT 的 loop 和手动展开；hybrid 可作为后续独立变体加入。

API 与 benchmark 都独立于 CannBench 的 operator plugin、CLI 和公共 backend。

该目录作为独立 CMake 工程使用，不修改仓库根构建入口：

```bash
cmake -S ascend_occupancy -B build/ascend_occupancy \
  -DASCEND_CANN_PACKAGE_PATH=<cann-path> \
  -DASC_OCCUPANCY_BUILD_PERF_TESTS=ON \
  -DCMAKE_ASC_ARCHITECTURES=dav-3510
cmake --build build/ascend_occupancy --parallel
ctest --test-dir build/ascend_occupancy --output-on-failure
```

Host/parser 测试由该子工程的 CTest 管理；仓库原有 `pytest -q` 继续作为 CannBench 回归检查，不负责发现 C++/ASC 测试。安装和导出 package config 不纳入第一版，使用方先通过 `add_subdirectory()` 或独立构建产物接入。

## 12. 原始六类优化假设的处理

原始 `occupancy.md` 提出六类寄存器优化写法。第一版将其保留为实验路线，不把它们写成已证实规律。

### 12.1 Pointer Aliasing / `__restrict__`

比较普通指针与 `__restrict__`。需要同时检查生成寄存器数、Stack size、正确性和性能。只有调用方能保证不别名时才允许使用优化版本。

### 12.2 局部作用域收缩

比较长生命周期变量与显式局部作用域。现代 SSA 优化可能已经完成相同生命周期分析，因此源码改写不保证降低寄存器数。

### 12.3 Loop Unrolling

比较编译器循环、部分展开、`#pragma unroll 1` 和手动展开。本次实测中手动展开明显变慢，因此这一项必须作为多候选实验，不能默认展开有利。

### 12.4 UB 充当显式临时存储

比较自动 Stack spill 与显式 `__ubuf__` 临时区。需要额外验证 UB 容量、Bank 冲突、同步和访存成本。它不是无条件优于 Stack 的“虚拟寄存器”。

### 12.5 显式变量复用

比较独立变量名与源码级变量复用。LLVM/Bisheng 的 SSA 和寄存器分配器可能重新拆分变量，因此必须以 `--cce-res-usage`、生成代码和实测为准。

### 12.6 数据类型降精度

比较 float 与 half/bfloat16，但必须先定义精度容差和语义等价边界。降精度同时改变计算、访存和输出契约，不能仅视为寄存器优化。

## 13. 测试与验收

### 13.1 Host 单元测试

- 950 寄存器档位边界：256/257、512/513、1024/1025、2048。
- Block Threads 的 0、非 Warp 对齐、上限和越界值。
- `launch_bounds` 小于实际 Block Threads。
- 动态 UB 为 0、恰好等于上限和超过上限。
- 静态 UB 已知和未知时的 `ub_capacity_check_complete`。
- Stack size 为 0 和非 0。
- 0/1 resident block 语义。
- theoretical occupancy 的尾 Warp 计算。
- 候选去重、排序和容量查询。

### 13.2 CMake/parser 测试

- 正常 Bisheng 输出解析。
- 多 kernel 输出按目标 symbol 选择。
- 缺失字段、重复字段、负数和格式变化时失败。
- 生成头文件内容稳定。
- 源码或关键编译宏变化会触发重新 probe。

### 13.3 950 实机测试

- 三个 benchmark 所有候选先通过正确性。
- 每个性能候选至少采集 3 个独立样本。
- 短 kernel 输出所有样本和中位数，不只报告最佳样本。
- 记录频率、CANN、Bisheng、驱动、固件和原始产物路径。
- profile 前列出所有 kernel，不按名字先过滤。
- 保留 probe 与正式编译资源输出；第一版实机报告人工核对结果，后续增强加入自动一致性检查。

### 13.4 仓库验证

- `pytest -q` 继续通过。
- 新代码只位于顶层 `ascend_occupancy/` 和对应设计文档。
- 不修改 `src/cannbench/cli.py`、公共 backend、core config 或具体 operator plugin。
- 不引入具体算子名分支。

## 14. 风险与边界

### 14.1 Bisheng 输出不是稳定结构化 ABI

`--cce-res-usage` 当前输出是文本。工具链升级可能改变文本格式，解析器必须 fail closed，并保留原始日志，不能静默使用默认值。

### 14.2 probe 与正式编译漂移

只要 source、宏、架构或编译选项不同，资源结果就可能不同。构建系统必须集中管理这些输入，生成常量不得参与 device 代码生成。

### 14.3 CMake ASC `try_compile` 兼容性

CANN 9.1、CMake 3.28.3 和 `dav-3510` 上已经用最小 repro 验证 project-signature `try_compile(... OUTPUT_VARIABLE ...)` 能完整保留 `--cce-res-usage` 输出。隔离的嵌套 CMake probe 加 `execute_process` 也已验证可用，继续作为工具链变化后的回退路径。该结论绑定当前版本；升级 CANN/CMake 后仍需重跑 parser 和 probe 测试。

### 14.4 节点测量长尾

本次 sincos 数据存在 170-180 us 长尾。性能判断应使用中位数、范围和独立重跑，不能使用单轮或最佳值。

### 14.5 occupancy 不是性能模型

高理论 occupancy 可能增加 Warp 排队、降低可用寄存器或改变 Grid 并行核数。任何 API 字段、日志和 README 都必须避免把 occupancy 写成性能保证。

### 14.6 当前只支持 950/dav-3510

寄存器档位和一核一 Block 语义必须绑定架构。其他 SoC 只有在取得对应设备证据后才能增加，不应默认复用 950 参数。

## 15. 第一版实施状态

第一版已经按以下范围实现；未列入第一版的能力继续保持显式边界。

### 实现前置验证门槛

以下四个最小可行性 probe 已在 20002 节点完成；原始日志保存在 `/tmp/occupancy-feasibility-20260801T153037+0800`：

1. project-signature `try_compile(... OUTPUT_VARIABLE ...)` 成功捕获 `--cce-res-usage`；嵌套 CMake 加 `execute_process` 回退路径也成功。
2. 950 所需的 Vector Core、Warp、线程和 UB 属性均可通过 ACL 查询；有效调用返回 0，受控无效属性和设备返回 `107000`。
3. `alpha`/`beta` 两个 symbol 的 256/1024 两档共四个变体均能唯一映射资源行。
4. 同一 `alpha_lb256` 变体的 probe 与正式编译 symbol、寄存器和 Stack size 人工核对一致。

### 已完成：资源分析闭环

- 独立目录和 CMake 工程；
- `try_compile + --cce-res-usage`；
- 严格 parser 和生成头文件；
- 设备属性查询；
- `ascOccupancyAnalyzeKernel`；
- host 单元测试；
- register-spill 实机样例。

### 已完成：候选与 benchmark

- launch-bounds 候选生成；
- Block/Grid 候选生成；
- 通用 benchmark runner；
- launch-geometry 和 loop-unroll 样例；
- JSON/CSV 输出。

### 后续增强：回归与扩展

- 正式编译资源自动一致性检查；
- CI 资源回归阈值；
- 可选 msopprof 辅助脚本；
- 取得证据后支持其他 Ascend 架构。

## 16. 最终结论

1. Ascend 950 需要的不是 CUDA Occupancy API 的逐函数复刻，而是一个诚实的资源分析与调优入口。
2. 一核一 Block 使 MaxActiveBlocks 退化为 0/1，使 MaxPotentialBlockSize 容易退化为最大合法线程数。
3. `--cce-res-usage` 能提供当前最关键的 kernel 资源证据；通过 CMake 双阶段编译可以把文本输出转成 host 可用常量。
4. API 可以可靠回答“是否满足已知启动约束、是否 spill、理论 Warp 占用率、哪些档位值得重编译”。
5. API 不能可靠回答“哪个线程数一定最快”；性能最优值必须由 benchmark 或 autotune 得到。
6. 20002 节点的当前三个 case 表明：无 spill、最大线程数和手动展开都不能单独作为性能更快的充分条件。
7. 因此值得实现第一版，但必须把确定的资源结论、资源候选和实测最优值明确分层。
