# Ascend 算子开发典型运行时错误与定位案例

本文从 CannBench 相关 Codex session 中提取算子开发期间实际遇到的 AICore Error、Kernel CoreDump、Kernel 卡死及邻近问题，用于：

- 评估算子开发过程中的典型错误覆盖度；
- 选择可稳定复现的问题定位演示案例；
- 建立从表象、触发条件到根因和修复方案的排查路径。

本文是工程案例总结，不是 CANN 官方错误码手册。同一错误码可能对应不同根因，不能只凭错误码直接下结论。

## 1. 扫描范围与完整性

本次扫描以 Codex rollout 的 `cwd` 为准，覆盖：

| 工作目录 | 顶层 session | subagent session | 合计 |
| --- | ---: | ---: | ---: |
| `/root/aiagent/cannbench` | 23 | 149 | 172 |
| `/root/aiagent/cannbench-2` | 5 | 0 | 5 |
| 总计 | 28 | 149 | 177 |

- 数据时间范围：2026-06-16 至 2026-08-11。
- rollout 总量约 812 MB。
- 所有 `cwd` 精确等于上述两个目录的 session 均已扫描。
- 另外排查了所有文本中引用 CannBench、但 `cwd` 不同的 3 个 session：
  - `/root/aiagent/HierarchicalKV-ascend`：CannBench 仅为旁路引用，没有新增 CannBench 故障案例；
  - `/root/aiagent/dsa-fa`：DSA 调研，没有相关运行时故障案例；
  - `/root/aiagent/dsa-fa/cannbench`：早期 predecessor 开发，只有循环导入等框架问题，没有新增 AICore Error、CoreDump 或 Kernel 卡死案例。

结论：目标范围和交叉引用范围均已扫描完。案例按同一故障的多次复现、续接 session 和 subagent 讨论去重，不把重复日志计为新案例。

## 2. 术语边界

- **Linux CoreDump**：Host 进程收到 `SIGSEGV`、`SIGABRT` 等信号后产生的进程 core 文件。根因可能在自定义 OPP、动态库 ABI、CANN runtime 或算子实现。
- **AICore dump/error info**：设备侧异常后由 runtime、driver 或 device log 报出的 AIC/AIV、MTE、Cube、Fixpipe 等错误信息，不等同于 Linux 进程 core。
- **Kernel 卡死**：本文同时覆盖真正的同步死锁、设备执行超时，以及 profiler 固定超时造成的“假卡死”。三者必须通过脱离 profiler 的单次同步执行来区分。
- **SIMT 相关**：区分“根因直接位于 SIMT/VF 路径”和“算子含 SIMT，但根因位于 Cube、Host ABI 或 profiler”。

## 3. 案例总览

| 编号 | 主要现象 | 根因摘要 | SIMT 关系 | 状态 | 演示推荐 |
| --- | --- | --- | --- | --- | --- |
| E01 | `507015`、timeout/trap、AIV scalar 指令异常 | AIC/AIV 跨核生产消费缺少真正的跨核同步 | 混合 AIC/AIV，直接相关 | 已定位并修复 | S |
| E02 | `error 171`、Cube/Fixpipe multi-bit ECC | L0A/L0B/L0C buffer 生命周期事件缺失 | 混合 Cube + Vector/SIMT，直接相关 | 已定位并修复 | S |
| E03 | `error 82`、MTE 写地址越界 | 将 `expand`/`narrow` 非连续 view 当成 dense tensor | SIMT gather-pack 数据路径，直接相关 | 已定位并修复 | S |
| E04 | `error 82`、MTE 写地址越界 | Cube Matmul 外维、转置或 offset 建模错误 | SIMT 邻接，根因在 AIC/Cube | 未完全闭环 | 不作为主案例 |
| E05 | Mean/Concat/Copy/`.cpu()` 等无辜后续操作报错 | 异步执行导致错误位置漂移 | 通用模式，实测多见于 SIMT/混合 kernel | 已形成定位方法 | S，作为诊断层 |
| E06 | Host `SIGSEGV`，workspace 查询处崩溃 | 自定义 OPP 与 `torch_npu` schema/ABI 不匹配 | 非 SIMT | 已定位并修复 | A |
| E07 | `msopprof` 下 `507015`，VF 参数非法 | 一个 ELF 中包含多个 VF/SIMT 入口 | 直接 SIMT | 已定位并规避 | A |
| E08 | `507046 / 0x7030010`，约 10 秒超时 | profiler replay 固定超时，正常 kernel 本身约 339 秒 | 目标是混合 SIMT kernel，根因在 profiler | 已定位 | A |
| E09 | 无显式 AICore Error，结果随机漂移 | SIMT GM store 对后续 MTE/Cube 不可见 | 直接 SIMT | 已定位并修复 | B |
| E10 | SSH 侧 255，真实进程 134/core dump | CANN 更新后 Softmax SIMT/hybrid 路径 abort | SIMT 路径，疑似环境兼容 | 未闭环 | 不作为主案例 |

推荐等级：`S` 表示最适合标准定位演示，`A` 表示适合专题演示，`B` 表示有价值但复现和解释成本较高。

## 4. 逐项案例

### E01：AIC/AIV 跨核同步缺失导致 timeout/trap

**表象**

- HD=128 等路径出现 `507015`；
- 伴随 kernel timeout、trap 或 AIV scalar 指令异常；
- 小 shape、低并发或改变流水顺序后可能偶尔通过，表现不稳定。

**触发场景**

混合 AIC/AIV kernel 中，AIC 生产中间数据，AIV/SIMT gather 侧消费数据。代码使用了 `Mutex` 或核内事件，但生产者与消费者实际位于不同核/不同 engine。

**根因**

`Mutex` 只能解决允许范围内的 kernel-local pipeline 同步，不能自动建立 AIC 与 AIV 之间的跨核完成关系。消费者可能在生产数据和 cache 可见性建立之前读取 buffer。

**历史解决方案**

当时通过完整的 `AIC_1_2` 调度、TSCM/L1 ping-pong、跨核 set/wait 和流水事件补齐生产消费协议后消除故障。

**当前实现约束**

上述历史修复用于解释根因，不应直接作为 CannBench 新算子的代码模板。当前仓库要求新设计收敛到 `C API + Tensor API + SIMT API`，不得新引入 `CrossCoreSetFlag`、`CrossCoreWaitFlag` 等 Basic API 跨核同步。优先通过分核职责、buffer 所有权和 launch/phase 拆分消除跨核依赖。

**演示价值**

非常高。它能展示“看似有锁但仍然卡死”、核内同步与跨核同步的区别，以及如何用 shape 和流水裁剪定位生产消费边界。

### E02：L0 buffer 生命周期事件缺失导致 ECC

**表象**

- runtime 报 `error 171`；
- 设备日志指向 Cube/Fixpipe multi-bit ECC；
- 常被误判为硬件 ECC 或随机设备故障。

**触发场景**

Cube pipeline 复用 L0A/L0B/L0C buffer 时，只保证了计算顺序，没有完整表达 MTE1、MTE2、M 和 Fixpipe 对 buffer 的占用与释放关系。多轮或 ping-pong 执行更容易触发。

**根因**

缺失 `M_MTE1`、`MTE1_MTE2`、`FIX_M` 等生命周期事件，导致上一阶段仍在读取或写回时下一阶段覆盖同一片 L0 buffer。ECC 是越界/覆盖后的设备侧表象，不代表物理内存一定损坏。

**解决方案**

补齐每个 buffer 从搬入、计算到 Fixpipe 写回的完整所有权协议；逐个关闭 ping-pong 或流水阶段验证是哪条依赖缺失。历史实现使用事件补齐，后续新实现需按仓库当前 API 边界选择等价的允许机制。

**演示价值**

非常高。错误特征明确、反直觉强，适合演示如何从 ECC 反推 pipeline 生命周期问题。

### E03：非连续 view 被当成 dense tensor 导致 MTE 越界

**表象**

- runtime 报 `error 82`；
- 设备日志为 MTE write address out of range；
- 输入 shape 表面合法，分配大小也看似足够。

**触发场景**

Host 侧通过 `expand`、`narrow`、切片或类似操作构造输入，得到带特殊 stride 或 storage offset 的非连续 view。SIMT gather-pack 或后续 score 路径按连续 dense layout 解释地址。

**根因**

kernel 的地址模型与真实 tensor stride/storage 不一致。尤其是 `expand` 产生的 stride 0 维度，不能仅凭 shape 推导物理地址。

**解决方案**

- 在算子 ABI 只接受连续输入时，调用前显式 `.contiguous()`；
- 在入口增加 `is_contiguous()`、stride、storage offset 和实际 storage 大小检查；
- 如果必须支持 strided tensor，将 stride 明确加入 tiling 和 kernel 地址计算，不能隐式假设 dense。

**演示价值**

最高。容易构造最小复现，错误稳定，能清楚展示 shape 正确不等于地址模型正确。

### E04：Cube Matmul 地址建模错误导致相同的 error 82

**表象**

同样是 `error 82` 和 MTE 地址越界，但对输入做 `.contiguous()` 后仍不消失。

**触发场景**

混合算子的 Cube Matmul/GEMV 路径改变外维、转置关系、尾块或基地址 offset 后触发。部分 shape 正常，跨越特定 tile 边界后失败。

**根因**

候选根因集中在外维展开、GEMV/Matmul 建模、A/B 矩阵转置语义和 tile offset。它与 E03 共享错误码，但不是同一问题。

**处理建议**

- 先用连续输入排除 Host layout；
- 将 M、N、K 和 batch/outer 维缩到单 tile，再逐维放大；
- 对每个 GM/L1/L0 offset 做边界公式核对；
- 单独验证 transpose 前后的逻辑 shape 与物理 layout。

**状态与演示价值**

原 session 中没有形成完全闭环的唯一根因，因此不建议作为标准答案型演示。可作为高级开放式排查题。

### E05：异步错误漂移到后续无辜算子

**表象**

真正失败的自定义 kernel 返回后，异常可能在 Mean、Concat、InplaceCopy、`.cpu()`、下一次 launch，甚至进程退出时才暴露。

**触发场景**

默认异步 launch 下，Host 只完成任务下发，没有立即等待设备执行完成。第一个隐式或显式同步点替前面的错误“背锅”。

**根因**

报错栈反映的是错误被观察到的位置，不一定是错误发生的位置。

**解决方案**

```bash
ASCEND_LAUNCH_BLOCKING=1 <原始复现命令>
```

并在每个候选自定义算子之后立即执行 NPU synchronize。通过二分移同步点，定位第一个失败 launch。确认根因后再恢复异步执行，避免把同步调试模式误留在性能测试中。

**演示价值**

非常高，但它更适合作为 E01、E02 或 E03 的第一层诊断技巧，而不是单独的人造故障。

### E06：自定义 OPP 与 torch-npu ABI 不匹配导致 Host SIGSEGV

**表象**

- Python 进程收到 `SIGSEGV` 并产生 Linux core；
- 栈位于 `aclnnSparseFlashAttentionGetWorkspaceSize`；
- 还没有真正进入目标 AICore kernel，设备侧错误信息可能为空。

**触发场景**

加载了自定义 OPP，但 PyTorch extension 注册的 schema、参数顺序、可选参数或动态库版本与当前 `torch_npu`/CANN 环境不一致。

**根因**

Host ABI/注册不匹配。workspace 查询函数崩溃不等于 workspace 大小计算本身有 bug。

**解决方案**

- 确认加载与当前环境匹配的 `_C_ascend` schema；
- 将版本差异封装在算子本地 ABI adapter 中；
- 对照实际 schema 检查参数数量、顺序、dtype 和 optional 语义；
- 在跑 kernel 前先做最小的 op import、schema 查询和 workspace-only 调用。

**演示价值**

高。适合展示如何先判断 Host crash 还是 Device crash，避免一上来调 AICore 源码。

### E07：一个 ELF 中多个 VF 入口导致 msopprof 失败

**表象**

- 正常执行可能通过；
- 使用 `msopprof` 时出现 `507015`；
- 设备日志指向 VEC VF instruction parameter invalid。

**触发场景**

同一个 device `.so`/ELF 中链接了多个包含 `vector_simt_entry` 的 `.asc` 对象，profiler replay 或符号选择无法稳定确定目标 VF 入口。

**根因**

产物组织违反了 profiler 对 VF 入口唯一性的实际要求。它不是普通输入 shape 或计算精度问题。

**解决方案**

每个 device `.so` 只链接一个包含 VF/SIMT entry 的 `.asc`；不同版本或不同 kernel 入口拆成独立 device 产物，由 Host 侧选择加载。

**演示价值**

高。适合做“正常跑通过，但 profiler 报设备错”的工具链专题案例。

### E08：msopprof 固定 replay 超时造成假卡死

**表象**

- profiler 约 10 秒后报告 `507046 / 0x7030010`；
- 看起来像 kernel 卡死或设备执行超时；
- 脱离 profiler 后，单次执行能完成，但观察到约 339 秒的真实耗时。

**触发场景**

对极慢的混合 Cube/SIMT kernel 做 `msopprof` replay，kernel 正常耗时远超 profiler 的固定等待窗口。

**根因**

这是 profiler-side timeout，不是同步死锁。工具给出的错误码不能替代真实 wall time 测量。

**解决方案**

- 先脱离 profiler 做单次同步执行并记录 wall time；
- 缩小 shape、减少 replay/launch 次数或使用能调整超时的采集方式；
- 只有在无 profiler 时也无法在合理上界内完成，才继续按真正卡死排查。

**演示价值**

高。非常适合训练“卡死、超时、只是很慢”三者的区分方法，但演示时应缩小 shape，避免真的等待数分钟。

### E09：SIMT GM store 对后续 MTE/Cube 不可见

**表象**

- 没有明确 AICore Error；
- 同一输入多次运行结果漂移，或者只有部分 tile 使用旧数据；
- 加普通线程 fence 后仍可能复现。

**触发场景**

SIMT VF 将中间结果写入 GM，随后 MTE/Cube engine 在同一混合 pipeline 中读取该区域。执行顺序看似正确，但 cache 可见性没有建立。

**根因**

普通 SIMT fence 只覆盖其定义的线程/内存顺序，不能保证 GM cache 内容已对后续 MTE/Cube engine 全局可见。

**解决方案**

在生产者完成点使用 `asc_dcci_entire` 建立所需 cache 可见性，并同时保留正确的 phase/engine 完成关系。DCCI 不能替代同步，同步也不能替代 cache flush。

**演示价值**

中等。技术价值高，但它通常表现为非确定性精度问题，自动判定和稳定复现比显式错误码案例困难。

### E10：CANN 更新后 Softmax SIMT/hybrid 路径 abort

**表象**

- 外层 SSH 只看到退出码 255；
- 目标机真实进程退出码为 134，并产生 core dump；
- fast/spatial 路径在 CANN 更新后失败。

**触发场景**

同一 Softmax v3 SIMT/hybrid 代码在更新后的 CANN 环境执行。旧代码和新代码均失败，说明回归不随单次源码改动消失。

**根因判断**

现有证据更支持 CANN/toolchain/runtime 兼容性变化，而不是新提交独有的 kernel bug，但 session 中未完成最终根因闭环。

**处理建议**

- 在目标机读取真实退出码和 core backtrace，不以 SSH 255 作为根因；
- 固定代码，做旧/新 CANN A/B；
- 对比编译器、runtime、driver、OPP 和 `torch_npu` 版本矩阵；
- 先用最小 kernel 验证 SIMT/VF 入口和 launch ABI，再恢复完整 fast/spatial 路径。

**演示价值**

不适合标准定位演示：环境依赖强且根因未闭环。适合作为“如何识别环境回归”的讨论案例。

## 5. 哪些案例涉及 SIMT

### 5.1 根因或故障路径直接涉及 SIMT：6 项

- **E01**：AIC/AIV 混合路径，SIMT/AIV gather 消费侧参与跨核同步问题；
- **E02**：Cube 与 Vector/SIMT 混合流水，buffer 生命周期不完整；
- **E03**：SIMT gather-pack 按错误的 dense layout 访问输入；
- **E07**：多个 `vector_simt_entry` 导致 profiler/VF 入口冲突；
- **E09**：SIMT GM store 与后续 MTE/Cube 的 cache 可见性；
- **E10**：Softmax v3 SIMT/hybrid 路径在环境更新后 abort。

### 5.2 发生在 SIMT/混合算子上下文，但根因不在 SIMT：3 项

- **E04**：算子支持 SIMT 混合路径，但已知故障集中在 AIC/Cube Matmul 地址建模；
- **E05**：异步错误漂移是通用 runtime 行为，只是本次大量实例来自 SIMT/混合 kernel；
- **E08**：被 profile 的目标是混合 Cube/SIMT kernel，但超时根因在 `msopprof`。

### 5.3 与 SIMT 无直接关系：1 项

- **E06**：自定义 OPP、`torch_npu` 和 ACLNN 的 Host schema/ABI 问题。

因此，10 项中有 6 项的故障路径直接涉及 SIMT，另有 3 项是 SIMT 邻接问题；只有 E06 与 SIMT 无直接关系。不能据此把 9 项都归因于 SIMT。

## 6. 推荐的问题定位演示集

### 6.1 S 级：建议优先建设

| 案例 | 推荐演示主题 | 推荐原因 | 建议控制变量 |
| --- | --- | --- | --- |
| E03 | 非连续 view -> MTE 越界 | 复现稳定、修复小、因果链清楚 | 连续/非连续输入，shape 保持一致 |
| E02 | L0 生命周期缺失 -> ECC | 能展示 pipeline 事件和 buffer 所有权 | 单/双 buffer，逐步打开流水 |
| E01 | 核内锁误作跨核同步 -> timeout/trap | 能展示混合 AIC/AIV 的核心难点 | HD=64/128、单阶段/跨阶段 |
| E05 | 异步错误位置漂移 | 几乎适用于所有设备侧案例 | blocking 开关、同步点二分 |

E05 建议叠加在 E01、E02 或 E03 上：先展示错误落在后续算子，再开启 blocking 把错误拉回真实 launch。

### 6.2 A 级：适合专题演示

| 案例 | 主题 | 使用建议 |
| --- | --- | --- |
| E06 | Host ABI mismatch -> SIGSEGV | 强调 Host/Device 分流和 schema 校验 |
| E07 | 多 VF 入口 -> profiler 507015 | 准备一个错误 ELF 和拆分后的正确产物 |
| E08 | profiler 10 秒假卡死 | 使用缩小但仍超过 replay 阈值的 shape，避免 339 秒演示 |

### 6.3 B 级或不建议作为主案例

- **E09**：适合 cache 一致性专题，但非确定性输出不利于稳定演示；可通过循环运行和 hash/误差统计增强可观测性。
- **E04**：根因未完全闭环，不适合作为带标准答案的样例。
- **E10**：依赖特定 CANN 版本组合且未闭环，不适合作为日常定位演示。
- 非致命的 `RegisterFuncSymbol failed` 等噪声日志不建议单独做案例，除非能证明它与实际失败存在因果关系。

## 7. 通用定位流程

1. **先判断 Host 还是 Device**：`SIGSEGV` 且未进入 kernel，优先查 schema/ABI/动态库；出现 AIC/AIV/MTE/Cube 信息再进入设备侧排查。
2. **强制同步定位首个失败 launch**：开启 `ASCEND_LAUNCH_BLOCKING=1`，在每个候选 kernel 后立即 synchronize。
3. **脱离 profiler 单次运行**：先确认 kernel 是崩溃、死锁，还是仅仅超过 profiler 超时。
4. **检查 tensor 物理布局**：同时记录 shape、stride、storage offset、contiguous 状态、dtype 和实际 storage 大小。
5. **缩小到单 tile/单 phase**：关闭 ping-pong、减少核数，逐步恢复 MTE、Cube、SIMT 和 Fixpipe 阶段。
6. **核对 buffer 所有权**：明确每块 GM/L1/L0/TSCM buffer 的生产者、消费者、完成事件和复用时刻。
7. **分开验证同步与可见性**：同步解决先后关系，DCCI/cache 操作解决跨 engine 可见性，两者不可互相替代。
8. **最后核对环境矩阵**：代码不变而 CANN 更新后回归时，固定 compiler、runtime、driver、OPP、`torch_npu` 版本做 A/B。

## 8. 代表性 source session 索引

以下为去重后案例的代表性 rollout，完整扫描还包括对应的续接和 subagent session：

- `2026/07/08/rollout-2026-07-08T16-16-35-019f40cc-dd74-7751-9700-9fd304201de7.jsonl`：MTE 地址越界、异步报错定位等；
- `2026/07/31/rollout-2026-07-31T12-04-22-019fb658-386c-7573-913e-603b57623f90.jsonl`：Softmax CANN 更新后 SIGABRT/core dump；
- `2026/08/01/rollout-2026-08-01T11-38-20-019fbb66-bd00-70c3-9eeb-12dd6f70174f.jsonl`：多 VF/SIMT entry 与 profiler 故障；
- `2026/08/02/rollout-2026-08-02T19-47-17-019fc24c-c08f-7501-b458-ab9acd8b6e09.jsonl`：SIMT GM store 和 DCCI 可见性；
- `2026/08/05/rollout-2026-08-05T16-13-07-019fd0fb-be6c-7e53-b3f4-1193d3eaa416.jsonl`：跨核同步、L0 生命周期 ECC、profiler timeout 等集中排查；
- `2026/08/06/rollout-2026-08-06T22-47-26-019fd78b-1ca0-7832-8573-5bd818ab0530.jsonl`：自定义 OPP/`torch_npu` ABI SIGSEGV。

原始文件位于 `/root/.codex/sessions/`。索引只用于回溯证据；实际培训样例应将最小复现、环境版本、期望日志和修复 diff 固化到独立测试目录，避免依赖 Codex session 本身。
