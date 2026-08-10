# DSA V2 Sparse Attention 六项差距实施计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use
> `superpowers:executing-plans` to implement this plan task-by-task. Steps use
> checkbox (`- [ ]`) syntax for tracking.

**Goal:** 逐项验证五个尚未完成的 vLLM-Ascend 差距候选，只保留精度通过且两轮性能
不劣化的改动。

**Architecture:** 以 `main@7dbe8a2` 的 P1 selected128 三槽 rolling + PV512 为固定
基线，在同一个 operator-local mixed kernel 中依次测试 E1-E5。每个候选通过独立
red/green、真实设备 correctness 和 paired profile 后才成为下一候选的基线。

**Tech Stack:** Ascend C/Tensor/SIMT API、CANN 9.2、dav-3510、pytest、CannBench
framework profiling。

## Global Constraints

- 只修改 `src/cannbench/operators/builtin/sparse_attention/`、本实验文档和最终
  canonical published record。
- 不修改公共 backend、core、CLI 或 published schema。
- 不新增 Basic API include 或新的 API 家族；只复用当前 rolling source 已存在的
  过渡性搬运和同步设施。
- 每个性能候选必须先通过 all-ones、seed 7、seed 19 output/LSE 精度。
- 两轮 fresh-process BasicInfo candidate 均不劣于同组 baseline 才允许保留。
- 负面候选撤销源码和源码契约测试，但把数据和结论保留在实验文档中。

---

### Task 1: 固定实验基线和证据模板

**Files:**
- Modify: `docs/optimization/dsa-v2-sparse-attention-six-gap-experiments.zh-CN.md`
- Test: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v2_fused_source.py`

- [ ] 核对 worktree HEAD、状态和 baseline `_C.so`/kernel ELF provenance。
- [ ] 运行 operator-local 测试并保存准确的 pass/fail 数。
- [ ] 对 baseline 执行两轮 canonical BasicInfo，记录 raw artifact 路径、kernel、
  launch count、频率和两轮 latency。
- [ ] 将 baseline 数据写入实验文档并提交。

### Task 2: E1 Gather-produced Validity Mask

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v2_fused_source.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Modify: `docs/optimization/dsa-v2-sparse-attention-six-gap-experiments.zh-CN.md`

- [ ] 新增源码契约测试，要求每个 gather quarter 生成一个 `uint32_t` mask、mask 使用
  pair/slot offset、softmax 不再接收 `indices`，并确认测试因实现缺失而失败。
- [ ] 在 workspace 常量、gather 和 softmax 中实现四 mask 的生产、发布和寄存器复用。
- [ ] 运行目标测试至通过，并运行完整 operator-local SIMT 测试目录。
- [ ] 远端干净编译；执行 all-ones、seed 7、seed 19 精度。
- [ ] 执行两轮 paired BasicInfo；满足门槛则提交源码，否则恢复源码。
- [ ] 无论接受或拒绝，都记录数据、provenance 和 raw artifacts 并提交文档。

### Task 3: E2 Lane-local Score Cache

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v2_fused_source.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Modify: `docs/optimization/dsa-v2-sparse-attention-six-gap-experiments.zh-CN.md`

- [ ] 新增源码契约测试，要求 max pass 产生四个固定局部 scaled-score 标量，exp pass
  不再读取 `scores[...]` 或重复乘 scale，并观察预期失败。
- [ ] 以固定标量和 mask bit 实现 lane-local score cache，不手工展开 head 循环。
- [ ] 运行目标测试、operator-local 测试、远端编译和资源元数据检查。
- [ ] 执行三组精度和两轮 paired BasicInfo；按门槛接受或恢复。
- [ ] 记录结果和 raw artifacts 并提交。

### Task 4: E3 AIC Direct Query GM-to-L1

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v2_fused_source.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Modify: `docs/optimization/dsa-v2-sparse-attention-six-gap-experiments.zh-CN.md`

- [ ] 新增源码契约测试，要求 rolling AIC 接收 Query、使用 `nValue=64`、
  `dValue=576`、`srcDValue=query_tokens*576` 的 ND2NZ，并删除 rolling AIV Query VF、
  UB-to-L1 copy 和 Query-ready flag。
- [ ] 运行测试确认因 direct path 缺失而失败。
- [ ] 实现 canonical rolling AIC Query direct copy，通用路径不变。
- [ ] 运行目标测试、operator-local 测试、远端编译和三组精度。
- [ ] 执行两轮 paired BasicInfo；按门槛接受或恢复。
- [ ] 记录结果和 raw artifacts 并提交。

### Task 5: E4 QK128 Double Staging

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v2_fused_source.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Modify: `docs/optimization/dsa-v2-sparse-attention-six-gap-experiments.zh-CN.md`

- [ ] 新增源码契约测试，固定两个 QK128 L0A/L0B slot、交替 slot 和独立 event，确认
  测试先失败。
- [ ] 实现 `128+128+128+128+64` QK 累加及双 staging，不改变 L0C/score layout。
- [ ] 运行目标测试、operator-local 测试和远端编译，核对 L0/Stack/register 资源。
- [ ] 执行三组精度和两轮 paired BasicInfo；按门槛接受或恢复。
- [ ] 记录结果；若回退，明确 MMAD 次数与 overlap 的权衡并提交文档。

### Task 6: E5 PV-free 同步收敛

**Files:**
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/test/test_sparse_attention_v2_fused_source.py`
- Modify: `src/cannbench/operators/builtin/sparse_attention/simt/v2/aten_dsa_sparse_attention_v2/csrc/simt/sparse_attention_head64_fused_hd576.asc`
- Modify: `docs/optimization/dsa-v2-sparse-attention-six-gap-experiments.zh-CN.md`

- [ ] 新增源码契约测试，要求 output update 后由 `PIPE_V` 发布 PV-free，且该位置没有
  V-to-MTE3 notify/wait；确认测试先失败。
- [ ] 只修改 PV-free 发布边界，保持其他 ready/free 和搬运事件原样。
- [ ] 运行目标测试、operator-local 测试、远端编译和三组精度；hang 视为直接拒绝。
- [ ] 执行两轮 paired BasicInfo；按门槛接受或恢复。
- [ ] 记录结果和 raw artifacts 并提交。

### Task 7: 最终组合验证和发布

**Files:**
- Modify: `docs/optimization/dsa-v2-sparse-attention-six-gap-experiments.zh-CN.md`
- Modify if accepted: `published/opbench-ascend-950pr-simt-v2-dsa_decode-realistic-bfloat16/meta/benchmark-records.json`

- [ ] 从最终 retained source 做一次全新远端编译，核对 binary provenance。
- [ ] 重跑 all-ones、seed 7、seed 19 和两轮相对 `main@7dbe8a2` 的 BasicInfo。
- [ ] 若最终组合回退，按 E5 至 E1 的逆序移除最近保留项并重复验证。
- [ ] 运行 operator-local 测试、`pytest -q` 和 `git diff --check`。
- [ ] 搜索公共层 concrete operator hardcoding，并确认没有公共层修改。
- [ ] 仅在完整 Sparse Attention 和 DSA decode workflow 均不劣化时更新 published
  canonical latency，保持 schema 和 run id 不变。
- [ ] 将最终完成度、接受/拒绝项、性能和 artifact 路径写入实验文档并提交。
