# 历史 Correctness Workaround 复验规格

## 目标与根因

本案例用 `[4099, 257]` `float32` row sum 展示历史 correctness workaround 的正确生命周期。旧实现的 scratch ownership 只有 `threadIdx.x`，直接把 launch 从一个 block 扩为多 block 时，不同 block 会重叠写同一 512-slot scratch。历史 workaround 因此限制为单 block；它保证正确，但长期限制并行度。

unsafe removal 是“只把 block dim 改为 64、仍用 `scratch[threadIdx.x]`”。该配置的原始失败条件由 host ownership 模型重现：64 个 block 的 slot interval 重叠。它不会作为可执行 device scenario，避免默认生成错误输出。

根因修复把 slot 改为 `block_idx * 512 + threadIdx.x`，为每个物理 worker 分配唯一 scratch ownership，然后才恢复 64-block 并行。不使用 barrier、flag 或跨核同步掩盖竞态。

## 数据与语义

- 输入 `[4099, 257]`、输出 `[4099]`，均为 `float32`。
- 每行按 col 0 到 256 顺序执行 FP32 sum；两个 scenario 算法语义、固定随机输入和完整输出完全一致。
- 4099 和 257 均为奇数，覆盖 row 边界、reduction tail 和多 block 尾部。
- host oracle 使用相同累加顺序；量化输入保证精确比较可用。
- 每个 scenario 在一个进程内重复 launch 3 次，最后验证完整输出，覆盖 scratch 重用。

## 场景与 launch

- `SCENARIO_NUM=0`：`row_sum_single_block_workaround_kernel`，1 block x 512 SIMT 线程，512-slot scratch，安全保留历史 correctness workaround。
- `SCENARIO_NUM=1`：`row_sum_unique_scratch_multiblock_kernel`，64 blocks x 512 SIMT 线程，32768-slot scratch，根因修复后恢复多 block。
- 两个 scenario 都是一个 kernel family，每次 application call 预期 3 个物理 launch。

## 复验顺序

1. host 枚举 unsafe multi-block ownership，确认旧 slot 发生碰撞。
2. host 枚举 fixed ownership，确认所有 physical worker interval 唯一。
3. 单 block workaround 在固定输入上重复 launch 并过完整 host oracle。
4. fixed 多 block 在相同输入上重复 launch并覆盖 tail、边界与完整 oracle。
5. 只有上述正确性证据成立后才比较 `Task Duration`。

## 性能边界

msopprof raw 中同一 kernel family 的 3 个 selected row 合计为一次 application call 的 device-work boundary，不包含 host、H2D/D2H、allocation 和 launch gap。结构化 parser 必须核对目标 kernel 恰好出现 3 行并输出总和，拒绝缺失、额外 replay 或旧 CSV 污染。脚本不设置 profiler warmup；性能结果必须记录 kernel 名、block dim、expected launch count 和 raw 路径。

预期瓶颈：Case 0 是单 block 可用并行度；Case 1 恢复多 block，但增加 scratch footprint，实际收益必须实测。不能引用候选文档历史百分比填表。
