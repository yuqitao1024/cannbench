# Sparse Attention Head64 P=4 Ping-Pong 设计

本文记录 V3.2 realistic decode Head64 P=4 fused kernel 的 K/V gather ping-pong
后续优化。基础融合设计见
[`HEAD64_DESIGN.zh-CN.md`](HEAD64_DESIGN.zh-CN.md)，本文只描述单 buffer 验收后
增加的双缓冲调度、同步生命周期和实测结果。

设计与验证日期：2026-07-29。

## 1. 范围

目标 shape 保持不变：

```text
B=2 Q=2 H=128 KV_H=1 C=32768 S=2048 Dqk=576 Dv=512 BF16
head_tile=64 selected_partitions=4
32 AIC + 64 AIV, 1024 SIMT threads/AIV
```

本阶段只优化 fused 主 kernel 内的 K/V gather：

- K/V 在 L1 各使用两个 slot。
- 下一 K tile gather 与当前 QK MMAD 重叠。
- 下一 V tile gather 与当前 PV MMAD 重叠。
- online softmax、FP32 running output 和独立 Combine 保持不变。
- AIC/AIV 继续使用 `CrossCoreFlag` mode 2；slot request flag 为 `0/1`，slot
  ready flag 为 `2/3`。

不融合 Combine，不改变 P=1/P=2，不扩展 prefill 或其他 family，也不改变默认 tuning。

## 2. Buffer 与握手

每个物理 MIX task 的 L1 layout 为 K 和 V 分别预留两个 slot。UB gather buffer 保持
单 slot，因为同一 AIV 上的 gather VF 与 UB-to-L1 copy 仍按本地 pipe 顺序复用 UB；
需要跨 tile 保活的是已写入 L1、等待 AIC 消费的数据。

每个 slot 的跨核生命周期为：

```text
AIC request(slot 0/1)
  -> AIV wait request
  -> AIV gather GM -> UB -> L1[slot]
  -> AIV publish ready(slot 2/3)
  -> AIC wait ready
  -> AIC copy L1[slot] -> L0
  -> AIC request(next slot)
  -> current MMAD
```

两个 AIV 都执行各自 32-row half 的 gather 和 output update，没有空闲 sub-AIV。

## 3. PV 消费完成边界

V ping-pong 的初版出现稳定精度错误：LSE 正确，但 output 最大绝对误差为
`1.30859375`。串行 V gather 后精度恢复，说明数据布局和 slot 地址正确，问题位于
overlap 生命周期。

根因是 `head64_fused_output_update_vf` 在 `PIPE_V` 上异步执行，而 AIV 原先在 VF
完成前就通过 `PIPE_MTE3` 发布 `kAivToAicReady`。AIC 收到 flag 后可以进入下一 tile
并覆写 `ub_pv`，导致仍在读取当前 PV 的 output-update VF 看到新数据。LSE 不依赖 PV，
因此不受影响。

修复要求在发布 PV 已消费 flag 前建立本地完成边界：

```cpp
asc_vf_call<head64_fused_output_update_vf>(...);
asc_sync_notify(PIPE_V, PIPE_MTE3, EVENT_ID1);
asc_sync_wait(PIPE_V, PIPE_MTE3, EVENT_ID1);
AscendC::CrossCoreSetFlag<2, PIPE_MTE3>(kAivToAicReady);
```

该同步只关闭当前 `ub_pv` 的生命周期，不取消下一 V gather 与当前 PV MMAD 的重叠。

## 4. 验证结果

目标环境为 `Ascend950PR_9589`、CANN 9.2.0、`dav-3510`。正式实现提交为
`9cca869`，device source SHA-256 为
`f7d573ae26b846e47f9bc9b45e4c0b59d57cb396f1c99ef3f0af5bbbca9c75bb`。

精度与稳定性：

- `valid_s64` 连续 5 次通过，output/LSE 最大绝对误差固定为
  `0.015625/0.0149774551`。
- reduced/boundary/物理核复用共 `36/36` 通过，最大 output/LSE 绝对误差为
  `0.0185546875/0.0199785233`。
- full realistic decode 的 `262144` 个 output 和 `512` 个 LSE 元素均为 0 mismatch；
  最大绝对误差为 `0.009765625/0.0092763901`。
- 本地完整测试为 `457 passed, 2 skipped`，operator-local source 测试为
  `121 passed`。

端到端 wall time 使用 3 次预热、7 轮、每轮 5 次调用并在每次调用后同步：

| 实现 | 中位延迟 |
| --- | ---: |
| single buffer `a38d5cd` | `0.583123 ms` |
| K-only ping-pong `c40c78c` | `0.572862 ms` |
| K+V ping-pong `9cca869` | `0.554370 ms` / `0.565802 ms` |

两次 K+V 运行显示约 `0.01 ms` 的机器状态漂移，因此端到端总收益保守记录为约
`3%`，观测范围约 `3%` 到 `5%`。

相同 `msopprof --aic-metrics=Default --warm-up=5 --launch-count=1` 条件下：

| 实现 | Fused | Combine | Kernel 合计 |
| --- | ---: | ---: | ---: |
| single buffer `a38d5cd` | `388.953979 us` | `36.410000 us` | `425.363979 us` |
| K+V ping-pong `9cca869` | `365.545990 us` | `36.264000 us` | `401.809990 us` |

fused 主 kernel 提升 `6.02%`，kernel 合计提升 `5.54%`；Combine 基本不变。当前
fused launch 仍为 `32 / 64`，保持 32 AIC 和 64 AIV 的工作映射。

## 5. 结论

K/V ping-pong 达到预期的 kernel 侧小幅优化，并保持完整精度。收益没有更大，是因为
online softmax、PV/output update、Combine、Host launch 和同步固定开销仍在关键路径。
本阶段到此收敛，不继续堆叠同步或扩大 buffer；后续若要继续优化，应以 profiler 为
依据单独设计 Combine 融合或更深的多级流水。
