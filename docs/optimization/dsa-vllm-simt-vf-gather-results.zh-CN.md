# DSA V3.2 vLLM SIMT VF Gather 结果

## 范围

- 基线：vLLM-Ascend `a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca` 的
  `SparseFlashAttention` arch35 实现。
- 设备：`Ascend950PR_9589`，`NPU_ARCH=dav-3510`，device 0。
- 软件：CANN 9.2.0，Python 3.11，PyTorch/torch-npu 2.9 环境
  `cannbench-vllm-ascend-py311`。
- case：`deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048`，BF16，
  `B=2, Q=2, Hq=128, Hkv=1, S=32768, topk=2048, Dqk=576, Dv=512`。
- workflow：`dsa_decode = lightning_indexer + sparse_attention`。

实现位于 operator-local 路径
`src/cannbench/operators/builtin/sparse_attention/simt/vllm/`。本次按
function-first 要求保留上游 Basic API；新增 gather 使用 SIMT VF。

## 实现

复制版先完整复现上游 QK/V 合并搬运和 BMM1/softmax/BMM2 decode 流程。
QK-VF 版本使用 `sfa_qk_gather_vf` 将 512 维 nope key 与 64 维 rope key
写入 576 维 rolling workspace。最终版本再用 `sfa_v_gather_vf` 从
`valueGm` 覆盖同一行前 512 维。当前支持的 DSA shared-KV wrapper 明确构造
`value = key.clone()`，因此覆盖不改变 BMM1 的 key。

SIMT GM store 后必须执行 `asc_dcci_entire`。只使用
`asc_threadfence`、`asc_syncthreads` 或 V/MTE event 时，Cube 侧会读到旧 V；
典型现象是 LSE 全部正确而 output 约 24 万元素不匹配。加入 DCCI 后完整
随机精度通过。

## 精度

容差为 `atol=0.05, rtol=0.05`，校验全部 512 个 query-head 行和全部
262144 个 output 元素。

| 版本 | output mismatch | LSE mismatch | max abs output | max abs LSE |
| --- | ---: | ---: | ---: | ---: |
| 复制 a5b0ce | 0 / 262144 | 0 / 512 | 基线通过 | 基线通过 |
| QK-VF + DCCI | 0 / 262144 | 0 / 512 | 0.024414 | 0.033373 |
| QK+V-VF + DCCI | 0 / 262144 | 0 / 512 | 0.021057 | 0.031164 |

原始输出：

```text
evidence/copied-a5b0ce/accuracy.json
evidence/qk-vf/accuracy-dcci.json
evidence/qkv-vf/accuracy.json
```

## 性能

CannBench `BasicInfo` 在复制基线上得到两次 workflow 结果：

| run | lightning_indexer (us) | sparse_attention (us) | 合计 (us) |
| --- | ---: | ---: | ---: |
| 1 | 1984.614 | 54.563 | 2039.177 |
| 2 | 1983.935 | 54.249 | 2038.184 |

2026-08-07 通过正式 CannBench 子命令重跑复制基线和最终 QK+V-VF 版。
两边都显式移除 `ASCEND_LAUNCH_BLOCKING`，使用同一 case、device 0、
`BasicInfo` 和 1650 MHz 频率：

```bash
python -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version vllm \
  --op dsa_decode \
  --dtype bfloat16 \
  --dataset realistic \
  --case-id deepseek_v32_flashmla_decode_b2_q2_ctx32768_top2048 \
  --aic-metrics BasicInfo
```

| 版本 | run | lightning_indexer (us) | SparseFlashAttention (us) | workflow (us) |
| --- | ---: | ---: | ---: | ---: |
| 复制 a5b0ce | 1 | 1984.074 | 54.483 | 2038.557 |
| QK+V-VF + DCCI | 1 | 1983.277 | 882.803 | 2866.080 |
| QK+V-VF + DCCI | 2 | 1983.249 | 883.764 | 2867.013 |
| QK+V-VF + DCCI | 均值 | 1983.263 | 883.283 | 2866.547 |

正式 workflow 聚合只累加插件选中的 Indexer 和 `SparseFlashAttention`
主 kernel，不包含 profile 中可见的 Slice、Transpose、Log 和 Add。最终版主
kernel 相对复制版为 `16.21x`，workflow 回退 `40.62%`。两次最终版
相差不到 `1 us`，说明该回退不是单次 profiler 抖动，也不是
`ASCEND_LAUNCH_BLOCKING=1` 造成的。

早期 VF 实验中 `msopprof` 曾生成空 CSV，当时的失败日志仍保留在
`evidence/qk-vf/cannbench-*.log` 和 `evidence/qkv-vf/cannbench-*.log`；上表是恢复完整
CANN `PATH` 后的有效重测。新原始产物位于：

```text
evidence/cannbench-cli-no-launch-blocking/copied-a5b0ce-run-1/
evidence/cannbench-cli-no-launch-blocking/qkv-vf-final-run-{2,3}/
```

为完成同口径比较，operator-local
`vllm_decode_workflow_benchmark.py` 仍通过 CannBench 的 workflow、
`OperatorBenchmarkRequest`、`AscendBackend` 和插件 callable 构造路径执行，
仅将采样器替换为 NPU event。每个版本使用两个独立 Python 进程；每进程
warmup 5 次、采样 20 次，并保留完整 LSE 路径。下表给出两次进程中位数及其
均值：

| 版本 | sparse run 1/2 (us) | sparse 均值 (us) | workflow run 1/2 (us) | workflow 均值 (us) | 相对复制版 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 复制 a5b0ce | 739.551 / 720.989 | 730.270 | 2765.081 / 2746.015 | 2755.548 | - |
| QK-VF + DCCI | 1088.129 / 1063.773 | 1075.951 | 3112.096 / 3090.419 | 3101.257 | +12.55% |
| QK+V-VF + DCCI | 1249.314 / 1347.309 | 1298.312 | 3269.071 / 3370.003 | 3319.537 | +20.47% |

原始 40-sample 数据：

```text
evidence/event-comparison-lse/copied-a5b0ce/run-{1,2}.json
evidence/event-comparison-lse/qk-vf-dcci/run-{1,2}.json
evidence/event-comparison-lse/qkv-vf-final/run-{1,2}.json
```

结论：当前逐 selected-row 启动 VF、逐行执行 DCCI 的方案精度正确，但正式
CannBench `BasicInfo` 显示最终 QK+V-VF 使主 kernel 从 `54.483 us` 增至
约 `883.283 us`，workflow 回退 `40.62%`。NPU event 边界下 QK-VF 和
QK+V-VF 的回退分别为 12.55% 和 20.47%，该数据仅用于同计时边界的版本间
A/B。下一步优化应减少 VF invoke 和 DCCI 次数，例如每个 VF 批量处理多行并在
批次边界发布，而不是继续调整 Cube/softmax 主流程。

## 可复现证据

远程根目录：

```text
/root/cannbench-dsa-vllm-simt-vf-gather-20260806
```

关键哈希：

```text
05ebe36a3ae1248aaeca94c3eb1ddb6d0cdf6d2e96938a50706ed390a3ba76c7  copied package
5e84a254c5e40e264174d4cf4a72ed20a9eca783e84e693b65b203a8962cd705  final vector source
8b45423d6bdb709d7a2f17e56315d9711014beb75eb20fd13ad01d5b5223f6e3  final package
```

最终包部署路径：

```text
/root/cannbench-dsa-vllm-simt-vf-gather-20260806/opp/qkv-vf-final/vendors/cannbench_vllm_transformer
```
