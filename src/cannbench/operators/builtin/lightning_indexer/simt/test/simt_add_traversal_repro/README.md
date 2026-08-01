# SIMT Add Traversal Repro

该样例使用固定规模的原地 `float` 累加，对比以下三个二元因子组成的
`2 x 2 x 2 = 8` 个 case：

- 执行方式：direct SIMT / SIMD-SIMT hybrid (`asc_vf_call`)
- 遍历方式：global stride / block contiguous
- 代码形态：四次循环 / 手动展开

样例仅依赖 ASC 编译器和 ACL，不依赖 Python、PyTorch、torch_npu 或
CannBench 运行时。

当前实现设计见 [`DESIGN.md`](DESIGN.md)。

## 固定负载

所有 case 使用相同配置：

| 配置 | 值 |
| --- | ---: |
| 数据类型 | `float` |
| 操作 | `output[i] += input_x[i] + input_y[i]` |
| Grid Dim | 64 |
| Block Dim | 2048 |
| Launch Bounds | 2048 |
| 每线程元素数 | 4 |
| 总元素数 | `64 * 2048 * 4 = 524288` |

每次执行都会生成确定性的两个输入和非零 `output` 初值。普通运行默认
独立期望一次累加；profile replay 可通过可执行文件的第三个参数指定独立
期望次数。实际次数必须等于期望次数，且全部输出元素与逐次累加的 host
golden 完全一致，程序才返回成功。

## 八个 Case

| 执行方式 | 遍历方式 | 代码形态 | 可执行文件 | mode |
| --- | --- | --- | --- | --- |
| direct SIMT | global stride | loop | `simt_add_traversal_direct` | `global-stride` |
| direct SIMT | global stride | unrolled | `simt_add_traversal_direct` | `global-stride-unrolled` |
| direct SIMT | block contiguous | loop | `simt_add_traversal_direct` | `block-contiguous` |
| direct SIMT | block contiguous | unrolled | `simt_add_traversal_direct` | `block-contiguous-unrolled` |
| hybrid | global stride | loop | `simt_add_traversal_hybrid` | `global-stride` |
| hybrid | global stride | unrolled | `simt_add_traversal_hybrid` | `global-stride-unrolled` |
| hybrid | block contiguous | loop | `simt_add_traversal_hybrid` | `block-contiguous` |
| hybrid | block contiguous | unrolled | `simt_add_traversal_hybrid` | `block-contiguous-unrolled` |

direct target 使用 `--enable-simt`，host 侧以
`<<<64, 2048, 0, stream>>>` 启动。hybrid target 不使用
`--enable-simt`，外层 `__global__ __vector__` kernel 通过
`asc_vf_call` 启动 2048 个 SIMT 线程。

两个手动展开版本使用相同的 load-all 指令顺序：

1. 计算 `index0-index3`；
2. 读取 `input_x0-input_x3`；
3. 读取 `input_y0-input_y3`；
4. 累加并写回 `output0-output3`。

因此 global/block 展开版本之间只保留索引和步长差异。

## 编译

重新采集完整 8-case profile 时使用远端节点上的 CANN 9.1 环境：

```bash
source /home/l00848653/Ascend/cann-9.1.0/set_env.sh

cmake -S . -B build \
  -DASCEND_CANN_PACKAGE_PATH=/home/l00848653/Ascend/cann-9.1.0
cmake --build build -j2
```

CANN 9.1 编译的 direct SIMT target 调用普通的
`aclrtLaunchKernelWithHostArgs`，可以被对应 msopprof 捕获。当前 CANN
9.2 编译的 direct target 调用 `aclrtLaunchSIMTKernelWithHostArgs`；已测试
的 msopprof 注入库未 hook 该接口，因此 direct target 虽然精度通过，
但 profile 为空。hybrid target 在 CANN 9.2 下可以正常采集。

## Profile 方法

每个 case 使用独立输出目录，采集一次 kernel：

```bash
msopprof \
  --output=<profile-output> \
  --aic-metrics=Default \
  --launch-count=1 \
  ./build/<executable> <mode> 3
```

replay 和 warmup 使用 msopprof 默认行为。默认 kernel replay 会在同一
`output` 上重复执行原地累加；host 从一个固定的非零增量元素推导统一的
实际 `accumulation_count`，并要求它等于可执行文件参数给出的独立期望值。
上例末尾的 `3` 是 CANN 9.1.0 当前默认 replay 行为对应的应用参数，不是
msopprof 参数。每个 case 独立采集两轮。主指标取 `OpBasicInfo.csv` 中的
`Task Duration(us)`；辅助指标取 `ArithmeticUtilization.csv` 中 64 个
block 的 `aiv_total_cycles` 平均值。

## 当前验证状态

先前记录的数据来自覆盖写 `output = input_x + input_y` workload，不能用于
当前原地累加 workload，因而已移除。

2026-08-01 在 endpoint `ascend-950pr-lightning-indexer-v2` 上完成以下验证：

- PCI device `19e5:d806`，编译目标 `dav-3510`，driver/firmware
  `7.0.t9.0.B791`；
- 同一份 `main.asc` 的 SHA-256 为
  `016d39be7117799baa1b59ad584ddcb270464661959406c19a34b44dae359ba8`；
- CANN 9.1.0 与 9.2.0 均成功构建 direct 和 hybrid target；
- 两个 CANN 环境下的 8 个 executable/mode 组合均为
  `accumulation_count=1`、`mismatch_count=0`、`validation=pass`；
- CANN 9.1.0 下使用默认 replay/warmup 完成 8-case 两轮 `Default` metric
  profile，16 次采集均为 `expected_accumulation_count=3`、
  `accumulation_count=3`、`mismatch_count=0`、`validation=pass`；
- 16 个 profile 各生成 1 条目标 kernel 基础记录和 64 条 block 级
  arithmetic utilization 记录。

## 当前测试数据

| 执行方式 | 遍历 | 形态 | Round 1 (us) | Round 2 (us) | 平均 (us) | 平均 AIV cycles | 两轮差/均值 |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| direct SIMT | global | loop | 7.177 | 8.061 | 7.619 | 9550.2 | 11.6% |
| direct SIMT | global | unrolled | 9.572 | 9.707 | 9.640 | 13430.1 | 1.4% |
| direct SIMT | block | loop | 9.094 | 8.143 | 8.618 | 11948.7 | 11.0% |
| direct SIMT | block | unrolled | 10.207 | 10.137 | 10.172 | 14600.6 | 0.7% |
| hybrid | global | loop | 6.516 | 7.450 | **6.983** | 9326.3 | 13.4% |
| hybrid | global | unrolled | 8.923 | 8.805 | 8.864 | 12215.7 | 1.3% |
| hybrid | block | loop | 7.267 | 7.338 | 7.302 | 9848.7 | 1.0% |
| hybrid | block | unrolled | 8.951 | 9.039 | 8.995 | 12542.2 | 1.0% |

16 份日志都提示超过 108 个 sub-block 时可能丢失部分动态插桩数据。因此
`Task Duration(us)` 仍作为主指标，AIV cycles 仅作为可能不完整的辅助证据。

原始产物和逐 case 日志位于远端：

```text
/tmp/cannbench-simt-accumulate-DHbV8Y/profiles-default-independent-cann91-round1
/tmp/cannbench-simt-accumulate-DHbV8Y/profiles-default-independent-cann91-round2
/tmp/cannbench-simt-accumulate-DHbV8Y/profile-logs-default-independent
```

## 测试结论

- loop 在四组一一配对中均快于 unrolled，耗时降低 `15.3%-21.2%`；四个
  loop case 平均 `7.631 us`，unrolled 平均 `9.418 us`，总体低 `19.0%`。
- hybrid 在四组一一配对中均快于 direct SIMT，耗时降低 `8.0%-15.3%`；
  hybrid 总体平均 `8.036 us`，direct SIMT 平均 `9.012 us`，总体低
  `10.8%`。
- global 总体平均 `8.276 us`，block 平均 `8.772 us`，总体低 `5.7%`，
  但这是最弱的因子。hybrid loop 的两轮配对方向不一致，不能据此断言
  global 必然快于 block。
- 当前平均最快组合为 `hybrid + global stride + loop = 6.983 us`。

loop case 的两轮差/均值最高为 `13.4%`，明显高于 unrolled case 的
`0.7%-1.4%`。因此 loop 与 unrolled、hybrid 与 direct 的方向在所有逐轮
配对中一致，但涉及数个百分点的 traversal 差异仍需更多轮次确认。
