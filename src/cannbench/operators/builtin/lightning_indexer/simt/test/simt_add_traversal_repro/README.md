# SIMT Add Traversal Repro

该样例使用固定规模的 `float` 加法，对比以下三个二元因子组成的
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
| 操作 | `output[i] = input_x[i] + input_y[i]` |
| Grid Dim | 64 |
| Block Dim | 2048 |
| Launch Bounds | 2048 |
| 每线程元素数 | 4 |
| 总元素数 | `64 * 2048 * 4 = 524288` |

每次执行都会生成确定性输入并校验全部输出元素。只有
`mismatch_count=0` 且 `validation=pass` 才返回成功。

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
4. 执行四次加法并写回。

因此 global/block 展开版本之间只保留索引和步长差异。

## 编译

完整 8-case profile 使用远端节点上的 CANN 9.1 环境：

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
  ./build/<executable> <mode>
```

命令不显式设置 `--warm-up`，使用 msopprof 默认行为。每个 case 独立
采集两轮。主指标取 `OpBasicInfo.csv` 中的 `Task Duration(us)`；辅助指标
取 `ArithmeticUtilization.csv` 中 64 个 block 的
`aiv_total_cycles` 平均值。

## 当前测试数据

测试环境：远端端口 20002 节点，CANN 9.1 编译器、运行时和 msopprof。
以下数据对应当前统一 load-all 排序的源码，8 个 case 均为
`mismatch_count=0`、`validation=pass`。

| 执行方式 | 遍历 | 形态 | Round 1 (us) | Round 2 (us) | 平均 (us) | 平均 AIV cycles |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| direct SIMT | global | loop | 6.460 | 6.737 | 6.599 | 8387.5 |
| direct SIMT | global | unrolled | 7.717 | 7.666 | 7.692 | 10341.8 |
| direct SIMT | block | loop | 6.056 | 6.696 | 6.376 | 8218.2 |
| direct SIMT | block | unrolled | 8.130 | 8.196 | 8.163 | 11022.1 |
| hybrid | global | loop | 5.831 | 5.829 | **5.830** | 7381.4 |
| hybrid | global | unrolled | 7.752 | 7.861 | 7.807 | 10473.2 |
| hybrid | block | loop | 6.304 | 6.336 | 6.320 | 8119.3 |
| hybrid | block | unrolled | 7.889 | 7.628 | 7.759 | 10305.4 |

原始 profile 产物位于远端：

```text
/tmp/simt-add-load-all-cann91-6XNAkA/full-profiles-round1
/tmp/simt-add-load-all-cann91-6XNAkA/full-profiles-round2
```

## 测试结论

### Loop 与手动展开

loop 在四组一一配对中全部更快，是当前最稳定的结论：

| 执行方式与遍历 | Loop 相对 unrolled 的耗时降低 |
| --- | ---: |
| direct global | 14.2% |
| direct block | 21.9% |
| hybrid global | 25.3% |
| hybrid block | 18.5% |

四个 loop case 平均为 `6.281 us`，四个 unrolled case 平均为
`7.855 us`，loop 总体低约 20.0%。手工展开没有带来收益。

### Global 与 Block

global 四个 case 的总体平均为 `6.982 us`，block 为 `7.154 us`，global
总体低约 2.4%，但该结论不在所有配对中成立：

- direct loop：block 快约 3.4%；
- direct unrolled：global 快约 5.8%；
- hybrid loop：global 快约 7.8%；
- hybrid unrolled：block 快约 0.6%，差距接近噪声。

因此不能概括为“global 必然比 block 快”。

### Hybrid 与 Direct SIMT

hybrid 四个 case 的总体平均为 `6.929 us`，direct SIMT 为 `7.207 us`，
hybrid 总体低约 3.9%，但同样不在所有配对中成立：

- global loop：hybrid 快约 11.6%；
- global unrolled：direct 快约 1.5%；
- block loop：hybrid 快约 0.9%，差距接近噪声；
- block unrolled：hybrid 快约 5.0%。

因此不能概括为“hybrid 必然比 direct SIMT 快”。

### 当前最优 Case

当前最优组合为：

```text
hybrid + global stride + loop = 5.830 us
```

direct block loop 的两轮时间分别为 `6.056 us` 和 `6.696 us`，波动约
10.6%，明显高于其他 case。涉及该 case 的小幅差异需要更多独立样本
才能形成稳定结论。相比之下，loop 优于手动展开的幅度足够大，且四组
配对方向一致。
