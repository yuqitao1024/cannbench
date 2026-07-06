# 新增算子性能测试接入指南

本文说明在 CannBench 中新增一个算子性能测试需要补齐哪些代码、数据和验证步骤，覆盖 NVIDIA CUDA、Ascend CANN ops / vLLM Ascend、Ascend SIMT op 三类实现。

## 是否能直接测试

一个 patch 只是“新增了算子文件”时，通常还不能直接测试。至少要满足下面条件：

1. `cannbench bench --op <op>` 能在 CLI choices 中识别该算子。
2. `--dataset <split> --case-id <case>` 能加载到真实 case。
3. backend 能把 case materialize 成 tensor，并调用目标实现。
4. benchmark 能生成 `perf/*.json`，可选地能采集 profile 和发布到前端。
5. 对于非 PyTorch baseline，需要明确 `--implementation` 对应的是哪个库或自定义模块。

最小 smoke 验证命令：

```bash
PYTHONPATH=src python3 -m cannbench bench \
  --backend nvidia \
  --op <op> \
  --dataset smoke \
  --case-id <case_id> \
  --dtype float16 \
  --warmup 1 \
  --iterations 1 \
  --output-dir runs
```

如果这里报 `invalid choice`，说明 registry 或 CLI 未接入；如果报 `Unknown <op> dataset/case`，说明 dataset loader 未接入；如果报 `Unsupported operator for <backend> backend`，说明 backend dispatch 未接入。

## 代码接入清单

### 1. 注册算子

在 `src/cannbench/operators/registry.py` 添加 `OperatorSpec`：

```python
"my_op": OperatorSpec(
    name="my_op",
    supported_dtypes=("float32", "float16", "bfloat16"),
    dataset_namespace="my_op",
    runner_name="my_op",
),
```

这一步决定：

- `cannbench bench --op my_op` 是否能被 CLI 接受。
- release 构建时是否会生成 prepared inputs。
- 前端 published 结果里算子名如何分组。

### 2. 添加 dataset model 和 JSON

新增 `src/cannbench/datasets/my_op.py`，建议结构参考 `softmax.py`、`topk.py` 或 DSA 的 `lightning_indexer.py`。

至少提供：

- `MyOpCase` dataclass。
- `MyOpDataset` dataclass。
- `get_my_op_dataset(name: str)`。
- `get_my_op_case(dataset_name: str, case_id: str)`。
- `case.payload`，用于结果记录和前端展示。

新增数据目录：

```text
src/cannbench/datasets/data/my_op/
  __init__.py
  smoke.json
  realistic.json
  stress.json
```

JSON case 建议字段：

```json
{
  "case_id": "tiny_case",
  "family": "smoke",
  "source_kind": "synthetic_smoke",
  "source_project": "cannbench",
  "source_model": "smoke_fixture",
  "source_file": "built-in",
  "source_op": "my_op"
}
```

真实模型 shape 要在 `source_*` 字段里写清楚来源。前端不会理解算子的业务语义，只会展示 benchmark record 里的 `operator`、`dataset`、`case_id`、`family`、`source_*` 和 `payload`。

### 3. 接入 dataset loader

在 `src/cannbench/datasets/loader.py` 中补齐三处：

- import `get_my_op_case` / `get_my_op_dataset`。
- `OperatorDataset.get()` 支持 `dataset_namespace == "my_op"`。
- `get_operator_dataset("my_op")`。
- `get_operator_case("my_op", dataset_name, case_id)`。

同时在 `src/cannbench/datasets/__init__.py` 导出新 dataset API。

### 4. 添加 materialize 逻辑

在 `src/cannbench/datasets/materialize.py` 添加：

```python
def materialize_my_op_inputs(case: MyOpCase, *, dtype: str, seed: int) -> dict[str, object]:
    ...
```

要求：

- 同一 `case + dtype + seed` 生成完全确定的数据。
- shape、indices、mask、stride、layout 等信息都放进 payload。
- 数值数据用 tuple 或 `array("f")` 兼容现有 `_tensor()` 创建路径。
- index 类 tensor 明确 dtype，后端里通常用 `torch.long` 或 `torch.int32`。

### 5. 接入 backend baseline

通用 PyTorch baseline 在 `src/cannbench/backends/torch_backend_base.py`。

需要补三类路径：

- `_operator_callable()`：性能测试 warmup/iteration 调用。
- `_capture_operator_tensor()`：输出捕获，用于 compare。
- `run_operator()`：如果算子不走 `_operator_callable()` 通用分支，就要单独写 warmup/iteration。

建议新算子优先复用 `_operator_callable()`，让 `run_operator()` 只需进入通用分支；如果当前结构里没有通用分支，就按现有算子模式补一段。

### 6. 测试覆盖

至少补这些测试：

- `tests/test_operators.py`：算子已注册。
- `tests/test_datasets.py`：dataset 能加载、case metadata 和 payload 正确。
- `tests/test_operator_dispatch.py` 或 backend 相关测试：prepared input、backend dispatch 能跑到新算子。
- 如果有外部 adapter，补 adapter 解析和错误信息测试。

本地最低验证：

```bash
PYTHONPATH=src pytest -q tests/test_operators.py tests/test_datasets.py
PYTHONPATH=src pytest -q
git diff --check
```

## CUDA 接入

### PyTorch CUDA baseline

如果 `--backend nvidia` 不指定 `--implementation cuda_library`，默认走 PyTorch CUDA baseline。只要 `TorchOperatorBackend` 的 baseline 调用已实现，就可以运行：

```bash
PYTHONPATH=src python3 -m cannbench bench \
  --backend nvidia \
  --op my_op \
  --dataset smoke \
  --case-id tiny_case \
  --dtype float16 \
  --warmup 10 \
  --iterations 100 \
  --output-dir runs
```

### CUDA library / 外部 CUDA 算子

如果要对接开源 CUDA 库或业务 CUDA 算子，建议使用 adapter module，而不是把 CUDA kernel 放进 CannBench。

现有 DSA 示例：

- 标准 adapter：`cannbench_cuda_dsa`
- 标准 FlashMLA/DeepGEMM wrapper：`cannbench_cuda_dsa_flashmla_deepgemm`
- 环境变量：

```bash
CANNBENCH_CUDA_DSA_ADAPTER=cannbench_cuda_dsa
CANNBENCH_CUDA_DSA_LIGHTNING_INDEXER=cannbench_cuda_dsa_flashmla_deepgemm:lightning_indexer
CANNBENCH_CUDA_DSA_SPARSE_ATTENTION=cannbench_cuda_dsa_flashmla_deepgemm:sparse_attention
```

新增普通算子可采用同样模式：

1. 在 `NvidiaBackend._operator_callable()` 中识别 `request.implementation == "cuda_library"` 和 `request.op == "my_op"`。
2. 定义环境变量，例如 `CANNBENCH_CUDA_MY_OP_ADAPTER=my_cuda_my_op_adapter`。
3. adapter 暴露 `my_op(**kwargs)`。
4. backend 把 `torch`、`request`、`case`、`payload`、`device`、`dtype` 和已 materialize 的 tensors 传给 adapter。
5. adapter 内部调用真实 CUDA 库。

运行示例：

```bash
CANNBENCH_CUDA_MY_OP_ADAPTER=my_cuda_my_op_adapter \
PYTHONPATH=src python3 -m cannbench bench \
  --backend nvidia \
  --implementation cuda_library \
  --op my_op \
  --dataset realistic \
  --case-id real_case \
  --dtype float16 \
  --warmup 10 \
  --iterations 100 \
  --output-dir runs
```

如果要采集 Nsight Compute：

```bash
PYTHONPATH=src python3 -m cannbench bench \
  --backend nvidia \
  --implementation cuda_library \
  --op my_op \
  --dataset realistic \
  --case-id real_case \
  --warmup 10 \
  --iterations 20 \
  --output-dir runs \
  --capture-output
```

如果需要 profile kernel 过滤，在 `src/cannbench/core/profile.py` 的 `expected_kernel_name_patterns()` 添加该 op 的 kernel 名关键字。

## Ascend CANN ops / vLLM Ascend 接入

### 默认 CANN ops 路径

`--backend ascend --implementation cann_ops_library` 默认会走 PyTorch / torch_npu 暴露的算子。如果新算子能用 `torch` 或 `torch_npu` 直接表达，在 `AscendBackend` 里不需要特殊分支，通用 baseline 即可运行。

命令：

```bash
PYTHONPATH=src python3 -m cannbench bench \
  --backend ascend \
  --implementation cann_ops_library \
  --op my_op \
  --dataset smoke \
  --case-id tiny_case \
  --dtype float16 \
  --warmup 10 \
  --iterations 100 \
  --output-dir runs
```

### 自定义 CANN / vLLM Ascend op

如果目标是 CANN 自定义 op 或 vLLM Ascend op，需要在 `AscendBackend._operator_callable()` 中增加专用分支。

DSA 示例使用：

- `torch.ops._C_ascend.npu_vllm_quant_lightning_indexer`
- `torch.ops._C_ascend.npu_kv_quant_sparse_attn_sharedkv`
- metadata op 和 compute op 成对探测。

新增算子建议：

1. 明确 Python 调用入口，例如 `torch.ops._C_ascend.my_op` 或 `torch_npu.npu_my_op`。
2. 如有 metadata / tiling op，先在 setup 阶段调用 metadata，benchmark loop 中只跑 compute op。
3. backend 中单独实现 `_vllm_ascend_my_op_callable()` 或 `_cann_ops_my_op_callable()`。
4. 输入 layout 在 materialize 或 backend 中固定，不能让 CUDA/Ascend/SIMT 使用不同 case 语义。
5. 缺依赖时抛明确错误，包含缺失 module/op 名。

运行示例：

```bash
PYTHONPATH=src python3 -m cannbench bench \
  --backend ascend \
  --implementation vllm_ascend \
  --op my_op \
  --dataset realistic \
  --case-id real_case \
  --dtype float16 \
  --warmup 10 \
  --iterations 100 \
  --output-dir runs
```

远程 Ascend 测试用 `--endpoint`：

```bash
PYTHONPATH=src python3 -m cannbench bench \
  --backend ascend \
  --implementation cann_ops_library \
  --op my_op \
  --dataset realistic \
  --case-id real_case \
  --endpoint configs/<endpoint>.json \
  --warmup 10 \
  --iterations 100 \
  --output-dir runs
```

## Ascend SIMT op 接入

SIMT op 当前按算子和版本放在 dataset 目录下：

```text
src/cannbench/datasets/data/my_op/simt/
  README.md
  v1/
    install.sh
    pyproject.toml
    setup.py
    <python_module>/
      __init__.py
      ops.py
      csrc/
        register.asc
        simt/
          *.asc
```

需要补：

1. 在 `src/cannbench/backends/pytorch_backend.py` 的 `_ASCEND_SIMT_OP_MODULES` 中注册：

```python
_ASCEND_SIMT_OP_MODULES = {
    ("my_op", "v1"): "aten_my_op",
}
```

2. 在 `AscendBackend` 中覆盖目标算子的调用逻辑。softmax 当前通过 `_softmax()` 特判：

```python
if request.use_simt_op or request.deploy_simt_op:
    module = importlib.import_module(module_name)
    return module.ops.<simt_entry>(...)
```

新增算子也需要类似分支，不能只放 SIMT 目录。

3. `install.sh` 必须能在 Ascend 节点上完成编译/安装，并让 Python 能 import `<python_module>`。

4. 如果要前端展示 SIMT diff，目录结构要保持 `simt/v1`、`simt/v2` 这种版本目录。前端通过 published metadata 和源码 diff 展示 SIMT 版本差异。

运行示例：

```bash
PYTHONPATH=src python3 -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v1 \
  --op my_op \
  --dataset smoke \
  --case-id tiny_case \
  --dtype float16 \
  --deploy-simt-op \
  --warmup 10 \
  --iterations 100 \
  --output-dir runs
```

如果节点已经安装过 SIMT op，可以用：

```bash
PYTHONPATH=src python3 -m cannbench bench \
  --backend ascend \
  --implementation simt \
  --implementation-version v1 \
  --op my_op \
  --dataset smoke \
  --case-id tiny_case \
  --use-simt-op \
  --warmup 10 \
  --iterations 100 \
  --output-dir runs
```

## 前端展示和发布

前端展示主要消费 published 目录下的 benchmark records。新增算子通常不需要改前端，只要 benchmark record 字段完整。

建议保持：

- `case.family`：前端分组维度。
- `source_project` / `source_model` / `source_op`：真实来源。
- `payload`：shape、axis、layout、topK、phase 等用于解释性能的关键参数。
- `implementation`：区分 `cann_ops_library`、`cuda_library`、`vllm_ascend`、`simt`。

发布命令：

```bash
PYTHONPATH=src python3 -m cannbench publish \
  --source runs/<run_name> \
  --dest published
```

## 新增算子的验收清单

代码层面：

- `cannbench bench --help` 中 `--op` 包含新算子。
- `PYTHONPATH=src python3 -m cannbench prepare --op my_op ...` 能生成 prepared input。
- `PYTHONPATH=src python3 -m cannbench bench --backend nvidia --op my_op ...` 能跑 smoke。
- Ascend CANN ops 或 vLLM Ascend 路径能跑 smoke。
- 如果有 SIMT op，`--deploy-simt-op` 能完成安装并运行。
- `PYTHONPATH=src pytest -q` 通过。
- `git diff --check` 通过。
- `make release` 能把新 dataset、SIMT 文件、tools 打进 `dist/cannbench-release.tar.gz`。

结果层面：

- `perf/*.json` 中有正确 `op`、`backend`、`implementation`、`case_id`。
- `meta/benchmark-records.json` 能生成。
- 前端 published 页面能按算子和 implementation 展示。
- CUDA、CANN ops、SIMT 使用同一组 case 语义；如果某实现不支持某 shape，要拆 dataset 或显式标注原因，不能静默换 shape。

## 常见遗漏

- 只加 JSON，没加 `get_operator_dataset()`，CLI 找不到 case。
- 只加 registry，没加 backend dispatch，运行时报 unsupported operator。
- 只加 SIMT 目录，没注册 `_ASCEND_SIMT_OP_MODULES`，`--implementation simt` 实际没有调用 SIMT op。
- CANN ops 需要 metadata op，但把 metadata 放进 benchmark loop，导致统计包含 setup 开销。
- CUDA 和 Ascend 使用不同 layout 或不同 topK，前端看起来在比较同一个 case，实际不可比。
- 新 dataset 没有 `__init__.py`，release 或 importlib resources 找不到资源。
- profile kernel 名过滤只支持 softmax，新增算子采集 profile 后 summary 为空。
