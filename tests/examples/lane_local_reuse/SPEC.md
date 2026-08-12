# Lane-local Reuse 实验规格

## 目标

在 Ascend 950 上构造“重复加载与重算 / shared UB 缓存 / lane-local 缓存”三组受控实验，验证所有权局部的不变量复用是否值得保留。源代码只提出预期瓶颈，性能结论必须来自设备实测。

## 固定合同

- 输入是固定 seed `20260812` 生成的 `uint32` value 与 raw metadata。
- 固定 launch geometry 为 64 blocks、每 block 512 个 SIMT threads；每 lane 固定拥有最多 4 个元素。
- 每个有效元素先生成 compact metadata 和 prepared value，再以完全相同的顺序执行 8 个 pass，输出精确的 `uint32` 累加值。
- 每次调用 1 次 kernel launch；三个场景不得改变 grid、线程数、pass 数、输入、输出或 host oracle。
- host oracle 必须逐元素精确比较，并覆盖 `1`、`tile-1`、`tile`、`tile+1`、容量前 13 个元素的 tail 以及满容量边界；未使用的输出区必须保持 `0xa5a5a5a5` sentinel。

## 三个场景

- `SCENARIO_NUM=0`：每个 pass 都从 GM 重复加载 lane-owned input/raw metadata，重新计算 compact metadata 和 prepared value。volatile GM load 是实验合同的一部分，用于阻止不变量被编译器自动提升。
- `SCENARIO_NUM=1`：每 lane 生产一次 value/metadata/validity 到 shared UB，经一次 SIMT block barrier 后，每个 pass 从 shared UB 重新加载。volatile UB load 用于保留共享访存对照。
- `SCENARIO_NUM=2`：每 lane 在进入 pass loop 前生成 `value[4]`、`metadata[4]` 和 validity mask，并跨全部 pass 复用；这些局部值不得逃逸到共享存储。

Scenario 1 的 producer 和 consumer 都位于同一 SIMT VF，没有跨 pipeline 或跨 core 通信；`asc_syncthreads()` 已建立所需顺序，因此不使用 Mutex。若未来把 producer 移到另一 pipeline，只有在 copy/compute ordering 无法表达同等行为时，才允许使用 `AscendC::Mutex::Lock/Unlock`，且仍不得扩展到其他 Basic API。

## API 边界

- host 只使用函数、POD struct、`malloc/free` 与裸 ACL，不得使用 class、STL 容器或自定义 RAII。
- device 只使用 C API、Tensor API 与 SIMT API，不得包含 `basic_api/*`、`kernel_operator.h`，不得使用 `LocalTensor`、Set/WaitFlag、PipeBarrier 或 CrossCore 同步。
- 三个场景在一个 `.asc` 中通过 `SCENARIO_NUM` 编译期互斥，避免未选 VF 进入同一 ELF。

## 测量合同

- 目标为 Ascend 950 系列 `dav-3510`；实测必须记录具体 SoC、CANN/Bisheng、驱动、固件和频率。
- `scripts/run.sh` 必须先运行完整边界/tail 正确性套件，再用 msopprof 默认参数 profile 一个 `ELEMENT_CAPACITY-13` 调用；不得显式覆盖 profiler warmup。
- 每次 profile 只应出现所选场景的一个目标 kernel。保留完整 raw profiler 目录，并从 CSV 提取目标行用于审计。
- 比较相同的单 kernel `Task Duration` 边界；检查 compiler resource metadata，确认 Scenario 2 的局部值没有意外 spill 后才能解释结果。
- README 中的性能表已由保留的原始设备数据回填；没有 raw 证据时仍不得填入估算时延或加速比。
