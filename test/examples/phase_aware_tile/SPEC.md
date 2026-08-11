# Phase-aware Tile 规格

## 目标与非目标

案例冻结相同逐元素计算，比较保守容量求和、真实 live-set 复用和过大 outer tile/粗粒度 copy 反例。目标是提供可审计的容量 worksheet、同步轮数和逻辑传输计数；非目标是从源码预测 Ascend 950 性能或填入未采集数据。

## 数据与正确性合同

- shape 固定为 `128 x 8192` float32。
- `input` 与 `operand` 由确定公式生成，所有场景共享相同 device buffers 和 host oracle。
- 输出公式固定为 `(input * operand + 1) + operand`，全部值逐元素精确比较。
- 输出尾部额外分配 64 个 sentinel guard，kernel 不得改写。
- device 回传 `WorkCounters`，host 逐字段验证场景、tile、round、barrier、逻辑传输和容量计数。

## 生命周期 worksheet

`SCENARIO_NUM=0` 声明 persistent operand、phase A scratch、phase B scratch，各 1024 float。容量规划保守求和是 12288 bytes；phase A/B 不同时存活，真实 peak live-set 是 operand 加一个 scratch，即 8192 bytes。每 row 有 8 个 outer round，每 round 5 次 block barrier，并通过 GM workspace 写入和读回 intermediate。

`SCENARIO_NUM=1` 声明 persistent operand 和 shared phase scratch，各 2048 float，声明容量与真实 peak 都是 16384 bytes。scratch 在两个 phase 间原地复用，operand 保留到输出。每 row 有 4 个 outer round，每 round 3 次 barrier，不访问 GM workspace。

`SCENARIO_NUM=2` 声明 outer input、persistent operand 和 coarse replay operand，各 4096 float，声明容量与 peak 都是 49152 bytes。每 row 有 2 个 outer round，每个 outer 含 4 个 1024 compute subtile；每个 subtile 都复制完整 4096 operand，因此逻辑 UB copy 是有效元素数的 4 倍。该场景使大 live-set 与重复粗 copy 可在 compiler metadata 和 profile 中核对，但不预设一定回退。

所有 barrier 都是 block 内 `asc_syncthreads`。block 之间没有数据依赖或跨核同步。

## 实验解释合同

容量、同步轮数、传输粒度是独立因子。三个场景不是严格 factorial matrix，不能用任意一对场景的 Task Duration 差直接归因到单个因子。保留候选前必须建立控制变量实验、验证相同输出，并检查编译器资源元数据、raw kernel timeline 和重复 device time。

程序计数的是源码定义的逻辑访问和同步，不证明 cache 行为、物理带宽、occupancy 或 latency。任何实测结果必须绑定 Ascend 950 具体型号、软件栈、输入、同步边界、raw profile 目录和统计方法。

## 工程边界

- 一个 standalone `.asc` translation unit，通过 `SCENARIO_NUM=0/1/2` 编译期选择 launch。
- host 使用 C 风格函数、POD、`malloc/free` 和裸 `aclrtMalloc/aclrtFree`。
- device 仅使用 C API 与 SIMT API，不使用 Basic API、C++ Basic API buffer/event、Mutex 或跨核同步。
- 每个场景一次应用调用预期 1 次相关 kernel launch；若后续拆分 stage，必须按调用聚合完整边界。
- `run.sh` 正确性通过后才 profile，保留唯一 `raw` 目录，默认不传 warmup 参数。
- 设备实测环境、Task Duration、正确性和 raw 目录记录在 README；容量和逻辑计数仍不得替代 profiler 证据。
