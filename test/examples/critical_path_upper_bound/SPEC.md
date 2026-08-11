# Critical Path Upper Bound 规格

## 正确性合同

输入固定为 262144 个整数可精确表示的 float32。lane A 的三阶段输出为 `input+192`；lane B 的 baseline 与 counterfactual 输出均为 `input+32`。host 必须验证完整输出、guard、每 lane launch count、物理 loop iterations 和等价 increment。

`SCENARIO_NUM=0` 的 lane B 执行 32 次 `+1`。`SCENARIO_NUM=1` 执行 8 次 `+4`，只构造声明 4x work reduction 的等价 counterfactual，不代表设备上必然达到 4x。

## 并发与完整调用边界

host 创建两个 ACL streams。stream A 顺序运行 `lane_a_stage0/1/2`；stream B 运行一个 candidate kernel。两 lane 仅共享只读输入，输出、workspace 和计数互不重叠，不使用 device 跨核同步。

每 lane 使用同 stream ACL event pair 记录设备 interval。完整 join 边界使用从 enqueue 前到两个 end event 均完成后的 synchronized wall interval。ACL event 与 wall timer 是不同边界，均需保留；event max 不包含 host dispatch，wall 包含。

## max-not-sum 与归因合同

profile 必须先验证一次调用只出现 lane A 三个 stage 和 lane B 一个 stage。baseline 关键路径估算固定为 `max(lane_A_sum, lane_B_sum)`，不能相加。声明 ideal speedup 为 4，retention gate 为 5%。理论 upper bound 只回答“候选即使理想加速，完整并发边界最多能省多少”。

scenario 1 的 raw critical max 和完整 join wall 用于反事实对照，但单次观测不证明收益。若 streams 实际没有 overlap、launch offset 明显、profiler replay 改变执行或 scenario 资源竞争不同，必须回到 raw timeline 重新定义完整调用边界，不能继续套用 max 模型。

## 工程边界

- 一个 standalone `.asc`，编译期选择 `SCENARIO_NUM=0/1`。
- host 使用 C 函数、POD、`malloc/free` 和裸 ACL memory/stream/event。
- device 仅使用 C API 与 SIMT API，禁止 Basic API、C++ Basic API event/buffer、Mutex 和跨核同步。
- `run.sh` 先 correctness 后 profile，保留 raw，不显式覆盖 warmup。
- parser 使用结构化 CSV/JSON；launch count 不符时失败，禁止猜测调用分组。
- 实际 Task Duration、event interval、wall interval、环境和 raw 路径记录在 README；当前单次设备复核的 retention gate 结论为编码前拒绝。
