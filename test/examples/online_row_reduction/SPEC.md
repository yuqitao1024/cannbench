# Online Row Reduction 规格

## 数值语义

所有场景输入 float32，输出每行完整 float32 softmax。host 采用 double host oracle，逐行执行稳定 max、double exp/sum 和 normalize。逐元素判定使用 `absolute tolerance=2e-6` 与 `relative tolerance=3e-3` 的组合界；输出行和 tolerance 为 `3e-3`。

固定 case 为 ordinary `64 x 1024`、含 `+/-1000` 的 extreme `8 x 4096`、非 tile 整除 tail `4 x 65537`。tail case 的 tile width 为 4096，最后 tile 只有 1 个有效元素；无效位置不得参与 max 或 sum。

## 三种实现

`SCENARIO_NUM=0` 对每行执行 max scan、稳定 exp/sum scan、normalize scan。逻辑输入访问为 3 遍，保留多扫描 control。

`SCENARIO_NUM=1` 在一次 stats scan 中在线更新 `(m,s)`，随后 normalize。若新值 `x <= m`，更新 `s += exp(x-m)`；否则更新 `s = s*exp(m-x)+1, m=x`。逻辑输入访问为 2 遍。

`SCENARIO_NUM=2` 以最多 4096 元素为界，在 tile 内在线得到 `(tile_m,tile_s)`，再用稳定公式合并到 row `(m,s)`，最后 normalize。tile state 和 row state 都是固定大小标量，容量不依赖整行宽度。逻辑输入访问为 2 遍，额外成本是 tile pair 初始化与合并。

## 执行与测量合同

当前三种实现采用相同 `1 block x 512 threads`，每个 thread 独立负责一行，每次进程运行一个 case，并产生 `1 次相关 kernel launch`。没有跨核同步、辅助 stage 或不同 parallelism。因此完整 softmax 调用边界为一个完整 kernel；各 case 必须分别 profile，不能跨 shape 聚合。

若后续改变 parallelism 或 launch count，Task Duration 对比必须聚合同一次调用的全部相关 kernel，并补充覆盖 launch gap 的同步端到端边界。算法扫描减少、launch 融合、并行度和 inner schedule 是不同因子，不能从一个总延迟差同时得出多个因果结论。

## 工程边界

- 单 standalone `.asc`，由 `SCENARIO_NUM=0/1/2` 编译期选择。
- host 使用 C 风格函数、POD、`malloc/free`、裸 `aclrtMalloc/aclrtFree`。
- device 仅使用 C API 与 SIMT API，禁止 Basic API、C++ Basic API event/buffer、Mutex 和跨核同步。
- `run.sh` 必须 correctness 后 profile，每个 scenario/case 使用唯一 raw 目录，不显式覆盖 profiler warmup。
- 设备实测环境、各 shape 的 Task Duration、正确性和 raw 目录记录在 README；源码不嵌入实测延迟或加速比。
