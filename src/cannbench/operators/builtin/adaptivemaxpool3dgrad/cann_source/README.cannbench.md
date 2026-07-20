# CANN Source Snapshot

这个目录是从以下路径复制来的 CANN `AdaptiveMaxPool3DGrad` 源码快照：

```text
/home/lizhaoqi/ops-nn/pooling/adaptive_max_pool3d_grad
```

它用于在 CannBench 的算子目录内保存 CANN 默认实现的参考源码，方便：

- 阅读 `op_api`、`op_host`、`op_kernel` 的实现。
- 对照 `msprof`/plog 中的 kernel 名、tiling key 和异常信息。
- 本地修改源码并准备后续自定义算子构建。

注意：仅复制源码不会改变 CannBench 当前运行的 CANN 默认实现。

当前 `--implementation cann_ops_library` 仍然通过 `torch_npu` 调用已安装 CANN
环境中的算子二进制，通常来自：

```text
$ASCEND_HOME_PATH/opp
$ASCEND_HOME_PATH/lib64
$ASCEND_HOME_PATH/lib64/plugin/opskernel
```

如果要让这里的修改真正参与 benchmark，需要进一步把该源码编译成可加载的自定义
算子实现，并通过 CannBench 的 operator-local implementation hook 接入。

