# Softmax V3 Compiler Repro

This directory contains a minimal compiler repro for the row-wise persistent softmax issue.

Files:
- `single_tu_switch_repro.asc`: reproduces the issue
- `split_tu_main.asc` + `split_tu_1024_helper.asc`: same logic split into two ASC compilation units, does not reproduce

Expected behavior:
- `single_tu_switch_repro`: `aclrtSynchronizeStream ret = 507035`, with device log `The configured UB size exceeds 224 KB`
- `split_tu_switch_repro`: `aclrtSynchronizeStream ret = 0`

Build:

```bash
cd src/cannbench/operators/builtin/softmax/simt/test/compiler_repro
mkdir -p build
cd build
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-3510
make -j1
```

Run:

```bash
cd src/cannbench/operators/builtin/softmax/simt/test/compiler_repro/build
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=1
./single_tu_switch_repro
./split_tu_switch_repro
```

Comparison point:
- Both executables contain a `256` path that actually runs
- Both keep a `1024` template instantiation that is never executed
- The only intended difference is whether `256` and `1024` live in the same ASC compilation unit
