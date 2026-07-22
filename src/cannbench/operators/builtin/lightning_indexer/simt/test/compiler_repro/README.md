# Lightning Indexer BF16 VF Repro

This directory contains a minimal runtime repro for the `bf16 + __simt_vf__ + asc_vf_call` path on Atlas 350.

Files:
- `fp16_vf_control.asc`: control executable using the same VF launch shape with `half`
- `bf16_vf_repro.asc`: repro executable using the same VF launch shape with `bfloat16_t`
- `bf16_vf_mixed_abi_repro.asc`: variant that keeps the VF signature as `bfloat16_t*`, but changes the outer `__global__` kernel ABI to `uint16_t*` and casts at `asc_vf_call`

What the executables do:
- allocate a one-element device input/output buffer
- launch a `__simt_vf__` kernel through `asc_vf_call`
- copy `input[0]` to `output[0]`
- synchronize, print `aclrtSynchronizeStream`, `aclGetRecentErrMsg`, and the output raw bits

Expected comparison point:
- `fp16_vf_control`: synchronize succeeds and output raw bits become `0x3c00`
- `bf16_vf_repro`: if the runtime cannot load the bf16 VF kernel, device log should show `Get kernel function failure! ret 107000`, and output raw bits typically remain `0x0000`
- `bf16_vf_mixed_abi_repro`: verifies whether changing only the outer `__global__` kernel ABI is enough, while the VF symbol still exposes `bfloat16_t*`

Build:

```bash
cd src/cannbench/operators/builtin/lightning_indexer/simt/test/compiler_repro
mkdir -p build
cd build
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
cmake .. -DCMAKE_ASC_ARCHITECTURES=dav-3510
make -j1
```

Run:

```bash
cd src/cannbench/operators/builtin/lightning_indexer/simt/test/compiler_repro/build
. /usr/local/Ascend/ascend-toolkit/set_env.sh
export ASCEND_HOME_PATH=/usr/local/Ascend/ascend-toolkit/latest
export ASCEND_SLOG_PRINT_TO_STDOUT=1
export ASCEND_GLOBAL_LOG_LEVEL=1
./fp16_vf_control
./bf16_vf_repro
./bf16_vf_mixed_abi_repro
```
