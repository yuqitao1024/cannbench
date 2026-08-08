# vLLM-Ascend Sparse Flash Attention Source Provenance

The reproducible operator project under `vllm_ascend_a5b0ce/` comes from
vLLM-Ascend commit:

```text
a5b0ce10d84bf76dd9c5c9e7ab9a5ddeed5af7ca
```

The source headers identify Huawei Technologies Co., Ltd. as copyright holder
and use the CANN Open Software License Agreement Version 2.0. The vendored
subset preserves the upstream `csrc` paths required to build
`SparseFlashAttention` for Ascend 950.

The copied source required operator-local host compatibility changes to
register Ascend 950 and accept its SoC version. The final arch35 implementation
also replaces the QK and V gather paths with `sfa_qk_gather_vf` and
`sfa_v_gather_vf`. Both publish SIMT GM stores with `asc_dcci_entire` before the
existing AIV-to-Cube synchronization boundary. This version intentionally
retains upstream Basic API code under the task's function-first exception.

## Copied Baseline SHA-256

```text
d78d073dfcc4bd2bf7eab378be62557209152bfdf325dbace8af3632bfc249e2  sparse_flash_attention.cpp
a993dd4b25433ba43cb4b1af1178f412c3881e5e42d9c497ddd1aa4c6734d2c5  sparse_flash_attention_common.h
a9594eab83b8e3038a4a68f6ba1159078020cf6a725e84aa26eda687155d8163  sparse_flash_attention_template_tiling_key.h
5122768ed0c019dd37a78cefb18091ea7a2a3b8a714a9f5a43fe491d5f7b6de2  arch35/sparse_flash_attention_kernel_mla.h
b1ff97cc03546234e9acea99ae6497b487a879ba3cf37b39213913d473874894  arch35/sparse_flash_attention_service_cube_mla.h
332679ebb84a571582d7f3fd3f5cd086415ff24188b3c4ef0370187ec3de02b3  arch35/sparse_flash_attention_service_vector_mla.h
05ebe36a3ae1248aaeca94c3eb1ddb6d0cdf6d2e96938a50706ed390a3ba76c7  copied operator package
```

## Final VF SHA-256

```text
5e84a254c5e40e264174d4cf4a72ed20a9eca783e84e693b65b203a8962cd705  arch35/sparse_flash_attention_service_vector_mla.h
8b45423d6bdb709d7a2f17e56315d9711014beb75eb20fd13ad01d5b5223f6e3  final operator package
```

Remote build, accuracy, timing, and package evidence is retained under:

```text
/root/cannbench-dsa-vllm-simt-vf-gather-20260806/evidence/
```
