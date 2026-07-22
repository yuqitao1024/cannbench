from pathlib import Path


def test_lightning_indexer_compiler_repro_files_exist():
    base = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/test/compiler_repro"
    )

    assert (base / "CMakeLists.txt").exists()
    assert (base / "README.md").exists()
    assert (base / "bf16_vf_repro.asc").exists()
    assert (base / "bf16_vf_mixed_abi_repro.asc").exists()
    assert (base / "fp16_vf_control.asc").exists()


def test_lightning_indexer_compiler_repro_cmake_defines_bf16_and_fp16_targets():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/test/compiler_repro/CMakeLists.txt"
    ).read_text(encoding="utf-8")

    assert "project(lightning_indexer_bf16_vf_compiler_repro LANGUAGES ASC CXX)" in source
    assert "add_executable(fp16_vf_control" in source
    assert "add_executable(bf16_vf_repro" in source
    assert "add_executable(bf16_vf_mixed_abi_repro" in source
    assert "LINKER_LANGUAGE ASC" in source


def test_lightning_indexer_bf16_vf_repro_uses_asc_vf_call_and_bfloat16_kernel():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/test/compiler_repro/bf16_vf_repro.asc"
    ).read_text(encoding="utf-8")

    assert '#include "simt_api/asc_bf16.h"' in source
    assert "__simt_vf__" in source
    assert "asc_vf_call<" in source
    assert "__gm__ const bfloat16_t* input" in source
    assert "__gm__ bfloat16_t* output" in source


def test_lightning_indexer_fp16_vf_control_uses_asc_vf_call_and_half_kernel():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/test/compiler_repro/fp16_vf_control.asc"
    ).read_text(encoding="utf-8")

    assert '#include "simt_api/asc_fp16.h"' in source
    assert "__simt_vf__" in source
    assert "asc_vf_call<" in source
    assert "__gm__ const half* input" in source
    assert "__gm__ half* output" in source


def test_lightning_indexer_bf16_vf_mixed_abi_repro_casts_uint16_global_abi_to_bfloat16_vf():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/test/compiler_repro/bf16_vf_mixed_abi_repro.asc"
    ).read_text(encoding="utf-8")

    assert '#include "simt_api/asc_bf16.h"' in source
    assert "__simt_vf__" in source
    assert "asc_vf_call<" in source
    assert "__gm__ const bfloat16_t* input" in source
    assert "__gm__ bfloat16_t* output" in source
    assert "__gm__ const uint16_t* input" in source
    assert "__gm__ uint16_t* output" in source
    assert "reinterpret_cast<__gm__ const bfloat16_t*>(input)" in source
    assert "reinterpret_cast<__gm__ bfloat16_t*>(output)" in source
