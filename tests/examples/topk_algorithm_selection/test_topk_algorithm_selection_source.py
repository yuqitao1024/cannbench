from pathlib import Path
import re
import subprocess


CASE_DIR = Path(__file__).resolve().parent
ASC_PATH = CASE_DIR / "topk_algorithm_selection.asc"
README_PATH = CASE_DIR / "README.md"
SPEC_PATH = CASE_DIR / "SPEC.md"
RUN_PATH = CASE_DIR / "scripts" / "run.sh"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_DIR)}"
    return path.read_text(encoding="utf-8")


def test_case_has_one_asc_translation_unit_and_required_files():
    assert sorted(path.name for path in CASE_DIR.glob("*.asc")) == [
        "topk_algorithm_selection.asc"
    ]
    for relative in ("CMakeLists.txt", "README.md", "SPEC.md", "scripts/run.sh"):
        assert (CASE_DIR / relative).is_file(), f"missing {relative}"


def test_cmake_selects_exactly_two_compile_time_scenarios():
    cmake = read_required(CASE_DIR / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-1\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(topk_algorithm_selection LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert "topk_algorithm_selection.asc" in cmake


def test_asc_source_stays_inside_c_tensor_simt_api_and_c_style_boundary():
    source = read_required(ASC_PATH)
    required = (
        '#include "acl/acl.h"',
        '#include "c_api/asc_simd.h"',
        '#include "simt_api/asc_simt.h"',
        "__simt_vf__",
        "asc_vf_call",
        "aclrtMalloc",
        "aclrtFree",
    )
    for token in required:
        assert token in source

    forbidden = (
        "kernel_operator.h",
        "basic_api/",
        "LocalTensor",
        "SetFlag",
        "WaitFlag",
        "PipeBarrier",
        "CrossCoreSetFlag",
        "CrossCoreWaitFlag",
        "AscendC::Mutex",
        "class ",
        "std::",
        "vector<",
        "unordered_map",
        "new ",
        "delete ",
    )
    for token in forbidden:
        assert token not in source

    assert "typedef struct" in source
    assert "DeviceBuffers" in source


def test_scenarios_expose_algorithmic_difference_before_microarchitecture():
    source = read_required(ASC_PATH)
    assert "#define MERGE_CHUNK 16U" in source
    assert "#define BITONIC_PADDED_WIDTH 64U" in source
    assert "topk_repeated_padded_bitonic_vf" in source
    assert "bitonic_sort_64_descending" in source
    assert "topk_streaming_bounded_vf" in source
    assert "bounded_insert" in source
    assert re.search(r"#if SCENARIO_NUM == 0\s+topk_repeated_padded_bitonic_kernel", source)
    assert "topk_streaming_bounded_kernel" in source


def test_host_uses_fixed_input_oracle_and_checks_values_and_indices():
    source = read_required(ASC_PATH)
    for token in (
        "#define RANDOM_SEED 20260811U",
        "fill_input_fixed_seed",
        "host_topk_oracle",
        "qsort",
        "verify_result",
        "ACL_MEMCPY_HOST_TO_DEVICE",
        "ACL_MEMCPY_DEVICE_TO_HOST",
        "Verification PASSED",
        "Verification FAILED",
    ):
        assert token in source
    assert "value != expected_values" in source
    assert "index != expected_indices" in source


def test_run_script_verifies_before_profiling_and_preserves_raw_data():
    script = read_required(RUN_PATH)
    subprocess.run(["bash", "-n", str(RUN_PATH)], check=True)
    assert 'run_scenario 0 "repeated_padded_bitonic"' in script
    assert 'run_scenario 1 "streaming_bounded"' in script
    assert "Verification PASSED" in script
    assert "msopprof" in script
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert 'profiles/scenario_${scenario}/raw' in script
    assert 'profiles/scenario_${scenario}/parsed' in script
    assert 'kernel_rows.csv' in script
    assert 'find "${raw_dir}"' in script
    assert "rm -rf" not in script


def test_docs_freeze_semantics_boundary_and_record_device_evidence():
    readme = read_required(README_PATH)
    spec = read_required(SPEC_PATH)
    combined = readme + "\n" + spec
    for token in (
        "[32, 1024]",
        "float32",
        "Top-32",
        "值降序",
        "索引升序",
        "SCENARIO_NUM=0",
        "SCENARIO_NUM=1",
        "1 个 block",
        "512 个 SIMT 线程",
        "固定随机种子",
        "host oracle",
        "msopprof",
        "Task Duration",
        "预期瓶颈",
    ):
        assert token in combined
    for token in (
        "CANN 9.2.0",
        "默认 warmup",
        "182261.843750",
        "7616.294922",
        "95.82%",
        "23.93x",
        "20260812-002736",
    ):
        assert token in readme
    assert "待实测" not in readme
