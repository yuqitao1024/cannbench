from pathlib import Path
import re
import subprocess


CASE_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = CASE_ROOT / "row_width_dispatch.asc"
RUN_PATH = CASE_ROOT / "scripts" / "run.sh"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_ROOT)}"
    return path.read_text(encoding="utf-8")


def test_row_width_dispatch_has_required_standalone_layout():
    assert sorted(path.name for path in CASE_ROOT.glob("*.asc")) == [
        "row_width_dispatch.asc"
    ]
    for relative in ("CMakeLists.txt", "README.md", "SPEC.md", "scripts/run.sh"):
        assert (CASE_ROOT / relative).is_file(), f"missing {relative}"


def test_row_width_dispatch_cmake_selects_three_compile_time_strategies():
    cmake = read_required(CASE_ROOT / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-2\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(row_width_dispatch LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert cmake.count("row_width_dispatch.asc") == 1


def test_row_width_dispatch_source_stays_in_c_tensor_simt_boundary():
    source = read_required(SOURCE_PATH)
    for token in (
        '#include "acl/acl.h"',
        '#include "simt_api/asc_simt.h"',
        "aclrtMalloc",
        "aclrtMemcpy",
        "aclrtFree",
        "typedef struct DeviceBuffers",
    ):
        assert token in source

    forbidden = (
        "kernel_operator.h",
        "basic_api/",
        "AscendC::LocalTensor",
        "SetFlag",
        "WaitFlag",
        "PipeBarrier",
        "CrossCoreSetFlag",
        "CrossCoreWaitFlag",
        "AscendC::Mutex",
        "class ",
        "std::",
        "vector<",
        "array<",
        "new ",
        "delete ",
    )
    for token in forbidden:
        assert token not in source


def test_row_width_dispatch_exposes_generic_bucketed_and_exact_overfit_paths():
    source = read_required(SOURCE_PATH)
    for token in (
        "#define SMALL_WIDTH_LIMIT 256U",
        "#define MEDIUM_WIDTH_LIMIT 4096U",
        "#define EXACT_WIDTH 1024U",
        "#define MAX_WIDTH 8192U",
        "generic_row_normalize_kernel",
        "small_row_normalize_kernel",
        "medium_row_normalize_kernel",
        "wide_row_normalize_kernel",
        "exact_1024_row_normalize_kernel",
        "exact_overfit_fallback_kernel",
        "dim3(SMALL_THREAD_COUNT",
        "dim3(MEDIUM_THREAD_COUNT",
        "dim3(WIDE_THREAD_COUNT",
    ):
        assert token in source

    assert "if (width <= SMALL_WIDTH_LIMIT)" in source
    assert "else if (width <= MEDIUM_WIDTH_LIMIT)" in source
    assert "if (width == EXACT_WIDTH)" in source
    assert "scan_width = MAX_WIDTH" in source
    assert "launch_count=1" in source


def test_row_width_dispatch_host_oracle_checks_sum_normalize_and_row_tail():
    source = read_required(SOURCE_PATH)
    for token in (
        "#define RANDOM_SEED 20260812U",
        "#define NORMALIZE_SCALE 1000000U",
        "#define OUTPUT_SENTINEL 0xa5a5a5a5U",
        "fill_input_fixed_seed",
        "build_host_oracle",
        "verify_row_sums",
        "verify_normalized_output",
        "verify_row_tail_sentinel",
        "row sum mismatch",
        "normalized output mismatch",
        "row tail overwrite",
        "Verification PASSED",
        "Verification FAILED",
    ):
        assert token in source
    assert "width == 0U || width > MAX_WIDTH" in source


def test_row_width_dispatch_run_script_profiles_boundaries_and_neighbors_after_correctness():
    script = read_required(RUN_PATH)
    subprocess.run(["bash", "-n", str(RUN_PATH)], check=True)
    for width in (255, 256, 257, 1023, 1024, 1025, 4095, 4096, 4097, 8192):
        assert str(width) in script
    assert 'run_scenario 0 "generic"' in script
    assert 'run_scenario 1 "bucketed"' in script
    assert 'run_scenario 2 "exact_overfit"' in script
    assert 'grep -q "Verification PASSED"' in script
    assert script.index("correctness_widths") < script.index("profile_widths")
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert "--warm-up" not in script
    assert "--warmup" not in script
    assert '${PROFILE_ROOT}/scenario_${scenario}/width_${width}/raw' in script
    assert "kernel_rows.csv" in script
    assert 'find "${raw_dir}"' in script


def test_row_width_dispatch_docs_define_auditable_launches_without_fake_results():
    readme = read_required(CASE_ROOT / "README.md")
    spec = read_required(CASE_ROOT / "SPEC.md")
    combined = readme + "\n" + spec
    for token in (
        "SCENARIO_NUM=0",
        "SCENARIO_NUM=1",
        "SCENARIO_NUM=2",
        "<=256",
        "<=4096",
        ">4096",
        "exact width",
        "邻近",
        "padding",
        "row sum",
        "normalize",
        "host oracle",
        "每次调用 1 次 kernel launch",
        "默认参数",
        "Ascend 950",
        "预期瓶颈",
    ):
        assert token in combined
    assert "5.829" in readme
    assert "9.343" in readme
    assert "8.47%" in readme
    assert "真实 shape 权重" in readme
