from pathlib import Path
import re
import subprocess


CASE_ROOT = Path(__file__).resolve().parent
SOURCE = CASE_ROOT / "online_row_reduction.asc"
README = CASE_ROOT / "README.md"
SPEC = CASE_ROOT / "SPEC.md"
RUN = CASE_ROOT / "scripts" / "run.sh"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_ROOT)}"
    return path.read_text(encoding="utf-8")


def test_online_row_reduction_has_one_asc_and_required_tracked_files():
    assert sorted(path.name for path in CASE_ROOT.glob("*.asc")) == [
        "online_row_reduction.asc"
    ]
    for relative in (
        "CMakeLists.txt",
        "README.md",
        "SPEC.md",
        "scripts/run.sh",
    ):
        assert (CASE_ROOT / relative).is_file(), f"missing {relative}"


def test_online_row_reduction_cmake_selects_exactly_three_scenarios():
    cmake = read_required(CASE_ROOT / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-2\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(online_row_reduction LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert "target_link_libraries(online_row_reduction PRIVATE m)" in cmake
    assert cmake.count("online_row_reduction.asc") == 1


def test_online_row_reduction_source_is_c_style_and_stays_in_allowed_api_boundary():
    source = read_required(SOURCE)
    for token in (
        '#include "acl/acl.h"',
        '#include "c_api/asc_simd.h"',
        '#include "simt_api/asc_simt.h"',
        "typedef struct",
        "DeviceBuffers",
        "aclrtMalloc",
        "aclrtFree",
        "__simt_vf__",
        "asc_vf_call",
    ):
        assert token in source

    for token in (
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
        "new ",
        "delete ",
    ):
        assert token not in source


def test_online_row_reduction_implements_three_stable_scan_structures():
    source = read_required(SOURCE)
    for token in (
        "#define TILE_ELEMENTS 4096U",
        "three_scan_softmax_kernel",
        "online_stats_softmax_kernel",
        "tiled_online_softmax_kernel",
        "three_scan_softmax_vf",
        "online_stats_softmax_vf",
        "tiled_online_softmax_vf",
        "online_update",
        "merge_online_stats",
        "tile_start",
        "tile_end",
        "expf",
    ):
        assert token in source
    assert re.search(r"#if SCENARIO_NUM == 0\s+three_scan_softmax_kernel", source)
    assert re.search(r"#elif SCENARIO_NUM == 1\s+online_stats_softmax_kernel", source)


def test_online_row_reduction_fixed_cases_oracle_tolerance_and_full_output():
    source = read_required(SOURCE)
    for token in (
        "ORDINARY_WIDTH 1024U",
        "EXTREME_WIDTH 4096U",
        "TAIL_WIDTH 65537U",
        "1000.0F",
        "-1000.0F",
        "fill_fixed_input",
        "build_double_softmax_oracle",
        "double* expected",
        "ABS_TOLERANCE",
        "REL_TOLERANCE",
        "verify_full_output",
        "verify_row_sums",
        "verify_output_guard",
        "Verification PASSED",
    ):
        assert token in source


def test_online_row_reduction_reports_scans_and_freezes_launch_geometry():
    source = read_required(SOURCE)
    for token in (
        "stats_scans_per_row",
        "normalize_scans_per_row",
        "input_element_visits",
        "tile_merges",
        "tail_elements",
        "verify_work_counters",
        "#define BLOCK_DIM 1U",
        "#define THREAD_COUNT 512U",
    ):
        assert token in source


def test_online_row_reduction_run_profiles_each_case_after_correctness_and_keeps_raw():
    script = read_required(RUN)
    subprocess.run(["bash", "-n", str(RUN)], check=True)
    assert 'run_scenario 0 "three_scan_softmax_kernel"' in script
    assert 'run_scenario 1 "online_stats_softmax_kernel"' in script
    assert 'run_scenario 2 "tiled_online_softmax_kernel"' in script
    assert "for case_id in 0 1 2" in script
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert "mktemp -d" in script
    assert 'scenario_${scenario}_case_${case_id}_' in script
    assert 'raw_dir="${profile_dir}/raw"' in script
    assert '--output="${raw_dir}"' in script
    assert "--warm-up" not in script
    assert "--warmup" not in script
    assert "--kernel-name" not in script


def test_online_row_reduction_docs_define_call_boundary_and_leave_performance_unmeasured():
    readme = read_required(README)
    spec = read_required(SPEC)
    combined = readme + "\n" + spec
    for token in (
        "SCENARIO_NUM=0",
        "SCENARIO_NUM=1",
        "SCENARIO_NUM=2",
        "double host oracle",
        "absolute tolerance",
        "relative tolerance",
        "65537",
        "1 block",
        "512",
        "1 次相关 kernel launch",
        "完整 softmax 调用边界",
        "Task Duration",
        "raw",
        "340.984985",
    ):
        assert token in combined
    assert "12981.391602" in readme
    assert "13026.042969" in readme
    assert "状态容量不依赖整行宽度" in readme
