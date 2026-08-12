from pathlib import Path
import re
import subprocess


CASE_ROOT = Path(__file__).resolve().parent
SOURCE_PATH = CASE_ROOT / "lane_local_reuse.asc"
RUN_PATH = CASE_ROOT / "scripts" / "run.sh"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_ROOT)}"
    return path.read_text(encoding="utf-8")


def function_body(source: str, name: str, next_name: str) -> str:
    return source[source.index(name) : source.index(next_name)]


def test_lane_local_reuse_case_has_required_standalone_layout():
    assert sorted(path.name for path in CASE_ROOT.glob("*.asc")) == [
        "lane_local_reuse.asc"
    ]
    for relative in ("CMakeLists.txt", "README.md", "SPEC.md", "scripts/run.sh"):
        assert (CASE_ROOT / relative).is_file(), f"missing {relative}"


def test_lane_local_reuse_cmake_selects_three_compile_time_scenarios():
    cmake = read_required(CASE_ROOT / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-2\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(lane_local_reuse LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert cmake.count("lane_local_reuse.asc") == 1


def test_lane_local_reuse_source_respects_api_and_c_style_boundary():
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
        "class ",
        "std::",
        "vector<",
        "array<",
        "new ",
        "delete ",
    )
    for token in forbidden:
        assert token not in source

    mutex_tokens = re.findall(r"AscendC::Mutex::[A-Za-z_]+", source)
    assert set(mutex_tokens) <= {"AscendC::Mutex::Lock", "AscendC::Mutex::Unlock"}


def test_lane_local_reuse_scenarios_isolate_metadata_lifetimes():
    source = read_required(SOURCE_PATH)
    assert "#define PASS_COUNT 8U" in source
    assert "#define LANE_ELEMENT_COUNT 4U" in source
    assert "#if SCENARIO_NUM == 0" in source
    assert "#elif SCENARIO_NUM == 1" in source
    assert "recompute_lane_metadata_kernel" in source
    assert "shared_ub_metadata_kernel" in source
    assert "lane_local_reuse_kernel" in source

    recompute = function_body(
        source, "recompute_lane_metadata_vf", "shared_ub_metadata_vf"
    )
    assert "for (uint32_t pass = 0; pass < PASS_COUNT; ++pass)" in recompute
    assert "__gm__ volatile" in recompute
    assert recompute.index("for (uint32_t pass") < recompute.index("__gm__ volatile")

    shared = function_body(source, "shared_ub_metadata_vf", "lane_local_reuse_vf")
    assert "shared_values" in shared
    assert "shared_metadata" in shared
    assert "shared_validity" in shared
    assert "asc_syncthreads();" in shared
    assert "__ubuf__ volatile" in shared
    assert shared.index("asc_syncthreads();") < shared.index("for (uint32_t pass")

    lane_local = function_body(source, "lane_local_reuse_vf", "recompute_lane_metadata_kernel")
    assert "lane_values[LANE_ELEMENT_COUNT]" in lane_local
    assert "lane_metadata[LANE_ELEMENT_COUNT]" in lane_local
    assert "lane_validity_mask" in lane_local
    assert lane_local.index("lane_validity_mask") < lane_local.index("for (uint32_t pass")


def test_lane_local_reuse_host_oracle_covers_boundaries_and_tail_exactly():
    source = read_required(SOURCE_PATH)
    for token in (
        "#define RANDOM_SEED 20260812U",
        "BOUNDARY_CASE_COUNTS",
        "TILE_ELEMENT_COUNT - 1U",
        "TILE_ELEMENT_COUNT",
        "TILE_ELEMENT_COUNT + 1U",
        "ELEMENT_CAPACITY - 13U",
        "fill_input_fixed_seed",
        "build_host_oracle",
        "verify_output",
        "OUTPUT_SENTINEL",
        "verify_untouched_tail",
        "output mismatch",
        "Verification PASSED",
        "Verification FAILED",
    ):
        assert token in source
    assert "for (uint32_t index = 0; index < element_count; ++index)" in source


def test_lane_local_reuse_run_script_verifies_then_profiles_raw_default_runs():
    script = read_required(RUN_PATH)
    subprocess.run(["bash", "-n", str(RUN_PATH)], check=True)
    assert 'run_scenario 0 "recompute_lane_metadata_kernel"' in script
    assert 'run_scenario 1 "shared_ub_metadata_kernel"' in script
    assert 'run_scenario 2 "lane_local_reuse_kernel"' in script
    assert 'grep -q "Verification PASSED"' in script
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert "msopprof" in script
    assert "--warm-up" not in script
    assert "--warmup" not in script
    assert '${PROFILE_ROOT}/scenario_${scenario}/raw' in script
    assert 'find "${raw_dir}"' in script
    assert "kernel_rows.csv" in script


def test_lane_local_reuse_docs_freeze_fair_boundary_without_fake_results():
    readme = read_required(CASE_ROOT / "README.md")
    spec = read_required(CASE_ROOT / "SPEC.md")
    combined = readme + "\n" + spec
    for token in (
        "SCENARIO_NUM=0",
        "SCENARIO_NUM=1",
        "SCENARIO_NUM=2",
        "重复加载",
        "shared UB",
        "lane-local",
        "validity",
        "tail",
        "固定 seed",
        "host oracle",
        "每次调用 1 次 kernel launch",
        "Ascend 950",
        "预期瓶颈",
        "默认参数",
    ):
        assert token in combined
    assert "8.312000" in readme
    assert "6.038000" in readme
    assert "4.468000" in readme
    assert "不使用 Mutex" in readme
