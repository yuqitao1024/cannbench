from pathlib import Path
import re
import subprocess


CASE_DIR = Path(__file__).resolve().parent
ASC_PATH = CASE_DIR / "consumer_native_layout.asc"
RUN_PATH = CASE_DIR / "scripts" / "run.sh"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_DIR)}"
    return path.read_text(encoding="utf-8")


def test_consumer_native_layout_has_one_asc_and_tracked_deliverables():
    assert sorted(path.name for path in CASE_DIR.glob("*.asc")) == [
        "consumer_native_layout.asc"
    ]
    for relative in ("CMakeLists.txt", "README.md", "SPEC.md", "scripts/run.sh"):
        assert (CASE_DIR / relative).is_file(), f"missing {relative}"


def test_consumer_native_layout_cmake_selects_two_compile_time_scenarios():
    cmake = read_required(CASE_DIR / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-1\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(consumer_native_layout LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert "consumer_native_layout.asc" in cmake


def test_source_stays_in_c_tensor_simt_api_and_c_style_resource_boundary():
    source = read_required(ASC_PATH)
    for token in (
        '#include "acl/acl.h"',
        '#include "c_api/asc_simd.h"',
        '#include "simt_api/asc_simt.h"',
        "__simt_vf__",
        "asc_vf_call",
        "typedef struct",
        "DeviceBuffers",
        "malloc(",
        "free(",
        "aclrtMalloc",
        "aclrtFree",
    ):
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
        "AscendC::",
        "class ",
        "std::",
        "vector<",
        "unordered_map",
        "new ",
        "delete ",
    )
    for token in forbidden:
        assert token not in source


def test_case_zero_materializes_row_major_then_packs_and_case_one_writes_native_offset():
    source = read_required(ASC_PATH)
    for token in (
        "blocked_offset_16x16",
        "row_major_producer_vf",
        "explicit_blocked_pack_vf",
        "consumer_native_layout_vf",
        "row_major_producer_kernel",
        "explicit_blocked_pack_kernel",
        "consumer_native_layout_kernel",
        "row_major_workspace",
    ):
        assert token in source
    assert "row_major_workspace[logical_offset] = producer_transform(input[logical_offset])" in source
    assert "consumer_output[consumer_offset] = row_major_workspace[logical_offset]" in source
    assert "consumer_output[consumer_offset] = producer_transform(input[logical_offset])" in source

    launch = source.split("static void launch_selected_pipeline", 1)[1]
    launch = launch.split("int main", 1)[0]
    case_zero, case_one = launch.split("#else", 1)
    assert case_zero.count("<<<BLOCK_DIM, 0, stream>>>") == 2
    assert "row_major_producer_kernel" in case_zero
    assert "explicit_blocked_pack_kernel" in case_zero
    assert case_one.count("<<<BLOCK_DIM, 0, stream>>>") == 1
    assert "consumer_native_layout_kernel" in case_one


def test_host_uses_fixed_input_and_full_blocked_layout_oracle():
    source = read_required(ASC_PATH)
    for token in (
        "#define RANDOM_SEED 20260812U",
        "fill_input_fixed_seed",
        "host_blocked_offset_16x16",
        "host_consumer_layout_oracle",
        "verify_complete_consumer_layout",
        "actual != expected_output",
        "ACL_MEMCPY_HOST_TO_DEVICE",
        "ACL_MEMCPY_DEVICE_TO_HOST",
        "Verification PASSED",
        "Verification FAILED",
    ):
        assert token in source
    assert re.search(r"#if SCENARIO_NUM == 0\s+if \(!check_acl\(aclrtMalloc\(", source)


def test_profile_script_keeps_raw_rows_and_aggregates_every_kernel_in_complete_boundary():
    script = read_required(RUN_PATH)
    subprocess.run(["bash", "-n", str(RUN_PATH)], check=True)
    assert 'run_scenario 0 "explicit_pack"' in script
    assert 'run_scenario 1 "native_layout"' in script
    assert "Verification PASSED" in script
    assert "msopprof" in script
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert 'profiles/scenario_${scenario}/raw' in script
    assert 'profiles/scenario_${scenario}/parsed' in script
    assert "row_major_producer_kernel,explicit_blocked_pack_kernel" in script
    assert '"consumer_native_layout_kernel"' in script
    assert "expected_launches_per_call" in script
    assert '--launch-count="${expected_launches_per_call}"' in script
    assert "mean_complete_boundary_us" in script
    assert "kernel_rows.csv" in script
    assert "aggregate.csv" in script
    assert "warmup" not in script.lower()
    assert "rm -rf" not in script


def test_docs_define_layout_workspace_launch_and_leave_measurements_blank():
    readme = read_required(CASE_DIR / "README.md")
    spec = read_required(CASE_DIR / "SPEC.md")
    combined = readme + "\n" + spec
    for token in (
        "[4096, 256]",
        "float32",
        "16x16 blocked",
        "[256, 16, 16, 16]",
        "SCENARIO_NUM=0",
        "SCENARIO_NUM=1",
        "row-major GM workspace",
        "4 MiB",
        "64 个 block",
        "512 个 SIMT 线程",
        "两个 kernel",
        "一个 kernel",
        "完整生产边界",
        "host oracle",
        "Task Duration",
        "预期瓶颈",
        "msopprof",
    ):
        assert token in combined
    assert "29.364000" in readme
    assert "15.025000" in readme
    assert "不含两次 launch 间 gap" in readme
