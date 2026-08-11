from pathlib import Path
import csv
import re
import subprocess
import sys


CASE_DIR = Path(__file__).resolve().parent
ASC_PATH = CASE_DIR / "workaround_revalidation.asc"
RUN_PATH = CASE_DIR / "scripts" / "run.sh"
PARSER_PATH = CASE_DIR / "scripts" / "parse_profile.py"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_DIR)}"
    return path.read_text(encoding="utf-8")


def test_workaround_revalidation_has_one_asc_and_tracked_deliverables():
    assert sorted(path.name for path in CASE_DIR.glob("*.asc")) == [
        "workaround_revalidation.asc"
    ]
    for relative in (
        "CMakeLists.txt",
        "README.md",
        "SPEC.md",
        "scripts/run.sh",
        "scripts/parse_profile.py",
    ):
        assert (CASE_DIR / relative).is_file(), f"missing {relative}"


def test_cmake_selects_exactly_two_scenarios():
    cmake = read_required(CASE_DIR / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-1\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(workaround_revalidation LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake


def test_source_uses_allowed_api_and_c_style_resources():
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
        "calloc(",
        "free(",
        "aclrtMalloc",
        "aclrtFree",
    ):
        assert token in source
    for token in (
        "kernel_operator.h", "basic_api/", "LocalTensor", "SetFlag", "WaitFlag",
        "PipeBarrier", "CrossCoreSetFlag", "CrossCoreWaitFlag", "AscendC::",
        "class ", "std::", "vector<", "unordered_map", "new ", "delete ",
    ):
        assert token not in source


def test_safe_workaround_and_root_cause_fix_have_auditable_ownership_and_launches():
    source = read_required(ASC_PATH)
    for token in (
        "#define WORKAROUND_BLOCK_DIM 1U",
        "#define FIXED_BLOCK_DIM 64U",
        "#define REPEAT_COUNT 3U",
        "UNSAFE_REMOVAL_REPRODUCTION",
        "scratch_slot = (uint32_t)threadIdx.x",
        "scratch_slot = block_index * (uint32_t)blockDim.x + (uint32_t)threadIdx.x",
        "row_sum_single_block_workaround_kernel",
        "row_sum_unique_scratch_multiblock_kernel",
        "validate_ownership_model",
        "unsafe_collision_count",
        "fixed_collision_count",
        "Ownership model PASSED",
    ):
        assert token in source
    launch = source.split("static void launch_selected_kernel", 1)[1].split("int main", 1)[0]
    assert "<<<WORKAROUND_BLOCK_DIM, 0, stream>>>" in launch
    assert "<<<FIXED_BLOCK_DIM, 0, stream>>>" in launch
    assert "for (uint32_t repeat = 0U; repeat < REPEAT_COUNT; ++repeat)" in launch


def test_fixed_seed_tail_shape_and_complete_oracle_are_shared():
    source = read_required(ASC_PATH)
    for token in (
        "#define ROW_COUNT 4099U",
        "#define ROW_WIDTH 257U",
        "#define RANDOM_SEED 20260812U",
        "fill_input_fixed_seed",
        "host_row_sum_oracle",
        "verify_complete_output",
        "actual != expected_output",
        "ACL_MEMCPY_HOST_TO_DEVICE",
        "ACL_MEMCPY_DEVICE_TO_HOST",
        "Verification PASSED",
        "Verification FAILED",
    ):
        assert token in source


def test_run_script_verifies_then_profiles_and_records_kernel_block_repeat_manifest():
    script = read_required(RUN_PATH)
    subprocess.run(["bash", "-n", str(RUN_PATH)], check=True)
    assert 'run_scenario 0 "single_block_workaround"' in script
    assert 'run_scenario 1 "fixed_multiblock"' in script
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert 'profiles/scenario_${scenario}/raw' in script
    assert 'profiles/scenario_${scenario}/parsed' in script
    assert "row_sum_single_block_workaround_kernel" in script
    assert "row_sum_unique_scratch_multiblock_kernel" in script
    assert "block_dim=1" in script
    assert "block_dim=64" in script
    assert "expected_launches_per_call=3" in script
    assert '--launch-count="${expected_launches_per_call}"' in script
    assert "launch_manifest.txt" in script
    assert "kernel_rows.csv" in script
    assert "aggregate.csv" in script
    assert "parse_profile.py" in script
    assert "warmup" not in script.lower()
    assert "rm -rf" not in script


def test_profile_parser_requires_exact_launch_count_and_writes_aggregate(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    profile = raw / "OpBasicInfo_0.csv"
    fieldnames = ["Op Name", "Task Duration(us)"]

    def write_rows(count: int) -> None:
        with profile.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for index in range(count):
                writer.writerow(
                    {
                        "Op Name": "row_sum_single_block_workaround_kernel",
                        "Task Duration(us)": str(10.0 + index),
                    }
                )

    rows = tmp_path / "kernel_rows.csv"
    aggregate = tmp_path / "aggregate.csv"
    command = [
        sys.executable,
        str(PARSER_PATH),
        "--raw",
        str(raw),
        "--kernel",
        "row_sum_single_block_workaround_kernel",
        "--expected-launches",
        "3",
        "--rows-output",
        str(rows),
        "--aggregate-output",
        str(aggregate),
    ]

    write_rows(2)
    failed = subprocess.run(command, text=True, capture_output=True)
    assert failed.returncode != 0
    assert "expected 3 selected rows, observed 2" in failed.stderr

    write_rows(3)
    subprocess.run(command, check=True)
    parsed = list(csv.DictReader(rows.open(encoding="utf-8")))
    summary = list(csv.DictReader(aggregate.open(encoding="utf-8")))
    assert len(parsed) == 3
    assert [float(row["task_duration_us"]) for row in parsed] == [10.0, 11.0, 12.0]
    assert summary == [
        {
            "kernel": "row_sum_single_block_workaround_kernel",
            "observed_launches": "3",
            "task_duration_sum_us": "33.000000",
        }
    ]

    with profile.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["Op Name", "Task Duration(ns)"])
        writer.writeheader()
        for index in range(3):
            writer.writerow(
                {
                    "Op Name": "row_sum_single_block_workaround_kernel",
                    "Task Duration(ns)": str(10000 + index),
                }
            )
    wrong_unit = subprocess.run(command, text=True, capture_output=True)
    assert wrong_unit.returncode != 0
    assert "required column Task Duration(us)" in wrong_unit.stderr


def test_docs_require_root_cause_and_record_device_evidence():
    readme = read_required(CASE_DIR / "README.md")
    spec = read_required(CASE_DIR / "SPEC.md")
    combined = readme + "\n" + spec
    for token in (
        "[4099, 257]", "float32", "SCENARIO_NUM=0", "SCENARIO_NUM=1",
        "历史 correctness workaround", "unsafe removal", "scratch ownership",
        "原始失败条件", "根因修复", "边界", "tail", "多 block", "重复 launch",
        "完整输出", "host oracle", "Task Duration", "预期瓶颈",
        "dav-3510", "CANN 9.2.0", "2431.649048", "309.659996",
        "87.27%", "7.85x", "32256", "修复后为 0", "20260812-063112",
    ):
        assert token in combined
    assert "待实测" not in readme
    assert re.search(r"\|\s*0\s*\|[^\n]*2431\.649048", readme)
    assert re.search(r"\|\s*1\s*\|[^\n]*309\.659996", readme)
