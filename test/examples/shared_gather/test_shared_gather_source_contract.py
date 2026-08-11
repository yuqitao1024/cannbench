from pathlib import Path
import re
import subprocess


CASE_DIR = Path(__file__).resolve().parent
ASC_PATH = CASE_DIR / "shared_gather.asc"
RUN_PATH = CASE_DIR / "scripts" / "run.sh"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_DIR)}"
    return path.read_text(encoding="utf-8")


def test_shared_gather_has_one_asc_and_complete_tracked_deliverables():
    assert sorted(path.name for path in CASE_DIR.glob("*.asc")) == ["shared_gather.asc"]
    for relative in ("CMakeLists.txt", "README.md", "SPEC.md", "scripts/run.sh"):
        assert (CASE_DIR / relative).is_file(), f"missing {relative}"


def test_shared_gather_cmake_selects_two_scenarios_at_compile_time():
    cmake = read_required(CASE_DIR / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-1\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(shared_gather LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert "shared_gather.asc" in cmake


def test_shared_gather_source_uses_only_allowed_api_and_c_style_resources():
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


def test_scenario_zero_has_two_explicit_volatile_gathers_and_scenario_one_has_one_producer():
    source = read_required(ASC_PATH)
    for token in (
        "consumer_a_independent_gather",
        "consumer_b_independent_gather",
        "__gm__ volatile const float*",
        "shared_gather_producer",
        "consumer_a_transform",
        "consumer_b_transform",
        "shared_gather_independent_consumers_vf",
        "shared_gather_reuse_vf",
        "shared_gather_independent_kernel",
        "shared_gather_reuse_kernel",
    ):
        assert token in source
    independent_body = source.split("__simt_vf__ inline void shared_gather_independent_consumers_vf", 1)[1]
    independent_body = independent_body.split("__simt_vf__ inline void shared_gather_reuse_vf", 1)[0]
    assert independent_body.count("consumer_a_independent_gather(") == 1
    assert independent_body.count("consumer_b_independent_gather(") == 1
    reuse_body = source.split("__simt_vf__ inline void shared_gather_reuse_vf", 1)[1]
    reuse_body = reuse_body.split("__global__", 1)[0]
    assert reuse_body.count("shared_gather_producer(") == 1
    assert "independent_gather(" not in reuse_body
    assert re.search(r"#if SCENARIO_NUM == 0\s+shared_gather_independent_kernel", source)
    assert "shared_gather_reuse_kernel" in source


def test_host_builds_fixed_inputs_and_checks_both_complete_outputs_against_oracle():
    source = read_required(ASC_PATH)
    for token in (
        "#define RANDOM_SEED 20260812U",
        "fill_inputs_fixed_seed",
        "host_shared_gather_oracle",
        "verify_complete_outputs",
        "expected_consumer_a",
        "expected_consumer_b",
        "consumer_a != expected_consumer_a",
        "consumer_b != expected_consumer_b",
        "ACL_MEMCPY_HOST_TO_DEVICE",
        "ACL_MEMCPY_DEVICE_TO_HOST",
        "Verification PASSED",
        "Verification FAILED",
    ):
        assert token in source


def test_run_script_verifies_before_msopprof_and_retains_raw_evidence_without_warmup_override():
    script = read_required(RUN_PATH)
    subprocess.run(["bash", "-n", str(RUN_PATH)], check=True)
    assert 'run_scenario 0 "independent_consumers"' in script
    assert 'run_scenario 1 "shared_producer"' in script
    assert "Verification PASSED" in script
    assert "msopprof" in script
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert 'profiles/scenario_${scenario}/raw' in script
    assert 'profiles/scenario_${scenario}/parsed' in script
    assert "kernel_rows.csv" in script
    assert 'find "${raw_dir}"' in script
    assert "warmup" not in script.lower()
    assert "rm -rf" not in script


def test_docs_freeze_shape_semantics_launch_boundary_and_blank_performance_table():
    readme = read_required(CASE_DIR / "README.md")
    spec = read_required(CASE_DIR / "SPEC.md")
    combined = readme + "\n" + spec
    for token in (
        "[65536, 128]",
        "[8192]",
        "[8192, 128]",
        "float32",
        "uint32",
        "SCENARIO_NUM=0",
        "SCENARIO_NUM=1",
        "64 个 block",
        "512 个 SIMT 线程",
        "固定随机种子",
        "host oracle",
        "两个完整输出",
        "Task Duration",
        "预期瓶颈",
        "msopprof",
    ):
        assert token in combined
    assert "22.200001" in readme
    assert "18.719999" in readme
    assert "单次短 kernel raw" in readme
