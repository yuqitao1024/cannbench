from pathlib import Path
import re
import subprocess


CASE_ROOT = Path(__file__).resolve().parent
SOURCE = CASE_ROOT / "phase_aware_tile.asc"
README = CASE_ROOT / "README.md"
SPEC = CASE_ROOT / "SPEC.md"
RUN = CASE_ROOT / "scripts" / "run.sh"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_ROOT)}"
    return path.read_text(encoding="utf-8")


def test_phase_aware_tile_has_one_asc_and_required_tracked_files():
    assert sorted(path.name for path in CASE_ROOT.glob("*.asc")) == [
        "phase_aware_tile.asc"
    ]
    for relative in (
        "CMakeLists.txt",
        "README.md",
        "SPEC.md",
        "scripts/run.sh",
    ):
        assert (CASE_ROOT / relative).is_file(), f"missing {relative}"


def test_phase_aware_tile_cmake_selects_exactly_three_scenarios():
    cmake = read_required(CASE_ROOT / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-2\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(phase_aware_tile LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert cmake.count("phase_aware_tile.asc") == 1


def test_phase_aware_tile_source_keeps_c_style_and_allowed_device_api_boundary():
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


def test_phase_aware_tile_exposes_capacity_reuse_and_oversized_counterexample():
    source = read_required(SOURCE)
    for token in (
        "#define CONSERVATIVE_TILE 1024U",
        "#define REUSE_TILE 2048U",
        "#define OVERSIZED_OUTER_TILE 4096U",
        "#define STREAMED_SUBTILE 1024U",
        "conservative_phase_buffers_kernel",
        "peak_live_set_reuse_kernel",
        "oversized_outer_tile_kernel",
        "phase_a_scratch",
        "phase_b_scratch",
        "shared_phase_scratch",
        "persistent_operand",
        "coarse_replay_operand",
        "workspace",
    ):
        assert token in source
    assert "asc_syncthreads" in source
    assert re.search(r"#if SCENARIO_NUM == 0\s+conservative_phase_buffers_kernel", source)
    assert re.search(r"#elif SCENARIO_NUM == 1\s+peak_live_set_reuse_kernel", source)


def test_phase_aware_tile_freezes_oracle_output_guard_and_work_counters():
    source = read_required(SOURCE)
    for token in (
        "fill_fixed_inputs",
        "build_host_oracle",
        "verify_output",
        "verify_output_guard",
        "verify_work_counters",
        "logical_gm_read_elements",
        "logical_gm_write_elements",
        "logical_ub_copy_elements",
        "barrier_rounds",
        "declared_ub_bytes",
        "peak_live_set_bytes",
        "Verification PASSED",
    ):
        assert token in source


def test_phase_aware_tile_run_verifies_before_profile_and_keeps_default_warmup():
    script = read_required(RUN)
    subprocess.run(["bash", "-n", str(RUN)], check=True)
    assert 'run_scenario 0 "conservative_phase_buffers_kernel"' in script
    assert 'run_scenario 1 "peak_live_set_reuse_kernel"' in script
    assert 'run_scenario 2 "oversized_outer_tile_kernel"' in script
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert "mktemp -d" in script
    assert 'scenario_${scenario}_' in script
    assert 'raw_dir="${profile_dir}/raw"' in script
    assert '--output="${raw_dir}"' in script
    assert "--warm-up" not in script
    assert "--warmup" not in script
    assert "--kernel-name" not in script


def test_phase_aware_tile_docs_keep_factors_independent_and_results_unmeasured():
    readme = read_required(README)
    spec = read_required(SPEC)
    combined = readme + "\n" + spec
    for token in (
        "SCENARIO_NUM=0",
        "SCENARIO_NUM=1",
        "SCENARIO_NUM=2",
        "容量",
        "同步轮数",
        "传输粒度",
        "独立因子",
        "host oracle",
        "Task Duration",
        "raw",
        "88.457001",
    ):
        assert token in combined
    assert "64.139999" in readme
    assert "81.013000" in readme
    assert "有效反例" in readme
