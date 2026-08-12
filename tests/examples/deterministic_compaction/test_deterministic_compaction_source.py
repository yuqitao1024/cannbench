from pathlib import Path
import re


CASE_ROOT = Path(__file__).resolve().parent
SOURCE = CASE_ROOT / "deterministic_compaction.asc"


def _text(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.name}"
    return path.read_text(encoding="utf-8")


def test_standalone_case_layout_and_scenario_switch():
    required = {
        "CMakeLists.txt",
        "deterministic_compaction.asc",
        "README.md",
        "SPEC.md",
        "scripts/run.sh",
    }
    assert required <= {
        str(path.relative_to(CASE_ROOT))
        for path in CASE_ROOT.rglob("*")
        if path.is_file()
    }

    cmake = _text(CASE_ROOT / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0"' in cmake
    assert 'SCENARIO_NUM MATCHES "^[0-1]$"' in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert cmake.count("deterministic_compaction.asc") == 1


def test_single_asc_uses_only_allowed_device_api_boundary_and_c_style_host_resources():
    source = _text(SOURCE)
    assert '#include "acl/acl.h"' in source
    assert '#include "simt_api/asc_simt.h"' in source
    assert "aclrtMalloc" in source
    assert "aclrtFree" in source
    assert "typedef struct" in source
    assert "} DeviceBuffers;" in source

    forbidden = (
        "kernel_operator.h",
        "basic_api/",
        "AscendC::LocalTensor",
        "SetFlag",
        "WaitFlag",
        "PipeBarrier",
        "CrossCoreSetFlag",
        "CrossCoreWaitFlag",
        "std::vector",
        "std::array",
        "std::unique_ptr",
        "std::shared_ptr",
    )
    for token in forbidden:
        assert token not in source
    assert "std::" not in source
    assert "class " not in source
    assert "constexpr" not in source
    assert re.search(r"\bclass\s+[A-Za-z_]", source) is None


def test_host_cleanup_does_not_jump_over_count_initialization():
    source = _text(SOURCE)
    run_body = source[
        source.index("static bool run_compaction") : source.index("int32_t main")
    ]
    first_cleanup_jump = run_body.index("goto cleanup;")
    assert run_body.index("uint32_t expected_count = 0;") < first_cleanup_jump
    assert run_body.index("uint32_t actual_count = 0;") < first_cleanup_jump


def test_both_scenarios_preserve_input_index_order_without_atomic_rank_assignment():
    source = _text(SOURCE)
    assert "stable_atomic_scan_compaction_kernel" in source
    assert "packed_prefix_compaction_kernel" in source
    assert "asc_atomic_add(selected_count, 1U)" in source
    assert "count_selected_before" in source
    assert "asc_ballot" in source
    assert "__popc" in source
    assert "warp_counts" in source
    assert "warp_offsets" in source
    assert "running_offset" in source
    assert "atomic_output_offset" not in source
    assert "atomic_output_rank" not in source


def test_atomic_counter_is_reinitialized_inside_the_kernel_for_profiler_replay():
    source = _text(SOURCE)
    vf = source[
        source.index("__simt_vf__ inline void stable_atomic_scan_compaction_vf") :
        source.index("__global__ __vector__ void stable_atomic_scan_compaction_kernel")
    ]
    assert "selected_count[0] = 0U;" in vf
    assert vf.index("selected_count[0] = 0U;") < vf.index("asc_syncthreads();")
    assert vf.index("asc_syncthreads();") < vf.index("asc_atomic_add(selected_count, 1U)")


def test_fixed_input_host_oracle_checks_count_order_and_tail():
    source = _text(SOURCE)
    for symbol in (
        "fill_fixed_input",
        "build_host_oracle",
        "verify_selected_count",
        "verify_stable_order",
        "verify_untouched_tail",
        "OUTPUT_SENTINEL",
        "Verification PASSED",
    ):
        assert symbol in source


def test_run_script_builds_and_profiles_each_scenario_in_unique_directories():
    script = _text(CASE_ROOT / "scripts/run.sh")
    assert "run_scenario 0" in script
    assert "run_scenario 1" in script
    assert 'cmake -S "${CASE_ROOT}"' in script
    assert '-DSCENARIO_NUM="${scenario}"' in script
    assert "msopprof" in script
    assert "mktemp -d" in script
    assert 'scenario_${scenario}_' in script
    assert "--kernel-name" not in script
    assert 'grep -q "Verification PASSED"' in script


def test_readme_states_fair_aggregation_boundary_and_records_measured_claim_scope():
    readme = _text(CASE_ROOT / "README.md")
    assert "输入 index" in readme
    assert "每次程序调用 1 个相关 kernel" in readme
    assert "所有相关 kernel 的 Task Duration 求和" in readme
    assert "不能只挑最快的一行" in readme
    assert "1620.112061" in readme
    assert "18.013000" in readme
    assert "单次 raw" in readme
    assert "端到端" in readme
