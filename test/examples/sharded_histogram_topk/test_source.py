from pathlib import Path
import re


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "sharded_histogram_topk.asc"
CMAKE = ROOT / "CMakeLists.txt"
RUN = ROOT / "scripts" / "run.sh"
README = ROOT / "README.md"
SPEC = ROOT / "SPEC.md"


def read(path: Path) -> str:
    assert path.is_file(), f"missing required sample file: {path.name}"
    return path.read_text(encoding="utf-8")


def test_standalone_sample_files_and_scenario_switch_exist():
    cmake = read(CMAKE)
    source = read(SOURCE)

    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert 'SCENARIO_NUM=${SCENARIO_NUM}' in cmake
    assert "SCENARIO_NUM must be 0 or 1" in cmake
    assert "static_assert(SCENARIO_NUM == 0 || SCENARIO_NUM == 1" in source
    assert "project(sharded_histogram_topk LANGUAGES ASC CXX)" in cmake


def test_source_keeps_c_style_host_and_allowed_device_api_boundary():
    source = read(SOURCE)

    assert '#include "acl/acl.h"' in source
    assert '#include "simt_api/asc_simt.h"' in source
    assert "aclrtMalloc" in source
    assert "aclrtMemcpy" in source
    assert "aclrtCreateStream" in source
    assert "class " not in source
    assert not re.search(r"\bclass\s+[A-Za-z_]", source)
    assert "std::vector" not in source

    forbidden = (
        "kernel_operator.h",
        "basic_api/",
        "AscendC::LocalTensor",
        "SetFlag",
        "WaitFlag",
        "PipeBarrier",
        "CrossCoreSetFlag",
        "CrossCoreWaitFlag",
    )
    for token in forbidden:
        assert token not in source


def test_case0_is_one_block_and_case1_is_two_stage_sharded_reduction():
    source = read(SOURCE)

    assert "BASELINE_BLOCK_COUNT = 1" in source
    assert "SHARD_COUNT = 64" in source
    assert "baseline_histogram_threshold_kernel<<<BASELINE_BLOCK_COUNT" in source
    assert "shard_histogram_kernel<<<SHARD_COUNT" in source
    assert "reduce_histogram_threshold_kernel<<<REDUCE_BLOCK_COUNT" in source
    assert "partial_histograms" in source
    assert re.search(r"#if\s+SCENARIO_NUM\s*==\s*0", source)
    assert re.search(r"#else.*shard_histogram_kernel<<<", source, re.DOTALL)


def test_fixed_seed_oracle_checks_every_bin_total_threshold_and_tail():
    source = read(SOURCE)

    assert "RANDOM_SEED" in source
    assert "for (uint32_t bin = 0; bin < BIN_COUNT; ++bin)" in source
    assert "histogram mismatch" in source
    assert "total mismatch" in source
    assert "threshold mismatch" in source
    assert "threshold tail mismatch" in source
    assert "Verification PASSED" in source


def test_run_script_profiles_complete_calls_and_sums_case1_stages():
    script = read(RUN)

    assert "run_case 0" in script
    assert "run_case 1" in script
    assert "baseline_histogram_threshold_kernel" in script
    assert "shard_histogram_kernel" in script
    assert "reduce_histogram_threshold_kernel" in script
    assert "aggregate_profile.py" not in script
    assert "stages of one operator call" in script
    assert "stage_sum_us" in script
    assert "repeated samples" not in script
    assert "--warm-up" not in script
    assert 'RUN_ID=${RUN_ID:-$(date +%Y%m%d-%H%M%S)}' in script
    assert 'RUN_ROOT="${PROFILE_ROOT}/${RUN_ID}"' in script
    assert 'rm -rf "${PROFILE_ROOT}"' not in script


def test_docs_define_contract_launch_counts_and_scope_measured_performance():
    readme = read(README)
    spec = read(SPEC)

    for text in (readme, spec):
        assert "256" in text
        assert "SCENARIO_NUM=0" in text
        assert "SCENARIO_NUM=1" in text
        assert "1 次 kernel launch" in text
        assert "2 次 kernel launch" in text
        assert "Ascend 950" in text

    assert "16328.507812" in readme
    assert "292.433994" in readme
    assert "不包含两次 launch" in readme
    assert "stage_sum_us" in readme
    assert "不得" in spec
