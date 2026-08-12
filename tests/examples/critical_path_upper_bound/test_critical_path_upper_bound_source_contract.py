from pathlib import Path
import csv
import json
import re
import subprocess
import sys


CASE_ROOT = Path(__file__).resolve().parent
SOURCE = CASE_ROOT / "critical_path_upper_bound.asc"
README = CASE_ROOT / "README.md"
SPEC = CASE_ROOT / "SPEC.md"
RUN = CASE_ROOT / "scripts" / "run.sh"
ANALYZER = CASE_ROOT / "scripts" / "analyze_profile.py"


def read_required(path: Path) -> str:
    assert path.is_file(), f"missing required file: {path.relative_to(CASE_ROOT)}"
    return path.read_text(encoding="utf-8")


def test_critical_path_upper_bound_has_one_asc_and_required_tracked_files():
    assert sorted(path.name for path in CASE_ROOT.glob("*.asc")) == [
        "critical_path_upper_bound.asc"
    ]
    for relative in (
        "CMakeLists.txt",
        "README.md",
        "SPEC.md",
        "scripts/run.sh",
        "scripts/analyze_profile.py",
    ):
        assert (CASE_ROOT / relative).is_file(), f"missing {relative}"


def test_critical_path_upper_bound_cmake_selects_two_scenarios():
    cmake = read_required(CASE_ROOT / "CMakeLists.txt")
    assert 'set(SCENARIO_NUM "0" CACHE STRING' in cmake
    assert re.search(r'SCENARIO_NUM MATCHES "\^\[0-1\]\$"', cmake)
    assert "find_package(ASC REQUIRED)" in cmake
    assert "project(critical_path_upper_bound LANGUAGES ASC CXX)" in cmake
    assert "SCENARIO_NUM=${SCENARIO_NUM}" in cmake
    assert cmake.count("critical_path_upper_bound.asc") == 1


def test_critical_path_upper_bound_source_is_c_style_and_allowed_api_only():
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


def test_critical_path_upper_bound_uses_two_streams_lane_events_and_join_wall():
    source = read_required(SOURCE)
    assert source.count("aclrtCreateStream") >= 2
    assert source.count("aclrtCreateEvent") >= 4
    for token in (
        "aclrtRecordEvent",
        "aclrtSynchronizeEvent",
        "aclrtEventElapsedTime",
        "clock_gettime",
        "CLOCK_MONOTONIC",
        "lane_a_event_ms",
        "lane_b_event_ms",
        "join_wall_ms",
        "max_lane_ms",
        "additive_lane_sum_ms_forbidden",
    ):
        assert token in source


def test_critical_path_upper_bound_has_fixed_chain_and_equivalent_counterfactual():
    source = read_required(SOURCE)
    for token in (
        "#define LANE_A_STAGE_REPEATS 64U",
        "#define CANDIDATE_BASELINE_REPEATS 32U",
        "#define CANDIDATE_COUNTERFACTUAL_REPEATS 8U",
        "#define DECLARED_IDEAL_SPEEDUP 4.0",
        "#define RETENTION_GATE_PERCENT 5.0",
        "lane_a_stage0_kernel",
        "lane_a_stage1_kernel",
        "lane_a_stage2_kernel",
        "lane_b_candidate_baseline_kernel",
        "lane_b_candidate_counterfactual_kernel",
        "volatile float accumulator",
    ):
        assert token in source
    assert re.search(r"#if SCENARIO_NUM == 0\s+lane_b_candidate_baseline_kernel", source)
    assert re.search(r"#else\s+lane_b_candidate_counterfactual_kernel", source)


def test_critical_path_upper_bound_verifies_complete_outputs_and_counters():
    source = read_required(SOURCE)
    for token in (
        "fill_fixed_input",
        "build_host_oracles",
        "verify_lane_a_output",
        "verify_lane_b_output",
        "verify_output_guards",
        "verify_lane_counters",
        "lane_a_launch_count",
        "lane_b_launch_count",
        "physical_loop_iterations",
        "Verification PASSED",
    ):
        assert token in source


def test_critical_path_upper_bound_run_is_correctness_first_raw_and_default_warmup():
    script = read_required(RUN)
    subprocess.run(["bash", "-n", str(RUN)], check=True)
    assert 'run_scenario 0 "lane_b_candidate_baseline_kernel"' in script
    assert 'run_scenario 1 "lane_b_candidate_counterfactual_kernel"' in script
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert "--launch-count=4" in script
    assert "mktemp -d" in script
    assert 'scenario_${scenario}_' in script
    assert 'raw_dir="${profile_dir}/raw"' in script
    assert '--output="${raw_dir}"' in script
    assert "analyze_profile.py" in script
    assert "profile" in script
    assert "compare" in script
    assert "--warm-up" not in script
    assert "--warmup" not in script
    assert "--kernel-name" not in script


def test_critical_path_analyzer_uses_max_not_sum_and_applies_gate(tmp_path):
    raw0 = tmp_path / "raw0" / "OPPROF_BASELINE"
    raw1 = tmp_path / "raw1" / "OPPROF_COUNTERFACTUAL"
    raw0.mkdir(parents=True)
    raw1.mkdir(parents=True)

    def write_profile(root: Path, lane_b_name: str, lane_b_duration: float) -> None:
        rows = [
            (f"lane_a_stage{stage}_kernel", 10.0)
            for stage in range(3)
        ] + [(lane_b_name, lane_b_duration)]
        for index, (name, duration) in enumerate(rows):
            kernel_dir = root / name / "0"
            kernel_dir.mkdir(parents=True)
            with (kernel_dir / f"OpBasicInfo_{index}.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(handle, fieldnames=["Op Name", "Task Duration(us)"])
                writer.writeheader()
                writer.writerow({"Op Name": name, "Task Duration(us)": str(duration)})

    write_profile(raw0, "lane_b_candidate_baseline_kernel", 40.0)
    write_profile(raw1, "lane_b_candidate_counterfactual_kernel", 10.0)
    log0 = tmp_path / "baseline.log"
    log1 = tmp_path / "counterfactual.log"
    log0.write_text("TIMING lane_a_event_ms=0.030 lane_b_event_ms=0.040 join_wall_ms=0.050\n", encoding="utf-8")
    log1.write_text("TIMING lane_a_event_ms=0.030 lane_b_event_ms=0.010 join_wall_ms=0.041\n", encoding="utf-8")
    analysis0 = tmp_path / "analysis0.json"
    analysis1 = tmp_path / "analysis1.json"
    comparison = tmp_path / "comparison.json"

    for scenario, raw, log, output in (
        (0, raw0.parent, log0, analysis0),
        (1, raw1.parent, log1, analysis1),
    ):
        subprocess.run(
            [
                sys.executable,
                str(ANALYZER),
                "profile",
                "--raw",
                str(raw),
                "--log",
                str(log),
                "--scenario",
                str(scenario),
                "--output",
                str(output),
            ],
            check=True,
        )

    baseline = json.loads(analysis0.read_text(encoding="utf-8"))
    assert baseline["lane_a_sum_us"] == 30.0
    assert baseline["lane_b_sum_us"] == 40.0
    assert baseline["critical_path_us"] == 40.0
    assert baseline["additive_lane_sum_us_forbidden"] == 70.0
    assert baseline["theoretical_upper_bound_percent"] == 25.0
    assert baseline["retention_gate_passed"] is True

    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "compare",
            "--baseline",
            str(analysis0),
            "--counterfactual",
            str(analysis1),
            "--output",
            str(comparison),
        ],
        check=True,
    )
    compared = json.loads(comparison.read_text(encoding="utf-8"))
    assert compared["baseline_critical_path_us"] == 40.0
    assert compared["counterfactual_critical_path_us"] == 30.0
    assert compared["single_run_join_wall_change_percent"] == 18.0


def test_critical_path_upper_bound_docs_forbid_lane_sum_and_record_device_evidence():
    readme = read_required(README)
    spec = read_required(SPEC)
    combined = readme + "\n" + spec
    for token in (
        "SCENARIO_NUM=0",
        "SCENARIO_NUM=1",
        "max(lane_A_sum, lane_B_sum)",
        "不能相加",
        "declared ideal speedup",
        "retention gate",
        "ACL event",
        "synchronized wall interval",
        "完整 join 边界",
        "Task Duration",
        "raw",
        "dav-3510",
        "CANN 9.2.0",
        "默认 warmup 为 5 次",
        "119.976998",
        "24.084000",
        "12.196000",
        "0.00%",
        "编码前拒绝",
        "profiler 扰动证据",
        "scenario_0_fQT61d/raw",
        "scenario_1_MzdT1C/raw",
    ):
        assert token in combined
    assert "未提供实测收益" not in readme
    assert "待实测" not in readme
