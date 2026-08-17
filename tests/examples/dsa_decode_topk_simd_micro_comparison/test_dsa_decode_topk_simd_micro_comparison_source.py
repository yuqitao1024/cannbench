from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess
import sys


CASE_DIR = Path(__file__).resolve().parent
RUN_PATH = CASE_DIR / "scripts" / "run.sh"
PARSER_PATH = CASE_DIR / "scripts" / "parse_profile.py"


def read_required(relative: str) -> str:
    path = CASE_DIR / relative
    assert path.is_file(), f"missing required file: {relative}"
    return path.read_text(encoding="utf-8")


def test_exact_deliverables_and_two_targets():
    expected = {
        "CMakeLists.txt",
        "README.md",
        "SPEC.md",
        "host_common.h",
        "scripts/parse_profile.py",
        "scripts/run.sh",
        "simd_micro_topk.asc",
        "simt_v2_topk.asc",
        "test_dsa_decode_topk_simd_micro_comparison_source.py",
    }
    actual = {
        str(path.relative_to(CASE_DIR))
        for path in CASE_DIR.rglob("*")
        if path.is_file() and "__pycache__" not in path.parts
    }
    assert actual == expected

    cmake = read_required("CMakeLists.txt")
    assert "project(dsa_decode_topk_simd_micro_comparison LANGUAGES ASC CXX)" in cmake
    assert 'set(CMAKE_ASC_ARCHITECTURES "dav-3510"' in cmake
    assert re.search(
        r"add_executable\(dsa_decode_topk_simt_v2_baseline\s+simt_v2_topk\.asc\)",
        cmake,
    )
    assert re.search(
        r"add_executable\(dsa_decode_topk_simd_micro\s+simd_micro_topk\.asc\)",
        cmake,
    )


def test_fixed_shape_launch_layout_and_frozen_baseline():
    host = read_required("host_common.h")
    baseline_path = CASE_DIR / "simt_v2_topk.asc"
    baseline = read_required("simt_v2_topk.asc")
    candidate = read_required("simd_micro_topk.asc")

    for token in (
        "kRowCount = 4",
        "kContextCount = 32768",
        "kTopK = 2048",
        "bfloat16_t",
        "int32_t",
    ):
        assert token in host + baseline + candidate

    digest = hashlib.sha256(baseline_path.read_bytes()).hexdigest()
    assert digest == "d0f5821eeb739d00c2ab783c0bc19a401a1500e3b5918746b0c8d07ded13a0ae"
    for source in (baseline, candidate):
        assert "kTotalUbufBytes" in source
        assert "kRowCount * kCanonicalContextShardCount" in source
        assert source.count("AscendC::SyncAll()") == 4
        assert source.count("<<<") == 1
    assert "kTotalUbufBytes =" in candidate
    assert "45056" in candidate
    assert "constexpr int32_t kStateWordsPerRow" not in candidate
    assert "constexpr int32_t kOffsetWordsPerShard" not in candidate


def test_candidate_uses_five_simd_micro_stages_without_simt_or_vllm_algorithm():
    candidate = read_required("simd_micro_topk.asc")

    assert candidate.count("__simd_vf__") == 5
    for token in (
        "distributed_high_histogram_simd_vf",
        "distributed_select_high_simd_vf",
        "distributed_low_histogram_simd_vf",
        "distributed_select_low_offsets_simd_vf",
        "distributed_compact_simd_vf",
        "MicroAPI::Histograms",
        "MicroAPI::Compare",
        "MicroAPI::Squeeze",
        "MicroAPI::LoadAlign",
        "MicroAPI::StoreAlign",
    ):
        assert token in candidate
    for forbidden in ("__simt_vf__", "asc_vf_call", "LiTopKVF", "16384"):
        assert forbidden not in candidate
    assert "__simd_callee__" not in candidate


def test_histogram_stages_emit_frequencies_for_scalar_suffix_scans():
    candidate = read_required("simd_micro_topk.asc")

    assert candidate.count("MicroAPI::HistogramsType::FREQUENCY") == 4
    assert "MicroAPI::HistogramsType::ACCUMULATE" not in candidate


def test_reducer_blocks_address_state_by_reducer_row():
    candidate = read_required("simd_micro_topk.asc")

    assert (
        "static_cast<int64_t>(block_index) * kStateWordsPerRow"
        in candidate
    )
    assert candidate.count("state[reducer_state_base +") == 2
    assert candidate.count("state + reducer_state_base +") == 4
    assert candidate.count("state[state_base +") == 3
    assert candidate.count("asc_store_dev(") == 6


def test_mixed_pipeline_dependencies_cover_each_ub_handoff():
    candidate = read_required("simd_micro_topk.asc")

    assert candidate.count("asc_sync_notify(PIPE_V, PIPE_S, EVENT_ID0)") == 5
    assert candidate.count("asc_sync_wait(PIPE_V, PIPE_S, EVENT_ID0)") == 5
    assert candidate.count("asc_sync_notify(PIPE_S, PIPE_MTE3, EVENT_ID0)") == 3
    assert candidate.count("asc_sync_wait(PIPE_S, PIPE_MTE3, EVENT_ID0)") == 3


def test_candidate_reuses_one_dma_staged_score_shard():
    candidate = read_required("simd_micro_topk.asc")

    assert candidate.count("mutable_gm_ptr(reduced_scores + score_base)") == 1
    assert candidate.count("asc_copy_gm2ub_align(") == 3
    assert "kScoreShardUbufBytes" in candidate
    for function_name in (
        "distributed_high_histogram_simd_vf",
        "distributed_low_histogram_simd_vf",
        "distributed_compact_simd_vf",
    ):
        start = candidate.index(function_name)
        signature = candidate[start : candidate.index(")", start)]
        assert "__ubuf__ uint16_t* score_shard" in signature
    assert "score_shard[" not in candidate


PROFILE_FIELDS = [
    "Op Name",
    "Task Duration(us)",
    "Block Dim",
    "Execution Time Current Frequency(MHz)",
    "Execution Time Rated Frequency(MHz)",
]


def write_profile_csv(
    path: Path,
    rows: list[dict[str, str]],
    fields: list[str] | None = None,
) -> None:
    path.parent.mkdir(parents=True)
    fields = fields or PROFILE_FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def run_parser(
    tmp_path: Path,
    rows: list[dict[str, str]],
    kernel: str,
    fields: list[str] | None = None,
    expected_frequency: str | None = None,
):
    raw = tmp_path / "raw"
    output = tmp_path / "parsed.json"
    write_profile_csv(raw / "OpBasicInfo_test.csv", rows, fields)
    command = [
        sys.executable,
        str(PARSER_PATH),
        "--raw",
        str(raw),
        "--kernel",
        kernel,
        "--expected-launches",
        "1",
        "--expected-block-dim",
        "64",
    ]
    if expected_frequency is not None:
        command.extend(["--expected-frequency", expected_frequency])
    command.extend(["--output", str(output)])
    result = subprocess.run(
        command,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, output


def test_profile_parser_accepts_one_exact_64_block_full_frequency_row(tmp_path):
    kernel = "dsa_decode_topk_simd_micro_kernel"
    rows = [
        {
            "Op Name": kernel,
            "Task Duration(us)": "12.5",
            "Block Dim": "64",
            "Execution Time Current Frequency(MHz)": "1650",
            "Execution Time Rated Frequency(MHz)": "1650",
        }
    ]
    result, output = run_parser(tmp_path, rows, kernel)
    assert result.returncode == 0, result.stderr
    parsed = json.loads(output.read_text(encoding="utf-8"))
    assert parsed["op_name"] == kernel
    assert parsed["task_duration_us"] == 12.5
    assert parsed["block_dim"] == 64
    assert parsed["frequency_mhz"] == 1650.0
    assert parsed["rated_frequency_mhz"] == 1650.0


def test_profile_parser_rejects_duplicate_target_rows(tmp_path):
    kernel = "dsa_decode_topk_simt_v2_kernel"
    row = {
        "Op Name": kernel,
        "Task Duration(us)": "14.5",
        "Block Dim": "64",
        "Execution Time Current Frequency(MHz)": "1650",
        "Execution Time Rated Frequency(MHz)": "1650",
    }
    result, output = run_parser(tmp_path, [row, row], kernel)
    assert result.returncode != 0
    assert not output.exists()
    assert "expected 1 exact target row, observed 2" in result.stderr


def test_profile_parser_rejects_exact_name_distractor_and_zero_match(tmp_path):
    result, output = run_parser(
        tmp_path,
        [
            {
                "Op Name": "dsa_decode_topk_simd_micro_kernel_suffix",
                "Task Duration(us)": "14.5",
                "Block Dim": "64",
                "Execution Time Current Frequency(MHz)": "1650",
                "Execution Time Rated Frequency(MHz)": "1650",
            }
        ],
        "dsa_decode_topk_simd_micro_kernel",
    )
    assert result.returncode != 0
    assert not output.exists()
    assert "expected 1 exact target row, observed 0" in result.stderr


def test_profile_parser_rejects_wrong_block_dimension(tmp_path):
    result, output = run_parser(
        tmp_path,
        [
            {
                "Op Name": "dsa_decode_topk_simd_micro_kernel",
                "Task Duration(us)": "14.5",
                "Block Dim": "32",
                "Execution Time Current Frequency(MHz)": "1650",
                "Execution Time Rated Frequency(MHz)": "1650",
            }
        ],
        "dsa_decode_topk_simd_micro_kernel",
    )
    assert result.returncode != 0
    assert not output.exists()
    assert "expected block dimension 64, observed 32" in result.stderr


def test_profile_parser_rejects_missing_frequency_columns(tmp_path):
    row = {
        "Op Name": "dsa_decode_topk_simd_micro_kernel",
        "Task Duration(us)": "14.5",
        "Block Dim": "64",
        "Execution Time Current Frequency(MHz)": "1650",
        "Execution Time Rated Frequency(MHz)": "1650",
    }
    for index, (missing, message) in enumerate((
        ("Execution Time Current Frequency(MHz)", "current frequency"),
        ("Execution Time Rated Frequency(MHz)", "rated frequency"),
    )):
        fields = [field for field in PROFILE_FIELDS if field != missing]
        result, output = run_parser(
            tmp_path / f"missing-{index}",
            [row],
            "dsa_decode_topk_simd_micro_kernel",
            fields=fields,
        )
        assert result.returncode != 0
        assert not output.exists()
        assert f"required {message} column missing" in result.stderr


def test_profile_parser_rejects_nonfinite_and_nonpositive_values(tmp_path):
    for field, value in (
        ("Task Duration(us)", "nan"),
        ("Task Duration(us)", "0"),
        ("Execution Time Current Frequency(MHz)", "inf"),
        ("Execution Time Rated Frequency(MHz)", "-1"),
    ):
        row = {
            "Op Name": "dsa_decode_topk_simd_micro_kernel",
            "Task Duration(us)": "14.5",
            "Block Dim": "64",
            "Execution Time Current Frequency(MHz)": "1650",
            "Execution Time Rated Frequency(MHz)": "1650",
        }
        row[field] = value
        result, output = run_parser(tmp_path / f"{field}-{value}", [row], row["Op Name"])
        assert result.returncode != 0
        assert not output.exists()
        assert "invalid numeric target row" in result.stderr


def test_profile_parser_rejects_unexpected_frequency_per_sample(tmp_path):
    result, output = run_parser(
        tmp_path,
        [
            {
                "Op Name": "dsa_decode_topk_simd_micro_kernel",
                "Task Duration(us)": "14.5",
                "Block Dim": "64",
                "Execution Time Current Frequency(MHz)": "1600",
                "Execution Time Rated Frequency(MHz)": "1650",
            }
        ],
        "dsa_decode_topk_simd_micro_kernel",
        expected_frequency="1650",
    )
    assert result.returncode != 0
    assert not output.exists()
    assert "expected frequency 1650.0/1650.0 MHz" in result.stderr


def test_runner_freezes_correctness_order_and_profiler_contract():
    runner = read_required("scripts/run.sh")

    assert "for round in $(seq 1 10)" in runner
    assert 'if (( round % 2 == 1 ))' in runner
    assert 'order=(simt_v2_baseline simd_micro)' in runner
    assert 'order=(simd_micro simt_v2_baseline)' in runner
    assert "Verification PASSED" in runner
    assert "--aic-metrics=Default --launch-count=1" in runner
    assert "--kernel-name" not in runner
    assert "warmup" not in runner.lower()
    assert "frequency parity rejected" in runner
    assert "--expected-frequency 1650" in runner


def test_runner_rejects_frequency_immediately_after_each_parser_invocation():
    runner = read_required("scripts/run.sh")
    parser_call = runner.index('python3 "${SCRIPT_DIR}/parse_profile.py"')
    next_loop_end = runner.index("    done\n", parser_call)

    assert runner.index("--expected-frequency 1650", parser_call) < next_loop_end


def test_runner_summary_python_is_syntactically_valid():
    runner = read_required("scripts/run.sh")
    summary = runner.split("<<'PY'\n", 1)[1].rsplit("\nPY", 1)[0]

    compile(summary, str(RUN_PATH), "exec")


def test_runner_publishes_sibling_archive_and_checksum():
    runner = read_required("scripts/run.sh")

    assert 'ARCHIVE_PATH="${RESULT_ROOT}.tar.gz"' in runner
    assert 'CHECKSUM_PATH="${ARCHIVE_PATH}.sha256"' in runner
    assert 'tar -czf "${ARCHIVE_PATH}"' in runner
    assert 'sha256sum "${ARCHIVE_PATH}" > "${CHECKSUM_PATH}"' in runner


def test_runner_configures_cmake_with_effective_build_directory():
    runner = read_required("scripts/run.sh")

    assert "cmake -S \"${EXAMPLE_DIR}\" -B \"${BUILD_DIR}\"" in runner
