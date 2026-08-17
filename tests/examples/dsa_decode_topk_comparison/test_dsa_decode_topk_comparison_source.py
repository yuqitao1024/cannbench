from pathlib import Path
import csv
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


def normalize_whitespace(text: str) -> str:
    return " ".join(text.split())


def function_source(source: str, name: str, next_name: str) -> str:
    start = source.index(name)
    end = source.index(next_name, start)
    return source[start:end]


def test_exact_deliverables_and_two_targets():
    expected = {
        "CMakeLists.txt",
        "README.md",
        "SPEC.md",
        "host_common.h",
        "scripts/parse_profile.py",
        "scripts/run.sh",
        "simt_v2_topk.asc",
        "test_dsa_decode_topk_comparison_source.py",
        "vllm_ascend_topk.asc",
    }
    actual = {
        str(path.relative_to(CASE_DIR))
        for path in CASE_DIR.rglob("*")
        if path.is_file()
        and "__pycache__" not in path.parts
        and "evidence" not in path.parts
    }
    assert actual == expected
    cmake = read_required("CMakeLists.txt")
    assert "project(dsa_decode_topk_comparison LANGUAGES ASC CXX)" in cmake
    assert "find_package(ASC REQUIRED)" in cmake
    assert 'set(CMAKE_ASC_ARCHITECTURES "dav-3510"' in cmake
    assert re.search(r"add_executable\(dsa_decode_topk_vllm_ascend\s+vllm_ascend_topk\.asc\)", cmake)
    assert re.search(r"add_executable\(dsa_decode_topk_simt_v2\s+simt_v2_topk\.asc\)", cmake)


def test_fixed_shape_bf16_contract_and_unique_single_launch_kernels():
    host = read_required("host_common.h")
    vllm = read_required("vllm_ascend_topk.asc")
    simt = read_required("simt_v2_topk.asc")
    combined = host + vllm + simt
    for token in (
        "kRowCount = 4",
        "kContextCount = 32768",
        "kTopK = 2048",
        "bfloat16_t",
        "int32_t",
    ):
        assert token in combined
    assert "dsa_decode_topk_vllm_ascend_kernel" in vllm
    assert "dsa_decode_topk_simt_v2_kernel" in simt
    assert "dsa_decode_topk_simt_v2_kernel" not in vllm
    assert "dsa_decode_topk_vllm_ascend_kernel" not in simt
    assert vllm.count("<<<") == 1
    assert simt.count("<<<") == 1


def test_sources_freeze_exact_upstream_provenance_and_algorithms():
    docs = normalize_whitespace(read_required("SPEC.md") + read_required("README.md"))
    for token in (
        "/root/aiagent/vllm-ascend-upstream/csrc/attention/lightning_indexer/op_kernel/arch35/",
        "bf9f4aae92041f3b300e64646f6261dfe1da53799944dc4ab13a00ead5966fa6",
        "06f83b05f1b879793089cdc9d2f8a865abdace88306f4eae6f4f723c27f60bce",
        "f35a4deea49ac37792e87191af5aa4f9e1d54e5b9702889e339aa10d439e8a7e",
        "6d1ceb35e2e9d7a9f961ab42043ce0f2bce2cd84adba447a130fb6e9e1d2a641",
        "src/cannbench/operators/builtin/lightning_indexer/simt/v2/aten_dsa_lightning_indexer_v2/csrc/simt/lightning_indexer_decode_distributed_topk_bfloat16.asc",
        "c33332a87b88b9c7e0b76fca16a10c7cdd7545ff54cf8c97aafbf6b2604f553e",
        "batch_size == 2 && query_count == 2 && context_shard_count == 16",
        "verbatim upstream Basic API",
    ):
        assert token in docs
    vllm = read_required("vllm_ascend_topk.asc")
    for token in ("16384", "LiTopKVF", "LiTopKGatherVF", "HistogramsHighVFImpl", "HistogramsLowVFImpl"):
        assert token in vllm
    simt = read_required("simt_v2_topk.asc")
    for token in (
        "ordered_bf16_key",
        "kCanonicalContextShardCount = 16",
        "distributed_high_histogram",
        "distributed_select_high",
        "distributed_low_histogram",
        "distributed_select_low_offsets",
        "distributed_compact",
        "AscendC::SyncAll()",
        "asc_atomic_add",
    ):
        assert token in simt
    assert "kRowCount * kCanonicalContextShardCount" in simt
    assert "decode_radix_topk" not in simt
    assert simt.index("} // namespace") < simt.index(
        "__global__ __vector__ void dsa_decode_topk_simt_v2_kernel")


def test_simt_score_consumers_reuse_one_dma_staged_ub_shard():
    simt = read_required("simt_v2_topk.asc")
    high_histogram = function_source(
        simt,
        "lightning_indexer_decode_distributed_high_histogram_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_select_high_bfloat16_v2_vf",
    )
    low_histogram = function_source(
        simt,
        "lightning_indexer_decode_distributed_low_histogram_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_select_low_offsets_bfloat16_v2_vf",
    )
    compact = function_source(
        simt,
        "lightning_indexer_decode_distributed_compact_bfloat16_v2_vf",
        "} // namespace",
    )

    assert '#include "c_api/asc_simd.h"' in simt
    assert simt.count("mutable_gm_ptr(reduced_scores + score_base)") == 1
    assert "kScoreShardUbufBytes" in simt
    assert "kTotalUbufBytes" in simt
    for score_consumer in (high_histogram, low_histogram, compact):
        assert "__ubuf__ const uint16_t* score_shard" in score_consumer
        assert "score_shard[shard_offset]" in score_consumer
        assert "reduced_score_bits[" not in score_consumer
    assert compact.count("score_shard[shard_offset]") == 1


def test_simt_reducers_stage_high_and_low_histograms_in_ub():
    simt = read_required("simt_v2_topk.asc")
    select_high = function_source(
        simt,
        "lightning_indexer_decode_distributed_select_high_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_low_histogram_bfloat16_v2_vf",
    )
    select_low = function_source(
        simt,
        "lightning_indexer_decode_distributed_select_low_offsets_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_compact_bfloat16_v2_vf",
    )
    kernel = simt[simt.index("__global__ __vector__ void dsa_decode_topk_simt_v2_kernel"):]

    assert "kHistogramShardWords" in simt
    assert "kHistogramShardUbufBytes" in simt
    assert simt.count("asc_copy_gm2ub_align(") == 3
    assert "__ubuf__ const uint32_t* high_histogram_shards" in select_high
    assert "__gm__ const uint32_t* high_histogram" not in select_high
    for token in (
        "__ubuf__ const uint32_t* high_histogram_shards",
        "__ubuf__ const uint32_t* low_histogram_shards",
    ):
        assert token in select_low
    assert "__gm__ const uint32_t* high_histogram" not in select_low
    assert "__gm__ const uint32_t* low_histogram" not in select_low
    assert "if (block_index < row_count)" in kernel
    assert "high_histogram_shards" in kernel
    assert "low_histogram_shards" in kernel


def test_simt_high_threshold_scan_uses_parallel_bin_groups():
    simt = read_required("simt_v2_topk.asc")
    select_high = function_source(
        simt,
        "lightning_indexer_decode_distributed_select_high_bfloat16_v2_vf",
        "lightning_indexer_decode_distributed_low_histogram_bfloat16_v2_vf",
    )

    assert "if (bucket == 0)" not in select_high
    assert "threshold_group_counts" in select_high
    assert "bucket < kThresholdGroupCount" in select_high
    assert "bucket * kThresholdBucketsPerGroup" in select_high
    assert "higher_group = bucket + 1" in select_high
    assert "group_bucket_begin + kThresholdBucketsPerGroup - 1" in select_high


def test_simt_compact_uses_warp_atomic_reservations_without_block_scan():
    simt = read_required("simt_v2_topk.asc")
    reservation = function_source(
        simt,
        "reserve_warp_output",
        "lightning_indexer_decode_distributed_compact_bfloat16_v2_vf",
    )
    compact = function_source(
        simt,
        "lightning_indexer_decode_distributed_compact_bfloat16_v2_vf",
        "} // namespace",
    )
    warp_compact = reservation + compact
    normalized_reservation = normalize_whitespace(reservation)
    normalized_compact = normalize_whitespace(compact)

    assert '#include "simt_api/device_warp_functions.h"' in simt
    for token in (
        "kCompactGreater",
        "kCompactEqual",
        "asc_ballot",
        "__popc",
        "lanemask_lt",
        "asc_atomic_add",
        "asc_shfl",
    ):
        assert token in warp_compact
    assert reservation.count("asc_ballot(") == 1
    assert reservation.count("asc_atomic_add(") == 1
    assert reservation.count("asc_shfl(") == 1
    ordered_reservation_tokens = (
        "const uint32_t selected_mask = asc_ballot(selected ? 1 : 0)",
        "const uint32_t warp_selected_count = static_cast<uint32_t>(__popc(selected_mask))",
        "if (laneid() == 0 && warp_selected_count != 0)",
        "warp_start = asc_atomic_add(counter, warp_selected_count)",
        "warp_start = asc_shfl(warp_start, 0, kWarpSize)",
        "__popc(selected_mask & static_cast<uint32_t>(lanemask_lt()))",
        "return warp_start + lane_offset",
    )
    positions = [
        normalized_reservation.index(token)
        for token in ordered_reservation_tokens
    ]
    assert positions == sorted(positions)

    assert normalized_compact.count("reserve_warp_output(") == 2
    ordered_compact_tokens = (
        "compact_counters[thread_index] = 0",
        "asc_syncthreads()",
        "for (int32_t shard_offset = thread_index",
        "const uint32_t warp_greater_rank = reserve_warp_output( greater, compact_counters + kCompactGreater)",
        "const uint32_t warp_equal_rank = reserve_warp_output( equal, compact_counters + kCompactEqual)",
        "if (greater)",
        "else if (equal)",
    )
    positions = [normalized_compact.index(token) for token in ordered_compact_tokens]
    assert positions == sorted(positions)

    equal_branch = normalized_compact[normalized_compact.index("else if (equal)"):]
    ordered_equal_tokens = (
        "const uint32_t equal_rank = shard_equal_offset + warp_equal_rank",
        "if (equal_rank < equal_count_needed)",
        "const uint32_t output_slot = total_greater_count + equal_rank",
        "output[row_offset + output_slot] = context_index",
    )
    positions = [equal_branch.index(token) for token in ordered_equal_tokens]
    assert positions == sorted(positions)
    assert compact.count("asc_syncthreads()") == 1
    for obsolete in (
        "scan_a",
        "scan_b",
        "scan_source",
        "scan_destination",
        "packed_inclusive_counts",
        "offset <<= 1",
    ):
        assert obsolete not in compact


def test_host_is_acl_runtime_only_and_uses_score_set_oracle():
    host = read_required("host_common.h")
    all_source = host + read_required("vllm_ascend_topk.asc") + read_required("simt_v2_topk.asc")
    for token in (
        '#include "acl/acl.h"',
        "aclInit",
        "aclrtSetDevice",
        "aclrtCreateStream",
        "aclrtMalloc",
        "aclrtMemcpy",
        "aclrtSynchronizeStream",
        "fill_deterministic_bf16_scores",
        "verify_score_set",
        "index < kContextCount",
        "duplicate index",
        "score > threshold",
        "threshold_equal_selected",
        "expected_threshold_equal",
        "Verification PASSED",
    ):
        assert token in host
    for forbidden in ("aclrtEvent", "aclrtRecordEvent", "torch/", "torch_npu", "at::Tensor", "pytorch"):
        assert forbidden not in all_source
    docs = normalize_whitespace(read_required("SPEC.md") + read_required("README.md"))
    assert "ordering is intentionally not compared" in docs
    assert "equal-score index identity is intentionally not compared" in docs


def test_run_script_correctness_first_five_default_profiles_and_raw_retention():
    script = read_required("scripts/run.sh")
    subprocess.run(["bash", "-n", str(RUN_PATH)], check=True)
    loop = "for implementation in vllm_ascend simt_v2; do"
    correctness_loop = script.index(loop)
    profiling_loop = script.index(loop, correctness_loop + len(loop))
    assert "Verification PASSED" in script[correctness_loop:profiling_loop]
    assert "msopprof" not in script[correctness_loop:profiling_loop]
    assert profiling_loop < script.index("msopprof")
    assert "for sample in 1 2 3 4 5" in script
    assert '--aic-metrics="Default"' in script
    assert "--launch-count=1" in script
    assert "expected-launches 1" in script
    assert 'vllm_ascend) expected_block_dim=4' in script
    assert 'simt_v2) expected_block_dim=64' in script
    assert '--expected-block-dim "${expected_block_dim}"' in script
    assert "frequency" in script.lower()
    assert "frequency parity" in script.lower()
    assert 'reference_frequency = summary["vllm_ascend"]["frequency_mhz"]' in script
    assert 'summary["simt_v2"]["frequency_mhz"] != reference_frequency' in script
    assert "profiles/${implementation}/sample_${sample}/raw" in script
    assert "parse_profile.py" in script
    assert "rm -rf" not in script


def test_profile_parser_accepts_one_exact_row_and_rejects_missing_or_extra(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    profile = raw / "OpBasicInfo_0.csv"
    fields = [
        "Op Name",
        "Task Duration(us)",
        "Block Dim",
        "Execution Time Current Frequency(MHz)",
        "Execution Time Rated Frequency(MHz)",
    ]

    def write_rows(count: int, block_dim: str = "64") -> None:
        with profile.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for index in range(count):
                writer.writerow({
                    "Op Name": "dsa_decode_topk_simt_v2_kernel",
                    "Task Duration(us)": str(10.5 + index),
                    "Block Dim": block_dim,
                    "Execution Time Current Frequency(MHz)": "1800",
                    "Execution Time Rated Frequency(MHz)": "1800",
                })

    output = tmp_path / "sample.json"
    command = [
        sys.executable,
        str(PARSER_PATH),
        "--raw",
        str(raw),
        "--kernel",
        "dsa_decode_topk_simt_v2_kernel",
        "--expected-launches",
        "1",
        "--expected-block-dim",
        "64",
        "--output",
        str(output),
    ]
    write_rows(0)
    missing = subprocess.run(command, text=True, capture_output=True)
    assert missing.returncode != 0
    assert "expected 1 exact target row, observed 0" in missing.stderr

    write_rows(2)
    extra = subprocess.run(command, text=True, capture_output=True)
    assert extra.returncode != 0
    assert "expected 1 exact target row, observed 2" in extra.stderr

    write_rows(1)
    subprocess.run(command, check=True)
    parsed = output.read_text(encoding="utf-8")
    assert '"task_duration_us": 10.5' in parsed
    assert '"frequency_mhz": 1800.0' in parsed
    assert '"block_dim": 64' in parsed

    write_rows(1, block_dim="4")
    wrong_block_dim = subprocess.run(command, text=True, capture_output=True)
    assert wrong_block_dim.returncode != 0
    assert "expected block dimension 64, observed 4" in wrong_block_dim.stderr

    with profile.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields[:-1])
        writer.writeheader()
        writer.writerow({
            "Op Name": "dsa_decode_topk_simt_v2_kernel",
            "Task Duration(us)": "10.5",
            "Block Dim": "64",
            "Execution Time Current Frequency(MHz)": "1800",
        })
    missing_rated = subprocess.run(command, text=True, capture_output=True)
    assert missing_rated.returncode != 0
    assert "required rated frequency column missing" in missing_rated.stderr


def test_docs_freeze_measurement_and_evidence_contract():
    combined = normalize_whitespace(read_required("SPEC.md") + read_required("README.md"))
    for token in (
        "BF16 scores [4, 32768]",
        "INT32 indices [4, 2048]",
        "one kernel launch",
        "Ascend 950",
        "dav-3510",
        "msopprof",
        "Task Duration",
        "Default",
        "five independent",
        "frequency parity",
        "raw",
        "median",
        "min",
        "max",
        "ratio",
        "64-block distributed kernel",
        "7.807",
        "27.882",
        "3.5714x",
        "1650/1650 MHz",
        "actual-distributed-20260813-200615",
        "28.629",
        "28.742001",
        "18.8459995",
        "8.0840%",
        "10/10",
        "18.874001",
        "0.7204%",
        "1/5",
        "893b6406afc1a6384ab6fae8a2247d03cc230d87",
        "dsa-topk-warp-atomic-evidence-20260817.tar.gz",
        "22e4ce773d40559262426f3236cddc51b811adb7344f5226666c44b09f83d3e0",
    ):
        assert token in combined
    assert "TODO" not in read_required("README.md")
    assert "TBD" not in read_required("README.md")
    assert "8.3474x" not in read_required("README.md")
    assert (
        "flock -x /tmp/cannbench-dsa-stage-comparison.lock "
        "bash -lc 'RESULT_ROOT=\"$PWD/evidence/<unique-run>\" bash scripts/run.sh'"
    ) in read_required("README.md")
