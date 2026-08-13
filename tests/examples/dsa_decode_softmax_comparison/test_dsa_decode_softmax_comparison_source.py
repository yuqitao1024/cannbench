from pathlib import Path
import csv
import re
import subprocess
import sys


CASE_DIR = Path(__file__).resolve().parent
PARSER = CASE_DIR / "scripts" / "parse_profile.py"


def required(relative: str) -> str:
    path = CASE_DIR / relative
    assert path.is_file(), f"missing required file: {relative}"
    return path.read_text(encoding="utf-8")


def test_exact_example_files_and_build_targets_exist():
    assert sorted(path.name for path in CASE_DIR.glob("*.asc")) == [
        "simt_v2_softmax.asc",
        "vllm_ascend_softmax.asc",
    ]
    for relative in (
        "SPEC.md", "README.md", "CMakeLists.txt", "host_common.h",
        "scripts/run.sh", "scripts/parse_profile.py",
    ):
        required(relative)
    cmake = required("CMakeLists.txt")
    assert "dsa_decode_softmax_vllm_ascend" in cmake
    assert "dsa_decode_softmax_simt_v2" in cmake
    assert "dav-3510" in cmake


def test_fixed_shape_scale_layout_and_separate_kernel_symbols():
    common = required("host_common.h")
    sources = required("vllm_ascend_softmax.asc") + required("simt_v2_softmax.asc")
    for token in (
        "SOFTMAX_ROWS 512", "SELECTED_TOKENS 2048", "INDICES_ROWS 4",
        "SOFTMAX_SCALE (1.0F / 24.0F)", "TILE_TOKENS 128",
        "SCORE_MIN (-24.0F)", "SCORE_MAX 24.0F",
        "bfloat16_t", "running_max", "running_sum", "old_scale",
    ):
        assert token in common + sources
    assert "dsa_decode_softmax_vllm_ascend_kernel" in sources
    assert "dsa_decode_softmax_simt_v2_kernel" in sources
    assert "final_normalization" not in sources.lower()


def test_provenance_and_original_online_state_markers_are_retained():
    vllm = required("vllm_ascend_softmax.asc")
    simt = required("simt_v2_softmax.asc")
    for token in (
        "csrc/attention/sparse_flash_attention/op_kernel/arch35/sparse_flash_attention_service_vector_mla.h",
        "332679ebb84a571582d7f3fd3f5cd086415ff24188b3c4ef0370187ec3de02b3",
        "csrc/attention/common/op_kernel/arch35/vf/vf_mul_sel_softmaxflashv2_cast_nz_sfa.h",
        "8ddafeda9671ac77f9fe1bc86bad512c1a821bd4229b19a825a43525e6e63d10",
        "vf_basic_block_aligned128_no_update_sfa.h",
        "7ec0f45a92cbfde57e4a50f859eb14372fe3ab95d80eed7c4b5f3a6473aea66b",
        "vf_basic_block_aligned128_update_sfa.h",
        "cf4b5c9b941b8c7634380a9524cdea52934694860bf950ca5eeb2fc36fcbf8cf",
        "ProcessVec1NoUpdate", "ProcessVec1Update", "SFAUpdateExpSumAndExpMax",
        "kernel_operator.h", "LocalTensor", "MicroAPI::RegTensor",
        "ProcessVec1NoUpdateImpl128VF", "ProcessVec1UpdateImpl128VF",
        "MicroAPI::Muls", "MicroAPI::Reduce", "MicroAPI::ExpSub",
        "MicroAPI::LoadDist::DIST_DINTLV_B32",
    ):
        assert token in vllm
    assert "vllm_warp_max" not in vllm
    assert "vllm_warp_sum" not in vllm
    assert "asc_vf_call<apply_validity_and_causal_mask_vf>" not in vllm
    assert "indicesUb" not in vllm
    assert "causal_limit" not in vllm
    assert "MicroAPI::Compare" not in vllm
    assert "MicroAPI::Select" not in vllm
    assert "MicroAPI::" not in vllm.split("__simd_vf__", 1)[0]
    for token in (
        "sparse_attention_head64_fused_hd576.asc",
        "5817bea44cfb8c86837b8f5cbf561725cf6b37f5a9a05f4784cd02690b2ea4eb",
        "head64_fused_vllm_softmax_vf", "lane_valid_mask", "old_scale",
        "__ubuf__ const float* scores", "selected_start",
        "current_selected", "tile_valid",
    ):
        assert token in simt


def test_simt_preserves_the_production_single_tile_vf_call_boundary():
    simt = required("simt_v2_softmax.asc")
    vf = simt.split("head64_fused_vllm_softmax_vf(", 1)[1].split(
        "__global__ __vector__ void", 1
    )[0]
    kernel = simt.split("dsa_decode_softmax_simt_v2_kernel(", 1)[1].split(
        "static void launch_stage", 1
    )[0]

    assert "for (uint32_t tile" not in vf
    assert "blockIdx" not in vf
    assert "__gm__ const float* scores" not in vf
    assert "__gm__ bfloat16_t* probabilities" not in vf
    assert "for (uint32_t tile = 0U; tile < TILE_COUNT; ++tile)" in kernel
    assert kernel.count("asc_vf_call<head64_fused_vllm_softmax_vf>") == 1
    assert "asc_vf_call<head64_fused_softmax_init_vf>" in kernel
    assert "tile * TILE_TOKENS" in kernel


def test_simt_preserves_production_dynamic_ub_layout_and_launch_size():
    simt = required("simt_v2_softmax.asc")
    kernel = simt.split("dsa_decode_softmax_simt_v2_kernel(", 1)[1].split(
        "static void launch_stage", 1
    )[0]
    launch = simt.split("static void launch_stage", 1)[1]

    for token in (
        "kHead64FusedVllmUbRunningMaxOffset = 0",
        "kHead64FusedVllmUbRunningSumOffset = 128",
        "kHead64FusedVllmUbOldScaleOffset = 256",
        "kHead64FusedVllmUbScoresOffset = 103040",
        "kHead64FusedVllmUbProbabilitiesOffset = 119424",
        "kHead64FusedDynamicUbBytes = 211584",
        "ub_pipe.InitBuffer(ub_buffer, kHead64FusedDynamicUbBytes)",
        "ub_workspace + kHead64FusedVllmUbScoresOffset",
        "ub_workspace + kHead64FusedVllmUbProbabilitiesOffset",
    ):
        assert token in simt
    assert "TBuf<TPosition::VECCALC> scores_buf" not in kernel
    assert "<<<BLOCK_DIM, kHead64FusedDynamicUbBytes, stream>>>" in launch


def test_fixed_shape_launches_the_production_number_of_softmax_row_owners():
    common = required("host_common.h")
    docs = required("SPEC.md") + required("README.md")
    assert "SOFTMAX_ROWS 512" in common
    assert "ROWS_PER_BLOCK 32" in common
    assert "BLOCK_DIM (SOFTMAX_ROWS / ROWS_PER_BLOCK)" in common
    assert "16 active Softmax row owners" in docs


def test_acl_runtime_only_host_and_fixed_oracle_tolerances():
    common = required("host_common.h")
    assert '#include "acl/acl.h"' in common
    for token in (
        "aclInit", "aclrtMalloc", "aclrtMemcpy", "aclrtSynchronizeStream",
        "fill_deterministic_scores", "build_online_softmax_oracle",
        "verify_online_softmax", "Verification PASSED",
        "MAX_ATOL 2.0e-5", "MAX_RTOL 2.0e-5",
        "SUM_ATOL 2.0e-3", "SUM_RTOL 2.0e-3",
        "NUMERATOR_ATOL 8.0e-3", "NUMERATOR_RTOL 8.0e-3",
        "PROBABILITY_ATOL 2.0e-4", "PROBABILITY_RTOL 2.0e-3",
        "ROW_SUM_ATOL 2.0e-3", "isfinite",
    ):
        assert token in common
    combined = common + required("CMakeLists.txt") + required("scripts/run.sh")
    assert "bfloat16_t* numerators" in common
    assert "reinterpret_cast<__gm__ bfloat16_t*>" not in (
        required("vllm_ascend_softmax.asc") + required("simt_v2_softmax.asc")
    )
    for forbidden in ("torch_npu", "PyTorch", "libtorch", "aclrtEvent", "ACL Event"):
        assert forbidden not in combined


def test_old_scale_contract_preserves_each_production_write_boundary():
    common = required("host_common.h")

    assert "float* old_scale_sentinel" in common
    assert "old_scale_sentinel[index] = NAN" in common
    assert "buffers.old_scale, scale_count * sizeof(float), old_scale_sentinel" in common
    assert "expected.old_scale,\n            scale_count * sizeof(float), ACL_MEMCPY_HOST_TO_DEVICE" not in common
    assert "strcmp(implementation, \"vllm_ascend\") == 0" in common
    assert "tile == 0U && vllm_ascend" in common
    assert "isnan(actual_scale)" in common


def test_vllm_stage_copy_separates_template_width_from_active_rows():
    common = required("host_common.h")
    vllm = required("vllm_ascend_softmax.asc")
    simt = required("simt_v2_softmax.asc")

    for token in (
        "S1_TEMPLATE_ROWS 64",
        "ROWS_PER_BLOCK 32",
        "VEC1_SRC_STRIDE ((S1_TEMPLATE_ROWS >> 1) + 1)",
        "STAGE_TILE_ELEMENTS (VEC1_SRC_STRIDE * TILE_TOKENS)",
        "OUTPUT_TILE_ELEMENTS (S1_TEMPLATE_ROWS * TILE_TOKENS)",
    ):
        assert token in common
    assert "InitBuffer(probabilities_buf, STAGE_TILE_ELEMENTS" in vllm
    assert vllm.count("const uint32_t blockStride = VEC1_SRC_STRIDE;") == 2
    assert "DataCopyParams stage_copy" in vllm
    assert "TILE_TOKENS / 16U" in vllm
    assert "VEC1_SRC_STRIDE - ROWS_PER_BLOCK" in vllm
    assert "S1_TEMPLATE_ROWS - ROWS_PER_BLOCK" in vllm
    assert "DataCopy(numerators_gm[output_offset], probability_ub, stage_copy)" in vllm
    assert "ROWS_PER_BLOCK * TILE_TOKENS);" not in vllm[vllm.index("DataCopy(numerators_gm"):]
    assert "block / 2U" in common + vllm + simt
    assert "block % 2U" in common + vllm + simt


def test_runner_is_correctness_first_five_sample_msopprof_only_and_retains_raw():
    script = required("scripts/run.sh")
    subprocess.run(["bash", "-n", str(CASE_DIR / "scripts" / "run.sh")], check=True)
    implementation_loops = [
        match.start()
        for match in re.finditer("for implementation in vllm_ascend simt_v2; do", script)
    ]
    assert len(implementation_loops) == 2
    assert script.index("Verification PASSED") < script.index("msopprof")
    assert implementation_loops[1] < script.index("msopprof")
    assert "for sample in 1 2 3 4 5" in script
    assert "--aic-metrics=Default" in script
    assert "--launch-count=1" in script
    assert "Task Duration" in script
    assert "raw/sample_${sample}" in script
    assert "rm -rf" not in script
    assert "warmup" not in script.lower()
    assert "frequency" in script.lower()
    assert 'rated_frequencies.add(parsed[0]["rated_frequency_mhz"])' in script


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Op Name", "Task Duration(us)", "Block Dim", "Current Freq",
                "Rated Freq",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_parser_requires_exactly_one_target_row_and_structured_fields(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "sample.csv"
    command = [
        sys.executable, str(PARSER), "--raw", str(raw), "--kernel",
        "dsa_decode_softmax_simt_v2_kernel", "--sample", "3",
        "--expected-block-dim", "16", "--output", str(output),
    ]
    row = {
        "Op Name": "dsa_decode_softmax_simt_v2_kernel",
        "Task Duration(us)": "42.125",
        "Block Dim": "16",
        "Current Freq": "1650",
        "Rated Freq": "1650",
    }

    missing = subprocess.run(command, text=True, capture_output=True)
    assert missing.returncode != 0
    assert "no OpBasicInfo CSV" in missing.stderr

    _write_csv(raw / "OpBasicInfo_0.csv", [row])
    subprocess.run(command, check=True)
    assert list(csv.DictReader(output.open(encoding="utf-8"))) == [{
        "sample": "3", "kernel": row["Op Name"], "task_duration_us": "42.125000",
        "block_dim": "16", "frequency_mhz": "1650",
        "rated_frequency_mhz": "1650",
        "source_csv": str(raw / "OpBasicInfo_0.csv"),
    }]

    _write_csv(raw / "extra" / "OpBasicInfo_1.csv", [row])
    extra = subprocess.run(command, text=True, capture_output=True)
    assert extra.returncode != 0
    assert "expected exactly 1 target row, observed 2" in extra.stderr


def test_parser_rejects_missing_or_under_rated_frequency(tmp_path):
    raw = tmp_path / "raw"
    output = tmp_path / "sample.csv"
    command = [
        sys.executable, str(PARSER), "--raw", str(raw), "--kernel",
        "dsa_decode_softmax_simt_v2_kernel", "--sample", "1",
        "--expected-block-dim", "16", "--output", str(output),
    ]
    row = {
        "Op Name": "dsa_decode_softmax_simt_v2_kernel",
        "Task Duration(us)": "59.5",
        "Block Dim": "16",
        "Current Freq": "1600",
    }
    profile = raw / "OpBasicInfo_0.csv"
    profile.parent.mkdir(parents=True, exist_ok=True)
    with profile.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    missing = subprocess.run(command, text=True, capture_output=True)
    assert missing.returncode != 0
    assert "required rated frequency column missing" in missing.stderr

    row["Rated Freq"] = "1650"
    _write_csv(raw / "OpBasicInfo_0.csv", [row])
    under_rated = subprocess.run(command, text=True, capture_output=True)
    assert under_rated.returncode != 0
    assert "current/rated frequency mismatch: 1600/1650" in under_rated.stderr


def test_docs_freeze_contract_and_evidence_fields():
    docs = required("SPEC.md") + required("README.md")
    for token in (
        "[512, 2048]", "[4, 2048]", "1 / 24", "[-24, 24]",
        "BF16 numerator", "FP32 running max", "FP32 running sum",
        "old scale", "causal", "Task Duration", "five", "median", "minimum",
        "maximum", "ratio", "raw", "device 0", "dav-3510",
    ):
        assert token in docs
    for token in (
        "17.766001", "59.452000", "3.3464x", "1650/1650 MHz",
        "dsa-decode-softmax-actual-evidence-20260813-2130",
        "76.810997", "77.406998", "55.884998", "56.605999",
        "effective Softmax AIV owners", "physical MIX launch capacity",
    ):
        assert token in docs
    assert "TODO" not in required("README.md")
    assert "TBD" not in required("README.md")
    assert re.search(r"correctness.*before.*profil", docs, re.IGNORECASE | re.DOTALL)
