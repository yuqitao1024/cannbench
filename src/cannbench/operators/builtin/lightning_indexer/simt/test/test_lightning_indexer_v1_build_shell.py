from pathlib import Path


def test_lightning_indexer_simt_v1_setup_uses_bisheng_toolchain():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "bisheng" in setup_py
    assert "--enable-simt" not in setup_py
    assert 'library_name = "aten_dsa_lightning_indexer"' in setup_py


def test_lightning_indexer_build_isolates_each_fused_family_device_library():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert '"liblightning_indexer_family_4x64_kernel.so"' in setup_py
    assert '"liblightning_indexer_family_64x128_kernel.so"' in setup_py
    assert "for kernel_library, kernel_source in KERNEL_LIBRARIES.items()" in setup_py
    assert "self._build_kernel_library(" in setup_py
    assert '"-Wl,-rpath,$ORIGIN"' in setup_py
    assert "outputs.extend(self._kernel_outputs)" in setup_py
    assert "def copy_extensions_to_source(self):" in setup_py
    assert "def get_output_mapping(self):" in setup_py
    assert "sources=HOST_SOURCES" in setup_py
    assert "glob.glob(os.path.join(EXTENSIONS_DIR, \"simt\", \"*.asc\"))" not in setup_py


def test_context_sharded_family_64x128_builds_a_separate_device_library():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "lightning_indexer_context_sharded_family_64x128.asc" in setup_py
    assert (
        '"liblightning_indexer_context_sharded_family_64x128_kernel.so"'
        in setup_py
    )


def test_prefill_q2_family_64x128_builds_a_separate_device_library():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "lightning_indexer_prefill_q2_family_64x128.asc" in setup_py
    assert (
        '"liblightning_indexer_prefill_q2_family_64x128_kernel.so"'
        in setup_py
    )


def test_prefill_q2_family_64x128_uses_16_tasks_and_both_aivs():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_prefill_q2_family_64x128.asc"
    ).read_text(encoding="utf-8")

    for expected in (
        "kQueryAtomSize = 2",
        "kQueryAtomCount = 2048",
        "kLogicalTaskCount = 16",
        "kContextTileSize = 32",
        "kThreadsPerBlock = 1024",
        "__launch_bounds__(kThreadsPerBlock)",
        "params.m = kQueryAtomRows",
        "kCrossCoreSyncMode = 2",
        "kScoreReadyFlag = 0",
        "fixpipe_params.dualDstCtl = 1",
        "AscendC::GetSubBlockIdx()",
        "atom_index += kLogicalTaskCount",
        "query_index = atom_index * kQueryAtomSize + query_in_atom",
    ):
        assert expected in source


def test_context_sharded_family_64x128_maps_dynamic_bq_query_atoms_and_shards():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_context_sharded_family_64x128.asc"
    ).read_text(encoding="utf-8")

    for expected in (
        "kQueryAtomSize = 2",
        "kContextCount = 32768",
        "kContextTileSize = 32",
        "kThreadsPerBlock = 1024",
        "__launch_bounds__(kThreadsPerBlock)",
        "kScoreReadyFlag = 0",
        "kSharedScoresEntries = kHeadCount * kContextTileSize",
        "query_atom_count = (query_count + kQueryAtomSize - 1) / kQueryAtomSize",
        "base_task_index = task_id / context_shard_count",
        "batch_index = base_task_index / query_atom_count",
        "atom_index = base_task_index % query_atom_count",
        "shard_index = task_id % context_shard_count",
        "query_in_atom = static_cast<int32_t>(AscendC::GetSubBlockIdx())",
        "query_index = atom_index * kQueryAtomSize + query_in_atom",
        "valid_query = query_index < query_count",
        "shard_size = kContextCount / context_shard_count",
        "context_start = shard_index * shard_size",
        "context_index >= valid_context_lengths[row_index]",
    ):
        assert expected in source
    for stale in (
        "kContextShardSize = 4096",
        "kContextShardCount = 8",
        "kLogicalTaskCount = 16",
    ):
        assert stale not in source
    assert (
        "shared_scores + query_in_atom * kHeadCount * kContextTileSize"
        not in source
    )


def test_context_sharded_family_64x128_odd_q_does_not_read_padding_row():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_context_sharded_family_64x128.asc"
    ).read_text(encoding="utf-8")

    for expected in (
        "first_query_index = atom_index * kQueryAtomSize",
        "second_query_index = first_query_index + 1",
        "atom_query_rows = second_query_index < query_count",
        "? kQueryAtomRows",
        ": kHeadCount",
        "params.m = atom_query_rows",
        "fixpipe_params.mSize = atom_query_rows",
        "fixpipe_params.dualDstCtl = atom_query_rows == kQueryAtomRows ? 1 : 0",
    ):
        assert expected in source
    assert "safe_second_query_index" not in source


def test_context_sharded_family_64x128_uses_c_api_mode2_sync_and_batch_schedule():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_context_sharded_family_64x128.asc"
    ).read_text(encoding="utf-8")

    for expected in (
        '#include "c_api/sync/sync.h"',
        "asc_sync_block_arrive(PIPE_V, kScoreReadyFlag)",
        "asc_sync_block_wait(PIPE_S, kScoreReadyFlag)",
        "asc_sync_block_arrive(PIPE_FIX, kScoreReadyFlag)",
        "__schedmode__(1)",
    ):
        assert expected in source
    assert "CrossCoreSetFlag" not in source
    assert "CrossCoreWaitFlag" not in source


def test_lightning_indexer_fused_kernels_use_all_32_aics_and_64_aivs():
    root = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )
    for family in ("4x64", "64x128"):
        source = (root / f"lightning_indexer_fused_family_{family}.asc").read_text(
            encoding="utf-8"
        )
        assert "constexpr int32_t kMaxUsedCoreNum = 32;" in source
        assert "constexpr int32_t kMaxUsedCoreNum = 16;" not in source
        assert "constexpr int32_t kMaxUsedCoreNum = 11;" not in source
        assert "constexpr int32_t kThreadsPerBlock = 1024;" in source
        assert "constexpr int32_t kThreadsPerBlock = 256;" not in source
        assert "constexpr uint8_t kCrossCoreSyncMode = 2;" in source
        assert "constexpr uint16_t kScoreReadyFlag = 0;" in source


def test_context_sharded_topk_is_compiled_into_the_score_device_library():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "lightning_indexer_topk_scores.asc" not in setup_py
    assert '"liblightning_indexer_topk_scores_kernel.so"' not in setup_py
    assert "multiple SIMT VF entries per device ELF" in setup_py


def test_context_sharded_topk_uses_1024_threads_and_2048_score_tiles():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_context_sharded_family_64x128.asc"
    ).read_text(encoding="utf-8")

    for expected in (
        "kThreadsPerBlock = 1024",
        "__launch_bounds__(kThreadsPerBlock)",
        "kTopK = 2048",
        "kScoreTileSize = 2048",
        "kSortCapacity = 4096",
        "candidate_index < other_index",
        "needs_local_topk = shard_size > kTopK",
    ):
        assert expected in source


def test_context_sharded_topk_strictly_skips_local_selection_for_s16():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_context_sharded_family_64x128.asc"
    ).read_text(encoding="utf-8")

    for expected in (
        "needs_local_topk = shard_size > kTopK",
        "if (valid_query && needs_local_topk)",
        "asc_sync_vec()",
        "asc_sync_data_barrier(mem_dsb_t::DSB_DDR)",
        "context_shard_count == 1",
    ):
        assert expected in source
    assert "shard_size >= kTopK" not in source


def test_context_sharded_topk_uses_one_global_barrier_and_shard0_final_owner():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_context_sharded_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert source.count("asc_sync_inter_arrive(PIPE_V, kScoreReadyFlag)") == 1
    assert source.count("asc_sync_inter_wait(PIPE_S, kScoreReadyFlag)") == 1
    assert "if (context_shard_count > 1)" in source
    assert (
        "if (valid_query && shard_index == 0 && context_shard_count > 1)"
        in source
    )
    for expected in (
        "per_shard_candidate_count = shard_size < kTopK ? shard_size : kTopK",
        "final_candidate_count = context_shard_count * per_shard_candidate_count",
        "raw_score_candidates = shard_size == kTopK",
    ):
        assert expected in source


def test_context_sharded_standalone_topk_source_is_removed():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_topk_scores.asc"
    )

    assert not source.exists()


def test_context_sharded_bridge_plans_runtime_query_atoms_and_shards():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    planner = source.split("int32_t select_context_shard_count(", 1)[1].split(
        "\n}\n", 1
    )[0]
    for expected in (
        "query_atom_count = (query_count + 1) / 2",
        "base_task_count = batch_size * query_atom_count",
        "base_task_count > 32",
        "{16, 8, 4, 2, 1}",
        "base_task_count * shard_count <= 32",
    ):
        assert expected in planner


def test_context_sharded_bridge_uses_dynamic_workspaces_and_one_launch():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "lightning_indexer_forward_decode_family_64x128_context_sharded_bfloat16(",
        1,
    )[1].split("\n}\n", 1)[0]

    assert "{batch_size, query_count, 32768}" in body
    assert "query.options().dtype(c10::kBFloat16)" in body
    assert "{batch_size, query_count, 2048}" in body
    assert "context_shard_count > 1 && context_shard_count < 16" in body
    assert "{batch_size, query_count, context_shard_count, 2048}" in body
    assert body.count("launch_lightning_indexer_context_sharded") == 1
    assert "launch_lightning_indexer_topk_scores_bfloat16" not in body
    for tensor in (
        "query",
        "keys",
        "weights",
        "valid_context_lengths",
        "reduced_scores",
        "output",
    ):
        assert f"record_tensor_on_stream({tensor}, npu_stream);" in body


def test_prefill_full_score_path_uses_q2_persistent_tasks_and_allowed_apis():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_prefill_full_score_family_64x128.asc"
    ).read_text(encoding="utf-8")

    for expected in (
        "kQueryAtomSize = 2",
        "kPersistentTaskCount = 32",
        "kContextTileSize = 32",
        "kThreadsPerBlock = 1024",
        "atom_index += kPersistentTaskCount",
        "dual_dst_ctl = 1",
        "asc_sync_block_arrive",
        "asc_sync_block_wait",
        "reduced_scores[output_base + context_index]",
    ):
        assert expected in source
    for forbidden in (
        "basic_api/",
        "kernel_operator.h",
        "AscendC::LocalTensor",
        "SetFlag",
        "WaitFlag",
        "PipeBarrier",
        "CrossCore",
        "lightning_indexer_merge_topk_ub",
    ):
        assert forbidden not in source


def test_prefill_full_score_and_radix_topk_build_as_separate_libraries():
    setup_py = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/setup.py"
    ).read_text(encoding="utf-8")

    assert "lightning_indexer_prefill_full_score_family_64x128.asc" in setup_py
    assert "lightning_indexer_radix_topk_bfloat16.asc" in setup_py
    assert (
        '"liblightning_indexer_prefill_full_score_family_64x128_kernel.so"'
        in setup_py
    )
    assert '"liblightning_indexer_radix_topk_bfloat16_kernel.so"' in setup_py


def test_exact_v32_prefill_dispatches_full_score_then_radix_topk():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "lightning_indexer_forward_prefill_full_score_bfloat16(", 1
    )[1].split("\n}\n", 1)[0]

    assert "{1, 4096, 32768}" in body
    assert "query.options().dtype(c10::kBFloat16)" in body
    assert "{1, 4096, 2048}" in body
    assert "query.options().dtype(c10::kInt)" in body
    assert body.index("launch_lightning_indexer_prefill_full_score") < body.index(
        "launch_lightning_indexer_radix_topk_bfloat16"
    )
    for tensor in (
        "query",
        "keys",
        "weights",
        "valid_context_lengths",
        "reduced_scores",
        "output",
    ):
        assert f"record_tensor_on_stream({tensor}, npu_stream);" in body

    dispatch = source.split(
        'if (phase == "prefill" && family == "family_64x128")', 1
    )[1].split("\n  }", 1)[0]
    for expected in (
        "query.scalar_type() == at::ScalarType::BFloat16",
        "query.size(0) == 1",
        "query.size(1) == 4096",
        "keys.size(1) == 32768",
        "top_k == 2048",
        "lightning_indexer_forward_prefill_full_score_bfloat16",
    ):
        assert expected in dispatch


def test_context_sharded_bridge_dispatches_dynamic_bq_fixed_v32_family():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        'if (phase == "decode" && family == "family_64x128") {',
        1,
    )[1].split(
        'if (phase == "decode" && family == "family_32x128") {',
        1,
    )[0]

    for predicate in (
        "query.scalar_type() == at::ScalarType::BFloat16",
        "query.size(2) == 64",
        "query.size(3) == 128",
        "keys.size(1) == 32768",
        "keys.size(2) == 128",
        "top_k == 2048",
    ):
        assert predicate in body
    assert "query.size(0) == 2" not in body
    assert "query.size(1) == 2" not in body
    assert "select_context_shard_count(query.size(0), query.size(1))" in body
    assert "context_shard_count != 0" in body
    assert body.index(
        "lightning_indexer_forward_decode_family_64x128_context_sharded_bfloat16("
    ) < body.index("lightning_indexer_forward_decode_family_64x128_float(")
    assert ".item" not in body


def test_prefill_q2_candidate_is_not_enabled_after_performance_gate():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        'if (phase == "prefill" && family == "family_64x128") {', 1
    )[1].split(
        'if (phase == "prefill" && family == "family_32x128") {', 1
    )[0]

    assert "lightning_indexer_forward_prefill_q2_family_64x128_bfloat16" not in body
    assert "launch_lightning_indexer_prefill_q2_family_64x128_bfloat16" not in source
    assert body.count("lightning_indexer_forward_prefill_family_64x128_float(") == 1


def test_prefill_q2_candidate_does_not_leave_a_hardcoded_false_predicate():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "if (false)" not in source
    assert "&& false" not in source


def test_lightning_indexer_simt_v1_register_has_python_module_entry():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/register.asc"
    ).read_text(encoding="utf-8")

    assert "PyInit__C" in source
    assert "PyModuleDef_HEAD_INIT" in source


def test_lightning_indexer_package_imports_torch_before_loading_cpp_extension():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/__init__.py"
    ).read_text(encoding="utf-8")

    assert 'import_module("torch")' in source
    assert source.index('import_module("torch")') < source.index(
        'import_module(f"{__name__}._C")'
    )


def test_lightning_indexer_bridge_uses_fused_family_launchers():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "launch_lightning_indexer_fused_family_4x64_float" in source
    assert "launch_lightning_indexer_fused_family_64x128_float" in source
    assert "launch_lightning_indexer_score_4x64_float" not in source
    assert "launch_lightning_indexer_score_64x128_float" not in source
    assert "launch_lightning_indexer_prefill_family_4x64_postprocess_float" not in source
    assert (
        "launch_lightning_indexer_prefill_family_64x128_postprocess_float"
        not in source
    )


def test_lightning_indexer_family_4x64_bridge_submits_one_full_shape_launch():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_family_4x64_score_tiled_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_prefill_family_4x64_float(",
        1,
    )[0]

    assert body.count("run_lightning_indexer_family_4x64_tile(") == 1
    assert "for (int64_t query_start" not in body
    assert "for (int64_t context_start" not in body
    assert ".narrow(" not in body
    assert "at::cat(" not in body
    assert "auto best_scores = at::full(" in body
    assert "auto best_indices = at::zeros(" in body


def test_lightning_indexer_prefill_family_4x64_bridge_avoids_key_repeat():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert ".repeat({query_count, 1, 1})" not in source


def test_lightning_indexer_prefill_family_4x64_bridge_has_no_host_tiling():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "for (int64_t query_start = 0; query_start < query_count;" not in source
    assert "for (int64_t context_start = 0; context_start < context_count;" not in source


def test_lightning_indexer_bridge_flushes_torch_npu_tasks_before_raw_launches():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    for family in ("4x64", "64x128"):
        body = source.split(
            f"void run_lightning_indexer_family_{family}_tile(", 1
        )[1].split("\n}\n", 1)[0]
        assert body.index("npu_stream.stream(true)") < body.index(
            f"launch_lightning_indexer_fused_family_{family}_float("
        )


def test_lightning_indexer_bridge_uses_full_input_storage():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert ".narrow(" not in source
    assert "{batch_size, query_count, top_k}" in source
    assert "best_index_tiles" not in source
    assert "at::cat(" not in source


def test_lightning_indexer_records_full_shape_storage_for_raw_launches():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    family_4x64 = source.split(
        "void run_lightning_indexer_family_4x64_tile(", 1
    )[1].split("void run_lightning_indexer_family_64x128_tile(", 1)[0]
    for tensor in ("query", "keys", "weights", "best_scores", "best_indices"):
        assert f"record_tensor_on_stream({tensor}, npu_stream);" in family_4x64

    family_64x128 = source.split(
        "void run_lightning_indexer_family_64x128_tile(", 1
    )[1].split(
        "at::Tensor lightning_indexer_forward_family_4x64_score_tiled_float(", 1
    )[0]
    for tensor in ("query", "keys", "weights", "best_scores", "best_indices"):
        assert f"record_tensor_on_stream({tensor}, npu_stream);" in family_64x128


def test_lightning_indexer_bridge_has_no_stale_host_tile_constants():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "constexpr int64_t kFamily4x64QueryTile" not in source
    assert "constexpr int64_t kFamily4x64ContextTile" not in source
    assert "constexpr int64_t kFamily64x128QueryTile" not in source
    assert "constexpr int64_t kFamily64x128ContextTile" not in source


def test_lightning_indexer_prefill_family_4x64_bridge_extracts_tile_postprocess_helper():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "run_lightning_indexer_family_4x64_tile(" in source


def test_lightning_indexer_split_family_4x64_sources_are_removed():
    base = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )

    assert not (base / "lightning_indexer_score_family_4x64.asc").exists()
    assert not (base / "lightning_indexer_prefill_family_4x64.asc").exists()


def test_lightning_indexer_fused_families_use_cube_scores_in_shared_ub():
    base = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )

    for family in ("4x64", "64x128"):
        source = (base / f"lightning_indexer_fused_family_{family}.asc").read_text(
            encoding="utf-8"
        )

        assert '#include "tensor_api/tensor.h"' in source
        assert "__global__ __aicore__" in source
        assert "KERNEL_TYPE_MIX_AIC_1_2" in source
        assert "MakeMmad(" in source
        assert "Fixpipe<float, float" in source
        assert "TPosition::LCM" in source
        assert "kCrossCoreSyncMode = 2" in source
        assert "CrossCoreSetFlag<kCrossCoreSyncMode" in source
        assert "CrossCoreWaitFlag<kCrossCoreSyncMode" in source
        assert "for (int32_t row_start = 0; row_start < total_rows;" not in source
        assert "linear_index += used_core_num" in source


def test_lightning_indexer_fused_families_sync_aic_with_both_aivs():
    base = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )

    for family in ("4x64", "64x128"):
        source = (base / f"lightning_indexer_fused_family_{family}.asc").read_text(
            encoding="utf-8"
        )
        aiv = source.split(
            f"__aicore__ inline void lightning_indexer_fused_family_{family}_aiv(",
            1,
        )[1].split(
            f"__global__ __aicore__ void lightning_indexer_fused_family_{family}_kernel(",
            1,
        )[0]

        assert "kCrossCoreSyncMode = 2" in source
        assert "kScoreReadyFlag = 0" in source
        assert "kScoreReadyFlagBase" not in source
        assert "kScoreReadyFlag + block_idx" not in source
        assert "if (AscendC::GetSubBlockIdx() != 0)" not in aiv
        assert "CrossCoreSetFlag<kCrossCoreSyncMode, PIPE_V>(kScoreReadyFlag)" in aiv
        assert "CrossCoreWaitFlag<kCrossCoreSyncMode, PIPE_V>(kScoreReadyFlag)" in aiv


def test_lightning_indexer_prefill_family_64x128_bridge_extracts_tile_helper():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "run_lightning_indexer_family_64x128_tile(" in source


def test_lightning_indexer_prefill_family_64x128_uses_decode_fused_path():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "launch_lightning_indexer_fused_family_64x128_float" in source
    assert "launch_lightning_indexer_prefill_family_64x128_postprocess_float" not in source


def test_lightning_indexer_prefill_family_64x128_fp16_avoids_split_score_then_postprocess():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_prefill_family_64x128_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_decode_family_4x64_float(",
        1,
    )[0]

    assert "return lightning_indexer_forward_decode_family_64x128_float(" in body
    assert "launch_lightning_indexer_prefill_family_64x128_postprocess_float" not in body
    assert "run_lightning_indexer_family_64x128_tile(" not in body


def test_lightning_indexer_decode_family_64x128_reuses_fused_helper():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_decode_family_64x128_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_prefill_family_64x128_float(",
        1,
    )[0]

    assert "run_lightning_indexer_family_64x128_tile(" in body
    assert "auto best_scores = at::full(" in body
    assert "auto best_indices = at::zeros(" in body
    assert "return best_indices;" in body


def test_lightning_indexer_family_64x128_bridge_submits_one_full_shape_launch():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_decode_family_64x128_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_prefill_family_64x128_float(",
        1,
    )[0]

    assert body.count("run_lightning_indexer_family_64x128_tile(") == 1
    assert "for (int64_t query_start" not in body
    assert "for (int64_t context_start" not in body
    assert "query.narrow(" not in body
    assert "keys.narrow(" not in body


def test_lightning_indexer_family_64x128_kernel_owns_row_and_context_loops():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_fused_family_64x128.asc"
    ).read_text(encoding="utf-8")
    launcher = source.split(
        'extern "C" void launch_lightning_indexer_fused_family_64x128_float(',
        1,
    )[1]

    assert "linear_index += used_core_num" in source
    assert "context_start += kContextTileCapacity" in source
    assert launcher.count("lightning_indexer_fused_family_64x128_kernel") == 1
    assert "for (int32_t row_start" not in launcher


def test_lightning_indexer_family_64x128_serializes_l0_reuse_between_rows():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_fused_family_64x128.asc"
    ).read_text(encoding="utf-8")
    aic = source.split(
        "__aicore__ inline void lightning_indexer_fused_family_64x128_aic(",
        1,
    )[1].split(
        "__aicore__ inline void lightning_indexer_fused_family_64x128_aiv(",
        1,
    )[0]

    reverse_event = "AscendC::HardEvent::M_MTE1>(EVENT_ID0)"
    assert aic.count(f"SetFlag<{reverse_event}") == 2
    assert aic.count(f"WaitFlag<{reverse_event}") == 3


def test_lightning_indexer_family_64x128_serializes_l1_reuse_between_tiles():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_fused_family_64x128.asc"
    ).read_text(encoding="utf-8")
    aic = source.split(
        "__aicore__ inline void lightning_indexer_fused_family_64x128_aic(",
        1,
    )[1].split(
        "__aicore__ inline void lightning_indexer_fused_family_64x128_aiv(",
        1,
    )[0]

    reverse_event = "AscendC::HardEvent::MTE1_MTE2>(EVENT_ID0)"
    assert aic.count(f"SetFlag<{reverse_event}") == 3
    assert aic.count(f"WaitFlag<{reverse_event}") == 3


def test_lightning_indexer_family_64x128_serializes_l0c_reuse_between_tiles():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_fused_family_64x128.asc"
    ).read_text(encoding="utf-8")
    aic = source.split(
        "__aicore__ inline void lightning_indexer_fused_family_64x128_aic(",
        1,
    )[1].split(
        "__aicore__ inline void lightning_indexer_fused_family_64x128_aiv(",
        1,
    )[0]

    reverse_event = "AscendC::HardEvent::FIX_M>(EVENT_ID0)"
    assert aic.count(f"SetFlag<{reverse_event}") == 2
    assert aic.count(f"WaitFlag<{reverse_event}") == 2


def test_lightning_indexer_bridge_declares_only_fused_launchers_with_c_linkage():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert 'extern "C" void launch_lightning_indexer_fused_family_4x64_float' in source
    assert 'extern "C" void launch_lightning_indexer_fused_family_64x128_float' in source
    assert (
        'extern "C" void launch_lightning_indexer_prefill_family_4x64_postprocess_float'
        not in source
    )
    assert (
        'extern "C" void launch_lightning_indexer_prefill_family_64x128_postprocess_float'
        not in source
    )
    assert 'extern "C" void launch_lightning_indexer_score_4x64_float' not in source
    assert 'extern "C" void launch_lightning_indexer_score_64x128_float' not in source


def test_lightning_indexer_split_family_64x128_sources_are_removed():
    base = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt"
    )

    assert not (base / "lightning_indexer_score_family_64x128.asc").exists()
    assert not (base / "lightning_indexer_postprocess_family_64x128.asc").exists()


def test_lightning_indexer_family_4x64_fp16_fused_path_does_not_materialize_score_tile():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_family_4x64_score_tiled_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_prefill_family_4x64_float(",
        1,
    )[0]

    assert "auto score_tile = at::empty(" not in body


def test_lightning_indexer_family_64x128_fp16_fused_path_does_not_materialize_score_tile():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")
    body = source.split(
        "at::Tensor lightning_indexer_forward_decode_family_64x128_float(",
        1,
    )[1].split(
        "at::Tensor lightning_indexer_forward_prefill_family_64x128_float(",
        1,
    )[0]

    assert "auto score_tile = at::empty(" not in body


def test_lightning_indexer_bridge_uses_bfloat16_kernel_inputs():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "const at::BFloat16* query" in source
    assert "const at::BFloat16* keys" in source
    assert "const at::BFloat16* weights" in source
    assert "const_data_ptr<at::BFloat16>()" in source
    assert "query.to(at::kHalf)" not in source
    assert "keys.to(at::kHalf)" not in source
    assert "weights.to(at::kFloat)" not in source


def test_lightning_indexer_fused_family_4x64_kernel_uses_bfloat16_storage():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_fused_family_4x64.asc"
    ).read_text(encoding="utf-8")

    assert '#include "simt_api/asc_bf16.h"' in source
    assert "__simt_vf__" in source
    assert "__cbuf__ bfloat16_t" in source
    assert "__ca__ bfloat16_t" in source
    assert "__cb__ bfloat16_t" in source
    assert "__gm__ const bfloat16_t* weights" in source
    assert "__global__ __aicore__ void lightning_indexer_fused_family_4x64_kernel(" in source
    assert "__gm__ const uint16_t* query" in source
    assert "__gm__ const uint16_t* keys" in source
    assert "__gm__ const uint16_t* weights" in source
    assert "reinterpret_cast<__gm__ bfloat16_t*>(" in source
    assert "reinterpret_cast<const uint16_t*>(query)" in source
    assert "static_cast<float>(static_cast<bfloat16_t>(reduced_score))" in source
    assert "half" not in source


def test_lightning_indexer_fused_family_64x128_kernel_uses_bfloat16_storage():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_fused_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert '#include "simt_api/asc_bf16.h"' in source
    assert "__simt_vf__" in source
    assert "__cbuf__ bfloat16_t" in source
    assert "__ca__ bfloat16_t" in source
    assert "__cb__ bfloat16_t" in source
    assert "__gm__ const bfloat16_t* weights" in source
    assert "__global__ __aicore__ void lightning_indexer_fused_family_64x128_kernel(" in source
    assert "__gm__ const uint16_t* query" in source
    assert "__gm__ const uint16_t* keys" in source
    assert "__gm__ const uint16_t* weights" in source
    assert "reinterpret_cast<__gm__ bfloat16_t*>(" in source
    assert "reinterpret_cast<const uint16_t*>(query)" in source
    assert "static_cast<float>(static_cast<bfloat16_t>(reduced_score))" in source
    assert "half" not in source


def test_lightning_indexer_x128_kernel_supports_runtime_32_or_64_heads():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/"
        "lightning_indexer_fused_family_64x128.asc"
    ).read_text(encoding="utf-8")

    assert "constexpr int32_t kFamilyX128MaxHeadCount = 64;" in source
    assert "int32_t head_count" in source
    assert "params.m = static_cast<uint16_t>(head_count);" in source
    assert "fixpipe_params.mSize = static_cast<uint32_t>(head_count);" in source
    assert "head_index < head_count" in source


def test_lightning_indexer_x128_bridge_accepts_top2048_for_both_head_counts():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/lightning_indexer.asc"
    ).read_text(encoding="utf-8")

    assert "family_32x128 custom op requires top_k <= 2048" in source
    assert "family_64x128 custom op requires top_k <= 2048" in source


def test_lightning_indexer_topk_ub_capacity_covers_top2048_plus_context_tile():
    source = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc/simt/lightning_indexer_topk_ub.h"
    ).read_text(encoding="utf-8")

    assert "kFusedTopkSortCapacity = 4096" in source


def test_lightning_indexer_fused_kernels_use_per_row_valid_context_lengths():
    root = Path(
        "src/cannbench/operators/builtin/lightning_indexer/simt/v1/"
        "aten_dsa_lightning_indexer/csrc"
    )
    bridge = (root / "lightning_indexer.asc").read_text(encoding="utf-8")

    assert "Tensor valid_context_lengths" in bridge
    assert "valid_context_lengths.const_data_ptr<int32_t>()" in bridge
    for family in ("4x64", "64x128"):
        source = (
            root / "simt" / f"lightning_indexer_fused_family_{family}.asc"
        ).read_text(encoding="utf-8")
        assert "__gm__ const int32_t* valid_context_lengths" in source
        assert "valid_context_lengths[linear_index]" in source
        assert "context_start < row_context_count" in source
