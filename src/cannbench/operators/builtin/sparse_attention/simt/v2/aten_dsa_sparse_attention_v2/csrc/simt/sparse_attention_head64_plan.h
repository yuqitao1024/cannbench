#pragma once

#include <cstdint>

namespace aten_dsa_sparse_attention_v2 {

constexpr int32_t kHead64Tile = 64;
constexpr int32_t kHead64SelectedTile = 64;
constexpr int32_t kHead64QkTile = 64;
constexpr int32_t kHead64ValueTile = 128;
constexpr int32_t kHead64Threads = 1024;
constexpr int32_t kHead64ThreadsPerHead = 32;
constexpr int32_t kHead64PhysicalAicLimit = 32;

enum SparseAttentionHead64OutputMode : int32_t {
  kHead64OutputPartialFloat = 0,
  kHead64OutputDirectBfloat16 = 1,
};

struct SparseAttentionHead64Plan {
  int32_t used_core_num;
  int32_t task_count;
  int32_t batch_size;
  int32_t query_heads;
  int32_t query_tokens;
  int32_t context_tokens;
  int32_t selected_tokens;
  int32_t qk_head_dim;
  int32_t value_head_dim;
  int32_t head_tile;
  int32_t head_group_count;
  int32_t selected_tile;
  int32_t selected_partitions;
  int32_t selected_partition_tile_capacity;
  int32_t output_mode;
};

}  // namespace aten_dsa_sparse_attention_v2
