#pragma once

#include <cstdint>

#include "simt_api/asc_simt.h"

#include "sparse_attention_head64_plan.h"

namespace aten_dsa_sparse_attention_v2::head64 {

struct Head64Task {
  int32_t batch_index;
  int32_t query_token;
  int32_t head_group;
  int32_t partition;
};

__SIMT_DEVICE_FUNCTIONS_DECL__ inline float head64_warp_max(float value) {
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    const float peer = asc_shfl_xor(value, static_cast<int32_t>(offset), 32);
    value = value < peer ? peer : value;
  }
  return value;
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline float head64_warp_sum(float value) {
  for (uint32_t offset = 16; offset > 0; offset >>= 1) {
    value += asc_shfl_xor(value, static_cast<int32_t>(offset), 32);
  }
  return value;
}

__aicore__ inline void copy_head64_plan(
    SparseAttentionHead64Plan* plan,
    __gm__ const uint8_t* plan_gm) {
  auto dst = reinterpret_cast<uint32_t*>(plan);
  auto src = reinterpret_cast<__gm__ const uint32_t*>(plan_gm);
  for (uint32_t index = 0;
       index < sizeof(SparseAttentionHead64Plan) / sizeof(uint32_t);
       ++index) {
    dst[index] = src[index];
  }
}

__aicore__ inline Head64Task decode_head64_task(
    int32_t task_id,
    const SparseAttentionHead64Plan& plan) {
  const int32_t partition = task_id % plan.selected_partitions;
  task_id /= plan.selected_partitions;
  const int32_t head_group = task_id % plan.head_group_count;
  task_id /= plan.head_group_count;
  return {
      task_id / plan.query_tokens,
      task_id % plan.query_tokens,
      head_group,
      partition,
  };
}

__aicore__ inline int32_t head64_score_row_stride(
    const SparseAttentionHead64Plan& plan) {
  return plan.selected_partition_tile_capacity * plan.selected_tile;
}

__aicore__ inline int32_t head64_partition_begin(
    const Head64Task& task,
    const SparseAttentionHead64Plan& plan) {
  return task.partition * head64_score_row_stride(plan);
}

__aicore__ inline int32_t head64_partition_end(
    const Head64Task& task,
    const SparseAttentionHead64Plan& plan) {
  const int32_t begin = head64_partition_begin(task, plan);
  const int32_t end = begin + head64_score_row_stride(plan);
  return end < plan.selected_tokens ? end : plan.selected_tokens;
}

}  // namespace aten_dsa_sparse_attention_v2::head64
