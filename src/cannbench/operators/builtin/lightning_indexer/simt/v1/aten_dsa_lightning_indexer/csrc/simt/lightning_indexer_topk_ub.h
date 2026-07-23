#ifndef ATEN_DSA_LIGHTNING_INDEXER_TOPK_UB_H
#define ATEN_DSA_LIGHTNING_INDEXER_TOPK_UB_H

#include <cstdint>
#include <limits>

#include "simt_api/asc_simt.h"
#include "simt_api/device_sync_functions.h"

constexpr int32_t kFusedTopkSortCapacity = 4096;
constexpr int32_t kFusedTopkDynamicUbufBytes =
    2 * kFusedTopkSortCapacity * sizeof(uint32_t);

__SIMT_DEVICE_FUNCTIONS_DECL__ inline bool fused_candidate_is_better(
    float left_score,
    int32_t left_index,
    float right_score,
    int32_t right_index) {
  return left_score > right_score ||
      (left_score == right_score && left_index < right_index);
}

__SIMT_DEVICE_FUNCTIONS_DECL__ inline void lightning_indexer_merge_topk_ub(
    __ubuf__ float* candidate_scores,
    __ubuf__ int32_t* candidate_indices,
    __gm__ float* topk_scores,
    __gm__ int32_t* topk_indices,
    int64_t topk_base,
    int32_t context_count,
    int32_t top_k) {
  const int32_t tid = static_cast<int32_t>(threadIdx.x);
  for (int32_t rank = tid; rank < top_k; rank += blockDim.x) {
    candidate_scores[rank] = topk_scores[topk_base + rank];
    candidate_indices[rank] = topk_indices[topk_base + rank];
  }
  asc_syncthreads();

  const uint32_t candidate_count =
      static_cast<uint32_t>(top_k + context_count);
  uint32_t sort_count = 1U;
  while (sort_count < candidate_count) {
    sort_count <<= 1U;
  }

  for (uint32_t position = candidate_count + static_cast<uint32_t>(tid);
       position < sort_count;
       position += static_cast<uint32_t>(blockDim.x)) {
    candidate_scores[position] = -std::numeric_limits<float>::infinity();
    candidate_indices[position] = std::numeric_limits<int32_t>::max();
  }
  asc_syncthreads();

  for (uint32_t bitonic_size = 2U; bitonic_size <= sort_count;
       bitonic_size <<= 1U) {
    for (uint32_t stride = bitonic_size >> 1U; stride > 0U;
         stride >>= 1U) {
      for (uint32_t position = static_cast<uint32_t>(tid);
           position < sort_count;
           position += static_cast<uint32_t>(blockDim.x)) {
        const uint32_t partner = position ^ stride;
        if (partner <= position) {
          continue;
        }

        const float left_score = candidate_scores[position];
        const int32_t left_index = candidate_indices[position];
        const float right_score = candidate_scores[partner];
        const int32_t right_index = candidate_indices[partner];
        const bool better_first = (position & bitonic_size) == 0U;
        const bool swap = better_first
            ? fused_candidate_is_better(
                  right_score, right_index, left_score, left_index)
            : fused_candidate_is_better(
                  left_score, left_index, right_score, right_index);
        if (swap) {
          candidate_scores[position] = right_score;
          candidate_indices[position] = right_index;
          candidate_scores[partner] = left_score;
          candidate_indices[partner] = left_index;
        }
      }
      asc_syncthreads();
    }
  }

  for (int32_t rank = tid; rank < top_k; rank += blockDim.x) {
    topk_scores[topk_base + rank] = candidate_scores[rank];
    topk_indices[topk_base + rank] = candidate_indices[rank];
  }
  asc_syncthreads();
}

#endif
