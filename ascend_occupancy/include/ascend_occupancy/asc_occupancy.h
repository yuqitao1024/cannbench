#ifndef ASCEND_OCCUPANCY_ASC_OCCUPANCY_H_
#define ASCEND_OCCUPANCY_ASC_OCCUPANCY_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define ASC_OCCUPANCY_ABI_VERSION 1U
#define ASC_OCCUPANCY_MAX_BLOCK_CANDIDATES 4U

/* Callers initialize public struct arguments with the matching *_INIT macro. */

typedef enum AscOccupancyStatus {
  ASC_OCCUPANCY_SUCCESS = 0,
  ASC_OCCUPANCY_INVALID_ARGUMENT,
  ASC_OCCUPANCY_UNSUPPORTED_DEVICE,
  ASC_OCCUPANCY_RESOURCE_DATA_MISSING,
  ASC_OCCUPANCY_RESOURCE_DATA_INCONSISTENT,
  ASC_OCCUPANCY_INSUFFICIENT_CAPACITY,
} AscOccupancyStatus;

typedef struct AscOccupancyDeviceProperties {
  uint32_t abi_version;
  uint32_t struct_size;
  int32_t device_id;
  uint32_t vector_core_count;
  uint32_t warp_size;
  uint32_t max_threads_per_vector_core;
  uint64_t ub_bytes_per_vector_core;
} AscOccupancyDeviceProperties;

typedef struct AscKernelResourceUsage {
  uint32_t abi_version;
  uint32_t struct_size;
  const char* kernel_symbol;
  uint32_t launch_bounds;
  uint32_t used_registers_per_thread;
  uint32_t stack_size_bytes;
  uint64_t static_ub_bytes;
  bool static_ub_bytes_known;
} AscKernelResourceUsage;

typedef uint32_t AscOccupancyConstraintFlags;

#define ASC_OCCUPANCY_CONSTRAINT_THREADS (1U << 0)
#define ASC_OCCUPANCY_CONSTRAINT_LAUNCH_BOUND (1U << 1)
#define ASC_OCCUPANCY_CONSTRAINT_UB (1U << 2)

typedef struct AscOccupancyAnalysis {
  uint32_t abi_version;
  uint32_t struct_size;
  bool launchable_under_known_constraints;
  bool has_register_spill;
  bool ub_capacity_check_complete;
  uint32_t resident_blocks_per_vector_core;
  uint32_t active_warps_per_vector_core;
  uint32_t max_warps_per_vector_core;
  uint32_t register_limit_per_thread;
  uint32_t register_headroom;
  uint64_t known_ub_headroom_bytes;
  double theoretical_warp_occupancy;
  AscOccupancyConstraintFlags violated_constraints;
} AscOccupancyAnalysis;

typedef struct AscLaunchBoundsCandidate {
  uint32_t launch_bounds;
  uint32_t register_limit_per_thread;
  bool requires_recompile;
  bool requires_benchmark;
} AscLaunchBoundsCandidate;

typedef struct AscOccupancyCandidates {
  uint32_t abi_version;
  uint32_t struct_size;
  /* This is a known-constraint limit, never a performance recommendation. */
  uint32_t max_block_threads_under_known_constraints;
  uint32_t candidate_block_threads[ASC_OCCUPANCY_MAX_BLOCK_CANDIDATES];
  size_t candidate_count;
  bool benchmark_required_for_optimum;
} AscOccupancyCandidates;

#define ASC_OCCUPANCY_DEVICE_PROPERTIES_INIT \
  { ASC_OCCUPANCY_ABI_VERSION, sizeof(AscOccupancyDeviceProperties), 0, 0, 0, 0, 0 }
#define ASC_OCCUPANCY_KERNEL_RESOURCE_USAGE_INIT \
  { ASC_OCCUPANCY_ABI_VERSION, sizeof(AscKernelResourceUsage), NULL, 0, 0, 0, 0, false }
#define ASC_OCCUPANCY_ANALYSIS_INIT \
  { ASC_OCCUPANCY_ABI_VERSION, sizeof(AscOccupancyAnalysis), false, false, false, 0, 0, 0, 0, 0, 0, 0.0, 0 }
#define ASC_OCCUPANCY_CANDIDATES_INIT \
  { ASC_OCCUPANCY_ABI_VERSION, sizeof(AscOccupancyCandidates), 0, {0, 0, 0, 0}, 0, false }

AscOccupancyStatus ascOccupancyGetDeviceProperties(
    int32_t device_id,
    AscOccupancyDeviceProperties* properties);

AscOccupancyStatus ascOccupancyAnalyzeKernel(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    uint32_t block_threads,
    uint64_t dynamic_ub_bytes,
    AscOccupancyAnalysis* analysis);

AscOccupancyStatus ascOccupancyEnumerateLaunchBounds(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    AscLaunchBoundsCandidate* candidates,
    size_t* candidate_count);

AscOccupancyStatus ascOccupancyEnumerateBlockCandidates(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    AscOccupancyCandidates* candidates);

const char* ascOccupancyStatusString(AscOccupancyStatus status);

#ifdef __cplusplus
}
#endif

#endif  // ASCEND_OCCUPANCY_ASC_OCCUPANCY_H_
