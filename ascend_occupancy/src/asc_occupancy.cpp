#include <ascend_occupancy/asc_occupancy.h>

#include <algorithm>
#include <limits>

#if defined(ASC_OCCUPANCY_WITH_ACL)
#include <acl/acl.h>
#endif

namespace {

constexpr uint32_t kDefaultLaunchBounds = 1024U;
constexpr uint32_t kLaunchBoundsValues[] = {256U, 512U, 1024U, 2048U};
constexpr uint32_t kRegisterLimits[] = {127U, 64U, 32U, 16U};
constexpr int64_t kDav3510NpuArchitecture = 3510;

template <typename T>
bool HasCompatibleHeader(const T* value) {
  return value != nullptr && value->abi_version == ASC_OCCUPANCY_ABI_VERSION &&
         value->struct_size >= sizeof(T);
}

uint32_t EffectiveLaunchBounds(const AscKernelResourceUsage* resources) {
  return resources->launch_bounds == 0U ? kDefaultLaunchBounds
                                        : resources->launch_bounds;
}

bool IsSupportedLaunchBounds(uint32_t launch_bounds) {
  return launch_bounds >= 1U && launch_bounds <= kLaunchBoundsValues[3];
}

uint32_t RegisterLimitForLaunchBounds(uint32_t launch_bounds) {
  if (launch_bounds <= kLaunchBoundsValues[0]) {
    return kRegisterLimits[0];
  }
  if (launch_bounds <= kLaunchBoundsValues[1]) {
    return kRegisterLimits[1];
  }
  if (launch_bounds <= kLaunchBoundsValues[2]) {
    return kRegisterLimits[2];
  }
  return kRegisterLimits[3];
}

uint32_t CeilDiv(uint32_t numerator, uint32_t denominator) {
  return numerator == 0U ? 0U : 1U + (numerator - 1U) / denominator;
}

void AppendDistinctLaunchBounds(uint32_t* selected_bounds,
                                size_t* selected_count,
                                uint32_t launch_bounds) {
  for (size_t index = 0U; index < *selected_count; ++index) {
    if (selected_bounds[index] == launch_bounds) {
      return;
    }
  }
  selected_bounds[(*selected_count)++] = launch_bounds;
}

AscOccupancyStatus ValidateDevice(const AscOccupancyDeviceProperties* device) {
  if (!HasCompatibleHeader(device) || device->device_id < 0 ||
      device->vector_core_count == 0U || device->warp_size == 0U ||
      device->max_threads_per_vector_core == 0U ||
      device->ub_bytes_per_vector_core == 0U) {
    return ASC_OCCUPANCY_INVALID_ARGUMENT;
  }
  return ASC_OCCUPANCY_SUCCESS;
}

AscOccupancyStatus ValidateResources(const AscKernelResourceUsage* resources) {
  if (!HasCompatibleHeader(resources)) {
    return ASC_OCCUPANCY_INVALID_ARGUMENT;
  }
  if (resources->kernel_symbol == nullptr) {
    return ASC_OCCUPANCY_RESOURCE_DATA_MISSING;
  }

  const uint32_t launch_bounds = EffectiveLaunchBounds(resources);
  if (!IsSupportedLaunchBounds(launch_bounds)) {
    return ASC_OCCUPANCY_RESOURCE_DATA_INCONSISTENT;
  }
  if (resources->used_registers_per_thread >
      RegisterLimitForLaunchBounds(launch_bounds)) {
    return ASC_OCCUPANCY_RESOURCE_DATA_INCONSISTENT;
  }
  return ASC_OCCUPANCY_SUCCESS;
}

AscOccupancyStatus ValidateInputs(const AscOccupancyDeviceProperties* device,
                                  const AscKernelResourceUsage* resources) {
  const AscOccupancyStatus device_status = ValidateDevice(device);
  if (device_status != ASC_OCCUPANCY_SUCCESS) {
    return device_status;
  }
  return ValidateResources(resources);
}

bool KnownUbExceedsCapacity(const AscKernelResourceUsage* resources,
                            uint64_t dynamic_ub_bytes,
                            uint64_t capacity) {
  const uint64_t static_ub_bytes =
      resources->static_ub_bytes_known ? resources->static_ub_bytes : 0U;
  return static_ub_bytes > capacity || dynamic_ub_bytes > capacity - static_ub_bytes;
}

uint64_t KnownUbHeadroom(const AscKernelResourceUsage* resources,
                         uint64_t dynamic_ub_bytes,
                         uint64_t capacity) {
  if (KnownUbExceedsCapacity(resources, dynamic_ub_bytes, capacity)) {
    return 0U;
  }
  const uint64_t static_ub_bytes =
      resources->static_ub_bytes_known ? resources->static_ub_bytes : 0U;
  return capacity - static_ub_bytes - dynamic_ub_bytes;
}

void InitializeAnalysis(AscOccupancyAnalysis* analysis) {
  *analysis = {};
  analysis->abi_version = ASC_OCCUPANCY_ABI_VERSION;
  analysis->struct_size = sizeof(*analysis);
}

void InitializeCandidates(AscOccupancyCandidates* candidates) {
  *candidates = {};
  candidates->abi_version = ASC_OCCUPANCY_ABI_VERSION;
  candidates->struct_size = sizeof(*candidates);
  candidates->benchmark_required_for_optimum = true;
}

}  // namespace

extern "C" AscOccupancyStatus ascOccupancyGetDeviceProperties(
    int32_t device_id,
    AscOccupancyDeviceProperties* properties) {
  if (!HasCompatibleHeader(properties)) {
    return ASC_OCCUPANCY_INVALID_ARGUMENT;
  }

#if !defined(ASC_OCCUPANCY_WITH_ACL)
  (void)device_id;
  return ASC_OCCUPANCY_UNSUPPORTED_DEVICE;
#else
  int64_t npu_architecture = 0;
  int64_t vector_core_count = 0;
  int64_t warp_size = 0;
  int64_t max_threads_per_vector_core = 0;
  int64_t ub_bytes_per_vector_core = 0;
  int64_t max_threads_per_block = 0;
  if (aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_NPU_ARCH, &npu_architecture) !=
          ACL_SUCCESS ||
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_VECTOR_CORE_NUM,
                          &vector_core_count) != ACL_SUCCESS ||
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_WARP_SIZE, &warp_size) !=
          ACL_SUCCESS ||
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE,
                          &max_threads_per_vector_core) != ACL_SUCCESS ||
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE,
                          &ub_bytes_per_vector_core) != ACL_SUCCESS ||
      aclrtGetDeviceInfo(device_id, ACL_DEV_ATTR_MAX_THREADS_PER_BLOCK,
                          &max_threads_per_block) != ACL_SUCCESS) {
    return ASC_OCCUPANCY_RESOURCE_DATA_MISSING;
  }
  if (npu_architecture != kDav3510NpuArchitecture) {
    return ASC_OCCUPANCY_UNSUPPORTED_DEVICE;
  }
  if (max_threads_per_vector_core != max_threads_per_block ||
      vector_core_count <= 0 || warp_size <= 0 ||
      max_threads_per_vector_core <= 0 || ub_bytes_per_vector_core <= 0 ||
      vector_core_count > std::numeric_limits<uint32_t>::max() ||
      warp_size > std::numeric_limits<uint32_t>::max() ||
      max_threads_per_vector_core > std::numeric_limits<uint32_t>::max()) {
    return ASC_OCCUPANCY_RESOURCE_DATA_INCONSISTENT;
  }

  *properties = {};
  properties->abi_version = ASC_OCCUPANCY_ABI_VERSION;
  properties->struct_size = sizeof(*properties);
  properties->device_id = device_id;
  properties->vector_core_count = static_cast<uint32_t>(vector_core_count);
  properties->warp_size = static_cast<uint32_t>(warp_size);
  properties->max_threads_per_vector_core =
      static_cast<uint32_t>(max_threads_per_vector_core);
  properties->ub_bytes_per_vector_core =
      static_cast<uint64_t>(ub_bytes_per_vector_core);
  return ASC_OCCUPANCY_SUCCESS;
#endif
}

extern "C" AscOccupancyStatus ascOccupancyAnalyzeKernel(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    uint32_t block_threads,
    uint64_t dynamic_ub_bytes,
    AscOccupancyAnalysis* analysis) {
  if (!HasCompatibleHeader(analysis)) {
    return ASC_OCCUPANCY_INVALID_ARGUMENT;
  }

  const AscOccupancyStatus validation = ValidateInputs(device, resources);
  if (validation != ASC_OCCUPANCY_SUCCESS) {
    return validation;
  }

  InitializeAnalysis(analysis);
  const uint32_t launch_bounds = EffectiveLaunchBounds(resources);
  analysis->has_register_spill = resources->stack_size_bytes > 0U;
  analysis->ub_capacity_check_complete = resources->static_ub_bytes_known;
  analysis->active_warps_per_vector_core =
      CeilDiv(block_threads, device->warp_size);
  analysis->max_warps_per_vector_core =
      CeilDiv(device->max_threads_per_vector_core, device->warp_size);
  analysis->theoretical_warp_occupancy =
      static_cast<double>(analysis->active_warps_per_vector_core) /
      static_cast<double>(analysis->max_warps_per_vector_core);
  analysis->register_limit_per_thread = RegisterLimitForLaunchBounds(launch_bounds);
  analysis->register_headroom =
      analysis->register_limit_per_thread - resources->used_registers_per_thread;
  analysis->known_ub_headroom_bytes = KnownUbHeadroom(
      resources, dynamic_ub_bytes, device->ub_bytes_per_vector_core);

  AscOccupancyConstraintFlags violated_constraints = 0U;
  if (block_threads == 0U || block_threads > device->max_threads_per_vector_core) {
    violated_constraints |= ASC_OCCUPANCY_CONSTRAINT_THREADS;
  }
  if (block_threads > launch_bounds) {
    violated_constraints |= ASC_OCCUPANCY_CONSTRAINT_LAUNCH_BOUND;
  }
  if (KnownUbExceedsCapacity(resources, dynamic_ub_bytes,
                             device->ub_bytes_per_vector_core)) {
    violated_constraints |= ASC_OCCUPANCY_CONSTRAINT_UB;
  }

  analysis->violated_constraints = violated_constraints;
  analysis->launchable_under_known_constraints = violated_constraints == 0U;
  analysis->resident_blocks_per_vector_core =
      analysis->launchable_under_known_constraints ? 1U : 0U;
  return analysis->launchable_under_known_constraints ? ASC_OCCUPANCY_SUCCESS
                                                       : ASC_OCCUPANCY_INSUFFICIENT_CAPACITY;
}

extern "C" AscOccupancyStatus ascOccupancyEnumerateLaunchBounds(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    AscLaunchBoundsCandidate* candidates,
    size_t* candidate_count) {
  if (candidate_count == nullptr) {
    return ASC_OCCUPANCY_INVALID_ARGUMENT;
  }
  const AscOccupancyStatus validation = ValidateInputs(device, resources);
  if (validation != ASC_OCCUPANCY_SUCCESS) {
    return validation;
  }

  const uint32_t current_bounds = EffectiveLaunchBounds(resources);
  uint32_t selected_bounds[4] = {};
  size_t selected_count = 0U;
  if (resources->stack_size_bytes > 0U) {
    for (uint32_t bound : kLaunchBoundsValues) {
      if (bound < current_bounds) {
        AppendDistinctLaunchBounds(selected_bounds, &selected_count, bound);
      }
    }
    AppendDistinctLaunchBounds(selected_bounds, &selected_count, current_bounds);
  } else {
    AppendDistinctLaunchBounds(selected_bounds, &selected_count, current_bounds);
    for (uint32_t bound : kLaunchBoundsValues) {
      if (bound > current_bounds &&
          resources->used_registers_per_thread <= RegisterLimitForLaunchBounds(bound)) {
        AppendDistinctLaunchBounds(selected_bounds, &selected_count, bound);
      }
    }
  }

  const size_t required_count = selected_count;
  const size_t supplied_capacity = *candidate_count;
  *candidate_count = required_count;
  if (candidates == nullptr) {
    return ASC_OCCUPANCY_SUCCESS;
  }
  if (supplied_capacity < required_count) {
    return ASC_OCCUPANCY_INSUFFICIENT_CAPACITY;
  }

  for (size_t index = 0U; index < required_count; ++index) {
    candidates[index].launch_bounds = selected_bounds[index];
    candidates[index].register_limit_per_thread =
        RegisterLimitForLaunchBounds(selected_bounds[index]);
    candidates[index].requires_recompile = selected_bounds[index] != current_bounds;
    candidates[index].requires_benchmark = true;
  }
  return ASC_OCCUPANCY_SUCCESS;
}

extern "C" AscOccupancyStatus ascOccupancyEnumerateBlockCandidates(
    const AscOccupancyDeviceProperties* device,
    const AscKernelResourceUsage* resources,
    AscOccupancyCandidates* candidates) {
  if (!HasCompatibleHeader(candidates)) {
    return ASC_OCCUPANCY_INVALID_ARGUMENT;
  }
  const AscOccupancyStatus validation = ValidateInputs(device, resources);
  if (validation != ASC_OCCUPANCY_SUCCESS) {
    return validation;
  }

  InitializeCandidates(candidates);
  if (resources->static_ub_bytes_known &&
      resources->static_ub_bytes > device->ub_bytes_per_vector_core) {
    return ASC_OCCUPANCY_INSUFFICIENT_CAPACITY;
  }

  const uint32_t max_block_threads = std::min(
      device->max_threads_per_vector_core, EffectiveLaunchBounds(resources));
  candidates->max_block_threads_under_known_constraints = max_block_threads;
  const uint32_t aligned_max_block_threads =
      max_block_threads - max_block_threads % device->warp_size;

  for (uint32_t bound : kLaunchBoundsValues) {
    if (bound <= aligned_max_block_threads) {
      candidates->candidate_block_threads[candidates->candidate_count++] = bound;
    }
  }
  if (aligned_max_block_threads != 0U &&
      (candidates->candidate_count == 0U ||
       candidates->candidate_block_threads[candidates->candidate_count - 1U] !=
           aligned_max_block_threads) &&
      candidates->candidate_count < ASC_OCCUPANCY_MAX_BLOCK_CANDIDATES) {
    candidates->candidate_block_threads[candidates->candidate_count++] =
        aligned_max_block_threads;
  }
  return ASC_OCCUPANCY_SUCCESS;
}

extern "C" const char* ascOccupancyStatusString(AscOccupancyStatus status) {
  switch (status) {
    case ASC_OCCUPANCY_SUCCESS:
      return "success";
    case ASC_OCCUPANCY_INVALID_ARGUMENT:
      return "invalid argument";
    case ASC_OCCUPANCY_UNSUPPORTED_DEVICE:
      return "unsupported device";
    case ASC_OCCUPANCY_RESOURCE_DATA_MISSING:
      return "resource data missing";
    case ASC_OCCUPANCY_RESOURCE_DATA_INCONSISTENT:
      return "resource data inconsistent";
    case ASC_OCCUPANCY_INSUFFICIENT_CAPACITY:
      return "insufficient capacity";
    default:
      return "unknown occupancy status";
  }
}
