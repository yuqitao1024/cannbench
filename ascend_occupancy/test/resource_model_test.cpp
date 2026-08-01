#include <ascend_occupancy/asc_occupancy.h>

#include <cstdlib>
#include <iostream>

namespace {

#define CHECK(expression)                                                        \
  do {                                                                           \
    if (!(expression)) {                                                         \
      std::cerr << __FILE__ << ':' << __LINE__ << ": CHECK(" #expression        \
                << ") failed\n";                                               \
      return false;                                                              \
    }                                                                            \
  } while (false)

AscOccupancyDeviceProperties MakeDevice(uint32_t max_threads = 2048U,
                                        uint64_t ub_bytes = 1024U) {
  AscOccupancyDeviceProperties device = ASC_OCCUPANCY_DEVICE_PROPERTIES_INIT;
  device.device_id = 0;
  device.vector_core_count = 64U;
  device.warp_size = 32U;
  device.max_threads_per_vector_core = max_threads;
  device.ub_bytes_per_vector_core = ub_bytes;
  return device;
}

AscKernelResourceUsage MakeResources(uint32_t launch_bounds = 1024U,
                                      uint32_t registers = 32U) {
  AscKernelResourceUsage resources = ASC_OCCUPANCY_KERNEL_RESOURCE_USAGE_INIT;
  resources.kernel_symbol = "test_kernel";
  resources.launch_bounds = launch_bounds;
  resources.used_registers_per_thread = registers;
  return resources;
}

AscOccupancyAnalysis MakeAnalysis() {
  AscOccupancyAnalysis analysis = ASC_OCCUPANCY_ANALYSIS_INIT;
  return analysis;
}

bool TestHostOnlyDeviceDiscoveryIsExplicitlyUnsupported() {
  AscOccupancyDeviceProperties properties = ASC_OCCUPANCY_DEVICE_PROPERTIES_INIT;
  CHECK(ascOccupancyGetDeviceProperties(0, &properties) ==
        ASC_OCCUPANCY_UNSUPPORTED_DEVICE);
  CHECK(ascOccupancyGetDeviceProperties(-1, &properties) ==
        ASC_OCCUPANCY_UNSUPPORTED_DEVICE);
  return true;
}

bool TestRegisterLimitBoundaries() {
  const uint32_t bounds[] = {256U, 257U, 512U, 513U, 1024U, 1025U, 2048U};
  const uint32_t limits[] = {127U, 64U, 64U, 32U, 32U, 16U, 16U};
  const AscOccupancyDeviceProperties device = MakeDevice();

  for (size_t index = 0; index < sizeof(bounds) / sizeof(bounds[0]); ++index) {
    AscKernelResourceUsage resources = MakeResources(bounds[index], limits[index]);
    AscOccupancyAnalysis analysis = MakeAnalysis();
    CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 32U, 0U, &analysis) ==
          ASC_OCCUPANCY_SUCCESS);
    CHECK(analysis.register_limit_per_thread == limits[index]);
    CHECK(analysis.register_headroom == 0U);
  }
  return true;
}

bool TestLaunchConstraintsAndTailWarpOccupancy() {
  const AscOccupancyDeviceProperties device = MakeDevice();
  AscKernelResourceUsage resources = MakeResources(512U, 48U);
  AscOccupancyAnalysis analysis = MakeAnalysis();

  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 33U, 0U, &analysis) ==
        ASC_OCCUPANCY_SUCCESS);
  CHECK(analysis.launchable_under_known_constraints);
  CHECK(analysis.resident_blocks_per_vector_core == 1U);
  CHECK(analysis.active_warps_per_vector_core == 2U);
  CHECK(analysis.max_warps_per_vector_core == 64U);
  CHECK(analysis.theoretical_warp_occupancy == 2.0 / 64.0);

  analysis = MakeAnalysis();
  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 0U, 0U, &analysis) ==
        ASC_OCCUPANCY_INSUFFICIENT_CAPACITY);
  CHECK(analysis.resident_blocks_per_vector_core == 0U);
  CHECK(analysis.violated_constraints == ASC_OCCUPANCY_CONSTRAINT_THREADS);

  analysis = MakeAnalysis();
  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 513U, 0U, &analysis) ==
        ASC_OCCUPANCY_INSUFFICIENT_CAPACITY);
  CHECK((analysis.violated_constraints & ASC_OCCUPANCY_CONSTRAINT_LAUNCH_BOUND) != 0U);

  analysis = MakeAnalysis();
  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 2049U, 0U, &analysis) ==
        ASC_OCCUPANCY_INSUFFICIENT_CAPACITY);
  CHECK((analysis.violated_constraints & ASC_OCCUPANCY_CONSTRAINT_THREADS) != 0U);
  CHECK((analysis.violated_constraints & ASC_OCCUPANCY_CONSTRAINT_LAUNCH_BOUND) != 0U);
  return true;
}

bool TestUbAccountingAndSpillRisk() {
  const AscOccupancyDeviceProperties device = MakeDevice();
  AscKernelResourceUsage resources = MakeResources();
  AscOccupancyAnalysis analysis = MakeAnalysis();

  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 256U, 0U, &analysis) ==
        ASC_OCCUPANCY_SUCCESS);
  CHECK(!analysis.ub_capacity_check_complete);
  CHECK(analysis.known_ub_headroom_bytes == 1024U);
  CHECK(!analysis.has_register_spill);

  resources.static_ub_bytes_known = true;
  analysis = MakeAnalysis();
  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 256U, 1024U, &analysis) ==
        ASC_OCCUPANCY_SUCCESS);
  CHECK(analysis.ub_capacity_check_complete);
  CHECK(analysis.known_ub_headroom_bytes == 0U);

  analysis = MakeAnalysis();
  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 256U, 1025U, &analysis) ==
        ASC_OCCUPANCY_INSUFFICIENT_CAPACITY);
  CHECK((analysis.violated_constraints & ASC_OCCUPANCY_CONSTRAINT_UB) != 0U);

  resources.static_ub_bytes_known = true;
  resources.static_ub_bytes = 400U;
  resources.stack_size_bytes = 16U;
  analysis = MakeAnalysis();
  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 256U, 624U, &analysis) ==
        ASC_OCCUPANCY_SUCCESS);
  CHECK(analysis.ub_capacity_check_complete);
  CHECK(analysis.known_ub_headroom_bytes == 0U);
  CHECK(analysis.has_register_spill);

  analysis = MakeAnalysis();
  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 256U, 625U, &analysis) ==
        ASC_OCCUPANCY_INSUFFICIENT_CAPACITY);
  CHECK((analysis.violated_constraints & ASC_OCCUPANCY_CONSTRAINT_UB) != 0U);
  CHECK(analysis.resident_blocks_per_vector_core == 0U);

  resources.static_ub_bytes_known = false;
  resources.static_ub_bytes = 900U;
  analysis = MakeAnalysis();
  CHECK(ascOccupancyAnalyzeKernel(&device, &resources, 256U, 1024U, &analysis) ==
        ASC_OCCUPANCY_SUCCESS);
  CHECK(!analysis.ub_capacity_check_complete);
  CHECK(analysis.known_ub_headroom_bytes == 0U);
  return true;
}

bool TestCandidateEnumeration() {
  const AscOccupancyDeviceProperties device = MakeDevice();
  AscKernelResourceUsage resources = MakeResources(1024U, 32U);
  resources.stack_size_bytes = 8U;

  size_t required_count = 0U;
  CHECK(ascOccupancyEnumerateLaunchBounds(&device, &resources, nullptr,
                                          &required_count) == ASC_OCCUPANCY_SUCCESS);
  CHECK(required_count == 3U);

  AscLaunchBoundsCandidate too_small[2] = {};
  size_t too_small_capacity = 2U;
  CHECK(ascOccupancyEnumerateLaunchBounds(&device, &resources, too_small,
                                          &too_small_capacity) ==
        ASC_OCCUPANCY_INSUFFICIENT_CAPACITY);
  CHECK(too_small_capacity == 3U);

  AscLaunchBoundsCandidate launch_candidates[3] = {};
  size_t capacity = 3U;
  CHECK(ascOccupancyEnumerateLaunchBounds(&device, &resources, launch_candidates,
                                          &capacity) == ASC_OCCUPANCY_SUCCESS);
  CHECK(capacity == 3U);
  CHECK(launch_candidates[0].launch_bounds == 256U);
  CHECK(launch_candidates[1].launch_bounds == 512U);
  CHECK(launch_candidates[2].launch_bounds == 1024U);
  for (const AscLaunchBoundsCandidate& candidate : launch_candidates) {
    CHECK(candidate.requires_benchmark);
  }
  CHECK(!launch_candidates[2].requires_recompile);

  resources = MakeResources(2048U, 16U);
  AscOccupancyCandidates block_candidates = ASC_OCCUPANCY_CANDIDATES_INIT;
  CHECK(ascOccupancyEnumerateBlockCandidates(&device, &resources,
                                              &block_candidates) == ASC_OCCUPANCY_SUCCESS);
  CHECK(block_candidates.max_block_threads_under_known_constraints == 2048U);
  CHECK(block_candidates.candidate_count == 4U);
  CHECK(block_candidates.candidate_block_threads[0] == 256U);
  CHECK(block_candidates.candidate_block_threads[1] == 512U);
  CHECK(block_candidates.candidate_block_threads[2] == 1024U);
  CHECK(block_candidates.candidate_block_threads[3] == 2048U);
  CHECK(block_candidates.benchmark_required_for_optimum);

  resources.static_ub_bytes_known = false;
  resources.static_ub_bytes = 1025U;
  block_candidates = ASC_OCCUPANCY_CANDIDATES_INIT;
  CHECK(ascOccupancyEnumerateBlockCandidates(&device, &resources,
                                              &block_candidates) == ASC_OCCUPANCY_SUCCESS);
  CHECK(block_candidates.max_block_threads_under_known_constraints == 2048U);
  return true;
}

bool TestNonEndpointLaunchBoundsCandidatesWithSpill() {
  const AscOccupancyDeviceProperties device = MakeDevice();
  const uint32_t current_bounds[] = {257U, 513U, 1025U};
  const uint32_t expected_counts[] = {2U, 3U, 4U};
  const uint32_t expected_bounds[][4] = {
      {256U, 257U, 0U, 0U},
      {256U, 512U, 513U, 0U},
      {256U, 512U, 1024U, 1025U},
  };

  for (size_t test_case = 0U; test_case < 3U; ++test_case) {
    AscKernelResourceUsage resources = MakeResources(current_bounds[test_case], 16U);
    resources.stack_size_bytes = 8U;
    AscLaunchBoundsCandidate candidates[4] = {};
    size_t count = 4U;
    CHECK(ascOccupancyEnumerateLaunchBounds(&device, &resources, candidates, &count) ==
          ASC_OCCUPANCY_SUCCESS);
    CHECK(count == expected_counts[test_case]);
    for (size_t index = 0U; index < count; ++index) {
      CHECK(candidates[index].launch_bounds == expected_bounds[test_case][index]);
      CHECK(candidates[index].requires_recompile ==
            (candidates[index].launch_bounds != current_bounds[test_case]));
    }
  }
  return true;
}

bool TestNonEndpointLaunchBoundsCandidatesWithoutSpill() {
  const AscOccupancyDeviceProperties device = MakeDevice();
  const uint32_t current_bounds[] = {257U, 513U, 1025U};
  const uint32_t registers[] = {64U, 32U, 16U};
  const uint32_t expected_bounds[][2] = {
      {257U, 512U},
      {513U, 1024U},
      {1025U, 2048U},
  };

  for (size_t test_case = 0U; test_case < 3U; ++test_case) {
    AscKernelResourceUsage resources =
        MakeResources(current_bounds[test_case], registers[test_case]);
    AscLaunchBoundsCandidate candidates[2] = {};
    size_t count = 2U;
    CHECK(ascOccupancyEnumerateLaunchBounds(&device, &resources, candidates, &count) ==
          ASC_OCCUPANCY_SUCCESS);
    CHECK(count == 2U);
    for (size_t index = 0U; index < count; ++index) {
      CHECK(candidates[index].launch_bounds == expected_bounds[test_case][index]);
      CHECK(candidates[index].requires_recompile == (index != 0U));
    }
  }
  return true;
}

bool TestMinimumLaunchBoundsWithoutSpillNeedsFiveCandidates() {
  const AscOccupancyDeviceProperties device = MakeDevice();
  const AscKernelResourceUsage resources = MakeResources(1U, 16U);

  size_t required_count = 0U;
  CHECK(ascOccupancyEnumerateLaunchBounds(&device, &resources, nullptr,
                                          &required_count) == ASC_OCCUPANCY_SUCCESS);
  CHECK(required_count == 5U);

  AscLaunchBoundsCandidate too_small[4] = {};
  size_t too_small_capacity = 4U;
  CHECK(ascOccupancyEnumerateLaunchBounds(&device, &resources, too_small,
                                          &too_small_capacity) ==
        ASC_OCCUPANCY_INSUFFICIENT_CAPACITY);
  CHECK(too_small_capacity == 5U);

  const uint32_t expected_bounds[] = {1U, 256U, 512U, 1024U, 2048U};
  AscLaunchBoundsCandidate candidates[5] = {};
  size_t capacity = 5U;
  CHECK(ascOccupancyEnumerateLaunchBounds(&device, &resources, candidates,
                                          &capacity) == ASC_OCCUPANCY_SUCCESS);
  CHECK(capacity == 5U);
  for (size_t index = 0U; index < capacity; ++index) {
    CHECK(candidates[index].launch_bounds == expected_bounds[index]);
  }
  return true;
}

}  // namespace

int main() {
  const bool passed = TestHostOnlyDeviceDiscoveryIsExplicitlyUnsupported() &&
                      TestRegisterLimitBoundaries() &&
                      TestLaunchConstraintsAndTailWarpOccupancy() &&
                      TestUbAccountingAndSpillRisk() &&
                      TestCandidateEnumeration() &&
                      TestNonEndpointLaunchBoundsCandidatesWithSpill() &&
                      TestNonEndpointLaunchBoundsCandidatesWithoutSpill() &&
                      TestMinimumLaunchBoundsWithoutSpillNeedsFiveCandidates();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
