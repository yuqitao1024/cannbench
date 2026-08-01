#include <ascend_occupancy/asc_occupancy.h>

#include "fake_acl.h"

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

void SetDav3510DeviceInfo() {
  fakeAclReset();
  fakeAclSetDeviceInfo(ACL_DEV_ATTR_NPU_ARCH, 3510);
  fakeAclSetDeviceInfo(ACL_DEV_ATTR_VECTOR_CORE_NUM, 64);
  fakeAclSetDeviceInfo(ACL_DEV_ATTR_WARP_SIZE, 32);
  fakeAclSetDeviceInfo(ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE, 2048);
  fakeAclSetDeviceInfo(ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, 221184);
  fakeAclSetDeviceInfo(ACL_DEV_ATTR_MAX_THREADS_PER_BLOCK, 2048);
}

bool TestAclDevicePropertiesAreMeasured() {
  SetDav3510DeviceInfo();
  AscOccupancyDeviceProperties properties = ASC_OCCUPANCY_DEVICE_PROPERTIES_INIT;
  CHECK(ascOccupancyGetDeviceProperties(37, &properties) == ASC_OCCUPANCY_SUCCESS);
  CHECK(fakeAclLastDeviceId() == 37);
  CHECK(properties.device_id == 37);
  CHECK(properties.vector_core_count == 64U);
  CHECK(properties.warp_size == 32U);
  CHECK(properties.max_threads_per_vector_core == 2048U);
  CHECK(properties.ub_bytes_per_vector_core == 221184U);
  return true;
}

bool TestAclRejectsNegativeDeviceIdBeforeRuntimeQuery() {
  SetDav3510DeviceInfo();
  AscOccupancyDeviceProperties properties = ASC_OCCUPANCY_DEVICE_PROPERTIES_INIT;
  CHECK(ascOccupancyGetDeviceProperties(-1, &properties) ==
        ASC_OCCUPANCY_UNSUPPORTED_DEVICE);
  CHECK(fakeAclDeviceInfoCallCount() == 0U);
  return true;
}

bool TestAclQueryFailureIsResourceDataMissing() {
  SetDav3510DeviceInfo();
  fakeAclSetDeviceInfoStatus(ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE, 107000);
  AscOccupancyDeviceProperties properties = ASC_OCCUPANCY_DEVICE_PROPERTIES_INIT;
  CHECK(ascOccupancyGetDeviceProperties(0, &properties) ==
        ASC_OCCUPANCY_RESOURCE_DATA_MISSING);
  return true;
}

bool TestAclRejectsNonDav3510PositiveDeviceId() {
  SetDav3510DeviceInfo();
  fakeAclSetDeviceInfo(ACL_DEV_ATTR_NPU_ARCH, 3511);
  AscOccupancyDeviceProperties properties = ASC_OCCUPANCY_DEVICE_PROPERTIES_INIT;
  CHECK(ascOccupancyGetDeviceProperties(12, &properties) ==
        ASC_OCCUPANCY_UNSUPPORTED_DEVICE);
  return true;
}

bool TestAclRejectsInconsistentThreadLimits() {
  SetDav3510DeviceInfo();
  fakeAclSetDeviceInfo(ACL_DEV_ATTR_MAX_THREADS_PER_BLOCK, 1024);
  AscOccupancyDeviceProperties properties = ASC_OCCUPANCY_DEVICE_PROPERTIES_INIT;
  CHECK(ascOccupancyGetDeviceProperties(5, &properties) ==
        ASC_OCCUPANCY_RESOURCE_DATA_INCONSISTENT);
  return true;
}

}  // namespace

int main() {
  const bool passed = TestAclDevicePropertiesAreMeasured() &&
                      TestAclRejectsNegativeDeviceIdBeforeRuntimeQuery() &&
                      TestAclQueryFailureIsResourceDataMissing() &&
                      TestAclRejectsNonDav3510PositiveDeviceId() &&
                      TestAclRejectsInconsistentThreadLimits();
  return passed ? EXIT_SUCCESS : EXIT_FAILURE;
}
