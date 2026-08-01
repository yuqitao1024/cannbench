#include "fake_acl.h"

namespace {

constexpr int kAttributeCount = 6;
aclError g_statuses[kAttributeCount] = {};
int64_t g_values[kAttributeCount] = {};
uint32_t g_last_device_id = UINT32_MAX;
uint32_t g_device_info_call_count = 0U;

int AttributeIndex(aclrtDevAttr attr) {
  return static_cast<int>(attr);
}

}  // namespace

void fakeAclReset(void) {
  for (int index = 0; index < kAttributeCount; ++index) {
    g_statuses[index] = ACL_SUCCESS;
    g_values[index] = 0;
  }
  g_last_device_id = UINT32_MAX;
  g_device_info_call_count = 0U;
}

void fakeAclSetDeviceInfo(aclrtDevAttr attr, int64_t value) {
  g_values[AttributeIndex(attr)] = value;
}

void fakeAclSetDeviceInfoStatus(aclrtDevAttr attr, aclError status) {
  g_statuses[AttributeIndex(attr)] = status;
}

uint32_t fakeAclLastDeviceId(void) {
  return g_last_device_id;
}

uint32_t fakeAclDeviceInfoCallCount(void) {
  return g_device_info_call_count;
}

extern "C" aclError aclrtGetDeviceInfo(uint32_t device_id,
                                         aclrtDevAttr attr,
                                         int64_t* value) {
  ++g_device_info_call_count;
  g_last_device_id = device_id;
  const int index = AttributeIndex(attr);
  if (g_statuses[index] != ACL_SUCCESS) {
    return g_statuses[index];
  }
  *value = g_values[index];
  return ACL_SUCCESS;
}
