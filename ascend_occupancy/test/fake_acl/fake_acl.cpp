#include "fake_acl.h"

namespace {

constexpr int kAttributeCount = 6;
aclError g_statuses[kAttributeCount] = {};
int64_t g_values[kAttributeCount] = {};
int32_t g_last_device_id = -1;

int AttributeIndex(aclrtDevAttr attr) {
  return static_cast<int>(attr);
}

}  // namespace

void fakeAclReset(void) {
  for (int index = 0; index < kAttributeCount; ++index) {
    g_statuses[index] = ACL_SUCCESS;
    g_values[index] = 0;
  }
  g_last_device_id = -1;
}

void fakeAclSetDeviceInfo(aclrtDevAttr attr, int64_t value) {
  g_values[AttributeIndex(attr)] = value;
}

void fakeAclSetDeviceInfoStatus(aclrtDevAttr attr, aclError status) {
  g_statuses[AttributeIndex(attr)] = status;
}

int32_t fakeAclLastDeviceId(void) {
  return g_last_device_id;
}

extern "C" aclError aclrtGetDeviceInfo(int32_t device_id,
                                         aclrtDevAttr attr,
                                         int64_t* value) {
  g_last_device_id = device_id;
  const int index = AttributeIndex(attr);
  if (g_statuses[index] != ACL_SUCCESS) {
    return g_statuses[index];
  }
  *value = g_values[index];
  return ACL_SUCCESS;
}
