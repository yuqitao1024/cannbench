#ifndef ASC_OCCUPANCY_TEST_FAKE_ACL_H_
#define ASC_OCCUPANCY_TEST_FAKE_ACL_H_

#include <acl/acl.h>

void fakeAclReset(void);
void fakeAclSetDeviceInfo(aclrtDevAttr attr, int64_t value);
void fakeAclSetDeviceInfoStatus(aclrtDevAttr attr, aclError status);
uint32_t fakeAclLastDeviceId(void);
uint32_t fakeAclDeviceInfoCallCount(void);

#endif  // ASC_OCCUPANCY_TEST_FAKE_ACL_H_
