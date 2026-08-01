#ifndef FAKE_ACL_ACL_H_
#define FAKE_ACL_ACL_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int32_t aclError;

typedef enum aclrtDevAttr {
  ACL_DEV_ATTR_NPU_ARCH,
  ACL_DEV_ATTR_VECTOR_CORE_NUM,
  ACL_DEV_ATTR_WARP_SIZE,
  ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE,
  ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE,
  ACL_DEV_ATTR_MAX_THREADS_PER_BLOCK,
} aclrtDevAttr;

#define ACL_SUCCESS 0

aclError aclrtGetDeviceInfo(uint32_t device_id, aclrtDevAttr attr, int64_t* value);

#ifdef __cplusplus
}
#endif

#endif  // FAKE_ACL_ACL_H_
