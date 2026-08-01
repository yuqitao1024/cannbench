#pragma once

#include <acl/acl.h>

#include <string>

namespace asc_occupancy {

class AclEventRuntime {
 public:
  bool create_event(void** event, std::string* error) {
    aclrtEvent created = nullptr;
    if (aclrtCreateEvent(&created) != ACL_SUCCESS) {
      *error = "aclrtCreateEvent failed";
      return false;
    }
    *event = created;
    return true;
  }
  bool record_event(void* event, void* stream, std::string* error) {
    if (aclrtRecordEvent(static_cast<aclrtEvent>(event), static_cast<aclrtStream>(stream)) != ACL_SUCCESS) {
      *error = "aclrtRecordEvent failed";
      return false;
    }
    return true;
  }
  bool synchronize_event(void* event, std::string* error) {
    if (aclrtSynchronizeEvent(static_cast<aclrtEvent>(event)) != ACL_SUCCESS) {
      *error = "aclrtSynchronizeEvent failed";
      return false;
    }
    return true;
  }
  bool elapsed_us(double* value, void* start, void* end, std::string* error) {
    float elapsed_ms = 0.0F;
    if (aclrtEventElapsedTime(&elapsed_ms, static_cast<aclrtEvent>(start),
                              static_cast<aclrtEvent>(end)) != ACL_SUCCESS) {
      *error = "aclrtEventElapsedTime failed";
      return false;
    }
    *value = static_cast<double>(elapsed_ms) * 1000.0;
    return true;
  }
  void destroy_event(void* event) {
    if (event != nullptr) aclrtDestroyEvent(static_cast<aclrtEvent>(event));
  }
};

}  // namespace asc_occupancy
