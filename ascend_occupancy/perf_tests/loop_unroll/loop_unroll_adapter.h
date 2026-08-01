#pragma once

#include <acl/acl.h>

#include "benchmark_runner.h"

#include <cstdint>
#include <string>
#include <vector>

#if defined(LOOP_UNROLL_LOOP) == defined(LOOP_UNROLL_MANUAL)
#error "Select exactly one loop-unroll variant"
#endif

#if defined(LOOP_UNROLL_LOOP)
extern "C" __global__ __launch_bounds__(ASC_OCCUPANCY_LAUNCH_BOUNDS)
void loop_unroll_loop_kernel(const float* input_x, const float* input_y, float* output) {
  const uint32_t global_thread = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t iteration = 0; iteration < 4U; ++iteration) {
    const uint32_t item = global_thread + iteration * stride;
    output[item] = input_x[item] + input_y[item];
  }
}
#else
extern "C" __global__ __launch_bounds__(ASC_OCCUPANCY_LAUNCH_BOUNDS)
void loop_unroll_manual_kernel(const float* input_x, const float* input_y, float* output) {
  const uint32_t item0 = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;
  const uint32_t item1 = item0 + stride;
  const uint32_t item2 = item1 + stride;
  const uint32_t item3 = item2 + stride;
  const float x0 = input_x[item0];
  const float x1 = input_x[item1];
  const float x2 = input_x[item2];
  const float x3 = input_x[item3];
  const float y0 = input_y[item0];
  const float y1 = input_y[item1];
  const float y2 = input_y[item2];
  const float y3 = input_y[item3];
  output[item0] = x0 + y0;
  output[item1] = x1 + y1;
  output[item2] = x2 + y2;
  output[item3] = x3 + y3;
}
#endif

class LoopUnrollAdapter {
 public:
  bool initialize(std::string* error) {
    input_x_.resize(kElements);
    input_y_.resize(kElements);
    output_.resize(kElements);
    for (uint32_t item = 0; item < kElements; ++item) {
      input_x_[item] = static_cast<float>(item % 1024U) * 0.25F;
      input_y_[item] = static_cast<float>(item % 17U) * 0.5F;
    }
    if (!check(aclInit(nullptr), "aclInit", error)) return false;
    acl_initialized_ = true;
    if (!check(aclrtSetDevice(0), "aclrtSetDevice", error)) return false;
    device_set_ = true;
    if (!check(aclrtCreateStream(&stream_), "aclrtCreateStream", error) ||
        !check(aclrtMalloc(reinterpret_cast<void**>(&input_x_device_), bytes(),
                          ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(input_x)", error) ||
        !check(aclrtMalloc(reinterpret_cast<void**>(&input_y_device_), bytes(),
                          ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(input_y)", error) ||
        !check(aclrtMalloc(reinterpret_cast<void**>(&output_device_), bytes(),
                          ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(output)", error) ||
        !check(aclrtMemcpy(input_x_device_, bytes(), input_x_.data(), bytes(),
                          ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy(input_x)", error) ||
        !check(aclrtMemcpy(input_y_device_, bytes(), input_y_.data(), bytes(),
                          ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy(input_y)", error)) {
      return false;
    }
    return true;
  }

  void shutdown() {
    if (output_device_ != nullptr) aclrtFree(output_device_);
    if (input_y_device_ != nullptr) aclrtFree(input_y_device_);
    if (input_x_device_ != nullptr) aclrtFree(input_x_device_);
    if (stream_ != nullptr) aclrtDestroyStream(stream_);
    if (device_set_) aclrtResetDevice(0);
    if (acl_initialized_) aclFinalize();
  }

  void* stream() const { return stream_; }
  const char* benchmark_name() const { return "loop_unroll"; }
  const char* variant_name() const {
#if defined(LOOP_UNROLL_LOOP)
    return "loop";
#else
    return "manual";
#endif
  }
  uint64_t work_items() const { return kElements; }
  const AscKernelResourceUsage& resource_usage() const {
#if defined(LOOP_UNROLL_LOOP)
    return kLoop_unroll_loop_occupancy_benchResources;
#else
    return kLoop_unroll_manual_occupancy_benchResources;
#endif
  }
  std::vector<asc_occupancy::LaunchGeometry> candidates() const {
    return {{kGridBlocks, kBlockThreads}};
  }
  bool reset_iteration_state(std::string*) { return true; }
  bool launch(const asc_occupancy::LaunchGeometry& geometry, std::string*) {
#if defined(LOOP_UNROLL_LOOP)
    loop_unroll_loop_kernel<<<geometry.grid_blocks, geometry.block_threads, 0, stream_>>>(
        input_x_device_, input_y_device_, output_device_);
#else
    loop_unroll_manual_kernel<<<geometry.grid_blocks, geometry.block_threads, 0, stream_>>>(
        input_x_device_, input_y_device_, output_device_);
#endif
    return true;
  }
  bool synchronize(std::string* error) {
    return check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream", error);
  }
  bool validate(std::string* error) {
    if (!check(aclrtMemcpy(output_.data(), bytes(), output_device_, bytes(),
                           ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy(output)", error)) {
      return false;
    }
    for (uint32_t item = 0; item < kElements; ++item) {
      const float expected = input_x_[item] + input_y_[item];
      if (output_[item] != expected) {
        *error = "first mismatch at index " + std::to_string(item) +
                 ", actual=" + std::to_string(output_[item]) +
                 ", expected=" + std::to_string(expected);
        return false;
      }
    }
    return true;
  }

 private:
  static constexpr uint32_t kGridBlocks = 64;
  static constexpr uint32_t kBlockThreads = 2048;
  static constexpr uint32_t kElements = kGridBlocks * kBlockThreads * 4U;
  size_t bytes() const { return static_cast<size_t>(kElements) * sizeof(float); }
  static bool check(aclError status, const char* operation, std::string* error) {
    if (status == ACL_SUCCESS) return true;
    *error = std::string(operation) + " failed with ACL error " + std::to_string(status);
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr) *error += ": " + std::string(recent);
    return false;
  }

  bool acl_initialized_ = false;
  bool device_set_ = false;
  aclrtStream stream_ = nullptr;
  float* input_x_device_ = nullptr;
  float* input_y_device_ = nullptr;
  float* output_device_ = nullptr;
  std::vector<float> input_x_;
  std::vector<float> input_y_;
  std::vector<float> output_;
};
