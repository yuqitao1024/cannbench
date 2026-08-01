#pragma once

#include <acl/acl.h>
#include <simt_api/asc_simt.h>

#include "benchmark_runner.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

#if defined(REGISTER_SPILL_LB1024) == defined(REGISTER_SPILL_LB512)
#error "Select exactly one register-spill variant"
#endif

extern "C" __global__ __launch_bounds__(ASC_OCCUPANCY_LAUNCH_BOUNDS)
void register_spill_sincos_kernel(const float* input, float* output_sin,
                                  float* output_cos, uint32_t count) {
  const uint32_t block_start = blockIdx.x * blockDim.x * 16U;
  for (uint32_t iteration = 0; iteration < 16U; ++iteration) {
    const uint32_t item = block_start + iteration * blockDim.x + threadIdx.x;
    if (item < count) sincosf(input[item], output_sin + item, output_cos + item);
  }
}

class RegisterSpillAdapter {
 public:
  bool initialize(std::string* error) {
    input_.resize(kElements);
    output_sin_.resize(kElements);
    output_cos_.resize(kElements);
    golden_sin_.resize(kElements);
    golden_cos_.resize(kElements);
    for (uint32_t item = 0; item < kElements; ++item) {
      input_[item] = static_cast<float>(item % 1024U) * 0.001F;
      golden_sin_[item] = std::sin(input_[item]);
      golden_cos_[item] = std::cos(input_[item]);
    }
    if (!check(aclInit(nullptr), "aclInit", error)) return false;
    acl_initialized_ = true;
    if (!check(aclrtSetDevice(0), "aclrtSetDevice", error)) return false;
    device_set_ = true;
    if (!check(aclrtCreateStream(&stream_), "aclrtCreateStream", error) ||
        !check(aclrtMalloc(reinterpret_cast<void**>(&input_device_), bytes(),
                          ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(input)", error) ||
        !check(aclrtMalloc(reinterpret_cast<void**>(&output_sin_device_), bytes(),
                          ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(output_sin)", error) ||
        !check(aclrtMalloc(reinterpret_cast<void**>(&output_cos_device_), bytes(),
                          ACL_MEM_MALLOC_HUGE_FIRST),
               "aclrtMalloc(output_cos)", error) ||
        !check(aclrtMemcpy(input_device_, bytes(), input_.data(), bytes(),
                          ACL_MEMCPY_HOST_TO_DEVICE),
               "aclrtMemcpy(input)", error)) {
      return false;
    }
    return true;
  }

  void shutdown() {
    if (output_cos_device_ != nullptr) aclrtFree(output_cos_device_);
    if (output_sin_device_ != nullptr) aclrtFree(output_sin_device_);
    if (input_device_ != nullptr) aclrtFree(input_device_);
    if (stream_ != nullptr) aclrtDestroyStream(stream_);
    if (device_set_) aclrtResetDevice(0);
    if (acl_initialized_) aclFinalize();
  }

  void* stream() const { return stream_; }
  const char* benchmark_name() const { return "register_spill"; }
  const char* variant_name() const {
#if defined(REGISTER_SPILL_LB1024)
    return "lb1024";
#else
    return "lb512";
#endif
  }
  uint64_t work_items() const { return kElements; }
  const AscKernelResourceUsage& resource_usage() const {
#if defined(REGISTER_SPILL_LB1024)
    return kRegister_spill_lb1024_occupancy_benchResources;
#else
    return kRegister_spill_lb512_occupancy_benchResources;
#endif
  }
  std::vector<asc_occupancy::LaunchGeometry> candidates() const {
    return {{kGridBlocks, kBlockThreads}};
  }
  bool reset_iteration_state(std::string*) { return true; }
  bool launch(const asc_occupancy::LaunchGeometry& geometry, std::string*) {
    register_spill_sincos_kernel<<<geometry.grid_blocks, geometry.block_threads, 0, stream_>>>(
        input_device_, output_sin_device_, output_cos_device_, kElements);
    return true;
  }
  bool synchronize(std::string* error) {
    return check(aclrtSynchronizeStream(stream_), "aclrtSynchronizeStream", error);
  }
  bool validate(std::string* error) {
    if (!check(aclrtMemcpy(output_sin_.data(), bytes(), output_sin_device_, bytes(),
                           ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy(output_sin)", error) ||
        !check(aclrtMemcpy(output_cos_.data(), bytes(), output_cos_device_, bytes(),
                           ACL_MEMCPY_DEVICE_TO_HOST),
               "aclrtMemcpy(output_cos)", error)) {
      return false;
    }
    size_t first_mismatch = kElements;
    double max_absolute_error = 0.0;
    double max_relative_error = 0.0;
    for (uint32_t item = 0; item < kElements; ++item) {
      compare(output_sin_[item], golden_sin_[item], item, &first_mismatch,
              &max_absolute_error, &max_relative_error);
      compare(output_cos_[item], golden_cos_[item], item, &first_mismatch,
              &max_absolute_error, &max_relative_error);
    }
    if (first_mismatch != kElements) {
      *error = "first mismatch at index " + std::to_string(first_mismatch) +
               ", max_abs=" + std::to_string(max_absolute_error) +
               ", max_rel=" + std::to_string(max_relative_error);
      return false;
    }
    return true;
  }

 private:
  static constexpr uint32_t kGridBlocks = 48;
  static constexpr uint32_t kBlockThreads = 512;
  static constexpr uint32_t kElements = kGridBlocks * kBlockThreads * 16U;
  size_t bytes() const { return static_cast<size_t>(kElements) * sizeof(float); }
  static bool check(aclError status, const char* operation, std::string* error) {
    if (status == ACL_SUCCESS) return true;
    *error = std::string(operation) + " failed with ACL error " + std::to_string(status);
    const char* recent = aclGetRecentErrMsg();
    if (recent != nullptr) *error += ": " + std::string(recent);
    return false;
  }
  static void compare(float actual, float expected, size_t item, size_t* first_mismatch,
                      double* max_absolute_error, double* max_relative_error) {
    const double absolute_error = std::abs(static_cast<double>(actual) - expected);
    const double relative_error =
        absolute_error / std::max(std::abs(static_cast<double>(expected)), 1.0e-6);
    *max_absolute_error = std::max(*max_absolute_error, absolute_error);
    *max_relative_error = std::max(*max_relative_error, relative_error);
    if (absolute_error > 1.0e-4 && *first_mismatch == kElements) *first_mismatch = item;
  }

  bool acl_initialized_ = false;
  bool device_set_ = false;
  aclrtStream stream_ = nullptr;
  float* input_device_ = nullptr;
  float* output_sin_device_ = nullptr;
  float* output_cos_device_ = nullptr;
  std::vector<float> input_;
  std::vector<float> output_sin_;
  std::vector<float> output_cos_;
  std::vector<float> golden_sin_;
  std::vector<float> golden_cos_;
};
