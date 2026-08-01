#pragma once

#include <acl/acl.h>

#include "benchmark_runner.h"

#include <cstdint>
#include <string>
#include <vector>

extern "C" __global__ __launch_bounds__(ASC_OCCUPANCY_LAUNCH_BOUNDS)
void launch_geometry_gather_strided_kernel(const float* input, const uint32_t* index,
                                           float* output, uint32_t count) {
  const uint32_t first = blockIdx.x * blockDim.x + threadIdx.x;
  const uint32_t stride = gridDim.x * blockDim.x;
  for (uint32_t item = first; item < count; item += stride) {
    output[item] = input[index[item]] + 1.0F;
  }
}

class LaunchGeometryAdapter {
 public:
  bool initialize(std::string* error) {
    input_.resize(kElements);
    index_.resize(kElements);
    output_.resize(kElements);
    for (uint32_t item = 0; item < kElements; ++item) {
      input_[item] = static_cast<float>(item) * 0.25F;
      index_[item] = kElements - 1U - item;
    }
    if (aclInit(nullptr) != ACL_SUCCESS || aclrtSetDevice(0) != ACL_SUCCESS ||
        aclrtCreateStream(&stream_) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void**>(&input_device_), bytes(), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void**>(&index_device_), kElements * sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMalloc(reinterpret_cast<void**>(&output_device_), bytes(), ACL_MEM_MALLOC_HUGE_FIRST) != ACL_SUCCESS ||
        aclrtMemcpy(input_device_, bytes(), input_.data(), bytes(), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS ||
        aclrtMemcpy(index_device_, kElements * sizeof(uint32_t), index_.data(),
                    kElements * sizeof(uint32_t), ACL_MEMCPY_HOST_TO_DEVICE) != ACL_SUCCESS) {
      *error = "ACL initialization or input upload failed";
      return false;
    }
    return true;
  }
  void shutdown() {
    if (input_device_ != nullptr) aclrtFree(input_device_);
    if (index_device_ != nullptr) aclrtFree(index_device_);
    if (output_device_ != nullptr) aclrtFree(output_device_);
    if (stream_ != nullptr) aclrtDestroyStream(stream_);
    aclrtResetDevice(0);
    aclFinalize();
  }
  void* stream() const { return stream_; }
  const char* benchmark_name() const { return "launch_geometry"; }
  const char* variant_name() const { return "lb2048"; }
  uint64_t work_items() const { return kElements; }
  const AscKernelResourceUsage& resource_usage() const {
    return kLaunch_geometry_lb2048_occupancy_benchResources;
  }
  std::vector<asc_occupancy::LaunchGeometry> candidates() const {
    return {{4, 2048}, {8, 2048}, {16, 1024}, {32, 512}, {64, 256}};
  }
  bool reset_iteration_state(std::string*) { return true; }
  bool launch(const asc_occupancy::LaunchGeometry& geometry, std::string*) {
    launch_geometry_gather_strided_kernel<<<geometry.grid_blocks, geometry.block_threads, 0, stream_>>>(
        input_device_, index_device_, output_device_, kElements);
    return true;
  }
  bool synchronize(std::string* error) {
    if (aclrtSynchronizeStream(stream_) != ACL_SUCCESS) {
      *error = "aclrtSynchronizeStream failed";
      return false;
    }
    return true;
  }
  bool validate(std::string* error) {
    if (!synchronize(error) || aclrtMemcpy(output_.data(), bytes(), output_device_, bytes(),
                                           ACL_MEMCPY_DEVICE_TO_HOST) != ACL_SUCCESS) {
      *error = "result download failed";
      return false;
    }
    for (uint32_t item = 0; item < kElements; ++item) {
      const float expected = input_[index_[item]] + 1.0F;
      if (output_[item] != expected) {
        *error = "first mismatch at index " + std::to_string(item);
        return false;
      }
    }
    return true;
  }

 private:
  static constexpr uint32_t kElements = 16384;
  size_t bytes() const { return kElements * sizeof(float); }
  aclrtStream stream_ = nullptr;
  float* input_device_ = nullptr;
  uint32_t* index_device_ = nullptr;
  float* output_device_ = nullptr;
  std::vector<float> input_;
  std::vector<uint32_t> index_;
  std::vector<float> output_;
};
