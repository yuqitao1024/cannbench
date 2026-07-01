#pragma once

#include <torch/library.h>

#include <acl/acl.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace aten_softmax_v2::simt {

constexpr int64_t kMaxThreads = 1024;
constexpr int64_t kDefaultMaxBlocksPerCoreCap = 4;
constexpr double kDefaultUbufSafetyRatio = 0.5;
constexpr int64_t kUnboundedUbufLimit = 1 << 30;

struct SimtOccupancyDeviceProps {
  int64_t device;
  int64_t aicpu_core_count;
  int64_t aicore_core_count;
  int64_t cube_core_count;
  int64_t vector_core_count;
  int64_t warp_size;
  int64_t max_threads_per_vector_core;
  int64_t ubuf_per_vector_core;
  int64_t total_global_mem_size;
  int64_t l2_cache_size;
};

struct SimtOccupancyEstimate {
  int64_t blocks_per_core_by_threads;
  int64_t blocks_per_core_by_ubuf;
  int64_t blocks_per_core_cap;
  int64_t blocks_per_core;
  int64_t max_active_blocks;
  int64_t effective_ubuf_bytes;
};

struct SpatialLaunchConfig {
  int64_t block_x;
  int64_t block_y;
  int64_t block_threads;
  int64_t dynamic_ubuf_bytes;
  int64_t grid_x;
  int64_t grid_y;
  SimtOccupancyEstimate estimate;
  SimtOccupancyDeviceProps device_props;
};

inline int64_t ceil_div(int64_t a, int64_t b) {
  TORCH_CHECK(b > 0, "divisor must be positive");
  if (a <= 0) {
    return 0;
  }
  return (a + b - 1) / b;
}

inline void check_acl_error(aclError ret, const char* func) {
  TORCH_CHECK(
      ret == ACL_SUCCESS,
      func,
      " failed with acl error code ",
      static_cast<int>(ret));
}

class ScopedAclInit {
 public:
  ScopedAclInit() {
    const aclError ret = aclInit(nullptr);
    if (ret == ACL_SUCCESS) {
      owns_init_ = true;
      return;
    }
    TORCH_CHECK(
        ret == ACL_ERROR_REPEAT_INITIALIZE,
        "aclInit failed with acl error code ",
        static_cast<int>(ret));
  }

  ~ScopedAclInit() {
    if (!owns_init_) {
      return;
    }
    const aclError ret = aclFinalize();
    if (ret != ACL_SUCCESS && ret != ACL_ERROR_REPEAT_FINALIZE) {
      return;
    }
  }

 private:
  bool owns_init_ = false;
};

inline int64_t get_device_attr(int64_t device, aclrtDevAttr attr) {
  int64_t value = 0;
  check_acl_error(
      aclrtGetDeviceInfo(static_cast<uint32_t>(device), attr, &value),
      "aclrtGetDeviceInfo");
  return value;
}

inline SimtOccupancyDeviceProps query_simt_occupancy_device_props(
    int64_t device) {
  ScopedAclInit guard;
  check_acl_error(
      aclrtSetDevice(static_cast<int32_t>(device)), "aclrtSetDevice");
  return SimtOccupancyDeviceProps{
      device,
      get_device_attr(device, ACL_DEV_ATTR_AICPU_CORE_NUM),
      get_device_attr(device, ACL_DEV_ATTR_AICORE_CORE_NUM),
      get_device_attr(device, ACL_DEV_ATTR_CUBE_CORE_NUM),
      get_device_attr(device, ACL_DEV_ATTR_VECTOR_CORE_NUM),
      get_device_attr(device, ACL_DEV_ATTR_WARP_SIZE),
      get_device_attr(device, ACL_DEV_ATTR_MAX_THREAD_PER_VECTOR_CORE),
      get_device_attr(device, ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE),
      get_device_attr(device, ACL_DEV_ATTR_TOTAL_GLOBAL_MEM_SIZE),
      get_device_attr(device, ACL_DEV_ATTR_L2_CACHE_SIZE),
  };
}

inline std::vector<int64_t> flatten_device_props(
    const SimtOccupancyDeviceProps& props) {
  return {
      props.device,
      props.aicpu_core_count,
      props.aicore_core_count,
      props.cube_core_count,
      props.vector_core_count,
      props.warp_size,
      props.max_threads_per_vector_core,
      props.ubuf_per_vector_core,
      props.total_global_mem_size,
      props.l2_cache_size,
  };
}

inline std::vector<int64_t> spatial_block_size(
    int64_t dim_size,
    int64_t inner_size) {
  TORCH_CHECK(dim_size >= 0, "dim_size must be non-negative");
  TORCH_CHECK(inner_size >= 0, "inner_size must be non-negative");

  const int64_t inner_threads = std::min(inner_size, kMaxThreads);
  int64_t dim_threads = 1;
  if (inner_threads <= 64 && dim_size >= 64) {
    while (inner_threads * dim_threads <= kMaxThreads &&
           dim_threads <= dim_size) {
      dim_threads *= 2;
    }
    dim_threads /= 2;
  }
  return {dim_threads, inner_threads};
}

inline std::vector<int64_t> spatial_grid_size(
    int64_t block_x,
    int64_t block_y,
    int64_t max_active_blocks,
    int64_t outer_size,
    int64_t inner_size) {
  TORCH_CHECK(block_x > 0, "block_x must be positive");
  TORCH_CHECK(block_y > 0, "block_y must be positive");

  if (max_active_blocks <= 0 || outer_size <= 0 || inner_size <= 0) {
    return {0, 0};
  }

  int64_t inner_blocks = ceil_div(inner_size, block_y);
  inner_blocks = std::min(inner_blocks, max_active_blocks);
  if (inner_blocks == 0) {
    return {0, 0};
  }

  int64_t outer_blocks = ceil_div(max_active_blocks, inner_blocks);
  outer_blocks = std::min(outer_blocks, outer_size);
  return {outer_blocks, inner_blocks};
}

inline SimtOccupancyEstimate estimate_spatial_active_blocks(
    const SimtOccupancyDeviceProps& props,
    int64_t block_threads,
    int64_t dynamic_ubuf_bytes,
    int64_t max_blocks_per_core_cap = kDefaultMaxBlocksPerCoreCap,
    double ubuf_safety_ratio = kDefaultUbufSafetyRatio) {
  TORCH_CHECK(block_threads > 0, "block_threads must be positive");
  TORCH_CHECK(
      dynamic_ubuf_bytes >= 0, "dynamic_ubuf_bytes must be non-negative");
  TORCH_CHECK(
      max_blocks_per_core_cap > 0,
      "max_blocks_per_core_cap must be positive");
  TORCH_CHECK(
      ubuf_safety_ratio > 0.0 && ubuf_safety_ratio <= 1.0,
      "ubuf_safety_ratio must be in (0, 1]");

  const int64_t blocks_per_core_by_threads =
      std::max<int64_t>(1, props.max_threads_per_vector_core / block_threads);
  const int64_t effective_ubuf_bytes =
      static_cast<int64_t>(props.ubuf_per_vector_core * ubuf_safety_ratio);
  const int64_t blocks_per_core_by_ubuf =
      dynamic_ubuf_bytes == 0
      ? kUnboundedUbufLimit
      : std::max<int64_t>(1, effective_ubuf_bytes / dynamic_ubuf_bytes);

  const int64_t blocks_per_core = std::max<int64_t>(
      1,
      std::min(
          std::min(blocks_per_core_by_threads, blocks_per_core_by_ubuf),
          max_blocks_per_core_cap));
  const int64_t max_active_blocks = blocks_per_core * props.vector_core_count;

  return SimtOccupancyEstimate{
      blocks_per_core_by_threads,
      blocks_per_core_by_ubuf,
      max_blocks_per_core_cap,
      blocks_per_core,
      max_active_blocks,
      effective_ubuf_bytes,
  };
}

inline std::vector<int64_t> flatten_launch_config(
    const SpatialLaunchConfig& config) {
  auto values = std::vector<int64_t>{
      config.block_x,
      config.block_y,
      config.block_threads,
      config.dynamic_ubuf_bytes,
      config.grid_x,
      config.grid_y,
      config.estimate.blocks_per_core_by_threads,
      config.estimate.blocks_per_core_by_ubuf,
      config.estimate.blocks_per_core_cap,
      config.estimate.blocks_per_core,
      config.estimate.max_active_blocks,
      config.estimate.effective_ubuf_bytes,
  };
  auto props = flatten_device_props(config.device_props);
  values.insert(values.end(), props.begin(), props.end());
  return values;
}

inline SpatialLaunchConfig spatial_launch_config_impl(
    int64_t outer_size,
    int64_t dim_size,
    int64_t inner_size,
    int64_t accscalar_size,
    int64_t device,
    int64_t max_blocks_per_core_cap = kDefaultMaxBlocksPerCoreCap,
    double ubuf_safety_ratio = kDefaultUbufSafetyRatio) {
  TORCH_CHECK(outer_size >= 0, "outer_size must be non-negative");
  TORCH_CHECK(dim_size >= 0, "dim_size must be non-negative");
  TORCH_CHECK(inner_size >= 0, "inner_size must be non-negative");
  TORCH_CHECK(accscalar_size > 0, "accscalar_size must be positive");

  const auto block = spatial_block_size(dim_size, inner_size);
  const int64_t block_x = block[0];
  const int64_t block_y = block[1];
  const int64_t block_threads = block_x * block_y;
  const int64_t dynamic_ubuf_bytes =
      block_x == 1 ? 0 : block_threads * accscalar_size;
  const auto props = query_simt_occupancy_device_props(device);
  const auto estimate = estimate_spatial_active_blocks(
      props,
      block_threads,
      dynamic_ubuf_bytes,
      max_blocks_per_core_cap,
      ubuf_safety_ratio);
  const auto grid = spatial_grid_size(
      block_x,
      block_y,
      estimate.max_active_blocks,
      outer_size,
      inner_size);

  return SpatialLaunchConfig{
      block_x,
      block_y,
      block_threads,
      dynamic_ubuf_bytes,
      grid[0],
      grid[1],
      estimate,
      props,
  };
}

} // namespace aten_softmax_v2::simt
