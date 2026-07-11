#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "acl/acl.h"
#include "c_api/asc_simd.h"
#include "simt_api/asc_fp16.h"
#include "simt_api/asc_simt.h"

namespace softmax_compiler_repro {

constexpr int32_t kDeviceId = 0;
constexpr uint32_t kOuterSize = 32;
constexpr uint32_t kDimSize = 1024;
constexpr uint32_t kBlockX = 32;
constexpr uint32_t kBlockY = 8;

#define CHECK_ACL(expr)                                                                                 \
    do {                                                                                                \
        aclError ret = (expr);                                                                          \
        if (ret != ACL_SUCCESS) {                                                                       \
            std::cerr << "ACL call failed: " #expr ", ret=" << static_cast<int32_t>(ret) << std::endl; \
            std::exit(1);                                                                               \
        }                                                                                               \
    } while (0)

template <typename T>
__simt_callee__ inline T device_zero();

template <>
__simt_callee__ inline float device_zero<float>()
{
    return 0.0f;
}

template <>
__simt_callee__ inline __fp16 device_zero<__fp16>()
{
    return (__fp16)0;
}

template <typename scalar_t, typename outscalar_t, int64_t kThreadsPerBlock>
__simt_vf__ inline void debug_persistent_store_vf(
    __gm__ outscalar_t* output, __gm__ const scalar_t* input, int64_t outer_size, int64_t dim_size)
{
    output[0] = device_zero<outscalar_t>();
    (void)input;
    (void)outer_size;
    (void)dim_size;
}

template <typename scalar_t, typename outscalar_t, int64_t kThreadsPerBlock>
__global__ __vector__ void debug_persistent_store_kernel(
    __gm__ outscalar_t* output,
    __gm__ const scalar_t* input,
    int64_t outer_size,
    int64_t dim_size,
    int64_t block_x,
    int64_t block_y)
{
    asc_vf_call<debug_persistent_store_vf<scalar_t, outscalar_t, kThreadsPerBlock>>(
        dim3(block_x, block_y), output, input, outer_size, dim_size);
}

template <typename scalar_t, typename outscalar_t, int64_t kThreadsPerBlock>
inline void launch_debug_persistent_store_kernel(
    const scalar_t* input_ptr,
    outscalar_t* output_ptr,
    int64_t outer_size,
    int64_t dim_size,
    int64_t block_x,
    int64_t block_y,
    int64_t grid_x,
    aclrtStream stream)
{
    debug_persistent_store_kernel<scalar_t, outscalar_t, kThreadsPerBlock>
        <<<grid_x, 0, stream>>>(output_ptr, input_ptr, outer_size, dim_size, block_x, block_y);
}

void instantiate_1024_kernel_but_do_not_run(
    __fp16* input_device, __fp16* output_device, int64_t outer_size, int64_t dim_size, aclrtStream stream);

} // namespace softmax_compiler_repro
