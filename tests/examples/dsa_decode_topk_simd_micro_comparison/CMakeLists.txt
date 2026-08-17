cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
find_package(ASC REQUIRED)
project(dsa_decode_topk_simd_micro_comparison LANGUAGES ASC CXX)

add_executable(dsa_decode_topk_simt_v2_baseline simt_v2_topk.asc)
add_executable(dsa_decode_topk_simd_micro simd_micro_topk.asc)
foreach(target dsa_decode_topk_simt_v2_baseline dsa_decode_topk_simd_micro)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
