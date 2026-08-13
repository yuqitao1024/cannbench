cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
find_package(ASC REQUIRED)
project(dsa_decode_topk_comparison LANGUAGES ASC CXX)

add_executable(dsa_decode_topk_vllm_ascend vllm_ascend_topk.asc)
add_executable(dsa_decode_topk_simt_v2 simt_v2_topk.asc)
foreach(target dsa_decode_topk_vllm_ascend dsa_decode_topk_simt_v2)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
endforeach()
