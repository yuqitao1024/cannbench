cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
find_package(ASC REQUIRED)
project(dsa_decode_softmax_comparison LANGUAGES ASC CXX)

add_executable(dsa_decode_softmax_vllm_ascend vllm_ascend_softmax.asc)
add_executable(dsa_decode_softmax_simt_v2 simt_v2_softmax.asc)

foreach(target IN ITEMS dsa_decode_softmax_vllm_ascend dsa_decode_softmax_simt_v2)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>)
    target_link_libraries(${target} PRIVATE m)
endforeach()
