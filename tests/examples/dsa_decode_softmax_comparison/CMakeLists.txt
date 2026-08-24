cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")
find_package(ASC REQUIRED)
project(dsa_decode_softmax_comparison LANGUAGES ASC CXX)

add_executable(dsa_decode_softmax_vllm_ascend vllm_ascend_softmax.asc)
add_executable(dsa_decode_softmax_simt_v2 simt_v2_softmax.asc)
add_executable(dsa_decode_softmax_simt_v3 simt_v3_softmax.asc)
add_executable(dsa_decode_softmax_simt_v4 simt_v4_softmax.asc)
add_executable(dsa_decode_softmax_simt_v4a simt_v4a_softmax.asc)
add_executable(dsa_decode_softmax_simt_v4b simt_v4b_softmax.asc)
add_executable(dsa_decode_softmax_simt_v5 simt_v4_register_opt.asc)
add_executable(dsa_decode_softmax_simt_v6 simt_v6_softmax.asc)
add_executable(dsa_decode_softmax_simt_v7 simt_v7_softmax.asc)
add_executable(dsa_decode_softmax_simt_v8 simt_v8_softmax.asc)
add_executable(dsa_decode_softmax_simt_v4_sim_probe simt_v4_sim_probe.asc)
add_executable(dsa_decode_softmax_simt_v4_copy simt_v4_softmax_copy.asc)

foreach(target IN ITEMS dsa_decode_softmax_vllm_ascend dsa_decode_softmax_simt_v2
                        dsa_decode_softmax_simt_v3
                        dsa_decode_softmax_simt_v4
                        dsa_decode_softmax_simt_v4a
                        dsa_decode_softmax_simt_v4b
                        dsa_decode_softmax_simt_v5
                        dsa_decode_softmax_simt_v6
                        dsa_decode_softmax_simt_v7
                        dsa_decode_softmax_simt_v8
                        dsa_decode_softmax_simt_v4_sim_probe
                        dsa_decode_softmax_simt_v4_copy)
    target_compile_options(${target} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}> --cce-res-usage
        -g)
    target_link_libraries(${target} PRIVATE m)
endforeach()
