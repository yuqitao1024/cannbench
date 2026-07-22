cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(lightning_indexer_bf16_vf_compiler_repro LANGUAGES ASC CXX)

add_executable(fp16_vf_control
    fp16_vf_control.asc
)

add_executable(bf16_vf_repro
    bf16_vf_repro.asc
)

add_executable(bf16_vf_mixed_abi_repro
    bf16_vf_mixed_abi_repro.asc
)

foreach(target_name IN ITEMS fp16_vf_control bf16_vf_repro bf16_vf_mixed_abi_repro)
    set_target_properties(${target_name} PROPERTIES
        LINKER_LANGUAGE ASC
    )

    target_compile_options(${target_name} PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    )

    target_include_directories(${target_name} PRIVATE
        $ENV{ASCEND_HOME_PATH}/include
        $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/include
    )
endforeach()
