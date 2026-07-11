cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(softmax_persistent_store_compiler_repro LANGUAGES ASC CXX)

add_executable(single_tu_switch_repro
    single_tu_switch_repro.asc
)

add_executable(split_tu_switch_repro
    split_tu_main.asc
    split_tu_1024_helper.asc
)

foreach(target_name IN ITEMS single_tu_switch_repro split_tu_switch_repro)
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
