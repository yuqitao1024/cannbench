cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")

find_package(ASC REQUIRED)

project(msopprof_two_vf_repro LANGUAGES ASC CXX)

add_library(two_vf_kernels SHARED
    copy_vf.asc
    add_one_vf.asc
)

add_library(single_vf_kernel SHARED
    copy_vf.asc
)

add_executable(two_vf_repro
    two_vf_main.asc
)

add_executable(single_vf_control
    single_vf_main.asc
)

target_link_libraries(two_vf_repro PRIVATE two_vf_kernels)
target_link_libraries(single_vf_control PRIVATE single_vf_kernel)

foreach(target_name IN ITEMS
        two_vf_kernels
        single_vf_kernel
        two_vf_repro
        single_vf_control)
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
