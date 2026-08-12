cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(vcv_per_aiv_ub_layout LANGUAGES ASC CXX)

add_executable(vcv_per_aiv_ub_layout
    vcv_per_aiv_ub_layout.asc
)

target_compile_options(vcv_per_aiv_ub_layout PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_link_libraries(vcv_per_aiv_ub_layout PRIVATE m)
