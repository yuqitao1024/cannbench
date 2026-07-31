cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(simt_add_traversal_repro LANGUAGES ASC CXX)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture")

set(ADD_TRAVERSAL_SOURCE main.asc)

add_executable(simt_add_traversal_direct ${ADD_TRAVERSAL_SOURCE})
add_executable(simt_add_traversal_hybrid ${ADD_TRAVERSAL_SOURCE})

set_target_properties(
    simt_add_traversal_direct
    simt_add_traversal_hybrid
    PROPERTIES LINKER_LANGUAGE ASC
)

target_compile_definitions(simt_add_traversal_direct PRIVATE
    ADD_TRAVERSAL_DIRECT_SIMT=1
)
target_compile_definitions(simt_add_traversal_hybrid PRIVATE
    ADD_TRAVERSAL_HYBRID=1
)

target_compile_options(simt_add_traversal_direct PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES} --enable-simt>
)
target_compile_options(simt_add_traversal_hybrid PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)
