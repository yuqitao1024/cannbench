cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=single-block, 1=sharded hierarchical reduction")
if (NOT SCENARIO_NUM MATCHES "^[0-1]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0 or 1")
endif()

find_package(ASC REQUIRED)

project(sharded_histogram_topk LANGUAGES ASC CXX)

add_executable(sharded_histogram_topk
    sharded_histogram_topk.asc
)

target_compile_definitions(sharded_histogram_topk PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(sharded_histogram_topk PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=single-block, 1=64-shard + reduction)")
