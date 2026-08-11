cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=generic, 1=behavior buckets, 2=exact-shape overfit")
if (NOT SCENARIO_NUM MATCHES "^[0-2]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0, 1, or 2")
endif()

find_package(ASC REQUIRED)

project(row_width_dispatch LANGUAGES ASC CXX)

add_executable(row_width_dispatch
    row_width_dispatch.asc
)

target_compile_definitions(row_width_dispatch PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(row_width_dispatch PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=generic, 1=bucketed, 2=exact overfit)")
