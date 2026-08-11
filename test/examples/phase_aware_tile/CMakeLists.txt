cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=conservative, 1=phase reuse, 2=oversized outer tile")
if (NOT SCENARIO_NUM MATCHES "^[0-2]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0, 1, or 2")
endif()

find_package(ASC REQUIRED)

project(phase_aware_tile LANGUAGES ASC CXX)

add_executable(phase_aware_tile
    phase_aware_tile.asc
)

target_compile_definitions(phase_aware_tile PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(phase_aware_tile PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=conservative, 1=phase reuse, 2=oversized outer tile)")
