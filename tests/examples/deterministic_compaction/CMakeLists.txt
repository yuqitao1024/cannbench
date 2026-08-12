cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=atomic scan, 1=packed prefix")
if (NOT SCENARIO_NUM MATCHES "^[0-1]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0 or 1")
endif()

find_package(ASC REQUIRED)

project(deterministic_compaction LANGUAGES ASC CXX)

add_executable(deterministic_compaction
    deterministic_compaction.asc
)

target_compile_definitions(deterministic_compaction PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(deterministic_compaction PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=atomic scan, 1=packed prefix)")
