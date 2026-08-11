cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=repeated padded bitonic merge, 1=streaming bounded heap")
if (NOT SCENARIO_NUM MATCHES "^[0-1]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0 or 1")
endif()

find_package(ASC REQUIRED)

project(topk_algorithm_selection LANGUAGES ASC CXX)

add_executable(topk_algorithm_selection
    topk_algorithm_selection.asc
)

target_compile_definitions(topk_algorithm_selection PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(topk_algorithm_selection PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=repeated padded bitonic, 1=streaming bounded heap)")
