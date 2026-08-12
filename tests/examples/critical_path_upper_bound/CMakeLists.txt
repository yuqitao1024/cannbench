cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=baseline candidate, 1=counterfactual candidate")
if (NOT SCENARIO_NUM MATCHES "^[0-1]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0 or 1")
endif()

find_package(ASC REQUIRED)

project(critical_path_upper_bound LANGUAGES ASC CXX)

add_executable(critical_path_upper_bound
    critical_path_upper_bound.asc
)

target_compile_definitions(critical_path_upper_bound PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(critical_path_upper_bound PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=baseline candidate, 1=counterfactual candidate)")
