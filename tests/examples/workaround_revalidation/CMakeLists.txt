cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=single-block workaround, 1=fixed multi-block ownership")
if (NOT SCENARIO_NUM MATCHES "^[0-1]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0 or 1")
endif()

find_package(ASC REQUIRED)
project(workaround_revalidation LANGUAGES ASC CXX)

add_executable(workaround_revalidation workaround_revalidation.asc)
target_compile_definitions(workaround_revalidation PRIVATE SCENARIO_NUM=${SCENARIO_NUM})
target_compile_options(workaround_revalidation PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=workaround, 1=root-cause fix)")
