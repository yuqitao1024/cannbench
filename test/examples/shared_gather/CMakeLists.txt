cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=independent consumer gathers, 1=single producer shared gather")
if (NOT SCENARIO_NUM MATCHES "^[0-1]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0 or 1")
endif()

find_package(ASC REQUIRED)

project(shared_gather LANGUAGES ASC CXX)

add_executable(shared_gather
    shared_gather.asc
)

target_compile_definitions(shared_gather PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(shared_gather PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=independent consumers, 1=shared producer)")
