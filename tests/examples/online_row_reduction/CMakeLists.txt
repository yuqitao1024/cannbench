cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=three scan, 1=online stats, 2=tiled online")
if (NOT SCENARIO_NUM MATCHES "^[0-2]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0, 1, or 2")
endif()

find_package(ASC REQUIRED)

project(online_row_reduction LANGUAGES ASC CXX)

add_executable(online_row_reduction
    online_row_reduction.asc
)

target_compile_definitions(online_row_reduction PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(online_row_reduction PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_link_libraries(online_row_reduction PRIVATE m)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=three scan, 1=online stats, 2=tiled online)")
