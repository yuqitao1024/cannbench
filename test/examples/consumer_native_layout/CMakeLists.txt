cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")
set(SCENARIO_NUM "0" CACHE STRING "Scenario: 0=row-major plus explicit pack, 1=consumer-native producer")
if (NOT SCENARIO_NUM MATCHES "^[0-1]$")
    message(FATAL_ERROR "SCENARIO_NUM must be 0 or 1")
endif()

find_package(ASC REQUIRED)

project(consumer_native_layout LANGUAGES ASC CXX)

add_executable(consumer_native_layout
    consumer_native_layout.asc
)

target_compile_definitions(consumer_native_layout PRIVATE
    SCENARIO_NUM=${SCENARIO_NUM}
)

target_compile_options(consumer_native_layout PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

message(STATUS "SCENARIO_NUM=${SCENARIO_NUM} (0=explicit pack, 1=native layout)")
