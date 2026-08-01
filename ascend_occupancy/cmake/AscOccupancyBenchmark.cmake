include_guard(GLOBAL)
include(CMakeParseArguments)
include("${CMAKE_CURRENT_LIST_DIR}/AscOccupancyProbe.cmake")

function(asc_occupancy_configure_benchmark_main)
  cmake_parse_arguments(ARG "" "OUTPUT;ADAPTER_HEADER;ADAPTER_TYPE;RESOURCE_HEADER" "" ${ARGN})
  foreach(required OUTPUT ADAPTER_HEADER ADAPTER_TYPE RESOURCE_HEADER)
    if(NOT ARG_${required})
      message(FATAL_ERROR "asc_occupancy_configure_benchmark_main requires ${required}")
    endif()
  endforeach()
  set(ASC_OCCUPANCY_BENCHMARK_ADAPTER_HEADER "${ARG_ADAPTER_HEADER}")
  set(ASC_OCCUPANCY_BENCHMARK_ADAPTER_TYPE "${ARG_ADAPTER_TYPE}")
  set(ASC_OCCUPANCY_BENCHMARK_RESOURCE_HEADER "${ARG_RESOURCE_HEADER}")
  get_filename_component(output_directory "${ARG_OUTPUT}" DIRECTORY)
  file(MAKE_DIRECTORY "${output_directory}")
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/benchmark_main.asc.in" "${ARG_OUTPUT}" @ONLY)
endfunction()

function(asc_occupancy_add_benchmark)
  cmake_parse_arguments(ARG "" "NAME;ADAPTER_HEADER;ADAPTER_TYPE;KERNEL_SYMBOL_REGEX;LAUNCH_BOUNDS;STATIC_UB_BYTES"
    "INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS;COMPILE_OPTIONS" ${ARGN})
  foreach(required NAME ADAPTER_HEADER ADAPTER_TYPE KERNEL_SYMBOL_REGEX LAUNCH_BOUNDS)
    if(NOT ARG_${required})
      message(FATAL_ERROR "asc_occupancy_add_benchmark requires ${required}")
    endif()
  endforeach()
  set(benchmark_directory "${CMAKE_CURRENT_BINARY_DIR}/occupancy-benchmarks/${ARG_NAME}")
  set(benchmark_source "${benchmark_directory}/main.asc")
  set(resource_header "${CMAKE_CURRENT_BINARY_DIR}/occupancy-generated/${ARG_NAME}_occupancy_resources.h")
  asc_occupancy_configure_benchmark_main(
    OUTPUT "${benchmark_source}" ADAPTER_HEADER "${ARG_ADAPTER_HEADER}"
    ADAPTER_TYPE "${ARG_ADAPTER_TYPE}" RESOURCE_HEADER "${resource_header}")
  get_filename_component(adapter_directory "${ARG_ADAPTER_HEADER}" DIRECTORY)
  set(static_ub_argument "")
  if(DEFINED ARG_STATIC_UB_BYTES)
    set(static_ub_argument STATIC_UB_BYTES "${ARG_STATIC_UB_BYTES}")
  endif()
  asc_occupancy_add_kernel_variant(
    NAME "${ARG_NAME}" SOURCE "${benchmark_source}"
    KERNEL_SYMBOL_REGEX "${ARG_KERNEL_SYMBOL_REGEX}" LAUNCH_BOUNDS "${ARG_LAUNCH_BOUNDS}"
    ${static_ub_argument}
    INCLUDE_DIRECTORIES
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../include"
      "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/../perf_tests/common"
      "${adapter_directory}"
      ${ARG_INCLUDE_DIRECTORIES}
    COMPILE_DEFINITIONS ${ARG_COMPILE_DEFINITIONS}
    COMPILE_OPTIONS ${ARG_COMPILE_OPTIONS})
  file(GENERATE OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/run-all-occupancy-benchmarks.sh"
    CONTENT "#!/usr/bin/env bash\nset -eu\n\"$<TARGET_FILE:${ARG_NAME}>\" \"$@\"\n")
endfunction()
