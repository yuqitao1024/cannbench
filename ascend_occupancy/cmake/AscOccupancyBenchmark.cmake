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

function(asc_occupancy_generate_benchmark_runner)
  cmake_parse_arguments(ARG "" "OUTPUT;PYTHON_EXECUTABLE;AGGREGATOR" "" ${ARGN})
  foreach(required OUTPUT PYTHON_EXECUTABLE AGGREGATOR)
    if(NOT ARG_${required})
      message(FATAL_ERROR "asc_occupancy_generate_benchmark_runner requires ${required}")
    endif()
  endforeach()
  get_property(benchmark_targets GLOBAL PROPERTY ASC_OCCUPANCY_BENCHMARK_TARGETS)
  get_property(result_stems GLOBAL PROPERTY ASC_OCCUPANCY_BENCHMARK_RESULT_STEMS)
  list(LENGTH benchmark_targets target_count)
  list(LENGTH result_stems stem_count)
  if(target_count EQUAL 0 OR NOT target_count EQUAL stem_count)
    message(FATAL_ERROR "Occupancy benchmark target/result registration is inconsistent")
  endif()
  set(script
    "#!/usr/bin/env bash\nset -u\nstatus=0\nresults_dir=\"\${ASC_OCCUPANCY_RESULTS_DIR:-occupancy-results}\"\nmkdir -p \"$results_dir\"\n")
  set(aggregate_inputs "")
  math(EXPR last_index "${target_count} - 1")
  foreach(index RANGE ${last_index})
    list(GET benchmark_targets ${index} benchmark_target)
    list(GET result_stems ${index} result_stem)
    string(APPEND script
      "\"$<TARGET_FILE:${benchmark_target}>\" \"$@\" --json \"$results_dir/${result_stem}.json\" --csv \"$results_dir/${result_stem}.csv\" || status=$?\n")
    string(APPEND aggregate_inputs " \"$results_dir/${result_stem}.json\"")
  endforeach()
  string(APPEND script
    "\"${ARG_PYTHON_EXECUTABLE}\" \"${ARG_AGGREGATOR}\" --json-out \"$results_dir/occupancy-summary.json\" --csv-out \"$results_dir/occupancy-summary.csv\"${aggregate_inputs} || status=$?\n")
  string(APPEND script "exit \"$status\"\n")
  file(GENERATE OUTPUT "${ARG_OUTPUT}" CONTENT "${script}"
    FILE_PERMISSIONS
      OWNER_READ OWNER_WRITE OWNER_EXECUTE
      GROUP_READ GROUP_EXECUTE
      WORLD_READ WORLD_EXECUTE)
endfunction()

function(asc_occupancy_add_benchmark)
  cmake_parse_arguments(ARG "" "NAME;RESULT_STEM;ADAPTER_HEADER;ADAPTER_TYPE;KERNEL_SYMBOL_REGEX;LAUNCH_BOUNDS;STATIC_UB_BYTES"
    "INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS;COMPILE_OPTIONS" ${ARGN})
  foreach(required NAME RESULT_STEM ADAPTER_HEADER ADAPTER_TYPE KERNEL_SYMBOL_REGEX LAUNCH_BOUNDS)
    if(NOT ARG_${required})
      message(FATAL_ERROR "asc_occupancy_add_benchmark requires ${required}")
    endif()
  endforeach()
  if(NOT ARG_RESULT_STEM MATCHES "^[A-Za-z0-9_.-]+$")
    message(FATAL_ERROR "RESULT_STEM must contain only filename-safe ASCII characters")
  endif()
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
  set_property(GLOBAL APPEND PROPERTY ASC_OCCUPANCY_BENCHMARK_TARGETS "${ARG_NAME}")
  set_property(GLOBAL APPEND PROPERTY ASC_OCCUPANCY_BENCHMARK_RESULT_STEMS
    "${ARG_RESULT_STEM}")
endfunction()
