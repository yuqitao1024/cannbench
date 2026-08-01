if(NOT DEFINED ASC_OCCUPANCY_SOURCE_DIR OR
   NOT DEFINED ASC_OCCUPANCY_TEST_BINARY_DIR)
  message(FATAL_ERROR "Missing performance-test contract inputs")
endif()

function(assert_file_contains path)
  if(NOT EXISTS "${path}")
    message(FATAL_ERROR "Required performance test file is missing: ${path}")
  endif()
  file(READ "${path}" contents)
  foreach(expected IN LISTS ARGN)
    string(FIND "${contents}" "${expected}" index)
    if(index EQUAL -1)
      message(FATAL_ERROR "${path} is missing required contract text: ${expected}")
    endif()
  endforeach()
endfunction()

assert_file_contains(
  "${ASC_OCCUPANCY_SOURCE_DIR}/CMakeLists.txt"
  "add_subdirectory(perf_tests/register_spill)"
  "add_subdirectory(perf_tests/launch_geometry)"
  "add_subdirectory(perf_tests/loop_unroll)")

assert_file_contains(
  "${ASC_OCCUPANCY_SOURCE_DIR}/perf_tests/register_spill/CMakeLists.txt"
  "NAME register_spill_lb1024_occupancy_bench"
  "NAME register_spill_lb512_occupancy_bench"
  "LAUNCH_BOUNDS 1024"
  "LAUNCH_BOUNDS 512"
  "REGISTER_SPILL_LB1024"
  "REGISTER_SPILL_LB512"
  "RESULT_STEM register_spill-lb1024"
  "RESULT_STEM register_spill-lb512"
  "register_spill_sincos_kernel_simt_entry")

assert_file_contains(
  "${ASC_OCCUPANCY_SOURCE_DIR}/perf_tests/loop_unroll/CMakeLists.txt"
  "NAME loop_unroll_loop_occupancy_bench"
  "NAME loop_unroll_manual_occupancy_bench"
  "LOOP_UNROLL_LOOP"
  "LOOP_UNROLL_MANUAL"
  "RESULT_STEM loop_unroll-loop"
  "RESULT_STEM loop_unroll-manual"
  "loop_unroll_loop_kernel_simt_entry"
  "loop_unroll_manual_kernel_simt_entry")

file(REMOVE_RECURSE "${ASC_OCCUPANCY_TEST_BINARY_DIR}")
set(aggregate_source "${ASC_OCCUPANCY_TEST_BINARY_DIR}/source")
set(aggregate_build "${ASC_OCCUPANCY_TEST_BINARY_DIR}/build")
file(MAKE_DIRECTORY "${aggregate_source}")
file(WRITE "${aggregate_source}/main.cpp" "int main() { return 0; }\n")
file(WRITE "${aggregate_source}/CMakeLists.txt"
  "cmake_minimum_required(VERSION 3.20)\n"
  "project(aggregate_contract LANGUAGES CXX)\n"
  "include(\"${ASC_OCCUPANCY_SOURCE_DIR}/cmake/AscOccupancyBenchmark.cmake\")\n"
  "add_executable(benchmark_a main.cpp)\n"
  "add_executable(benchmark_b main.cpp)\n"
  "set_property(GLOBAL PROPERTY ASC_OCCUPANCY_BENCHMARK_TARGETS benchmark_a benchmark_b)\n"
  "set_property(GLOBAL PROPERTY ASC_OCCUPANCY_BENCHMARK_RESULT_STEMS result-a result-b)\n"
  "asc_occupancy_generate_benchmark_runner(\n"
  "  OUTPUT \"\${CMAKE_BINARY_DIR}/run-all.sh\"\n"
  "  PYTHON_EXECUTABLE \"${CMAKE_COMMAND}\"\n"
  "  AGGREGATOR \"${ASC_OCCUPANCY_SOURCE_DIR}/perf_tests/common/aggregate_results.py\")\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -S "${aggregate_source}" -B "${aggregate_build}"
  RESULT_VARIABLE aggregate_result
  OUTPUT_VARIABLE aggregate_output
  ERROR_VARIABLE aggregate_error)
if(NOT aggregate_result EQUAL 0)
  message(FATAL_ERROR
    "Aggregate runner configuration failed:\n${aggregate_output}\n${aggregate_error}")
endif()
assert_file_contains(
  "${aggregate_build}/run-all.sh"
  "benchmark_a"
  "benchmark_b"
  "result-a.json"
  "result-b.json"
  "aggregate_results.py"
  "occupancy-summary.json"
  "occupancy-summary.csv"
  "status=0"
  "exit \"$status\"")
