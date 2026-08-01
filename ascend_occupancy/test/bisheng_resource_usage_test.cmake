if(NOT DEFINED ASC_OCCUPANCY_SOURCE_DIR OR
   NOT DEFINED ASC_OCCUPANCY_TEST_BINARY_DIR)
  message(FATAL_ERROR "Missing CMake/parser test inputs")
endif()

set(module_path "${ASC_OCCUPANCY_SOURCE_DIR}/cmake/ParseBishengResourceUsage.cmake")
set(probe_module_path "${ASC_OCCUPANCY_SOURCE_DIR}/cmake/AscOccupancyProbe.cmake")
include("${probe_module_path}")
file(REMOVE_RECURSE "${ASC_OCCUPANCY_TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${ASC_OCCUPANCY_TEST_BINARY_DIR}")

function(run_parser_case name expected_result input regex expected_symbol expected_stack expected_registers)
  set(case_script "${ASC_OCCUPANCY_TEST_BINARY_DIR}/${name}.cmake")
  file(WRITE "${case_script}"
    "include(\"${module_path}\")\n"
    "set(resource_output [==[${input}]==])\n"
    "asc_occupancy_parse_bisheng_resource_usage(\n"
    "  OUTPUT_PREFIX parsed\n"
    "  RESOURCE_USAGE_OUTPUT \"\${resource_output}\"\n"
    "  KERNEL_SYMBOL_REGEX \"${regex}\")\n"
    "if(NOT \"\${parsed_KERNEL_SYMBOL}\" STREQUAL \"${expected_symbol}\")\n"
    "  message(FATAL_ERROR \"Unexpected symbol: \${parsed_KERNEL_SYMBOL}\")\n"
    "endif()\n"
    "if(NOT \"\${parsed_STACK_SIZE_BYTES}\" STREQUAL \"${expected_stack}\")\n"
    "  message(FATAL_ERROR \"Unexpected stack: \${parsed_STACK_SIZE_BYTES}\")\n"
    "endif()\n"
    "if(NOT \"\${parsed_USED_REGISTER_NUMBER}\" STREQUAL \"${expected_registers}\")\n"
    "  message(FATAL_ERROR \"Unexpected registers: \${parsed_USED_REGISTER_NUMBER}\")\n"
    "endif()\n")
  execute_process(
    COMMAND "${CMAKE_COMMAND}" -P "${case_script}"
    RESULT_VARIABLE result
    OUTPUT_VARIABLE output
    ERROR_VARIABLE error)
  if(NOT result EQUAL expected_result)
    message(FATAL_ERROR
      "${name}: expected result ${expected_result}, got ${result}:\n${output}\n${error}")
  endif()
endfunction()

run_parser_case(
  normal 0
  "[BISHENG] Function properties for alpha_kernel_simt_entry: Stack size: 32 bytes, Used register number: 48\n[BISHENG] Function properties for beta_kernel_simt_entry: Stack size: 0 bytes, Used register number: 5"
  "^alpha_kernel_simt_entry$" alpha_kernel_simt_entry 32 48)
run_parser_case(
  missing 1
  "[BISHENG] Function properties for beta_kernel_simt_entry: Stack size: 0 bytes, Used register number: 5"
  "^alpha_kernel_simt_entry$" "" "" "")
run_parser_case(
  duplicate 1
  "[BISHENG] Function properties for alpha_kernel_simt_entry: Stack size: 0 bytes, Used register number: 5\n[BISHENG] Function properties for alpha_kernel_simt_entry: Stack size: 32 bytes, Used register number: 48"
  "^alpha_kernel_simt_entry$" "" "" "")
run_parser_case(
  negative 1
  "[BISHENG] Function properties for alpha_kernel_simt_entry: Stack size: -1 bytes, Used register number: 48"
  "^alpha_kernel_simt_entry$" "" "" "")
run_parser_case(
  format_change 1
  "[BISHENG] Function properties for alpha_kernel_simt_entry: Stack bytes: 32, Used register number: 48"
  "^alpha_kernel_simt_entry$" "" "" "")

set(contract_script "${ASC_OCCUPANCY_TEST_BINARY_DIR}/contract.cmake")
set(contract_header "${ASC_OCCUPANCY_TEST_BINARY_DIR}/example_occupancy_resources.h")
file(WRITE "${contract_script}"
  "include(\"${probe_module_path}\")\n"
  "asc_occupancy_validate_kernel_variant_arguments(\n"
  "  NAME example SOURCE \"${CMAKE_CURRENT_LIST_FILE}\"\n"
  "  KERNEL_SYMBOL_REGEX \"^example_entry$\" LAUNCH_BOUNDS 512\n"
  "  STATIC_UB_BYTES 4096)\n"
  "asc_occupancy_write_resource_header(\n"
  "  OUTPUT \"${contract_header}\" NAME example KERNEL_SYMBOL example_entry\n"
  "  LAUNCH_BOUNDS 512 USED_REGISTER_NUMBER 48 STACK_SIZE_BYTES 32\n"
  "  STATIC_UB_BYTES 4096 STATIC_UB_BYTES_KNOWN TRUE)\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${contract_script}"
  RESULT_VARIABLE contract_result
  OUTPUT_VARIABLE contract_output
  ERROR_VARIABLE contract_error)
if(NOT contract_result EQUAL 0)
  message(FATAL_ERROR "Input/header contract failed:\n${contract_output}\n${contract_error}")
endif()
file(READ "${contract_header}" contract_header_contents)
foreach(expected_line
    "#include <ascend_occupancy/asc_occupancy.h>"
    "inline constexpr AscKernelResourceUsage kExampleResources"
    "\"example_entry\""
    "    512,"
    "    48,"
    "    32,"
    "    4096,"
    "    true,")
  string(FIND "${contract_header_contents}" "${expected_line}" expected_index)
  if(expected_index EQUAL -1)
    message(FATAL_ERROR "Generated header missed: ${expected_line}")
  endif()
endforeach()

set(escaped_header "${ASC_OCCUPANCY_TEST_BINARY_DIR}/escaped_occupancy_resources.h")
set(escaped_symbol [=[entry\path"quoted]=])
asc_occupancy_write_resource_header(
  OUTPUT "${escaped_header}" NAME escaped KERNEL_SYMBOL "${escaped_symbol}"
  LAUNCH_BOUNDS 512 USED_REGISTER_NUMBER 48 STACK_SIZE_BYTES 32
  STATIC_UB_BYTES 4096 STATIC_UB_BYTES_KNOWN TRUE)
file(READ "${escaped_header}" escaped_header_contents)
set(expected_escaped_symbol [=["entry\\path\"quoted"]=])
string(FIND "${escaped_header_contents}" "${expected_escaped_symbol}"
  escaped_symbol_index)
if(escaped_symbol_index EQUAL -1)
  message(FATAL_ERROR "Generated header did not escape the kernel symbol")
endif()

set(unsafe_symbol_script "${ASC_OCCUPANCY_TEST_BINARY_DIR}/unsafe_symbol.cmake")
file(WRITE "${unsafe_symbol_script}"
  "include(\"${probe_module_path}\")\n"
  "asc_occupancy_write_resource_header(\n"
  "  OUTPUT \"${ASC_OCCUPANCY_TEST_BINARY_DIR}/unsafe.h\" NAME unsafe\n"
  "  KERNEL_SYMBOL [=[unsafe\nsymbol]=]\n"
  "  LAUNCH_BOUNDS 512 USED_REGISTER_NUMBER 48 STACK_SIZE_BYTES 32\n"
  "  STATIC_UB_BYTES 4096 STATIC_UB_BYTES_KNOWN TRUE)\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${unsafe_symbol_script}"
  RESULT_VARIABLE unsafe_symbol_result
  OUTPUT_VARIABLE unsafe_symbol_output
  ERROR_VARIABLE unsafe_symbol_error)
if(unsafe_symbol_result EQUAL 0)
  message(FATAL_ERROR "Unsafe kernel symbol containing a newline was accepted")
endif()

set(simt_disabled_script "${ASC_OCCUPANCY_TEST_BINARY_DIR}/simt_disabled.cmake")
file(WRITE "${simt_disabled_script}"
  "include(\"${probe_module_path}\")\n"
  "set(CMAKE_ASC_COMPILER \"/fake/asc\")\n"
  "set(CMAKE_ASC_ENABLE_SIMT OFF)\n"
  "asc_occupancy_add_kernel_variant(\n"
  "  NAME simt_disabled SOURCE \"${simt_disabled_script}\"\n"
  "  KERNEL_SYMBOL_REGEX \"^simt_disabled$\" LAUNCH_BOUNDS 512)\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${simt_disabled_script}"
  RESULT_VARIABLE simt_disabled_result
  OUTPUT_VARIABLE simt_disabled_output
  ERROR_VARIABLE simt_disabled_error)
if(simt_disabled_result EQUAL 0)
  message(FATAL_ERROR "SIMT-disabled parent project was accepted")
endif()
string(FIND "${simt_disabled_output}${simt_disabled_error}"
  "CMAKE_ASC_ENABLE_SIMT=ON" simt_disabled_error_index)
if(simt_disabled_error_index EQUAL -1)
  message(FATAL_ERROR "SIMT-disabled parent project failed for the wrong reason")
endif()

set(placeholder_contract_script "${ASC_OCCUPANCY_TEST_BINARY_DIR}/placeholder_contract.cmake")
set(placeholder_header "${ASC_OCCUPANCY_TEST_BINARY_DIR}/placeholder_occupancy_resources.h")
file(WRITE "${placeholder_contract_script}"
  "include(\"${probe_module_path}\")\n"
  "asc_occupancy_write_resource_header(\n"
  "  OUTPUT \"${placeholder_header}\" NAME placeholder KERNEL_SYMBOL \"\"\n"
  "  LAUNCH_BOUNDS 512 USED_REGISTER_NUMBER 0 STACK_SIZE_BYTES 0\n"
  "  STATIC_UB_BYTES 0 STATIC_UB_BYTES_KNOWN FALSE)\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${placeholder_contract_script}"
  RESULT_VARIABLE placeholder_contract_result
  OUTPUT_VARIABLE placeholder_contract_output
  ERROR_VARIABLE placeholder_contract_error)
if(NOT placeholder_contract_result EQUAL 0)
  message(FATAL_ERROR
    "Placeholder header generation failed:\n${placeholder_contract_output}\n${placeholder_contract_error}")
endif()
file(READ "${placeholder_header}" placeholder_header_contents)
string(FIND "${placeholder_header_contents}" "\"\"" placeholder_symbol_index)
if(placeholder_symbol_index EQUAL -1)
  message(FATAL_ERROR "Placeholder header did not keep an empty symbol")
endif()

set(parameter_file "${ASC_OCCUPANCY_TEST_BINARY_DIR}/probe_parameters.cmake")
set(parameter_reader "${ASC_OCCUPANCY_TEST_BINARY_DIR}/read_probe_parameters.cmake")
set(probe_source "${ASC_OCCUPANCY_TEST_BINARY_DIR}/source with spaces/kernel source.cpp")
set(probe_target "kernel_probe")
set(probe_architecture "ascend950")
set(probe_launch_bounds "1024")
set(probe_includes
  "${ASC_OCCUPANCY_TEST_BINARY_DIR}/include with spaces"
  "${ASC_OCCUPANCY_TEST_BINARY_DIR}/second include")
set(probe_definitions "MACRO_VALUE=with space" "OTHER_MACRO=plain")
set(probe_options "-O3")
list(APPEND probe_options "$<$<COMPILE_LANGUAGE:ASC>:--define=left\\;right>")
asc_occupancy_write_probe_parameters(
  OUTPUT "${parameter_file}"
  NAME "parameter_test"
  TARGET "${probe_target}"
  SOURCE "${probe_source}"
  ARCHITECTURES "${probe_architecture}"
  ENABLE_SIMT "ON"
  LAUNCH_BOUNDS "${probe_launch_bounds}"
  INCLUDE_DIRECTORIES_VARIABLE probe_includes
  COMPILE_DEFINITIONS_VARIABLE probe_definitions
  COMPILE_OPTIONS_VARIABLE probe_options)
file(WRITE "${parameter_reader}"
  "include(\"${parameter_file}\")\n"
  "function(assert_equal actual expected description)\n"
  "  if(NOT \"\${actual}\" STREQUAL \"\${expected}\")\n"
  "    message(FATAL_ERROR \"\${description}: expected <\${expected}>, got <\${actual}>\")\n"
  "  endif()\n"
  "endfunction()\n"
  "assert_equal(\"\${ASC_OCCUPANCY_PROBE_NAME}\" \"parameter_test\" \"name\")\n"
  "assert_equal(\"\${ASC_OCCUPANCY_PROBE_TARGET}\" \"${probe_target}\" \"target\")\n"
  "assert_equal(\"\${ASC_OCCUPANCY_PROBE_SOURCE}\" \"${probe_source}\" \"source\")\n"
  "assert_equal(\"\${ASC_OCCUPANCY_PROBE_ARCHITECTURES}\" \"${probe_architecture}\" \"architectures\")\n"
  "assert_equal(\"\${ASC_OCCUPANCY_PROBE_ENABLE_SIMT}\" \"ON\" \"enable simt\")\n"
  "assert_equal(\"\${ASC_OCCUPANCY_PROBE_LAUNCH_BOUNDS}\" \"${probe_launch_bounds}\" \"launch bounds\")\n"
  "list(LENGTH ASC_OCCUPANCY_PROBE_INCLUDE_DIRECTORIES include_count)\n"
  "if(NOT include_count EQUAL 2)\n"
  "  message(FATAL_ERROR \"include element count: expected 2, got \${include_count}\")\n"
  "endif()\n"
  "list(GET ASC_OCCUPANCY_PROBE_INCLUDE_DIRECTORIES 0 include_first)\n"
  "list(GET ASC_OCCUPANCY_PROBE_INCLUDE_DIRECTORIES 1 include_second)\n"
  "assert_equal(\"\${include_first}\" \"${ASC_OCCUPANCY_TEST_BINARY_DIR}/include with spaces\" \"first include\")\n"
  "assert_equal(\"\${include_second}\" \"${ASC_OCCUPANCY_TEST_BINARY_DIR}/second include\" \"second include\")\n"
  "list(LENGTH ASC_OCCUPANCY_PROBE_COMPILE_DEFINITIONS definition_count)\n"
  "if(NOT definition_count EQUAL 2)\n"
  "  message(FATAL_ERROR \"definition element count: expected 2, got \${definition_count}\")\n"
  "endif()\n"
  "list(GET ASC_OCCUPANCY_PROBE_COMPILE_DEFINITIONS 0 definition_first)\n"
  "list(GET ASC_OCCUPANCY_PROBE_COMPILE_DEFINITIONS 1 definition_second)\n"
  "assert_equal(\"\${definition_first}\" \"MACRO_VALUE=with space\" \"first definition\")\n"
  "assert_equal(\"\${definition_second}\" \"OTHER_MACRO=plain\" \"second definition\")\n"
  "list(LENGTH ASC_OCCUPANCY_PROBE_COMPILE_OPTIONS option_count)\n"
  "if(NOT option_count EQUAL 2)\n"
  "  message(FATAL_ERROR \"option element count: expected 2, got \${option_count}\")\n"
  "endif()\n"
  "list(GET ASC_OCCUPANCY_PROBE_COMPILE_OPTIONS 0 option_first)\n"
  "list(GET ASC_OCCUPANCY_PROBE_COMPILE_OPTIONS 1 option_second)\n"
  "assert_equal(\"\${option_first}\" \"-O3\" \"first option\")\n"
  "assert_equal(\"\${option_second}\" \"$<$<COMPILE_LANGUAGE:ASC>:--define=left;right>\" \"second option\")\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${parameter_reader}"
  RESULT_VARIABLE parameter_result
  OUTPUT_VARIABLE parameter_output
  ERROR_VARIABLE parameter_error)
if(NOT parameter_result EQUAL 0)
  message(FATAL_ERROR
    "Probe parameter round-trip failed:\n${parameter_output}\n${parameter_error}")
endif()

set(benchmark_module_path "${ASC_OCCUPANCY_SOURCE_DIR}/cmake/AscOccupancyBenchmark.cmake")
set(benchmark_main "${ASC_OCCUPANCY_TEST_BINARY_DIR}/generated_benchmark.asc")
set(benchmark_contract_script "${ASC_OCCUPANCY_TEST_BINARY_DIR}/benchmark_contract.cmake")
file(WRITE "${benchmark_contract_script}"
  "include(\"${benchmark_module_path}\")\n"
  "asc_occupancy_configure_benchmark_main(\n"
  "  OUTPUT \"${benchmark_main}\" ADAPTER_HEADER \"adapter.h\"\n"
  "  ADAPTER_TYPE TestAdapter RESOURCE_HEADER \"test_resources.h\")\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${benchmark_contract_script}"
  RESULT_VARIABLE benchmark_contract_result
  OUTPUT_VARIABLE benchmark_contract_output
  ERROR_VARIABLE benchmark_contract_error)
if(NOT benchmark_contract_result EQUAL 0)
  message(FATAL_ERROR
    "Benchmark main generation failed:\n${benchmark_contract_output}\n${benchmark_contract_error}")
endif()
file(READ "${benchmark_main}" benchmark_main_contents)
foreach(expected_line
    "#include \"test_resources.h\""
    "#include \"adapter.h\""
    "TestAdapter adapter;"
    "return asc_occupancy::run_benchmark(argc, argv, adapter, runtime, std::cerr);")
  string(FIND "${benchmark_main_contents}" "${expected_line}" expected_index)
  if(expected_index EQUAL -1)
    message(FATAL_ERROR "Generated benchmark main missed: ${expected_line}")
  endif()
endforeach()

set(invalid_contract_script "${ASC_OCCUPANCY_TEST_BINARY_DIR}/invalid_contract.cmake")
file(WRITE "${invalid_contract_script}"
  "include(\"${probe_module_path}\")\n"
  "asc_occupancy_validate_kernel_variant_arguments(\n"
  "  NAME bad SOURCE \"${CMAKE_CURRENT_LIST_FILE}\"\n"
  "  KERNEL_SYMBOL_REGEX \"^bad$\" LAUNCH_BOUNDS 768)\n")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -P "${invalid_contract_script}"
  RESULT_VARIABLE invalid_contract_result)
if(invalid_contract_result EQUAL 0)
  message(FATAL_ERROR "Unsupported launch bounds accepted")
endif()
