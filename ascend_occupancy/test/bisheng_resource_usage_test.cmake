if(NOT DEFINED ASC_OCCUPANCY_SOURCE_DIR OR
   NOT DEFINED ASC_OCCUPANCY_TEST_BINARY_DIR)
  message(FATAL_ERROR "Missing CMake/parser test inputs")
endif()

set(module_path "${ASC_OCCUPANCY_SOURCE_DIR}/cmake/ParseBishengResourceUsage.cmake")
set(probe_module_path "${ASC_OCCUPANCY_SOURCE_DIR}/cmake/AscOccupancyProbe.cmake")
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
