if(ASC_OCCUPANCY_INTEGRATION_DRIVER)
  if(NOT DEFINED ASC_OCCUPANCY_SOURCE_DIR OR
     NOT DEFINED ASC_OCCUPANCY_TEST_BINARY_DIR OR
     NOT DEFINED ASC_OCCUPANCY_INTEGRATION_VARIANT)
    message(FATAL_ERROR "Missing occupancy probe integration driver inputs")
  endif()

  set(TEST_ROOT "${ASC_OCCUPANCY_TEST_BINARY_DIR}")
  set(TEST_RECORD_DIR "${TEST_ROOT}/records-${ASC_OCCUPANCY_INTEGRATION_VARIANT}")
  file(MAKE_DIRECTORY "${TEST_RECORD_DIR}")
  # Script mode has no project directory. These paths let the module exercise
  # its normal generated-file layout without enabling an ASC compiler.
  set(CMAKE_CURRENT_SOURCE_DIR "${TEST_ROOT}")
  set(CMAKE_CURRENT_BINARY_DIR "${TEST_ROOT}/binary")
  set(CMAKE_ASC_COMPILER "/nonexistent/asc")
  set(CMAKE_ASC_ENABLE_SIMT ON)
  set(CMAKE_ASC_ARCHITECTURES "ascend950")

  function(record_call name)
    file(APPEND "${TEST_RECORD_DIR}/${name}.txt" "${ARGN}\n")
  endfunction()

  # These replacements test the CMake orchestration contract. They deliberately
  # avoid pretending that the local host can compile ASC code.
  function(try_compile result_variable)
    record_call(try_compile ${ARGN})
    list(FIND ARGN "OUTPUT_VARIABLE" output_variable_index)
    if(output_variable_index EQUAL -1)
      message(FATAL_ERROR "Probe did not request try_compile output")
    endif()
    math(EXPR output_variable_index "${output_variable_index} + 1")
    list(GET ARGN ${output_variable_index} output_variable)
    file(READ "${TEST_ROOT}/kernel.cpp" kernel_source)
    string(REGEX MATCH "RESOURCE_VERSION ([0-9]+)" resource_match "${kernel_source}")
    set(resource_version "${CMAKE_MATCH_1}")
    if(NOT resource_version)
      message(FATAL_ERROR "Fake probe could not read the current kernel source")
    endif()
    math(EXPR register_count "40 + ${resource_version}")
    set(probe_output
      "[BISHENG] Function properties for integration_${ASC_OCCUPANCY_INTEGRATION_VARIANT}_entry: Stack size: ${resource_version} bytes, Used register number: ${register_count}")
    set(${result_variable} TRUE PARENT_SCOPE)
    set(${output_variable} "${probe_output}" PARENT_SCOPE)
  endfunction()

  function(add_executable)
    record_call(add_executable ${ARGN})
  endfunction()
  function(set_target_properties)
    record_call(set_target_properties ${ARGN})
  endfunction()
  function(target_link_libraries)
    record_call(target_link_libraries ${ARGN})
  endfunction()
  function(target_include_directories)
    record_call(target_include_directories ${ARGN})
  endfunction()
  function(target_compile_definitions)
    record_call(target_compile_definitions ${ARGN})
  endfunction()
  function(target_compile_options)
    record_call(target_compile_options ${ARGN})
  endfunction()
  function(set_property)
    record_call(set_property ${ARGN})
    if("${ARGV0}" STREQUAL "DIRECTORY" AND
       "${ARGV2}" STREQUAL "PROPERTY" AND
       "${ARGV3}" STREQUAL "CMAKE_CONFIGURE_DEPENDS")
      list(GET ARGN 4 configure_depends_source)
      set(ASC_OCCUPANCY_RECORDED_CONFIGURE_DEPENDS "${configure_depends_source}"
        CACHE INTERNAL "Recorded CMAKE_CONFIGURE_DEPENDS source" FORCE)
    endif()
  endfunction()

  include("${ASC_OCCUPANCY_SOURCE_DIR}/cmake/AscOccupancyProbe.cmake")
  asc_occupancy_add_kernel_variant(
    NAME integration_kernel
    SOURCE "${TEST_ROOT}/kernel.cpp"
    KERNEL_SYMBOL_REGEX "^integration_${ASC_OCCUPANCY_INTEGRATION_VARIANT}_entry$"
    LAUNCH_BOUNDS 1024
    STATIC_UB_BYTES 2048
    INCLUDE_DIRECTORIES "${TEST_ROOT}/include-one" "${TEST_ROOT}/include-two"
    COMPILE_DEFINITIONS "TEST_VARIANT=${ASC_OCCUPANCY_INTEGRATION_VARIANT}" "FEATURE_LEVEL=7"
    COMPILE_OPTIONS "-O3" "--user-option=${ASC_OCCUPANCY_INTEGRATION_VARIANT}")

  function(assert_equal expected actual description)
    if(NOT "${expected}" STREQUAL "${actual}")
      message(FATAL_ERROR "${description}: expected <${expected}>, got <${actual}>")
    endif()
  endfunction()
  function(assert_contains contents expected description)
    string(FIND "${contents}" "${expected}" expected_index)
    if(expected_index EQUAL -1)
      message(FATAL_ERROR "${description}: missing <${expected}> in <${contents}>")
    endif()
  endfunction()

  set(generated_directory "${CMAKE_CURRENT_BINARY_DIR}/occupancy-generated")
  set(probe_parameters
    "${CMAKE_CURRENT_BINARY_DIR}/occupancy-probes/integration_kernel/source/OccupancyProbeParameters.cmake")
  include("${probe_parameters}")
  assert_equal("${TEST_ROOT}/kernel.cpp" "${ASC_OCCUPANCY_PROBE_SOURCE}"
    "Probe source")
  assert_equal("ascend950" "${ASC_OCCUPANCY_PROBE_ARCHITECTURES}" "Probe architecture")
  assert_equal("ON" "${ASC_OCCUPANCY_PROBE_ENABLE_SIMT}" "Probe SIMT setting")
  assert_equal("1024" "${ASC_OCCUPANCY_PROBE_LAUNCH_BOUNDS}" "Probe launch bounds")
  assert_equal("${TEST_ROOT}/kernel.cpp" "${ASC_OCCUPANCY_RECORDED_CONFIGURE_DEPENDS}"
    "Kernel source registered as CMAKE_CONFIGURE_DEPENDS")

  file(READ "${TEST_RECORD_DIR}/add_executable.txt" formal_source)
  file(READ "${TEST_RECORD_DIR}/target_link_libraries.txt" formal_libraries)
  file(READ "${TEST_RECORD_DIR}/target_include_directories.txt" formal_includes)
  file(READ "${TEST_RECORD_DIR}/target_compile_definitions.txt" formal_definitions)
  file(READ "${TEST_RECORD_DIR}/target_compile_options.txt" formal_options)
  file(READ "${TEST_RECORD_DIR}/try_compile.txt" probe_compile)
  file(READ "${CMAKE_CURRENT_BINARY_DIR}/occupancy-probes/integration_kernel/source/CMakeLists.txt"
    probe_project)

  # Direct inputs must arrive at both probe and formal targets. The probe has
  # the library and generated-header directories directly; the formal target
  # receives those same directories through its public generated include and
  # the ascend_occupancy target's public include interface.
  foreach(include_directory "${TEST_ROOT}/include-one" "${TEST_ROOT}/include-two")
    list(FIND ASC_OCCUPANCY_PROBE_INCLUDE_DIRECTORIES "${include_directory}" probe_include_index)
    if(probe_include_index EQUAL -1)
      message(FATAL_ERROR "Probe missed include directory ${include_directory}")
    endif()
    assert_contains("${formal_includes}" "${include_directory}" "Formal include directory")
  endforeach()
  assert_contains("${ASC_OCCUPANCY_PROBE_INCLUDE_DIRECTORIES}" "${generated_directory}"
    "Probe generated include directory")
  assert_contains("${formal_includes}" "${generated_directory}" "Formal generated include directory")
  assert_contains("${formal_libraries}" "ascend_occupancy" "Formal occupancy include interface")

  foreach(definition "TEST_VARIANT=${ASC_OCCUPANCY_INTEGRATION_VARIANT}" "FEATURE_LEVEL=7")
    list(FIND ASC_OCCUPANCY_PROBE_COMPILE_DEFINITIONS "${definition}" probe_definition_index)
    if(probe_definition_index EQUAL -1)
      message(FATAL_ERROR "Probe missed definition ${definition}")
    endif()
    assert_contains("${formal_definitions}" "${definition}" "Formal compile definition")
  endforeach()
  assert_contains("${probe_project}"
    "ASC_OCCUPANCY_LAUNCH_BOUNDS=\${ASC_OCCUPANCY_PROBE_LAUNCH_BOUNDS}"
    "Probe launch-bounds definition")
  assert_contains("${formal_definitions}" "ASC_OCCUPANCY_LAUNCH_BOUNDS=1024"
    "Formal launch-bounds definition")
  foreach(option "-O3" "--user-option=${ASC_OCCUPANCY_INTEGRATION_VARIANT}")
    list(FIND ASC_OCCUPANCY_PROBE_COMPILE_OPTIONS "${option}" probe_option_index)
    if(probe_option_index EQUAL -1)
      message(FATAL_ERROR "Probe missed option ${option}")
    endif()
    assert_contains("${formal_options}" "${option}" "Formal compile option")
  endforeach()
  assert_contains("${probe_project}" "--cce-res-usage" "Probe resource usage option")
  assert_contains("${formal_options}" "--cce-res-usage" "Formal resource usage option")
  assert_contains("${probe_compile}" "-DCMAKE_ASC_ARCHITECTURES:STRING=ascend950"
    "Probe architecture forwarding")
  assert_contains("${probe_compile}" "-DCMAKE_ASC_ENABLE_SIMT:BOOL=ON"
    "Probe SIMT forwarding")
  assert_contains("${formal_source}" "${TEST_ROOT}/kernel.cpp" "Formal source")
  assert_contains("${formal_source}" "integration_kernel" "Formal target name")

  set(resource_header "${generated_directory}/integration_kernel_occupancy_resources.h")
  file(READ "${resource_header}" resource_header_contents)
  file(READ "${TEST_ROOT}/kernel.cpp" current_kernel_source)
  string(REGEX MATCH "RESOURCE_VERSION ([0-9]+)" current_resource_match "${current_kernel_source}")
  set(current_resource_version "${CMAKE_MATCH_1}")
  math(EXPR expected_register_count "40 + ${current_resource_version}")
  assert_contains("${resource_header_contents}"
    "\"integration_${ASC_OCCUPANCY_INTEGRATION_VARIANT}_entry\"" "Resource header symbol")
  assert_contains("${resource_header_contents}" "    ${expected_register_count},"
    "Resource header register count")
  assert_contains("${resource_header_contents}" "    ${current_resource_version},"
    "Resource header stack size")
  return()
endif()

if(NOT DEFINED ASC_OCCUPANCY_SOURCE_DIR OR
   NOT DEFINED ASC_OCCUPANCY_TEST_BINARY_DIR)
  message(FATAL_ERROR "Missing occupancy probe integration test inputs")
endif()

file(REMOVE_RECURSE "${ASC_OCCUPANCY_TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${ASC_OCCUPANCY_TEST_BINARY_DIR}/include-one")
file(MAKE_DIRECTORY "${ASC_OCCUPANCY_TEST_BINARY_DIR}/include-two")
set(kernel_source "${ASC_OCCUPANCY_TEST_BINARY_DIR}/kernel.cpp")
file(WRITE "${kernel_source}" "// RESOURCE_VERSION 1\n")

function(run_probe_configuration variant)
  execute_process(
    COMMAND "${CMAKE_COMMAND}"
      "-DASC_OCCUPANCY_INTEGRATION_DRIVER=ON"
      "-DASC_OCCUPANCY_INTEGRATION_VARIANT=${variant}"
      "-DASC_OCCUPANCY_SOURCE_DIR=${ASC_OCCUPANCY_SOURCE_DIR}"
      "-DASC_OCCUPANCY_TEST_BINARY_DIR=${ASC_OCCUPANCY_TEST_BINARY_DIR}"
      -P "${CMAKE_CURRENT_LIST_FILE}"
    RESULT_VARIABLE driver_result
    OUTPUT_VARIABLE driver_output
    ERROR_VARIABLE driver_error)
  if(NOT driver_result EQUAL 0)
    message(FATAL_ERROR
      "${variant} probe configuration failed:\n${driver_output}\n${driver_error}")
  endif()
endfunction()

run_probe_configuration(FIRST)
file(READ "${ASC_OCCUPANCY_TEST_BINARY_DIR}/binary/occupancy-generated/integration_kernel_occupancy_resources.h"
  first_resource_header)
string(FIND "${first_resource_header}" "\"integration_FIRST_entry\"" first_symbol_index)
if(first_symbol_index EQUAL -1)
  message(FATAL_ERROR "First configuration did not consume its probe output")
endif()

# A changed source and compile definition must produce a new probe invocation
# on the next configuration, and its generated resource header must consume it.
file(WRITE "${kernel_source}" "// RESOURCE_VERSION 2\n")
run_probe_configuration(SECOND)
file(READ "${ASC_OCCUPANCY_TEST_BINARY_DIR}/binary/occupancy-generated/integration_kernel_occupancy_resources.h"
  second_resource_header)
string(FIND "${second_resource_header}" "\"integration_SECOND_entry\"" second_symbol_index)
string(FIND "${second_resource_header}" "    42," second_register_index)
if(second_symbol_index EQUAL -1 OR second_register_index EQUAL -1)
  message(FATAL_ERROR "Second configuration did not consume new source/macro probe output")
endif()
file(READ "${ASC_OCCUPANCY_TEST_BINARY_DIR}/records-SECOND/try_compile.txt" second_try_compile)
string(FIND "${second_try_compile}" "integration_kernel_probe" second_probe_index)
if(second_probe_index EQUAL -1)
  message(FATAL_ERROR "Second configuration did not execute the probe")
endif()
