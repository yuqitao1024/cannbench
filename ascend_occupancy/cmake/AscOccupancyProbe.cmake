include_guard(GLOBAL)
include(CMakeParseArguments)
include("${CMAKE_CURRENT_LIST_DIR}/ParseBishengResourceUsage.cmake")

function(asc_occupancy_validate_kernel_variant_arguments)
  cmake_parse_arguments(ARG "" "NAME;SOURCE;KERNEL_SYMBOL_REGEX;LAUNCH_BOUNDS;STATIC_UB_BYTES"
    "INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS;COMPILE_OPTIONS" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown asc_occupancy_add_kernel_variant arguments: ${ARG_UNPARSED_ARGUMENTS}")
  endif()
  foreach(required NAME SOURCE KERNEL_SYMBOL_REGEX LAUNCH_BOUNDS)
    if(NOT ARG_${required})
      message(FATAL_ERROR "asc_occupancy_add_kernel_variant requires ${required}")
    endif()
  endforeach()
  if(NOT EXISTS "${ARG_SOURCE}")
    message(FATAL_ERROR "Kernel source does not exist: ${ARG_SOURCE}")
  endif()
  if(NOT ARG_LAUNCH_BOUNDS MATCHES "^(256|512|1024|2048)$")
    message(FATAL_ERROR
      "LAUNCH_BOUNDS must be exactly one of 256, 512, 1024, or 2048; got ${ARG_LAUNCH_BOUNDS}")
  endif()
  if(DEFINED ARG_STATIC_UB_BYTES AND
     (NOT ARG_STATIC_UB_BYTES MATCHES "^[0-9]+$"))
    message(FATAL_ERROR "STATIC_UB_BYTES must be a non-negative integer")
  endif()
endfunction()

function(asc_occupancy_write_resource_header)
  cmake_parse_arguments(ARG "" "OUTPUT;NAME;KERNEL_SYMBOL;LAUNCH_BOUNDS;USED_REGISTER_NUMBER;STACK_SIZE_BYTES;STATIC_UB_BYTES;STATIC_UB_BYTES_KNOWN" "" ${ARGN})
  foreach(required OUTPUT NAME LAUNCH_BOUNDS USED_REGISTER_NUMBER STACK_SIZE_BYTES STATIC_UB_BYTES STATIC_UB_BYTES_KNOWN)
    if(NOT DEFINED ARG_${required})
      message(FATAL_ERROR "asc_occupancy_write_resource_header requires ${required}")
    endif()
  endforeach()
  string(SUBSTRING "${ARG_NAME}" 0 1 name_initial)
  string(TOUPPER "${name_initial}" name_initial)
  string(SUBSTRING "${ARG_NAME}" 1 -1 name_remainder)
  string(MAKE_C_IDENTIFIER "${name_initial}${name_remainder}" name_identifier)
  if(ARG_STATIC_UB_BYTES_KNOWN)
    set(static_ub_bytes_known_literal true)
  else()
    set(static_ub_bytes_known_literal false)
  endif()
  get_filename_component(output_directory "${ARG_OUTPUT}" DIRECTORY)
  file(MAKE_DIRECTORY "${output_directory}")
  file(WRITE "${ARG_OUTPUT}" "#pragma once\n\n#include <ascend_occupancy/asc_occupancy.h>\n\n// Host-side resource data. Do not reference it from device functions.\ninline constexpr AscKernelResourceUsage k${name_identifier}Resources{\n    ASC_OCCUPANCY_ABI_VERSION,\n    sizeof(AscKernelResourceUsage),\n    \"${ARG_KERNEL_SYMBOL}\",\n    ${ARG_LAUNCH_BOUNDS},\n    ${ARG_USED_REGISTER_NUMBER},\n    ${ARG_STACK_SIZE_BYTES},\n    ${ARG_STATIC_UB_BYTES},\n    ${static_ub_bytes_known_literal},\n};\n")
endfunction()

function(asc_occupancy_add_kernel_variant)
  cmake_parse_arguments(ARG "" "NAME;SOURCE;KERNEL_SYMBOL_REGEX;LAUNCH_BOUNDS;STATIC_UB_BYTES"
    "INCLUDE_DIRECTORIES;COMPILE_DEFINITIONS;COMPILE_OPTIONS" ${ARGN})
  asc_occupancy_validate_kernel_variant_arguments(${ARGN})
  if(NOT CMAKE_ASC_COMPILER)
    message(FATAL_ERROR "asc_occupancy_add_kernel_variant requires an enabled ASC compiler")
  endif()

  get_filename_component(source_absolute "${ARG_SOURCE}" ABSOLUTE BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${source_absolute}")
  set(generated_directory "${CMAKE_CURRENT_BINARY_DIR}/occupancy-generated")
  set(generated_header "${generated_directory}/${ARG_NAME}_occupancy_resources.h")
  asc_occupancy_write_resource_header(
    OUTPUT "${generated_header}" NAME "${ARG_NAME}" KERNEL_SYMBOL ""
    LAUNCH_BOUNDS "${ARG_LAUNCH_BOUNDS}" USED_REGISTER_NUMBER 0 STACK_SIZE_BYTES 0
    STATIC_UB_BYTES 0 STATIC_UB_BYTES_KNOWN false)

  set(probe_directory "${CMAKE_CURRENT_BINARY_DIR}/occupancy-probes/${ARG_NAME}")
  set(probe_source_directory "${probe_directory}/source")
  set(probe_binary_directory "${probe_directory}/build")
  # try_compile reuses its binary directory. A reused object emits no Bisheng
  # resource line, so each configure must force the single probe compilation.
  file(REMOVE_RECURSE "${probe_binary_directory}")
  file(MAKE_DIRECTORY "${probe_source_directory}")
  set(ASC_OCCUPANCY_PROBE_NAME "${ARG_NAME}")
  set(ASC_OCCUPANCY_PROBE_TARGET "${ARG_NAME}_probe")
  set(ASC_OCCUPANCY_PROBE_SOURCE "${source_absolute}")
  set(ASC_OCCUPANCY_PROBE_ARCHITECTURES "${CMAKE_ASC_ARCHITECTURES}")
  get_filename_component(occupancy_project_directory
    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/.." ABSOLUTE)
  set(ASC_OCCUPANCY_PROBE_INCLUDE_DIRECTORIES
    "${generated_directory};${occupancy_project_directory}/include;${ARG_INCLUDE_DIRECTORIES}")
  set(ASC_OCCUPANCY_PROBE_COMPILE_DEFINITIONS "${ARG_COMPILE_DEFINITIONS}")
  set(ASC_OCCUPANCY_PROBE_COMPILE_OPTIONS "${ARG_COMPILE_OPTIONS}")
  set(ASC_OCCUPANCY_PROBE_LAUNCH_BOUNDS "${ARG_LAUNCH_BOUNDS}")
  configure_file("${CMAKE_CURRENT_FUNCTION_LIST_DIR}/OccupancyProbeProject.cmake.in"
    "${probe_source_directory}/CMakeLists.txt" @ONLY)

  set(previous_try_compile_target_type "${CMAKE_TRY_COMPILE_TARGET_TYPE}")
  set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
  try_compile(probe_succeeded
    "${probe_binary_directory}"
    "${probe_source_directory}"
    "asc_occupancy_${ARG_NAME}"
    "${ARG_NAME}_probe"
    CMAKE_FLAGS
      "-DCMAKE_ASC_ARCHITECTURES:STRING=${CMAKE_ASC_ARCHITECTURES}"
      "-DCMAKE_AR:FILEPATH=/usr/bin/ar"
      "-DCMAKE_RANLIB:FILEPATH=/usr/bin/ranlib"
      "-DCMAKE_ASC_COMPILER_AR:FILEPATH=/usr/bin/ar"
    OUTPUT_VARIABLE probe_output)
  set(CMAKE_TRY_COMPILE_TARGET_TYPE "${previous_try_compile_target_type}")
  file(WRITE "${probe_directory}/occupancy-probe.log" "${probe_output}")
  if(NOT probe_succeeded)
    message(FATAL_ERROR "Occupancy resource probe failed; see ${probe_directory}/occupancy-probe.log")
  endif()

  asc_occupancy_parse_bisheng_resource_usage(
    OUTPUT_PREFIX parsed_resource_usage
    RESOURCE_USAGE_OUTPUT "${probe_output}"
    KERNEL_SYMBOL_REGEX "${ARG_KERNEL_SYMBOL_REGEX}")
  if(DEFINED ARG_STATIC_UB_BYTES)
    set(static_ub_bytes "${ARG_STATIC_UB_BYTES}")
    set(static_ub_bytes_known true)
  else()
    set(static_ub_bytes 0)
    set(static_ub_bytes_known false)
  endif()
  asc_occupancy_write_resource_header(
    OUTPUT "${generated_header}" NAME "${ARG_NAME}"
    KERNEL_SYMBOL "${parsed_resource_usage_KERNEL_SYMBOL}"
    LAUNCH_BOUNDS "${ARG_LAUNCH_BOUNDS}"
    USED_REGISTER_NUMBER "${parsed_resource_usage_USED_REGISTER_NUMBER}"
    STACK_SIZE_BYTES "${parsed_resource_usage_STACK_SIZE_BYTES}"
    STATIC_UB_BYTES "${static_ub_bytes}"
    STATIC_UB_BYTES_KNOWN "${static_ub_bytes_known}")

  add_executable("${ARG_NAME}" "${source_absolute}")
  set_target_properties("${ARG_NAME}" PROPERTIES LINKER_LANGUAGE ASC)
  target_link_libraries("${ARG_NAME}" PRIVATE ascend_occupancy)
  target_include_directories("${ARG_NAME}" PRIVATE ${ARG_INCLUDE_DIRECTORIES})
  target_compile_definitions("${ARG_NAME}" PRIVATE
    ${ARG_COMPILE_DEFINITIONS} ASC_OCCUPANCY_LAUNCH_BOUNDS=${ARG_LAUNCH_BOUNDS})
  target_compile_options("${ARG_NAME}" PRIVATE
    ${ARG_COMPILE_OPTIONS} $<$<COMPILE_LANGUAGE:ASC>:--cce-res-usage>)
  target_include_directories("${ARG_NAME}" PUBLIC "${generated_directory}")
endfunction()
