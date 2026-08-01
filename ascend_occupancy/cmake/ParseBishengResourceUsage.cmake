include_guard(GLOBAL)

function(asc_occupancy_parse_bisheng_resource_usage)
  cmake_parse_arguments(ARG "" "OUTPUT_PREFIX;RESOURCE_USAGE_OUTPUT;KERNEL_SYMBOL_REGEX" "" ${ARGN})
  if(ARG_UNPARSED_ARGUMENTS OR NOT ARG_OUTPUT_PREFIX OR
     NOT DEFINED ARG_RESOURCE_USAGE_OUTPUT OR NOT ARG_KERNEL_SYMBOL_REGEX)
    message(FATAL_ERROR
      "asc_occupancy_parse_bisheng_resource_usage requires OUTPUT_PREFIX, "
      "RESOURCE_USAGE_OUTPUT, and KERNEL_SYMBOL_REGEX")
  endif()

  string(REPLACE "\r\n" "\n" resource_output "${ARG_RESOURCE_USAGE_OUTPUT}")
  string(REPLACE "\r" "\n" resource_output "${resource_output}")
  string(REPLACE "\n" ";" resource_lines "${resource_output}")
  set(matches "")
  foreach(resource_line IN LISTS resource_lines)
    if(resource_line MATCHES "Function properties for ")
      string(REGEX REPLACE "^[ \t]*\\[BISHENG\\][ \t]*" "" resource_line
        "${resource_line}")
      if(NOT resource_line MATCHES
          "^Function properties for ([^:]+): Stack size: ([0-9]+) bytes, Used register number: ([0-9]+)$")
        message(FATAL_ERROR
          "Unrecognized Bisheng resource usage line: ${resource_line}")
      endif()
      set(symbol "${CMAKE_MATCH_1}")
      set(stack_size_bytes "${CMAKE_MATCH_2}")
      set(used_register_number "${CMAKE_MATCH_3}")
      if(symbol MATCHES "${ARG_KERNEL_SYMBOL_REGEX}")
        list(APPEND matches "${symbol}|${stack_size_bytes}|${used_register_number}")
      endif()
    endif()
  endforeach()

  list(LENGTH matches match_count)
  if(NOT match_count EQUAL 1)
    message(FATAL_ERROR
      "Expected exactly one Bisheng resource usage record matching "
      "'${ARG_KERNEL_SYMBOL_REGEX}', found ${match_count}")
  endif()

  list(GET matches 0 selected_match)
  string(REPLACE "|" ";" selected_fields "${selected_match}")
  list(GET selected_fields 0 selected_symbol)
  list(GET selected_fields 1 selected_stack_size_bytes)
  list(GET selected_fields 2 selected_used_register_number)
  set(${ARG_OUTPUT_PREFIX}_KERNEL_SYMBOL "${selected_symbol}" PARENT_SCOPE)
  set(${ARG_OUTPUT_PREFIX}_STACK_SIZE_BYTES "${selected_stack_size_bytes}" PARENT_SCOPE)
  set(${ARG_OUTPUT_PREFIX}_USED_REGISTER_NUMBER "${selected_used_register_number}" PARENT_SCOPE)
endfunction()
