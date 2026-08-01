if(NOT DEFINED ASC_OCCUPANCY_SOURCE_DIR OR
   NOT DEFINED ASC_OCCUPANCY_TEST_BINARY_DIR)
  message(FATAL_ERROR "Missing source or test binary directory")
endif()

set(fake_cann_root "${ASC_OCCUPANCY_TEST_BINARY_DIR}/fake_cann")
set(configure_binary_dir "${ASC_OCCUPANCY_TEST_BINARY_DIR}/configured")
file(REMOVE_RECURSE "${ASC_OCCUPANCY_TEST_BINARY_DIR}")
file(MAKE_DIRECTORY "${fake_cann_root}/include/acl")
file(MAKE_DIRECTORY "${fake_cann_root}/lib64")
file(WRITE "${fake_cann_root}/include/acl/acl.h" "#pragma once\n")
file(WRITE "${fake_cann_root}/lib64/libascendcl.so" "")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "ASCEND_HOME_PATH=" "${CMAKE_COMMAND}"
    -S "${ASC_OCCUPANCY_SOURCE_DIR}"
    -B "${configure_binary_dir}"
    "-DASCEND_CANN_PACKAGE_PATH:PATH=${fake_cann_root}"
  RESULT_VARIABLE configure_result
  OUTPUT_VARIABLE configure_output
  ERROR_VARIABLE configure_error)
if(NOT configure_result EQUAL 0)
  message(FATAL_ERROR
    "Configuration with ASCEND_CANN_PACKAGE_PATH failed:\n${configure_output}\n${configure_error}")
endif()

file(READ "${configure_binary_dir}/CMakeCache.txt" configure_cache)
foreach(expected_entry
    "ASC_OCCUPANCY_ENABLE_ACL:BOOL=ON"
    "ASC_OCCUPANCY_ACL_INCLUDE_DIR:PATH=${fake_cann_root}/include"
    "ASC_OCCUPANCY_ACL_LIBRARY:FILEPATH=${fake_cann_root}/lib64/libascendcl.so")
  string(FIND "${configure_cache}" "${expected_entry}" entry_index)
  if(entry_index EQUAL -1)
    message(FATAL_ERROR "Missing expected CANN discovery entry: ${expected_entry}")
  endif()
endforeach()

set(fake_cann_x86_root "${ASC_OCCUPANCY_TEST_BINARY_DIR}/fake_cann_x86")
set(x86_configure_binary_dir "${ASC_OCCUPANCY_TEST_BINARY_DIR}/configured_x86")
file(MAKE_DIRECTORY "${fake_cann_x86_root}/x86_64-linux/include/acl")
file(MAKE_DIRECTORY "${fake_cann_x86_root}/x86_64-linux/lib64")
file(WRITE "${fake_cann_x86_root}/x86_64-linux/include/acl/acl.h" "#pragma once\n")
file(WRITE "${fake_cann_x86_root}/x86_64-linux/lib64/libascendcl.so" "")

execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "ASCEND_HOME_PATH=" "${CMAKE_COMMAND}"
    -S "${ASC_OCCUPANCY_SOURCE_DIR}"
    -B "${x86_configure_binary_dir}"
    "-DASCEND_CANN_PACKAGE_PATH:PATH=${fake_cann_x86_root}"
  RESULT_VARIABLE x86_configure_result
  OUTPUT_VARIABLE x86_configure_output
  ERROR_VARIABLE x86_configure_error)
if(NOT x86_configure_result EQUAL 0)
  message(FATAL_ERROR
    "Configuration with x86_64-linux CANN root failed:\n${x86_configure_output}\n${x86_configure_error}")
endif()

file(READ "${x86_configure_binary_dir}/CMakeCache.txt" x86_configure_cache)
foreach(expected_entry
    "ASC_OCCUPANCY_ACL_INCLUDE_DIR:PATH=${fake_cann_x86_root}/x86_64-linux/include"
    "ASC_OCCUPANCY_ACL_LIBRARY:FILEPATH=${fake_cann_x86_root}/x86_64-linux/lib64/libascendcl.so")
  string(FIND "${x86_configure_cache}" "${expected_entry}" entry_index)
  if(entry_index EQUAL -1)
    message(FATAL_ERROR "Missing expected x86_64-linux CANN discovery entry: ${expected_entry}")
  endif()
endforeach()

set(disabled_binary_dir "${ASC_OCCUPANCY_TEST_BINARY_DIR}/configured_disabled")
execute_process(
  COMMAND "${CMAKE_COMMAND}" -E env "ASCEND_HOME_PATH=" "${CMAKE_COMMAND}"
    -S "${ASC_OCCUPANCY_SOURCE_DIR}"
    -B "${disabled_binary_dir}"
    "-DASCEND_CANN_PACKAGE_PATH:PATH=${fake_cann_root}"
    "-DASC_OCCUPANCY_ENABLE_ACL:BOOL=OFF"
  RESULT_VARIABLE disabled_configure_result
  OUTPUT_VARIABLE disabled_configure_output
  ERROR_VARIABLE disabled_configure_error)
if(NOT disabled_configure_result EQUAL 0)
  message(FATAL_ERROR
    "Configuration with explicit ACL disablement failed:\n${disabled_configure_output}\n${disabled_configure_error}")
endif()

file(READ "${disabled_binary_dir}/CMakeCache.txt" disabled_configure_cache)
string(FIND "${disabled_configure_cache}"
  "ASC_OCCUPANCY_ENABLE_ACL:BOOL=OFF" disabled_entry_index)
if(disabled_entry_index EQUAL -1)
  message(FATAL_ERROR "Explicit ASC_OCCUPANCY_ENABLE_ACL=OFF was not preserved")
endif()
