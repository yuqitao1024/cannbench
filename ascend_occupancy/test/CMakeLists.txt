add_library(ascend_occupancy_without_acl STATIC ../src/asc_occupancy.cpp)
target_compile_features(ascend_occupancy_without_acl PRIVATE cxx_std_17)
target_include_directories(ascend_occupancy_without_acl PUBLIC "${PROJECT_SOURCE_DIR}/include")

add_executable(asc_occupancy_resource_model_test resource_model_test.cpp)
target_link_libraries(asc_occupancy_resource_model_test PRIVATE ascend_occupancy_without_acl)
add_test(NAME asc_occupancy_resource_model_test COMMAND asc_occupancy_resource_model_test)

add_library(ascend_occupancy_fake_acl STATIC
  ../src/asc_occupancy.cpp
  fake_acl/fake_acl.cpp)
target_compile_features(ascend_occupancy_fake_acl PRIVATE cxx_std_17)
target_compile_definitions(ascend_occupancy_fake_acl PRIVATE ASC_OCCUPANCY_WITH_ACL=1)
target_include_directories(ascend_occupancy_fake_acl
  PUBLIC "${PROJECT_SOURCE_DIR}/include"
  PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fake_acl")

add_executable(asc_occupancy_acl_device_test acl_device_test.cpp)
target_include_directories(asc_occupancy_acl_device_test PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/fake_acl")
target_link_libraries(asc_occupancy_acl_device_test PRIVATE ascend_occupancy_fake_acl)
add_test(NAME asc_occupancy_acl_device_test COMMAND asc_occupancy_acl_device_test)

add_executable(asc_occupancy_benchmark_runner_test benchmark_runner_test.cpp)
target_compile_features(asc_occupancy_benchmark_runner_test PRIVATE cxx_std_17)
target_include_directories(asc_occupancy_benchmark_runner_test PRIVATE
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/perf_tests/common")
add_test(NAME asc_occupancy_benchmark_runner_test COMMAND asc_occupancy_benchmark_runner_test)

add_test(
  NAME asc_occupancy_cmake_discovery_test
  COMMAND "${CMAKE_COMMAND}"
    "-DASC_OCCUPANCY_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DASC_OCCUPANCY_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/cmake_discovery"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake_discovery_test.cmake")

add_test(
  NAME asc_occupancy_bisheng_resource_usage_test
  COMMAND "${CMAKE_COMMAND}"
    "-DASC_OCCUPANCY_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DASC_OCCUPANCY_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/bisheng_resource_usage"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/bisheng_resource_usage_test.cmake")

add_test(
  NAME asc_occupancy_probe_integration_test
  COMMAND "${CMAKE_COMMAND}"
    "-DASC_OCCUPANCY_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DASC_OCCUPANCY_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/probe_integration"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/occupancy_probe_integration_test.cmake")

add_test(
  NAME asc_occupancy_perf_test_contract_test
  COMMAND "${CMAKE_COMMAND}"
    "-DASC_OCCUPANCY_SOURCE_DIR=${PROJECT_SOURCE_DIR}"
    "-DASC_OCCUPANCY_TEST_BINARY_DIR=${CMAKE_CURRENT_BINARY_DIR}/perf_test_contract"
    -P "${CMAKE_CURRENT_SOURCE_DIR}/perf_test_contract_test.cmake")

find_package(Python3 REQUIRED COMPONENTS Interpreter)
add_test(
  NAME asc_occupancy_aggregate_results_test
  COMMAND "${Python3_EXECUTABLE}"
    "${CMAKE_CURRENT_SOURCE_DIR}/aggregate_results_test.py"
    "${PROJECT_SOURCE_DIR}/perf_tests/common/aggregate_results.py")
