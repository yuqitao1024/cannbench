cmake_minimum_required(VERSION 3.20)

project(ascend_occupancy LANGUAGES C CXX)

include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
check_c_source_compiles(
  "#include <ascend_occupancy/asc_occupancy.h>\nint main(void) { return ASC_OCCUPANCY_ABI_VERSION == 1U ? 0 : 1; }"
  ASC_OCCUPANCY_PUBLIC_HEADER_IS_C99)
if(NOT ASC_OCCUPANCY_PUBLIC_HEADER_IS_C99)
  message(FATAL_ERROR "asc_occupancy.h must be consumable from C99")
endif()

# The host resource model intentionally has no ACL or CANN dependency.
add_library(ascend_occupancy STATIC src/asc_occupancy.cpp)
target_compile_features(ascend_occupancy PUBLIC cxx_std_17)
target_include_directories(ascend_occupancy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

enable_testing()
add_subdirectory(test)
