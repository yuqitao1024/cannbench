cmake_minimum_required(VERSION 3.20)

project(ascend_occupancy LANGUAGES C CXX)

set(CMAKE_C_STANDARD 99)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

include(CheckCSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES "${CMAKE_CURRENT_SOURCE_DIR}/include")
check_c_source_compiles(
  "#include <ascend_occupancy/asc_occupancy.h>\nint main(void) { return ASC_OCCUPANCY_ABI_VERSION == 1U ? 0 : 1; }"
  ASC_OCCUPANCY_PUBLIC_HEADER_IS_C99)
if(NOT ASC_OCCUPANCY_PUBLIC_HEADER_IS_C99)
  message(FATAL_ERROR "asc_occupancy.h must be consumable from C99")
endif()

add_library(ascend_occupancy STATIC src/asc_occupancy.cpp)
target_compile_features(ascend_occupancy PUBLIC cxx_std_17)
target_include_directories(ascend_occupancy PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

option(ASC_OCCUPANCY_ENABLE_ACL "Enable CANN ACL runtime device discovery" OFF)
if(ASC_OCCUPANCY_ENABLE_ACL)
  find_path(ASC_OCCUPANCY_ACL_INCLUDE_DIR acl/acl.h REQUIRED)
  find_library(ASC_OCCUPANCY_ACL_LIBRARY NAMES ascendcl acl REQUIRED)
  target_compile_definitions(ascend_occupancy PRIVATE ASC_OCCUPANCY_WITH_ACL=1)
  target_include_directories(ascend_occupancy PRIVATE "${ASC_OCCUPANCY_ACL_INCLUDE_DIR}")
  target_link_libraries(ascend_occupancy PRIVATE "${ASC_OCCUPANCY_ACL_LIBRARY}")
endif()

enable_testing()
add_subdirectory(test)
