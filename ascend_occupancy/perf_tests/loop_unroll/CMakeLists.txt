asc_occupancy_add_benchmark(
  NAME loop_unroll_loop_occupancy_bench
  RESULT_STEM loop_unroll-loop
  ADAPTER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/loop_unroll_adapter.h"
  ADAPTER_TYPE LoopUnrollAdapter
  KERNEL_SYMBOL_REGEX "^loop_unroll_loop_kernel_simt_entry$"
  LAUNCH_BOUNDS 2048
  COMPILE_DEFINITIONS LOOP_UNROLL_LOOP)

asc_occupancy_add_benchmark(
  NAME loop_unroll_manual_occupancy_bench
  RESULT_STEM loop_unroll-manual
  ADAPTER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/loop_unroll_adapter.h"
  ADAPTER_TYPE LoopUnrollAdapter
  KERNEL_SYMBOL_REGEX "^loop_unroll_manual_kernel_simt_entry$"
  LAUNCH_BOUNDS 2048
  COMPILE_DEFINITIONS LOOP_UNROLL_MANUAL)
