asc_occupancy_add_benchmark(
  NAME register_spill_lb1024_occupancy_bench
  RESULT_STEM register_spill-lb1024
  ADAPTER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/register_spill_adapter.h"
  ADAPTER_TYPE RegisterSpillAdapter
  KERNEL_SYMBOL_REGEX "^register_spill_sincos_kernel_simt_entry$"
  LAUNCH_BOUNDS 1024
  COMPILE_DEFINITIONS REGISTER_SPILL_LB1024)

asc_occupancy_add_benchmark(
  NAME register_spill_lb512_occupancy_bench
  RESULT_STEM register_spill-lb512
  ADAPTER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/register_spill_adapter.h"
  ADAPTER_TYPE RegisterSpillAdapter
  KERNEL_SYMBOL_REGEX "^register_spill_sincos_kernel_simt_entry$"
  LAUNCH_BOUNDS 512
  COMPILE_DEFINITIONS REGISTER_SPILL_LB512)
