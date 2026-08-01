asc_occupancy_add_benchmark(
  NAME launch_geometry_lb2048_occupancy_bench
  ADAPTER_HEADER "${CMAKE_CURRENT_SOURCE_DIR}/launch_geometry_adapter.h"
  ADAPTER_TYPE LaunchGeometryAdapter
  KERNEL_SYMBOL_REGEX "^launch_geometry_gather_strided_kernel_simt_entry$"
  LAUNCH_BOUNDS 2048)
