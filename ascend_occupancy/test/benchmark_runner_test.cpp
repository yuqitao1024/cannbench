#include "benchmark_runner.h"

#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace {

struct FakeRuntime {
  bool fail_events = false;
  std::vector<double> samples{5.0, 1.0, 3.0};
  size_t sample_index = 0;
  int records = 0;

  bool create_event(void**, std::string*) { return !fail_events; }
  bool record_event(void*, void*, std::string*) { ++records; return true; }
  bool synchronize_event(void*, std::string*) { return true; }
  bool elapsed_us(double* value, void*, void*, std::string*) {
    *value = samples.at(sample_index++);
    return true;
  }
  void destroy_event(void*) {}
};

struct FakeAdapter {
  bool valid = true;
  int resets = 0;
  int launches = 0;
  AscKernelResourceUsage resources{ASC_OCCUPANCY_ABI_VERSION,
                                    sizeof(AscKernelResourceUsage),
                                    "fake_entry", 2048, 7, 0, 0, false};

  bool initialize(std::string*) { return true; }
  void shutdown() {}
  void* stream() const { return nullptr; }
  const char* benchmark_name() const { return "fake\"benchmark"; }
  const char* variant_name() const { return "lb2048"; }
  uint64_t work_items() const { return 16; }
  const AscKernelResourceUsage& resource_usage() const { return resources; }
  std::vector<asc_occupancy::LaunchGeometry> candidates() const {
    return {{4, 2048}};
  }
  bool reset_iteration_state(std::string*) { ++resets; return true; }
  bool launch(const asc_occupancy::LaunchGeometry&, std::string*) {
    ++launches;
    return true;
  }
  bool synchronize(std::string*) { return true; }
  bool validate(std::string* error) {
    if (!valid) *error = "first mismatch \"index 3\"";
    return valid;
  }
};

int run(FakeAdapter& adapter, FakeRuntime& runtime,
        const std::vector<std::string>& arguments) {
  std::vector<char*> argv;
  for (const std::string& argument : arguments) {
    argv.push_back(const_cast<char*>(argument.c_str()));
  }
  return asc_occupancy::run_benchmark(
      static_cast<int>(argv.size()), argv.data(), adapter, runtime, std::cerr);
}

bool contains_file(const std::string& path, const std::string& expected) {
  std::ifstream input(path);
  std::string contents((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
  return contents.find(expected) != std::string::npos;
}

}  // namespace

int main() {
  const std::string json = "/tmp/asc_occupancy_runner_test.json";
  const std::string csv = "/tmp/asc_occupancy_runner_test.csv";
  std::remove(json.c_str());
  std::remove(csv.c_str());

  FakeAdapter adapter;
  FakeRuntime runtime;
  if (run(adapter, runtime, {"runner", "--warmup", "2", "--iterations", "3",
                             "--json", json, "--csv", csv}) != 0) {
    return 1;
  }
  if (adapter.resets != 6 || adapter.launches != 6 || runtime.records != 6) {
    return 2;
  }
  if (!contains_file(json, "\"samples_us\":[5,1,3]") ||
      !contains_file(json, "\"median_us\":3") ||
      !contains_file(json, "fake\\\"benchmark") ||
      !contains_file(csv, "benchmark,variant,launch_bounds")) {
    return 3;
  }

  FakeAdapter invalid_adapter;
  invalid_adapter.valid = false;
  FakeRuntime invalid_runtime;
  if (run(invalid_adapter, invalid_runtime, {"runner", "--iterations", "1"}) == 0 ||
      invalid_adapter.launches != 1) {
    return 4;
  }

  FakeAdapter bad_args_adapter;
  FakeRuntime bad_args_runtime;
  if (run(bad_args_adapter, bad_args_runtime, {"runner", "--warmup", "zero"}) == 0) {
    return 5;
  }
  return 0;
}
