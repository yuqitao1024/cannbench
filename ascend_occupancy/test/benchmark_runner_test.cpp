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
  int created = 0;
  int destroyed = 0;

  bool create_event(void** event, std::string*) {
    if (fail_events) return false;
    *event = reinterpret_cast<void*>(static_cast<uintptr_t>(++created));
    return true;
  }
  bool record_event(void*, void*, std::string*) { ++records; return true; }
  bool synchronize_event(void*, std::string*) { return true; }
  bool elapsed_us(double* value, void*, void*, std::string*) {
    *value = samples.at(sample_index++);
    return true;
  }
  void destroy_event(void*) { ++destroyed; }
};

struct FakeAdapter {
  bool initialized = true;
  bool valid = true;
  int resets = 0;
  int launches = 0;
  int shutdowns = 0;
  std::vector<asc_occupancy::LaunchGeometry> candidate_list{{4, 2048}};
  AscKernelResourceUsage resources{ASC_OCCUPANCY_ABI_VERSION,
                                    sizeof(AscKernelResourceUsage),
                                    "fake_entry", 2048, 7, 0, 0, false};

  bool initialize(std::string* error) {
    if (!initialized) *error = "partial allocation";
    return initialized;
  }
  void shutdown() { ++shutdowns; }
  void* stream() const { return nullptr; }
  const char* benchmark_name() const { return "fake\"benchmark"; }
  const char* variant_name() const { return "lb2048"; }
  uint64_t work_items() const { return 16; }
  const AscKernelResourceUsage& resource_usage() const { return resources; }
  std::vector<asc_occupancy::LaunchGeometry> candidates() const {
    return candidate_list;
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

size_t count_in_file(const std::string& path, const std::string& expected) {
  std::ifstream input(path);
  std::string contents((std::istreambuf_iterator<char>(input)),
                       std::istreambuf_iterator<char>());
  size_t count = 0;
  for (size_t offset = 0;
       (offset = contents.find(expected, offset)) != std::string::npos;
       offset += expected.size()) {
    ++count;
  }
  return count;
}

}  // namespace

int main() {
  const std::string json = "/tmp/asc_occupancy_runner_test.json";
  const std::string csv = "/tmp/asc_occupancy_runner_test.csv";
  const std::string default_json = "fake_benchmark-lb2048.json";
  const std::string default_csv = "fake_benchmark-lb2048.csv";
  std::remove(json.c_str());
  std::remove(csv.c_str());
  std::remove(default_json.c_str());
  std::remove(default_csv.c_str());

  FakeAdapter adapter;
  FakeRuntime runtime;
  if (run(adapter, runtime, {"runner", "--warmup", "2", "--iterations", "3",
                             "--json", json, "--csv", csv}) != 0) {
    return 1;
  }
  if (adapter.resets != 6 || adapter.launches != 6 || runtime.records != 6) {
    return 2;
  }
  if (adapter.shutdowns != 1 || runtime.created != 2 || runtime.destroyed != 2) {
    return 6;
  }
  if (!contains_file(json, "\"samples_us\":[5,1,3]") ||
      !contains_file(json, "\"median_us\":3") ||
      !contains_file(json, "fake\\\"benchmark") ||
      !contains_file(csv, "benchmark,variant,environment_id,profile_path,launch_bounds")) {
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
  if (run(bad_args_adapter, bad_args_runtime,
          {"runner", "--candidate-index", "184467440737095516160"}) == 0) {
    return 12;
  }

  FakeAdapter init_failure_adapter;
  init_failure_adapter.initialized = false;
  FakeRuntime init_failure_runtime;
  if (run(init_failure_adapter, init_failure_runtime, {"runner"}) == 0 ||
      init_failure_adapter.shutdowns != 1) {
    return 7;
  }

  const std::string tie_json = "/tmp/asc_occupancy_runner_tie_test.json";
  const std::string tie_csv = "/tmp/asc_occupancy_runner_tie_test.csv";
  FakeAdapter tie_adapter;
  tie_adapter.candidate_list = {{4, 2048}, {8, 1024}};
  FakeRuntime tie_runtime;
  tie_runtime.samples = {5.0, 4.0, 6.0, 4.5, 3.0, 5.5};
  if (run(tie_adapter, tie_runtime,
          {"runner", "--warmup", "1", "--iterations", "3",
           "--environment-id", "tools2/cann-9.1", "--profile-path", "/tmp/profile raw",
           "--json", tie_json, "--csv", tie_csv}) != 0) {
    return 8;
  }
  if (!contains_file(tie_json, "\"environment_id\":\"tools2/cann-9.1\"") ||
      !contains_file(tie_json, "\"profile_path\":\"/tmp/profile raw\"") ||
      count_in_file(tie_json, "\"is_best_candidate\":true") != 2 ||
      !contains_file(tie_csv, "profile_path") ||
      !contains_file(tie_csv, ",true,")) {
    return 9;
  }

  const std::string selected_json = "/tmp/asc_occupancy_runner_selected_test.json";
  FakeAdapter selected_adapter;
  selected_adapter.candidate_list = {{4, 2048}, {16, 1024}};
  FakeRuntime selected_runtime;
  if (run(selected_adapter, selected_runtime,
          {"runner", "--candidate-index", "1", "--warmup", "1", "--iterations", "1",
           "--json", selected_json, "--csv", "/tmp/asc_occupancy_runner_selected_test.csv"}) != 0 ||
      selected_adapter.launches != 3 ||
      !contains_file(selected_json, "\"grid_blocks\":16,\"block_threads\":1024") ||
      contains_file(selected_json, "\"grid_blocks\":4,\"block_threads\":2048")) {
    return 11;
  }

  FakeAdapter default_output_adapter;
  FakeRuntime default_output_runtime;
  if (run(default_output_adapter, default_output_runtime,
          {"runner", "--warmup", "1", "--iterations", "1"}) != 0 ||
      !contains_file(default_json, "\"benchmark\":\"fake\\\"benchmark\"") ||
      !contains_file(default_csv, "is_best_candidate")) {
    return 10;
  }
  std::remove(default_json.c_str());
  std::remove(default_csv.c_str());
  return 0;
}
