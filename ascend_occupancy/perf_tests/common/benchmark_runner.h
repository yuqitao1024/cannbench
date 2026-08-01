#pragma once

#include <ascend_occupancy/asc_occupancy.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace asc_occupancy {

struct LaunchGeometry {
  uint32_t grid_blocks;
  uint32_t block_threads;
};

struct BenchmarkOptions {
  uint32_t warmup = 10;
  uint32_t iterations = 100;
  std::string json_path;
  std::string csv_path;
};

struct BenchmarkRecord {
  LaunchGeometry geometry{};
  std::vector<double> samples_us;
  bool validation_passed = false;
  std::string error;
  double median_us = 0;
  double min_us = 0;
  double max_us = 0;
};

inline bool parse_positive_u32(const char* value, uint32_t* result) {
  if (value == nullptr || *value == '\0') return false;
  uint64_t parsed = 0;
  for (const char* cursor = value; *cursor != '\0'; ++cursor) {
    if (*cursor < '0' || *cursor > '9') return false;
    parsed = parsed * 10 + static_cast<uint64_t>(*cursor - '0');
    if (parsed > UINT32_MAX) return false;
  }
  if (parsed == 0) return false;
  *result = static_cast<uint32_t>(parsed);
  return true;
}

inline bool parse_benchmark_options(int argc, char** argv, BenchmarkOptions* options,
                                    std::string* error) {
  for (int index = 1; index < argc; ++index) {
    const std::string argument(argv[index]);
    if (argument == "--warmup" || argument == "--iterations" ||
        argument == "--json" || argument == "--csv") {
      if (++index == argc) {
        *error = "missing value for " + argument;
        return false;
      }
      if (argument == "--warmup" && !parse_positive_u32(argv[index], &options->warmup)) {
        *error = "--warmup must be a positive integer";
        return false;
      }
      if (argument == "--iterations" &&
          !parse_positive_u32(argv[index], &options->iterations)) {
        *error = "--iterations must be a positive integer";
        return false;
      }
      if (argument == "--json") options->json_path = argv[index];
      if (argument == "--csv") options->csv_path = argv[index];
      continue;
    }
    *error = "unknown argument: " + argument;
    return false;
  }
  return true;
}

inline std::string json_escape(const std::string& value) {
  std::ostringstream escaped;
  for (unsigned char character : value) {
    switch (character) {
      case '"': escaped << "\\\""; break;
      case '\\': escaped << "\\\\"; break;
      case '\n': escaped << "\\n"; break;
      case '\r': escaped << "\\r"; break;
      case '\t': escaped << "\\t"; break;
      default:
        if (character < 0x20) {
          escaped << "\\u" << std::hex << std::setw(4) << std::setfill('0')
                  << static_cast<unsigned int>(character) << std::dec;
        } else {
          escaped << character;
        }
    }
  }
  return escaped.str();
}

inline std::string csv_escape(const std::string& value) {
  std::string escaped = "\"";
  for (char character : value) {
    if (character == '"') escaped += '"';
    escaped += character;
  }
  return escaped + "\"";
}

inline void finalize_statistics(BenchmarkRecord* record) {
  std::vector<double> sorted = record->samples_us;
  std::sort(sorted.begin(), sorted.end());
  record->min_us = sorted.front();
  record->max_us = sorted.back();
  const size_t middle = sorted.size() / 2;
  record->median_us = sorted.size() % 2 == 0
      ? (sorted[middle - 1] + sorted[middle]) / 2.0 : sorted[middle];
}

template <typename Adapter>
inline void write_json(std::ostream& output, const Adapter& adapter,
                       const BenchmarkOptions& options,
                       const std::vector<BenchmarkRecord>& records) {
  const auto& resources = adapter.resource_usage();
  output << "{\"benchmark\":\"" << json_escape(adapter.benchmark_name())
         << "\",\"variant\":\"" << json_escape(adapter.variant_name())
         << "\",\"launch_bounds\":" << resources.launch_bounds
         << ",\"used_registers_per_thread\":" << resources.used_registers_per_thread
         << ",\"stack_size_bytes\":" << resources.stack_size_bytes
         << ",\"work_items\":" << adapter.work_items()
         << ",\"warmup\":" << options.warmup
         << ",\"iterations\":" << options.iterations << ",\"records\":[";
  for (size_t index = 0; index < records.size(); ++index) {
    const auto& record = records[index];
    if (index != 0) output << ',';
    output << "{\"grid_blocks\":" << record.geometry.grid_blocks
           << ",\"block_threads\":" << record.geometry.block_threads
           << ",\"samples_us\":[";
    for (size_t sample_index = 0; sample_index < record.samples_us.size(); ++sample_index) {
      if (sample_index != 0) output << ',';
      output << record.samples_us[sample_index];
    }
    output << "],\"validation_passed\":" << (record.validation_passed ? "true" : "false")
           << ",\"error\":\"" << json_escape(record.error) << "\""
           << ",\"median_us\":" << record.median_us
           << ",\"min_us\":" << record.min_us << ",\"max_us\":" << record.max_us << '}';
  }
  output << "]}\n";
}

template <typename Adapter>
inline void write_csv(std::ostream& output, const Adapter& adapter,
                      const BenchmarkOptions& options,
                      const std::vector<BenchmarkRecord>& records) {
  const auto& resources = adapter.resource_usage();
  output << "benchmark,variant,launch_bounds,used_registers_per_thread,stack_size_bytes,"
            "grid_blocks,block_threads,work_items,warmup,iterations,samples_us,"
            "validation_passed,error,median_us,min_us,max_us\n";
  for (const auto& record : records) {
    std::ostringstream samples;
    for (size_t index = 0; index < record.samples_us.size(); ++index) {
      if (index != 0) samples << ';';
      samples << record.samples_us[index];
    }
    output << csv_escape(adapter.benchmark_name()) << ',' << csv_escape(adapter.variant_name())
           << ',' << resources.launch_bounds << ',' << resources.used_registers_per_thread
           << ',' << resources.stack_size_bytes << ',' << record.geometry.grid_blocks << ','
           << record.geometry.block_threads << ',' << adapter.work_items() << ',' << options.warmup
           << ',' << options.iterations << ',' << csv_escape(samples.str()) << ','
           << (record.validation_passed ? "true" : "false") << ',' << csv_escape(record.error)
           << ',' << record.median_us << ',' << record.min_us << ',' << record.max_us << '\n';
  }
}

template <typename Adapter, typename Runtime>
int run_benchmark(int argc, char** argv, Adapter& adapter, Runtime& runtime,
                  std::ostream& error_output) {
  BenchmarkOptions options;
  std::string error;
  if (!parse_benchmark_options(argc, argv, &options, &error)) {
    error_output << error << '\n';
    return 2;
  }
  if (!adapter.initialize(&error)) {
    error_output << "initialize failed: " << error << '\n';
    return 1;
  }
  std::vector<BenchmarkRecord> records;
  int status = 0;
  for (const LaunchGeometry geometry : adapter.candidates()) {
    BenchmarkRecord record;
    record.geometry = geometry;
    if (!adapter.reset_iteration_state(&record.error) || !adapter.launch(geometry, &record.error) ||
        !adapter.synchronize(&record.error)) {
      status = 1;
      records.push_back(record);
      break;
    }
    record.validation_passed = adapter.validate(&record.error);
    if (!record.validation_passed) {
      status = 1;
      records.push_back(record);
      break;
    }
    for (uint32_t iteration = 0; iteration < options.warmup; ++iteration) {
      if (!adapter.reset_iteration_state(&record.error) || !adapter.launch(geometry, &record.error) ||
          !adapter.synchronize(&record.error)) {
        status = 1;
        break;
      }
    }
    if (status != 0) { records.push_back(record); break; }
    void* start_event = nullptr;
    void* end_event = nullptr;
    if (!runtime.create_event(&start_event, &record.error) ||
        !runtime.create_event(&end_event, &record.error)) {
      status = 1;
    }
    for (uint32_t iteration = 0; status == 0 && iteration < options.iterations; ++iteration) {
      double elapsed_us = 0;
      if (!adapter.reset_iteration_state(&record.error) ||
          !runtime.record_event(start_event, adapter.stream(), &record.error) ||
          !adapter.launch(geometry, &record.error) ||
          !runtime.record_event(end_event, adapter.stream(), &record.error) ||
          !runtime.synchronize_event(end_event, &record.error) ||
          !runtime.elapsed_us(&elapsed_us, start_event, end_event, &record.error)) {
        status = 1;
        break;
      }
      record.samples_us.push_back(elapsed_us);
    }
    if (start_event != nullptr) runtime.destroy_event(start_event);
    if (end_event != nullptr) runtime.destroy_event(end_event);
    if (!record.samples_us.empty()) finalize_statistics(&record);
    records.push_back(record);
    if (status != 0) break;
  }
  if (!options.json_path.empty()) {
    std::ofstream json(options.json_path);
    if (!json) { error_output << "cannot open JSON output\n"; status = 1; }
    else write_json(json, adapter, options, records);
  }
  if (!options.csv_path.empty()) {
    std::ofstream csv(options.csv_path);
    if (!csv) { error_output << "cannot open CSV output\n"; status = 1; }
    else write_csv(csv, adapter, options, records);
  }
  adapter.shutdown();
  if (status != 0) error_output << "benchmark failed\n";
  return status;
}

}  // namespace asc_occupancy
