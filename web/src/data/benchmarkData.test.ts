import { describe, expect, it } from "vitest";
import { buildBenchmarkViewModel } from "./benchmarkData";
import type { BenchmarkRecord } from "../types";

const records: BenchmarkRecord[] = [
  {
    schema_version: 1,
    run_id: "run-a",
    operator: "softmax",
    dataset: "realistic",
    case_id: "gptj_attention",
    shape: [1, 16, 128, 128],
    dtype: "float16",
    backend: "nvidia",
    device_class: "H800",
    implementation: "cuda_event",
    implementation_version: "cuda-event",
    metrics: { latency_ms_avg: 0.011, latency_ms_p50: 0.011, latency_ms_p95: 0.012, sample_count: 1 },
    accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
    diff_ref: null
  },
  {
    schema_version: 1,
    run_id: "run-b",
    operator: "softmax",
    dataset: "realistic",
    case_id: "gptj_attention",
    shape: [1, 16, 128, 128],
    dtype: "float16",
    backend: "ascend",
    device_class: "Ascend",
    implementation: "simt",
    implementation_version: "dynamic-ubuf",
    metrics: { latency_ms_avg: 0.014, latency_ms_p50: 0.014, latency_ms_p95: 0.015, sample_count: 1 },
    accuracy: { passed: true, max_abs_error: 0.0004, max_rel_error: 0.0008 },
    diff_ref: "softmax/simt/dynamic-ubuf"
  }
];

describe("buildBenchmarkViewModel", () => {
  it("groups records by operator, dataset, case, and chart series", () => {
    const model = buildBenchmarkViewModel(records);

    expect(model.operators.map((operator) => operator.name)).toEqual(["softmax"]);
    expect(model.datasetsFor("softmax")).toEqual(["realistic"]);
    expect(model.casesFor("softmax", "realistic")).toHaveLength(1);
    expect(model.seriesFor("softmax", "realistic").map((series) => series.name)).toEqual([
      "GPU H800",
      "SIMT op: dynamic-ubuf"
    ]);
  });
});
