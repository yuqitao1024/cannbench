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
    family: "attention",
    shape: [1, 16, 128, 128],
    dtype: "float16",
    backend: "nvidia",
    device_class: "H800",
    implementation: "ncu",
    implementation_version: "ncu",
    source_kind: "real_model",
    source_project: "TritonBench",
    source_model: "GPTJForCausalLM",
    source_file: "hf_train/GPTJForCausalLM_train.json",
    source_op: "aten._softmax.default",
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
    family: "attention",
    shape: [1, 16, 128, 128],
    dtype: "float16",
    backend: "ascend",
    device_class: "950PR",
    implementation: "simt",
    implementation_version: "v1",
    source_kind: "real_model",
    source_project: "TritonBench",
    source_model: "GPTJForCausalLM",
    source_file: "hf_train/GPTJForCausalLM_train.json",
    source_op: "aten._softmax.default",
    metrics: { latency_ms_avg: 0.014, latency_ms_p50: 0.014, latency_ms_p95: 0.015, sample_count: 1 },
    accuracy: { passed: true, max_abs_error: 0.0004, max_rel_error: 0.0008 },
    diff_ref: "softmax/simt/v1"
  },
  {
    schema_version: 1,
    run_id: "run-c",
    operator: "softmax",
    dataset: "smoke",
    case_id: "tiny_logits",
    family: "lm_logits",
    shape: [128, 128],
    dtype: "float16",
    backend: "ascend",
    device_class: "950PR",
    implementation: "cann_ops_library",
    implementation_version: "cannops",
    source_kind: "synthetic_smoke",
    source_project: "cannbench",
    source_model: "smoke_fixture",
    source_file: "built-in",
    source_op: "softmax",
    metrics: { latency_ms_avg: 0.021, latency_ms_p50: 0.02, latency_ms_p95: 0.024, sample_count: 1 },
    accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
    diff_ref: null
  }
];

describe("buildBenchmarkViewModel", () => {
  it("groups records by operator, dataset, case, and chart series", () => {
    const model = buildBenchmarkViewModel(records);

    expect(model.operators.map((operator) => operator.name)).toEqual(["softmax"]);
    expect(model.datasetsFor("softmax")).toEqual(["ALL", "smoke", "realistic"]);
    expect(model.casesFor("softmax", "realistic")).toHaveLength(1);
    expect(model.casesFor("softmax", "ALL").map((item) => item.caseId)).toEqual([
      "tiny_logits",
      "gptj_attention"
    ]);
    expect(model.seriesFor("softmax", "realistic").map((series) => series.name)).toEqual([
      "NVIDIA H800 PyTorch",
      "Ascend 950PR SIMT v1"
    ]);
  });

  it("exposes dataset coverage metadata in case summaries", () => {
    const model = buildBenchmarkViewModel(records);
    const [realistic] = model.casesFor("softmax", "realistic");
    const [smoke] = model.casesFor("softmax", "smoke");

    expect(realistic.sourceLabel).toBe("TritonBench / GPTJForCausalLM");
    expect(realistic.coverageTag).toBe("real-model coverage");
    expect(realistic.availableSeries).toEqual(["NVIDIA H800 PyTorch", "Ascend 950PR SIMT v1"]);
    expect(smoke.sourceLabel).toBe("cannbench / smoke_fixture");
    expect(smoke.coverageTag).toBe("smoke coverage");
  });
});
