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
    implementation: "cuda-pytorch",
    implementation_version: "cuda-pytorch",
    source_kind: "real_model",
    source_project: "TritonBench",
    source_model: "GPTJForCausalLM",
    source_file: "hf_train/GPTJForCausalLM_train.json",
    source_op: "aten._softmax.default",
    metrics: { latency_ms: 0.011},
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
    metrics: { latency_ms: 0.014},
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
    metrics: { latency_ms: 0.021},
    accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
    diff_ref: null
  }
];

const dsaRecords: BenchmarkRecord[] = [
  {
    schema_version: 1,
    run_id: "dsa-decode-cuda",
    operator: "sparse_attention",
    dataset: "realistic_decode",
    case_id: "deepseek_a5_decode_b1_ctx512_top512",
    family: "decode_sparse_attention",
    shape: [1, 64, 512],
    dtype: "bfloat16",
    backend: "nvidia",
    device_class: "H800",
    implementation: "cuda_library",
    implementation_version: "cuda-library",
    source_kind: "library_compatible_realistic",
    source_project: "FlashMLA",
    source_model: "DeepSeek-V4-compatible",
    source_file: "benchmarks",
    source_op: "flash_mla_with_kvcache",
    metrics: { latency_ms: 0.06},
    accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
    diff_ref: null
  },
  {
    schema_version: 1,
    run_id: "dsa-decode",
    operator: "sparse_attention",
    dataset: "realistic_decode",
    case_id: "deepseek_a5_decode_b1_ctx512_top512",
    family: "decode_sparse_attention",
    shape: [1, 64, 512],
    dtype: "bfloat16",
    backend: "ascend",
    device_class: "950PR",
    implementation: "vllm_ascend",
    implementation_version: "vllm-ascend",
    source_kind: "library_compatible_realistic",
    source_project: "vllm-ascend",
    source_model: "DeepSeek-V4-compatible",
    source_file: "csrc/attention/kv_quant_sparse_attn_sharedkv/README.md",
    source_op: "npu_kv_quant_sparse_attn_sharedkv",
    metrics: { latency_ms: 0.08},
    accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
    diff_ref: null
  },
  {
    schema_version: 1,
    run_id: "dsa-prefill",
    operator: "sparse_attention",
    dataset: "realistic_prefill",
    case_id: "deepseek_a5_prefill_b1_q512_ctx512_top512",
    family: "prefill_sparse_attention",
    shape: [512, 64, 512],
    dtype: "bfloat16",
    backend: "ascend",
    device_class: "950PR",
    implementation: "vllm_ascend",
    implementation_version: "vllm-ascend",
    source_kind: "library_compatible_realistic",
    source_project: "vllm-ascend",
    source_model: "DeepSeek-V4-compatible",
    source_file: "csrc/attention/kv_quant_sparse_attn_sharedkv/README.md",
    source_op: "npu_kv_quant_sparse_attn_sharedkv",
    metrics: { latency_ms: 1.8},
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

  it("orders cases and chart points by shape element count", () => {
    const cudaRecord = records[0];
    const unorderedRecords: BenchmarkRecord[] = [
      { ...cudaRecord, run_id: "large", case_id: "large", shape: [3, 3, 3] },
      { ...cudaRecord, run_id: "same-shape-b", case_id: "same_shape_b", shape: [2, 8] },
      { ...cudaRecord, run_id: "small", case_id: "small", shape: [3, 4] },
      { ...cudaRecord, run_id: "same-volume", case_id: "same_volume", shape: [4, 4] },
      { ...cudaRecord, run_id: "same-shape-a", case_id: "same_shape_a", shape: [2, 8] }
    ];
    const expectedOrder = ["small", "same_shape_a", "same_shape_b", "same_volume", "large"];

    const model = buildBenchmarkViewModel(unorderedRecords);

    expect(model.casesFor("softmax", "realistic").map((item) => item.caseId)).toEqual(expectedOrder);
    expect(model.seriesFor("softmax", "realistic")[0].points.map((point) => point.caseId)).toEqual(
      expectedOrder
    );
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

  it("keeps DSA decode and prefill datasets as separate chart groups", () => {
    const model = buildBenchmarkViewModel(dsaRecords);

    expect(model.datasetsFor("sparse_attention")).toEqual([
      "ALL",
      "realistic_decode",
      "realistic_prefill"
    ]);
    expect(model.chartSegmentsFor("sparse_attention", "ALL")).toEqual([
      { key: "realistic_decode", label: "realistic_decode", start: 0, end: 0 },
      { key: "realistic_prefill", label: "realistic_prefill", start: 1, end: 1 }
    ]);
    expect(model.seriesFor("sparse_attention", "realistic_decode").map((series) => series.name)).toEqual([
      "NVIDIA H800 CUDA Library",
      "Ascend 950PR vLLM Ascend"
    ]);
    expect(model.seriesFor("sparse_attention", "realistic_prefill").map((series) => series.name)).toEqual([
      "Ascend 950PR vLLM Ascend"
    ]);
  });
});
