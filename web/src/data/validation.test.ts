import { describe, expect, it } from "vitest";
import { validateGpuBenchmarkUpload } from "./validation";

const validRecord = {
  schema_version: 1,
  run_id: "gpu-run-1",
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
};

describe("validateGpuBenchmarkUpload", () => {
  it("accepts allowlisted GPU performance records", () => {
    const result = validateGpuBenchmarkUpload({ records: [validRecord] });

    expect(result.ok).toBe(true);
    expect(result.acceptedCount).toBe(1);
  });

  it("rejects non-GPU records", () => {
    const result = validateGpuBenchmarkUpload({ records: [{ ...validRecord, backend: "ascend" }] });

    expect(result.ok).toBe(false);
    expect(result.errors).toContain("records[0].backend must be nvidia or gpu");
  });

  it("rejects sensitive fields recursively", () => {
    const result = validateGpuBenchmarkUpload({
      records: [{ ...validRecord, metrics: { ...validRecord.metrics, stdout: "secret" } }]
    });

    expect(result.ok).toBe(false);
    expect(result.errors[0]).toMatch(/sensitive field/i);
  });

  it("rejects unknown top-level fields", () => {
    const result = validateGpuBenchmarkUpload({ records: [{ ...validRecord, hostname: "build-host" }] });

    expect(result.ok).toBe(false);
    expect(result.errors[0]).toMatch(/sensitive field/i);
  });
});
