import { afterEach, describe, expect, it, vi } from "vitest";
import { loadBenchmarkRecords } from "./benchmarkRecordsApi";

describe("loadBenchmarkRecords", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("loads and merges benchmark records from the published run index", async () => {
    const fetchMock = vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      if (url === "/published/index.json") {
        return {
          ok: true,
          json: async () => ({
            runs: [
              "opbench-ascend-950pr-cann-cannops-softmax-realistic-float16",
              "opbench-ascend-950pr-simt-v1-softmax-realistic-float16"
            ]
          })
        };
      }
      if (url === "/published/opbench-ascend-950pr-cann-cannops-softmax-realistic-float16/meta/benchmark-records.json") {
        return {
          ok: true,
          json: async () => ({
            records: [{ run_id: "opbench-ascend-950pr-cann-cannops-softmax-realistic-float16" }]
          })
        };
      }
      if (url === "/published/opbench-ascend-950pr-simt-v1-softmax-realistic-float16/meta/benchmark-records.json") {
        return {
          ok: true,
          json: async () => ({
            records: [{ run_id: "opbench-ascend-950pr-simt-v1-softmax-realistic-float16" }]
          })
        };
      }
      throw new Error(`unexpected fetch: ${url}`);
    });
    vi.stubGlobal("fetch", fetchMock);

    const records = await loadBenchmarkRecords();

    expect(records).toEqual([
      { run_id: "opbench-ascend-950pr-cann-cannops-softmax-realistic-float16" },
      { run_id: "opbench-ascend-950pr-simt-v1-softmax-realistic-float16" }
    ]);
    expect(fetchMock).toHaveBeenNthCalledWith(1, "/published/index.json", { signal: undefined });
    expect(fetchMock).toHaveBeenNthCalledWith(
      2,
      "/published/opbench-ascend-950pr-cann-cannops-softmax-realistic-float16/meta/benchmark-records.json",
      { signal: undefined }
    );
    expect(fetchMock).toHaveBeenNthCalledWith(
      3,
      "/published/opbench-ascend-950pr-simt-v1-softmax-realistic-float16/meta/benchmark-records.json",
      { signal: undefined }
    );
  });
});
