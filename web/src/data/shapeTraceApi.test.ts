import { afterEach, describe, expect, it, vi } from "vitest";
import type { ShapeTrace, ShapeTraceIndexEntry } from "../shape-trace/types";
import { fetchShapeTrace, fetchShapeTraceIndex, shapeTraceKey } from "./shapeTraceApi";

const indexEntry: ShapeTraceIndexEntry = {
  operator: "dsa_decode",
  dataset: "realistic",
  case_id: "case-1",
  phase: "decode",
  group: "deepseek-v32"
};

const trace: ShapeTrace = {
  ...indexEntry,
  schema_version: 1,
  symbols: [],
  stages: [],
  device_execution: {
    status: "unavailable",
    implementation: "simt",
    version: null,
    message: "No device trace.",
    kernels: []
  }
};

function response(payload: unknown): Response {
  return { ok: true, status: 200, json: async () => payload } as Response;
}

afterEach(() => {
  vi.unstubAllGlobals();
});

describe("shapeTraceApi", () => {
  it("loads the trace index", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response({ traces: [indexEntry] })));
    await expect(fetchShapeTraceIndex()).resolves.toEqual([indexEntry]);
  });

  it("loads one encoded trace", async () => {
    const fetchMock = vi.fn(async () => response(trace));
    vi.stubGlobal("fetch", fetchMock);
    await expect(fetchShapeTrace("dsa decode", "realistic", "case/1")).resolves.toEqual(trace);
    expect(fetchMock).toHaveBeenCalledWith(
      "/api/shape-trace?operator=dsa+decode&dataset=realistic&case=case%2F1",
      { signal: undefined }
    );
  });

  it("rejects a malformed trace index", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response({ traces: null })));
    await expect(fetchShapeTraceIndex()).rejects.toThrow("invalid shape trace index payload");
  });

  it("builds a stable identity key", () => {
    expect(shapeTraceKey(indexEntry)).toBe("dsa_decode\u0000realistic\u0000case-1");
  });
});
