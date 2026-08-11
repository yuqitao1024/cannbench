import { afterEach, describe, expect, it, vi } from "vitest";
import type { ShapeTrace, ShapeTraceIndexEntry } from "../shape-trace/types";
import {
  fetchShapeTrace,
  fetchShapeTraceIndex,
  ShapeTraceApiError,
  shapeTraceKey
} from "./shapeTraceApi";

const indexEntry: ShapeTraceIndexEntry = {
  operator: "dsa_decode",
  dataset: "realistic",
  case_id: "case-1",
  phase: "decode",
  group: "deepseek-v32"
};

const axis = { symbol: "D", value: 128, meaning: "feature width", role: "preserved" as const };

const tensor = {
  id: "query",
  label: "Query",
  logical_only: false,
  axes: [axis]
};

const stage = {
  id: "projection",
  component: "projection",
  title: "Projection",
  operation: "transform",
  formula: "query -> output",
  scope: "one row",
  tensors: [tensor],
  input_ids: ["query"],
  output_ids: [],
  contracted_axes: [],
  insight: "Projects one row."
};

const trace: ShapeTrace = {
  ...indexEntry,
  schema_version: 1,
  symbols: [axis],
  stages: [stage],
  device_execution: {
    status: "unavailable",
    implementation: "simt",
    version: null,
    message: "No device trace.",
    kernels: []
  }
};

function traceWithStage(overrides: Record<string, unknown> = {}): unknown {
  return {
    ...trace,
    stages: [
      { ...stage, ...overrides }
    ]
  };
}

function response(payload: unknown): Response {
  return { ok: true, status: 200, json: async () => payload } as Response;
}

function errorResponse(status: number): Response {
  return { ok: false, status } as Response;
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

  it("reports an index request failure with typed request context", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => errorResponse(404)));

    const request = fetchShapeTraceIndex();
    await expect(request).rejects.toBeInstanceOf(ShapeTraceApiError);
    await expect(request).rejects.toMatchObject({ request: "index", status: 404 });
  });

  it("reports a detail request failure with typed request context", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => errorResponse(404)));

    const request = fetchShapeTrace("operator", "dataset", "case");
    await expect(request).rejects.toBeInstanceOf(ShapeTraceApiError);
    await expect(request).rejects.toMatchObject({ request: "detail", status: 404 });
  });

  it("rejects a malformed trace index", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response({ traces: null })));
    await expect(fetchShapeTraceIndex()).rejects.toThrow("invalid shape trace index payload");
  });

  it("rejects malformed primitive fields in a trace index entry", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({ traces: [{ ...indexEntry, phase: 3 }] }))
    );
    await expect(fetchShapeTraceIndex()).rejects.toThrow("invalid shape trace index entry");
  });

  it("rejects duplicate identities in the trace index", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({ traces: [indexEntry, { ...indexEntry }] }))
    );
    await expect(fetchShapeTraceIndex()).rejects.toThrow("duplicate shape trace index identity");
  });

  it("rejects an unsupported detail schema version", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => response({ ...trace, schema_version: 2 })));
    await expect(fetchShapeTrace("operator", "dataset", "case")).rejects.toThrow(
      "invalid shape trace schema_version"
    );
  });

  it.each([
    ["axis value", { ...trace, symbols: [{ ...axis, value: 0 }] }],
    ["tensor logical_only", traceWithStage({ tensors: [{ ...tensor, logical_only: "no" }] })],
    ["stage component", traceWithStage({ component: 1 })],
    ["contracted axis reference", traceWithStage({ contracted_axes: ["missing"] })]
  ])("rejects malformed %s fields", async (_label, payload) => {
    vi.stubGlobal("fetch", vi.fn(async () => response(payload)));
    await expect(fetchShapeTrace("operator", "dataset", "case")).rejects.toThrow(
      "invalid shape trace"
    );
  });

  it("rejects duplicate stage ids", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response({ ...trace, stages: [stage, { ...stage }] }))
    );
    await expect(fetchShapeTrace("operator", "dataset", "case")).rejects.toThrow(
      "duplicate shape trace stage id"
    );
  });

  it("rejects inconsistent device status fields", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          ...trace,
          device_execution: {
            status: "available",
            implementation: "simt",
            version: "v1",
            message: null,
            kernels: []
          }
        })
      )
    );
    await expect(fetchShapeTrace("operator", "dataset", "case")).rejects.toThrow(
      "invalid available device execution"
    );
  });

  it("rejects malformed device kernel counts", async () => {
    const kernel = {
      id: "kernel",
      title: "Kernel",
      summary: "Summary",
      task_count: 0,
      used_core_count: 1,
      task_formula: "B",
      task_axes: [axis],
      tile_tensors: [tensor],
      steps: ["Run."]
    };
    vi.stubGlobal(
      "fetch",
      vi.fn(async () =>
        response({
          ...trace,
          device_execution: {
            status: "available",
            implementation: "simt",
            version: "v1",
            message: null,
            kernels: [kernel]
          }
        })
      )
    );
    await expect(fetchShapeTrace("operator", "dataset", "case")).rejects.toThrow(
      "invalid device kernel task/core counts"
    );
  });

  it("rejects duplicate tensor ids at the fetched trace boundary", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response(traceWithStage({ tensors: [tensor, { ...tensor }] })))
    );

    await expect(fetchShapeTrace("operator", "dataset", "case")).rejects.toThrow(
      "duplicate tensor id: query"
    );
  });

  it("rejects missing tensor references at the fetched trace boundary", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => response(traceWithStage({ output_ids: ["missing"] })))
    );

    await expect(fetchShapeTrace("operator", "dataset", "case")).rejects.toThrow(
      "unknown tensor id: missing"
    );
  });

  it.each(["input_ids", "output_ids"] as const)(
    "rejects duplicate tensor ids within %s at the fetched trace boundary",
    async (fieldName) => {
      vi.stubGlobal(
        "fetch",
        vi.fn(async () =>
          response(traceWithStage({ [fieldName]: ["query", "query"] }))
        )
      );

      await expect(fetchShapeTrace("operator", "dataset", "case")).rejects.toThrow(
        `${fieldName} contains duplicate tensor id: query`
      );
    }
  );

  it("allows the same tensor id once in each reference list", async () => {
    const payload = traceWithStage({ output_ids: ["query"] });
    vi.stubGlobal("fetch", vi.fn(async () => response(payload)));

    await expect(fetchShapeTrace("operator", "dataset", "case")).resolves.toEqual(payload);
  });

  it("builds a stable identity key", () => {
    expect(shapeTraceKey(indexEntry)).toBe("dsa_decode\u0000realistic\u0000case-1");
  });
});
