import { describe, expect, it } from "vitest";
import type { ChartSeries } from "../types";
import {
  enforceSelectedSeries,
  relativePerformanceValue,
  selectPerformanceBaseline
} from "./performanceBaseline";

function series(
  key: string,
  backend: "nvidia" | "ascend",
  implementation: "cuda-pytorch" | "cuda_library" | "cann_ops_library" | "simt",
  values: Array<number | null>
): ChartSeries {
  return {
    key,
    name: key,
    records: [
      {
        backend,
        implementation
      } as ChartSeries["records"][number]
    ],
    points: values.map((latencyMs, index) => ({
      caseId: `case-${index}`,
      latencyMs,
      record: null
    }))
  };
}

describe("selectPerformanceBaseline", () => {
  it("prefers the first eligible CUDA series over CANN Ops", () => {
    const cann = series("cann", "ascend", "cann_ops_library", [0.02]);
    const cudaFirst = series("cuda-first", "nvidia", "cuda-pytorch", [0.01]);
    const cudaSecond = series("cuda-second", "nvidia", "cuda_library", [0.03]);

    expect(selectPerformanceBaseline([cann, cudaFirst, cudaSecond])).toEqual({
      seriesKey: "cuda-first",
      seriesName: "cuda-first",
      kind: "cuda"
    });
  });

  it("uses CANN Ops when CUDA has no finite positive point", () => {
    const cuda = series("cuda", "nvidia", "cuda-pytorch", [null, 0, Number.POSITIVE_INFINITY]);
    const cann = series("cann", "ascend", "cann_ops_library", [0.02]);

    expect(selectPerformanceBaseline([cuda, cann])).toEqual({
      seriesKey: "cann",
      seriesName: "cann",
      kind: "cann_ops"
    });
  });

  it("returns null without an eligible CUDA or CANN Ops series", () => {
    expect(selectPerformanceBaseline([series("simt", "ascend", "simt", [0.01])])).toBeNull();
  });
});

describe("relativePerformanceValue", () => {
  it("computes baseline latency divided by implementation latency", () => {
    expect(relativePerformanceValue(0.02, 0.01)).toBe(2);
    expect(relativePerformanceValue(0.02, 0.04)).toBe(0.5);
    expect(relativePerformanceValue(0.02, 0.02)).toBe(1);
  });

  it.each([
    [null, 0.01],
    [0.01, null],
    [0, 0.01],
    [0.01, 0],
    [Number.POSITIVE_INFINITY, 0.01],
    [0.01, Number.NaN]
  ])("returns null for an invalid pair %#", (baseline, implementation) => {
    expect(relativePerformanceValue(baseline, implementation)).toBeNull();
  });

  it("returns null when finite inputs overflow the quotient", () => {
    expect(relativePerformanceValue(Number.MAX_VALUE, Number.MIN_VALUE)).toBeNull();
  });

  it("returns null when finite inputs underflow the quotient", () => {
    expect(relativePerformanceValue(Number.MIN_VALUE, Number.MAX_VALUE)).toBeNull();
  });
});

describe("enforceSelectedSeries", () => {
  it("retains available selections and adds the mandatory baseline", () => {
    expect(enforceSelectedSeries(["cuda", "cann", "simt"], ["simt"], "cuda")).toEqual([
      "cuda",
      "simt"
    ]);
  });

  it("drops unavailable selections and selects all when nothing remains", () => {
    expect(enforceSelectedSeries(["cann", "simt"], ["old-cuda"], "cann")).toEqual([
      "cann",
      "simt"
    ]);
  });

  it("preserves available selections without a baseline", () => {
    expect(enforceSelectedSeries(["simt-v1", "simt-v2"], ["simt-v2"], null)).toEqual([
      "simt-v2"
    ]);
  });
});
