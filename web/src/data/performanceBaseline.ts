import type { ChartSeries, PerformanceBaseline } from "../types";

function hasEligiblePoint(series: ChartSeries): boolean {
  return series.points.some(
    (point) => point.latencyMs !== null && Number.isFinite(point.latencyMs) && point.latencyMs > 0
  );
}

function isCudaSeries(series: ChartSeries): boolean {
  return series.records.some((record) => record.backend === "nvidia" || record.backend === "gpu");
}

function isCannOpsSeries(series: ChartSeries): boolean {
  return series.records.some((record) => record.implementation === "cann_ops_library");
}

function descriptor(series: ChartSeries, kind: PerformanceBaseline["kind"]): PerformanceBaseline {
  return {
    seriesKey: series.key,
    seriesName: series.name,
    kind
  };
}

export function selectPerformanceBaseline(series: ChartSeries[]): PerformanceBaseline | null {
  const cuda = series.find((item) => hasEligiblePoint(item) && isCudaSeries(item));
  if (cuda) {
    return descriptor(cuda, "cuda");
  }
  const cannOps = series.find((item) => hasEligiblePoint(item) && isCannOpsSeries(item));
  return cannOps ? descriptor(cannOps, "cann_ops") : null;
}

export function relativePerformanceValue(
  baselineLatencyMs: number | null,
  implementationLatencyMs: number | null
): number | null {
  if (
    baselineLatencyMs === null ||
    implementationLatencyMs === null ||
    !Number.isFinite(baselineLatencyMs) ||
    !Number.isFinite(implementationLatencyMs) ||
    baselineLatencyMs <= 0 ||
    implementationLatencyMs <= 0
  ) {
    return null;
  }
  const ratio = baselineLatencyMs / implementationLatencyMs;
  return Number.isFinite(ratio) && ratio > 0 ? ratio : null;
}

export function enforceSelectedSeries(
  availableKeys: string[],
  selectedKeys: string[],
  baselineKey: string | null
): string[] {
  const kept = selectedKeys.filter((key) => availableKeys.includes(key));
  if (kept.length === 0) {
    return availableKeys;
  }
  return baselineKey && !kept.includes(baselineKey) ? [baselineKey, ...kept] : kept;
}
