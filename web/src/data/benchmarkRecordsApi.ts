import type { BenchmarkRecord } from "../types";

interface BenchmarkPayload {
  records: BenchmarkRecord[];
}

export const DEFAULT_PUBLISHED_RUN = "default";

export async function loadBenchmarkRecords(signal?: AbortSignal): Promise<BenchmarkRecord[]> {
  const response = await fetch(
    `/published/${DEFAULT_PUBLISHED_RUN}/meta/benchmark-records.json`,
    { signal }
  );
  if (!response.ok) {
    throw new Error(`failed to load benchmark records: ${response.status}`);
  }
  const payload = (await response.json()) as BenchmarkPayload;
  if (!payload || !Array.isArray(payload.records)) {
    throw new Error("invalid benchmark record payload");
  }
  return payload.records;
}
