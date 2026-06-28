import type { BenchmarkRecord } from "../types";

interface BenchmarkPayload {
  records: BenchmarkRecord[];
}

interface PublishedRunsIndex {
  runs: string[];
}

export async function loadBenchmarkRecords(signal?: AbortSignal): Promise<BenchmarkRecord[]> {
  const indexResponse = await fetch("/published/index.json", { signal });
  if (!indexResponse.ok) {
    throw new Error(`failed to load published run index: ${indexResponse.status}`);
  }

  const indexPayload = (await indexResponse.json()) as PublishedRunsIndex;
  if (!indexPayload || !Array.isArray(indexPayload.runs)) {
    throw new Error("invalid published run index payload");
  }

  const records: BenchmarkRecord[] = [];
  for (const runName of indexPayload.runs) {
    const response = await fetch(`/published/${runName}/meta/benchmark-records.json`, { signal });
    if (!response.ok) {
      throw new Error(`failed to load benchmark records for ${runName}: ${response.status}`);
    }
    const payload = (await response.json()) as BenchmarkPayload;
    if (!payload || !Array.isArray(payload.records)) {
      throw new Error(`invalid benchmark record payload for ${runName}`);
    }
    records.push(...payload.records);
  }
  return records;
}
