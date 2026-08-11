import type { ShapeTrace, ShapeTraceIndexEntry } from "../shape-trace/types";

export function shapeTraceKey(
  value: Pick<ShapeTraceIndexEntry, "operator" | "dataset" | "case_id">
): string {
  return `${value.operator}\u0000${value.dataset}\u0000${value.case_id}`;
}

export async function fetchShapeTraceIndex(signal?: AbortSignal): Promise<ShapeTraceIndexEntry[]> {
  const response = await fetch("/api/shape-traces", { signal });
  if (!response.ok) {
    throw new Error(`shape trace index request failed: ${response.status}`);
  }
  const payload = (await response.json()) as { traces?: ShapeTraceIndexEntry[] };
  if (!Array.isArray(payload.traces)) {
    throw new Error("invalid shape trace index payload");
  }
  return payload.traces;
}

export async function fetchShapeTrace(
  operator: string,
  dataset: string,
  caseId: string,
  signal?: AbortSignal
): Promise<ShapeTrace> {
  const params = new URLSearchParams({ operator, dataset, case: caseId });
  const response = await fetch(`/api/shape-trace?${params.toString()}`, { signal });
  if (!response.ok) {
    throw new Error(`shape trace request failed: ${response.status}`);
  }
  return (await response.json()) as ShapeTrace;
}
