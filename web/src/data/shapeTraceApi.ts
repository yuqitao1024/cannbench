import type { ShapeTrace, ShapeTraceIndexEntry } from "../shape-trace/types";

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function stringArray(value: unknown): string[] | null {
  return Array.isArray(value) && value.every((item) => typeof item === "string")
    ? value
    : null;
}

function validateStageTensorReferences(stage: unknown): void {
  if (!isObject(stage) || !Array.isArray(stage.tensors)) {
    throw new Error("invalid shape trace stage payload");
  }

  const seen = new Set<string>();
  for (const tensor of stage.tensors) {
    if (!isObject(tensor) || typeof tensor.id !== "string" || tensor.id.length === 0) {
      throw new Error("invalid shape trace tensor payload");
    }
    if (seen.has(tensor.id)) {
      throw new Error(`duplicate tensor id: ${tensor.id}`);
    }
    seen.add(tensor.id);
  }

  const inputIds = stringArray(stage.input_ids);
  const outputIds = stringArray(stage.output_ids);
  if (inputIds === null || outputIds === null) {
    throw new Error("invalid shape trace tensor references");
  }
  const missing = [...inputIds, ...outputIds].find((id) => !seen.has(id));
  if (missing !== undefined) {
    throw new Error(`unknown tensor id: ${missing}`);
  }
}

function validateShapeTrace(payload: unknown): ShapeTrace {
  if (!isObject(payload) || !Array.isArray(payload.stages)) {
    throw new Error("invalid shape trace payload");
  }
  payload.stages.forEach(validateStageTensorReferences);
  return payload as unknown as ShapeTrace;
}

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
  return validateShapeTrace(await response.json());
}
