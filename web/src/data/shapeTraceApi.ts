import type {
  AxisRole,
  DeviceKernelTrace,
  ShapeAxis,
  ShapeStage,
  ShapeTensor,
  ShapeTrace,
  ShapeTraceIndexEntry
} from "../shape-trace/types";

export type ShapeTraceApiRequest = "index" | "detail";

export class ShapeTraceApiError extends Error {
  constructor(
    public readonly request: ShapeTraceApiRequest,
    public readonly status: number
  ) {
    super(`shape trace ${request} request failed: ${status}`);
    this.name = "ShapeTraceApiError";
  }
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function nonEmptyString(value: unknown, fieldName: string): string {
  if (typeof value !== "string" || value.trim().length === 0) {
    throw new Error(`invalid shape trace ${fieldName}`);
  }
  return value;
}

function nullableNonEmptyString(value: unknown, fieldName: string): string | null {
  return value === null ? null : nonEmptyString(value, fieldName);
}

function positiveInteger(value: unknown, fieldName: string): number {
  if (!Number.isInteger(value) || (value as number) <= 0) {
    throw new Error(`invalid shape trace ${fieldName}`);
  }
  return value as number;
}

function uniqueValues(values: string[], fieldName: string): void {
  const seen = new Set<string>();
  for (const value of values) {
    if (seen.has(value)) throw new Error(`duplicate ${fieldName}: ${value}`);
    seen.add(value);
  }
}

function decodeStringArray(value: unknown, fieldName: string): string[] {
  if (!Array.isArray(value)) throw new Error(`invalid shape trace ${fieldName}`);
  return value.map((item) => nonEmptyString(item, fieldName));
}

function decodeAxisRole(value: unknown): AxisRole {
  if (
    value !== "preserved" &&
    value !== "contracted" &&
    value !== "reduced" &&
    value !== "produced"
  ) {
    throw new Error("invalid shape trace axis role");
  }
  return value;
}

function decodeAxis(value: unknown): ShapeAxis {
  if (!isObject(value)) throw new Error("invalid shape trace axis");
  return {
    symbol: nonEmptyString(value.symbol, "axis symbol"),
    value: positiveInteger(value.value, "axis value"),
    meaning: nonEmptyString(value.meaning, "axis meaning"),
    role: decodeAxisRole(value.role)
  };
}

function decodeAxisArray(value: unknown, fieldName: string): ShapeAxis[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`invalid shape trace ${fieldName}`);
  }
  const axes = value.map(decodeAxis);
  uniqueValues(
    axes.map((axis) => axis.symbol),
    `${fieldName} symbol`
  );
  return axes;
}

function decodeTensor(value: unknown): ShapeTensor {
  if (!isObject(value)) throw new Error("invalid shape trace tensor payload");
  const axes = decodeAxisArray(value.axes, "tensor axes");
  if (axes.length > 3 || typeof value.logical_only !== "boolean") {
    throw new Error("invalid shape trace tensor payload");
  }
  return {
    id: nonEmptyString(value.id, "tensor id"),
    label: nonEmptyString(value.label, "tensor label"),
    axes,
    logical_only: value.logical_only
  };
}

function decodeTensorArray(value: unknown, fieldName: string): ShapeTensor[] {
  if (!Array.isArray(value) || value.length === 0) {
    throw new Error(`invalid shape trace ${fieldName}`);
  }
  const tensors = value.map(decodeTensor);
  uniqueValues(
    tensors.map((tensor) => tensor.id),
    fieldName === "stage tensors" ? "tensor id" : `${fieldName} id`
  );
  return tensors;
}

function validateUniqueReferences(fieldName: string, ids: string[]): void {
  const seen = new Set<string>();
  for (const id of ids) {
    if (seen.has(id)) {
      throw new Error(`${fieldName} contains duplicate tensor id: ${id}`);
    }
    seen.add(id);
  }
}

function decodeStage(value: unknown): ShapeStage {
  if (!isObject(value)) throw new Error("invalid shape trace stage payload");
  const tensors = decodeTensorArray(value.tensors, "stage tensors");
  const inputIds = decodeStringArray(value.input_ids, "input_ids");
  const outputIds = decodeStringArray(value.output_ids, "output_ids");
  const contractedAxes = decodeStringArray(value.contracted_axes, "contracted_axes");
  validateUniqueReferences("input_ids", inputIds);
  validateUniqueReferences("output_ids", outputIds);
  uniqueValues(contractedAxes, "contracted_axes");

  const tensorIds = new Set(tensors.map((tensor) => tensor.id));
  const missingTensor = [...inputIds, ...outputIds].find((id) => !tensorIds.has(id));
  if (missingTensor !== undefined) throw new Error(`unknown tensor id: ${missingTensor}`);

  const tensorAxes = new Set(
    tensors.flatMap((tensor) => tensor.axes.map((axis) => axis.symbol))
  );
  const missingAxis = contractedAxes.find((symbol) => !tensorAxes.has(symbol));
  if (missingAxis !== undefined) {
    throw new Error(`invalid shape trace contracted axis reference: ${missingAxis}`);
  }

  return {
    id: nonEmptyString(value.id, "stage id"),
    component: nonEmptyString(value.component, "stage component"),
    title: nonEmptyString(value.title, "stage title"),
    operation: nonEmptyString(value.operation, "stage operation"),
    formula: nonEmptyString(value.formula, "stage formula"),
    scope: nonEmptyString(value.scope, "stage scope"),
    tensors,
    input_ids: inputIds,
    output_ids: outputIds,
    contracted_axes: contractedAxes,
    insight: nonEmptyString(value.insight, "stage insight")
  };
}

function decodeIndexEntry(value: unknown): ShapeTraceIndexEntry {
  if (!isObject(value)) throw new Error("invalid shape trace index entry");
  try {
    return {
      operator: nonEmptyString(value.operator, "operator"),
      dataset: nonEmptyString(value.dataset, "dataset"),
      case_id: nonEmptyString(value.case_id, "case_id"),
      phase: nonEmptyString(value.phase, "phase"),
      group: nonEmptyString(value.group, "group")
    };
  } catch {
    throw new Error("invalid shape trace index entry");
  }
}

function decodeKernel(value: unknown): DeviceKernelTrace {
  if (!isObject(value)) throw new Error("invalid device kernel payload");
  if (
    !Number.isInteger(value.task_count) ||
    (value.task_count as number) <= 0 ||
    !Number.isInteger(value.used_core_count) ||
    (value.used_core_count as number) <= 0 ||
    (value.used_core_count as number) > (value.task_count as number)
  ) {
    throw new Error("invalid device kernel task/core counts");
  }
  const taskCount = value.task_count as number;
  const usedCoreCount = value.used_core_count as number;
  const steps = decodeStringArray(value.steps, "device kernel steps");
  if (steps.length === 0) throw new Error("invalid shape trace device kernel steps");
  return {
    id: nonEmptyString(value.id, "device kernel id"),
    title: nonEmptyString(value.title, "device kernel title"),
    summary: nonEmptyString(value.summary, "device kernel summary"),
    task_count: taskCount,
    used_core_count: usedCoreCount,
    task_formula: nonEmptyString(value.task_formula, "device kernel task_formula"),
    task_axes: decodeAxisArray(value.task_axes, "device kernel task axes"),
    tile_tensors: decodeTensorArray(value.tile_tensors, "device kernel tile tensors"),
    steps
  };
}

function decodeDeviceExecution(value: unknown): ShapeTrace["device_execution"] {
  if (!isObject(value) || !Array.isArray(value.kernels)) {
    throw new Error("invalid shape trace device execution");
  }
  const implementation = nonEmptyString(value.implementation, "device implementation");
  const version = nullableNonEmptyString(value.version, "device version");
  const message = nullableNonEmptyString(value.message, "device message");
  const kernels = value.kernels.map(decodeKernel);
  uniqueValues(
    kernels.map((kernel) => kernel.id),
    "device kernel id"
  );

  if (value.status === "available") {
    if (version === null || message !== null || kernels.length === 0) {
      throw new Error("invalid available device execution");
    }
    return { status: "available", implementation, version, message, kernels };
  }
  if (value.status === "unavailable") {
    if (message === null || kernels.length !== 0) {
      throw new Error("invalid unavailable device execution");
    }
    return { status: "unavailable", implementation, version, message, kernels };
  }
  throw new Error("invalid shape trace device status");
}

function validateStageAxes(stages: ShapeStage[], symbols: ShapeAxis[]): void {
  const bySymbol = new Map(symbols.map((axis) => [axis.symbol, axis]));
  for (const stage of stages) {
    for (const tensor of stage.tensors) {
      for (const axis of tensor.axes) {
        const symbol = bySymbol.get(axis.symbol);
        if (
          symbol === undefined ||
          symbol.value !== axis.value ||
          symbol.meaning !== axis.meaning ||
          symbol.role !== axis.role
        ) {
          throw new Error(`invalid shape trace stage axis reference: ${axis.symbol}`);
        }
      }
    }
  }
}

function decodeShapeTrace(payload: unknown): ShapeTrace {
  if (!isObject(payload)) throw new Error("invalid shape trace payload");
  if (payload.schema_version !== 1) {
    throw new Error("invalid shape trace schema_version");
  }
  if (!Array.isArray(payload.stages) || payload.stages.length === 0) {
    throw new Error("invalid shape trace stages");
  }
  const identity = decodeIndexEntry(payload);
  const symbols = decodeAxisArray(payload.symbols, "symbols");
  const stages = payload.stages.map(decodeStage);
  uniqueValues(
    stages.map((stage) => stage.id),
    "shape trace stage id"
  );
  validateStageAxes(stages, symbols);
  return {
    ...identity,
    schema_version: 1,
    symbols,
    stages,
    device_execution: decodeDeviceExecution(payload.device_execution)
  };
}

export function shapeTraceKey(
  value: Pick<ShapeTraceIndexEntry, "operator" | "dataset" | "case_id">
): string {
  return `${value.operator}\u0000${value.dataset}\u0000${value.case_id}`;
}

export async function fetchShapeTraceIndex(signal?: AbortSignal): Promise<ShapeTraceIndexEntry[]> {
  const response = await fetch("/api/shape-traces", { signal });
  if (!response.ok) {
    throw new ShapeTraceApiError("index", response.status);
  }
  const payload: unknown = await response.json();
  if (!isObject(payload) || !Array.isArray(payload.traces)) {
    throw new Error("invalid shape trace index payload");
  }
  const entries = payload.traces.map(decodeIndexEntry);
  const identities = entries.map(shapeTraceKey);
  if (new Set(identities).size !== identities.length) {
    throw new Error("duplicate shape trace index identity");
  }
  return entries;
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
    throw new ShapeTraceApiError("detail", response.status);
  }
  return decodeShapeTrace(await response.json());
}
