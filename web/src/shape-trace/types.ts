export type AxisRole = "preserved" | "contracted" | "reduced" | "produced";

export interface ShapeAxis {
  symbol: string;
  value: number;
  meaning: string;
  role: AxisRole;
}

export interface ShapeTensor {
  id: string;
  label: string;
  axes: ShapeAxis[];
  logical_only: boolean;
}

export interface ShapeStage {
  id: string;
  component: string;
  title: string;
  operation: string;
  formula: string;
  scope: string;
  tensors: ShapeTensor[];
  input_ids: string[];
  output_ids: string[];
  contracted_axes: string[];
  insight: string;
}

export interface DeviceKernelTrace {
  id: string;
  title: string;
  summary: string;
  task_count: number;
  used_core_count: number;
  task_formula: string;
  task_axes: ShapeAxis[];
  tile_tensors: ShapeTensor[];
  steps: string[];
}

export interface ShapeTraceIndexEntry {
  operator: string;
  dataset: string;
  case_id: string;
  phase: string;
  group: string;
}

export interface ShapeTrace extends ShapeTraceIndexEntry {
  schema_version: 1;
  symbols: ShapeAxis[];
  stages: ShapeStage[];
  device_execution: {
    status: "available" | "unavailable";
    implementation: string;
    version: string | null;
    message: string | null;
    kernels: DeviceKernelTrace[];
  };
}
