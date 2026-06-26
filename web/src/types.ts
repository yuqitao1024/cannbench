export type DatasetName = "smoke" | "realistic" | "stress" | string;

export type BenchmarkBackend = "nvidia" | "gpu" | "ascend" | "npu";

export type BenchmarkImplementation = "cuda_event" | "ncu" | "cann_ops_library" | "simt";

export interface BenchmarkMetrics {
  latency_ms_avg: number;
  latency_ms_p50: number;
  latency_ms_p95: number;
  sample_count: number;
}

export interface AccuracySummary {
  passed: boolean;
  max_abs_error: number;
  max_rel_error: number;
}

export interface BenchmarkRecord {
  schema_version: 1;
  run_id: string;
  operator: string;
  dataset: DatasetName;
  case_id: string;
  shape: number[];
  dtype: string;
  backend: BenchmarkBackend;
  device_class: string;
  implementation: BenchmarkImplementation;
  implementation_version: string;
  metrics: BenchmarkMetrics;
  accuracy: AccuracySummary;
  diff_ref: string | null;
}

export interface OperatorSummary {
  name: string;
  recordCount: number;
  caseCount: number;
}

export interface CaseSummary {
  caseId: string;
  shape: number[];
  dtype: string;
  records: BenchmarkRecord[];
}

export interface ChartPoint {
  caseId: string;
  latencyMs: number | null;
  record: BenchmarkRecord | null;
}

export interface ChartSeries {
  key: string;
  name: string;
  records: BenchmarkRecord[];
  points: ChartPoint[];
}

export interface BenchmarkViewModel {
  records: BenchmarkRecord[];
  operators: OperatorSummary[];
  datasetsFor: (operator: string) => string[];
  casesFor: (operator: string, dataset: string) => CaseSummary[];
  recordsForCase: (operator: string, dataset: string, caseId: string) => BenchmarkRecord[];
  seriesFor: (operator: string, dataset: string) => ChartSeries[];
}

export interface SimtOperatorDiff {
  operator: string;
  base_version: string;
  compare_version: string;
  patch: string;
}
