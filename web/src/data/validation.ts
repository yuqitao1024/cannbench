const SENSITIVE_FIELDS = new Set([
  "hostname",
  "ip",
  "username",
  "env",
  "command",
  "workdir",
  "path",
  "log",
  "stdout",
  "stderr",
  "source_code",
  "diff",
  "profile_raw"
]);

const RECORD_FIELDS = new Set([
  "schema_version",
  "run_id",
  "operator",
  "dataset",
  "case_id",
  "shape",
  "dtype",
  "backend",
  "device_class",
  "implementation",
  "implementation_version",
  "metrics",
  "accuracy",
  "diff_ref"
]);

const METRIC_FIELDS = new Set(["latency_ms_avg", "latency_ms_p50", "latency_ms_p95", "sample_count"]);
const ACCURACY_FIELDS = new Set(["passed", "max_abs_error", "max_rel_error"]);

export interface ValidationResult {
  ok: boolean;
  acceptedCount: number;
  errors: string[];
  warnings: string[];
}

function isObject(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null && !Array.isArray(value);
}

function checkSensitiveFields(value: unknown, path: string, errors: string[]): void {
  if (Array.isArray(value)) {
    value.forEach((item, index) => checkSensitiveFields(item, `${path}[${index}]`, errors));
    return;
  }
  if (!isObject(value)) {
    return;
  }
  for (const [key, child] of Object.entries(value)) {
    if (SENSITIVE_FIELDS.has(key.toLowerCase())) {
      errors.push(`sensitive field rejected at ${path}.${key}`);
    }
    checkSensitiveFields(child, `${path}.${key}`, errors);
  }
}

function rejectUnknownFields(
  value: Record<string, unknown>,
  allowed: Set<string>,
  path: string,
  errors: string[]
): void {
  for (const key of Object.keys(value)) {
    if (!allowed.has(key)) {
      errors.push(`${path}.${key} is not allowed`);
    }
  }
}

function requireString(value: unknown, path: string, errors: string[]): string | null {
  if (typeof value !== "string" || value.length === 0 || value.length > 160) {
    errors.push(`${path} must be a non-empty string up to 160 characters`);
    return null;
  }
  return value;
}

function requireNumber(value: unknown, path: string, errors: string[]): number | null {
  if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
    errors.push(`${path} must be a non-negative finite number`);
    return null;
  }
  return value;
}

function validateRecord(record: unknown, index: number, errors: string[]): void {
  const path = `records[${index}]`;
  if (!isObject(record)) {
    errors.push(`${path} must be an object`);
    return;
  }

  rejectUnknownFields(record, RECORD_FIELDS, path, errors);
  if (record.schema_version !== 1) {
    errors.push(`${path}.schema_version must be 1`);
  }

  for (const key of ["run_id", "operator", "dataset", "case_id", "dtype", "device_class", "implementation", "implementation_version"]) {
    requireString(record[key], `${path}.${key}`, errors);
  }

  if (record.backend !== "nvidia" && record.backend !== "gpu") {
    errors.push(`${path}.backend must be nvidia or gpu`);
  }

  if (record.implementation !== "cuda_event" && record.implementation !== "ncu") {
    errors.push(`${path}.implementation must be cuda_event or ncu`);
  }

  if (!Array.isArray(record.shape) || record.shape.length === 0 || record.shape.length > 8) {
    errors.push(`${path}.shape must be a non-empty numeric array up to 8 dimensions`);
  } else {
    record.shape.forEach((dimension, dimensionIndex) => {
      if (!Number.isInteger(dimension) || dimension <= 0) {
        errors.push(`${path}.shape[${dimensionIndex}] must be a positive integer`);
      }
    });
  }

  if (!isObject(record.metrics)) {
    errors.push(`${path}.metrics must be an object`);
  } else {
    rejectUnknownFields(record.metrics, METRIC_FIELDS, `${path}.metrics`, errors);
    for (const key of METRIC_FIELDS) {
      requireNumber(record.metrics[key], `${path}.metrics.${key}`, errors);
    }
  }

  if (!isObject(record.accuracy)) {
    errors.push(`${path}.accuracy must be an object`);
  } else {
    rejectUnknownFields(record.accuracy, ACCURACY_FIELDS, `${path}.accuracy`, errors);
    if (typeof record.accuracy.passed !== "boolean") {
      errors.push(`${path}.accuracy.passed must be boolean`);
    }
    requireNumber(record.accuracy.max_abs_error, `${path}.accuracy.max_abs_error`, errors);
    requireNumber(record.accuracy.max_rel_error, `${path}.accuracy.max_rel_error`, errors);
  }

  if (record.diff_ref !== null) {
    errors.push(`${path}.diff_ref must be null for uploaded GPU records`);
  }
}

export function validateGpuBenchmarkUpload(payload: unknown): ValidationResult {
  const errors: string[] = [];
  const warnings: string[] = [];
  checkSensitiveFields(payload, "payload", errors);

  if (!isObject(payload)) {
    return { ok: false, acceptedCount: 0, errors: ["payload must be an object"], warnings };
  }
  rejectUnknownFields(payload, new Set(["records"]), "payload", errors);

  if (!Array.isArray(payload.records)) {
    errors.push("payload.records must be an array");
    return { ok: false, acceptedCount: 0, errors, warnings };
  }
  if (payload.records.length === 0 || payload.records.length > 10000) {
    errors.push("payload.records must contain 1 to 10000 records");
  }

  payload.records.forEach((record, index) => validateRecord(record, index, errors));
  return {
    ok: errors.length === 0,
    acceptedCount: errors.length === 0 ? payload.records.length : 0,
    errors,
    warnings
  };
}
