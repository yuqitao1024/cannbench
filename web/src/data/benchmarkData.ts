import type {
  BenchmarkRecord,
  BenchmarkViewModel,
  CaseSummary,
  ChartSeries,
  OperatorSummary
} from "../types";

const DATASET_ORDER = new Map([
  ["smoke", 0],
  ["realistic", 1],
  ["stress", 2]
]);

function datasetRank(dataset: string): number {
  return DATASET_ORDER.get(dataset) ?? 100;
}

function seriesKey(record: BenchmarkRecord): string {
  if (record.backend === "nvidia" || record.backend === "gpu") {
    return "gpu";
  }
  if (record.implementation === "library") {
    return "npu-library";
  }
  return `npu-custom:${record.implementation_version}`;
}

function seriesName(record: BenchmarkRecord): string {
  if (record.backend === "nvidia" || record.backend === "gpu") {
    return `GPU ${record.device_class}`;
  }
  if (record.implementation === "library") {
    return "NPU library";
  }
  return `NPU custom: ${record.implementation_version}`;
}

function seriesRank(key: string): [number, string] {
  if (key === "gpu") {
    return [0, key];
  }
  if (key === "npu-library") {
    return [1, key];
  }
  return [2, key];
}

function compareSeriesKeys(left: string, right: string): number {
  const [leftRank, leftName] = seriesRank(left);
  const [rightRank, rightName] = seriesRank(right);
  return leftRank - rightRank || leftName.localeCompare(rightName);
}

function uniqueSorted(values: string[], ranker?: (value: string) => number): string[] {
  return [...new Set(values)].sort((left, right) => {
    const leftRank = ranker?.(left) ?? 0;
    const rightRank = ranker?.(right) ?? 0;
    return leftRank - rightRank || left.localeCompare(right);
  });
}

export function buildBenchmarkViewModel(records: BenchmarkRecord[]): BenchmarkViewModel {
  const recordsByOperator = new Map<string, BenchmarkRecord[]>();
  for (const record of records) {
    const operatorRecords = recordsByOperator.get(record.operator) ?? [];
    operatorRecords.push(record);
    recordsByOperator.set(record.operator, operatorRecords);
  }

  const operators: OperatorSummary[] = [...recordsByOperator.entries()]
    .map(([name, operatorRecords]) => ({
      name,
      recordCount: operatorRecords.length,
      caseCount: new Set(operatorRecords.map((record) => record.case_id)).size
    }))
    .sort((left, right) => left.name.localeCompare(right.name));

  function filtered(operator: string, dataset?: string): BenchmarkRecord[] {
    return records.filter((record) => {
      if (record.operator !== operator) {
        return false;
      }
      return dataset === undefined || record.dataset === dataset;
    });
  }

  function datasetsFor(operator: string): string[] {
    return uniqueSorted(
      filtered(operator).map((record) => record.dataset),
      datasetRank
    );
  }

  function casesFor(operator: string, dataset: string): CaseSummary[] {
    const cases = new Map<string, CaseSummary>();
    for (const record of filtered(operator, dataset)) {
      const current = cases.get(record.case_id);
      if (current) {
        current.records.push(record);
        continue;
      }
      cases.set(record.case_id, {
        caseId: record.case_id,
        shape: record.shape,
        dtype: record.dtype,
        records: [record]
      });
    }
    return [...cases.values()];
  }

  function recordsForCase(operator: string, dataset: string, caseId: string): BenchmarkRecord[] {
    return filtered(operator, dataset).filter((record) => record.case_id === caseId);
  }

  function seriesFor(operator: string, dataset: string): ChartSeries[] {
    const cases = casesFor(operator, dataset);
    const groups = new Map<string, BenchmarkRecord[]>();
    const names = new Map<string, string>();
    for (const record of filtered(operator, dataset)) {
      const key = seriesKey(record);
      const group = groups.get(key) ?? [];
      group.push(record);
      groups.set(key, group);
      names.set(key, seriesName(record));
    }

    return [...groups.keys()].sort(compareSeriesKeys).map((key) => {
      const group = groups.get(key) ?? [];
      return {
        key,
        name: names.get(key) ?? key,
        records: group,
        points: cases.map((caseSummary) => {
          const record = group.find((item) => item.case_id === caseSummary.caseId) ?? null;
          return {
            caseId: caseSummary.caseId,
            latencyMs: record?.metrics.latency_ms_avg ?? null,
            record
          };
        })
      };
    });
  }

  return {
    records,
    operators,
    datasetsFor,
    casesFor,
    recordsForCase,
    seriesFor
  };
}
