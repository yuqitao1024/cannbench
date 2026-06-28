import type {
  BenchmarkRecord,
  BenchmarkViewModel,
  CaseSummary,
  ChartSegment,
  ChartSeries,
  MetricOption,
  OperatorSummary,
  SeriesOption
} from "../types";

const SPLIT_ORDER = ["smoke", "realistic", "stress"] as const;
const ALL_DATASET = "ALL";
const METRIC_OPTIONS: MetricOption[] = [{ key: "latency", name: "Latency" }];

function datasetRank(dataset: string): number {
  if (dataset === ALL_DATASET) {
    return -1;
  }
  const index = SPLIT_ORDER.indexOf(dataset as (typeof SPLIT_ORDER)[number]);
  return index === -1 ? 100 : index;
}

function orderedUnique(values: string[]): string[] {
  return [...new Set(values)];
}

function isGpuRecord(record: BenchmarkRecord): boolean {
  return record.backend === "nvidia" || record.backend === "gpu";
}

function seriesId(record: BenchmarkRecord): string {
  if (isGpuRecord(record)) {
    return `nvidia-${record.device_class.toLowerCase()}-cuda-pytorch`;
  }
  if (record.implementation === "cann_ops_library") {
    return `ascend-${record.device_class.toLowerCase()}-cann-cannops`;
  }
  return `ascend-${record.device_class.toLowerCase()}-simt-${record.implementation_version.toLowerCase()}`;
}

function seriesName(record: BenchmarkRecord): string {
  if (isGpuRecord(record)) {
    return `NVIDIA ${record.device_class} PyTorch`;
  }
  if (record.implementation === "cann_ops_library") {
    return `Ascend ${record.device_class} CANN Ops`;
  }
  return `Ascend ${record.device_class} SIMT ${record.implementation_version}`;
}

function seriesSortKey(record: BenchmarkRecord): [number, string] {
  if (isGpuRecord(record)) {
    return [0, seriesName(record)];
  }
  if (record.implementation === "cann_ops_library") {
    return [1, seriesName(record)];
  }
  return [2, seriesName(record)];
}

function compareSeries(left: SeriesOption, right: SeriesOption): number {
  const leftRank = left.key.startsWith("nvidia-") ? 0 : left.key.includes("-cann-") ? 1 : 2;
  const rightRank = right.key.startsWith("nvidia-") ? 0 : right.key.includes("-cann-") ? 1 : 2;
  return leftRank - rightRank || left.name.localeCompare(right.name);
}

function coverageTag(sourceKind: string): string {
  if (sourceKind === "real_model") {
    return "real-model coverage";
  }
  if (sourceKind.startsWith("synthetic_smoke")) {
    return "smoke coverage";
  }
  if (sourceKind.startsWith("synthetic_boundary")) {
    return "stress coverage";
  }
  return sourceKind.replaceAll("_", " ");
}

function sourceLabel(record: BenchmarkRecord): string {
  return `${record.source_project} / ${record.source_model}`;
}

function filteredRecords(records: BenchmarkRecord[], operator: string, dataset?: string): BenchmarkRecord[] {
  return records.filter((record) => {
    if (record.operator !== operator) {
      return false;
    }
    if (!dataset || dataset === ALL_DATASET) {
      return true;
    }
    return record.dataset === dataset;
  });
}

function orderedCaseKeys(records: BenchmarkRecord[], dataset: string): string[] {
  if (dataset !== ALL_DATASET) {
    return orderedUnique(records.map((record) => record.case_id));
  }
  const ordered: string[] = [];
  for (const split of SPLIT_ORDER) {
    const splitKeys = orderedUnique(records.filter((record) => record.dataset === split).map((record) => record.case_id));
    ordered.push(...splitKeys);
  }
  const remainder = orderedUnique(
    records
      .filter((record) => !SPLIT_ORDER.includes(record.dataset as (typeof SPLIT_ORDER)[number]))
      .map((record) => record.case_id)
  );
  ordered.push(...remainder);
  return ordered;
}

export function metricOptions(): MetricOption[] {
  return METRIC_OPTIONS;
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

  function datasetsFor(operator: string): string[] {
    const datasets = orderedUnique(filteredRecords(records, operator).map((record) => record.dataset)).sort(
      (left, right) => datasetRank(left) - datasetRank(right) || left.localeCompare(right)
    );
    return datasets.length === 0 ? [] : [ALL_DATASET, ...datasets];
  }

  function casesFor(operator: string, dataset: string): CaseSummary[] {
    const relevant = filteredRecords(records, operator, dataset);
    const caseKeys = orderedCaseKeys(relevant, dataset);
    return caseKeys.map((caseId) => {
      const caseRecords = relevant.filter((record) => record.case_id === caseId);
      const sample = caseRecords[0];
      const seriesNames = orderedUnique(caseRecords.map((record) => seriesName(record)));
      return {
        caseId,
        dataset: dataset === ALL_DATASET ? sample.dataset : dataset,
        family: sample.family,
        shape: sample.shape,
        dtype: sample.dtype,
        records: caseRecords,
        sourceLabel: sourceLabel(sample),
        coverageTag: coverageTag(sample.source_kind),
        availableSeries: seriesNames
      };
    });
  }

  function seriesOptionsFor(operator: string, dataset: string): SeriesOption[] {
    const operatorRecords = filteredRecords(records, operator);
    const availableIds = new Set(filteredRecords(records, operator, dataset).map((record) => seriesId(record)));
    const seen = new Map<string, SeriesOption>();
    for (const record of operatorRecords) {
      const key = seriesId(record);
      if (!seen.has(key)) {
        seen.set(key, {
          key,
          name: seriesName(record),
          available: availableIds.has(key)
        });
      }
    }
    return [...seen.values()].sort(compareSeries);
  }

  function seriesFor(operator: string, dataset: string): ChartSeries[] {
    const relevant = filteredRecords(records, operator, dataset);
    const caseKeys = orderedCaseKeys(relevant, dataset);
    const optionMap = new Map<string, BenchmarkRecord[]>();
    for (const record of relevant) {
      const key = seriesId(record);
      const group = optionMap.get(key) ?? [];
      group.push(record);
      optionMap.set(key, group);
    }

    return [...optionMap.entries()]
      .map(([key, group]) => ({
        key,
        name: seriesName(group[0]),
        records: group,
        points: caseKeys.map((caseId) => {
          const record = group.find((item) => item.case_id === caseId) ?? null;
          return {
            caseId,
            latencyMs: record?.metrics.latency_ms_avg ?? null,
            record
          };
        })
      }))
      .sort((left, right) => {
        const sampleLeft = left.records[0];
        const sampleRight = right.records[0];
        const [leftRank, leftName] = seriesSortKey(sampleLeft);
        const [rightRank, rightName] = seriesSortKey(sampleRight);
        return leftRank - rightRank || leftName.localeCompare(rightName);
      });
  }

  function chartSegmentsFor(operator: string, dataset: string): ChartSegment[] {
    const relevant = filteredRecords(records, operator, dataset);
    if (dataset !== ALL_DATASET) {
      const count = orderedCaseKeys(relevant, dataset).length;
      return count === 0 ? [] : [{ key: dataset, label: dataset, start: 0, end: count - 1 }];
    }

    const segments: ChartSegment[] = [];
    let cursor = 0;
    for (const split of SPLIT_ORDER) {
      const keys = orderedUnique(relevant.filter((record) => record.dataset === split).map((record) => record.case_id));
      if (keys.length === 0) {
        continue;
      }
      segments.push({
        key: split,
        label: split,
        start: cursor,
        end: cursor + keys.length - 1
      });
      cursor += keys.length;
    }
    return segments;
  }

  return {
    records,
    operators,
    datasetsFor,
    casesFor,
    seriesFor,
    seriesOptionsFor,
    chartSegmentsFor
  };
}
