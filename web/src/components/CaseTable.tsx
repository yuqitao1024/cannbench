import type { BenchmarkRecord, CaseSummary } from "../types";

interface CaseTableProps {
  cases: CaseSummary[];
  selectedCaseId: string;
  onSelectCase: (caseId: string) => void;
}

function findRecord(records: BenchmarkRecord[], predicate: (record: BenchmarkRecord) => boolean) {
  return records.find(predicate);
}

function formatLatency(record: BenchmarkRecord | undefined): string {
  return record ? record.metrics.latency_ms_avg.toFixed(4) : "-";
}

function customDelta(library: BenchmarkRecord | undefined, custom: BenchmarkRecord | undefined): string {
  if (!library || !custom) {
    return "-";
  }
  const delta = ((custom.metrics.latency_ms_avg - library.metrics.latency_ms_avg) / library.metrics.latency_ms_avg) * 100;
  return `${delta >= 0 ? "+" : ""}${delta.toFixed(1)}%`;
}

export function CaseTable({ cases, selectedCaseId, onSelectCase }: CaseTableProps) {
  return (
    <section className="table-panel" aria-label="Case results">
      <table>
        <thead>
          <tr>
            <th>case</th>
            <th>shape</th>
            <th>dtype</th>
            <th>GPU H800</th>
            <th>NPU library</th>
            <th>NPU custom</th>
            <th>custom delta</th>
            <th>accuracy</th>
          </tr>
        </thead>
        <tbody>
          {cases.map((caseSummary) => {
            const gpu = findRecord(caseSummary.records, (record) => record.backend === "nvidia" || record.backend === "gpu");
            const library = findRecord(caseSummary.records, (record) => record.backend === "ascend" && record.implementation === "library");
            const custom = findRecord(caseSummary.records, (record) => record.backend === "ascend" && record.implementation === "custom");
            const accuracyPassed = caseSummary.records.every((record) => record.accuracy.passed);
            return (
              <tr
                key={caseSummary.caseId}
                className={caseSummary.caseId === selectedCaseId ? "is-selected" : undefined}
                onClick={() => onSelectCase(caseSummary.caseId)}
              >
                <td>
                  <button type="button" className="case-link" onClick={() => onSelectCase(caseSummary.caseId)}>
                    {caseSummary.caseId}
                  </button>
                </td>
                <td>{caseSummary.shape.join(" x ")}</td>
                <td>{caseSummary.dtype}</td>
                <td>{formatLatency(gpu)}</td>
                <td>{formatLatency(library)}</td>
                <td>{formatLatency(custom)}</td>
                <td>{customDelta(library, custom)}</td>
                <td className={accuracyPassed ? "status-ok" : "status-bad"}>{accuracyPassed ? "pass" : "fail"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
