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

function simtDelta(cannOpsLibrary: BenchmarkRecord | undefined, simtOp: BenchmarkRecord | undefined): string {
  if (!cannOpsLibrary || !simtOp) {
    return "-";
  }
  const delta =
    ((simtOp.metrics.latency_ms_avg - cannOpsLibrary.metrics.latency_ms_avg) /
      cannOpsLibrary.metrics.latency_ms_avg) *
    100;
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
            <th>CANN ops library</th>
            <th>SIMT op</th>
            <th>SIMT delta</th>
            <th>accuracy</th>
          </tr>
        </thead>
        <tbody>
          {cases.map((caseSummary) => {
            const gpu = findRecord(caseSummary.records, (record) => record.backend === "nvidia" || record.backend === "gpu");
            const cannOpsLibrary = findRecord(
              caseSummary.records,
              (record) => record.backend === "ascend" && record.implementation === "cann_ops_library"
            );
            const simtOp = findRecord(
              caseSummary.records,
              (record) => record.backend === "ascend" && record.implementation === "simt"
            );
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
                <td>{formatLatency(cannOpsLibrary)}</td>
                <td>{formatLatency(simtOp)}</td>
                <td>{simtDelta(cannOpsLibrary, simtOp)}</td>
                <td className={accuracyPassed ? "status-ok" : "status-bad"}>{accuracyPassed ? "pass" : "fail"}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
