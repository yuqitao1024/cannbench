import type { CaseSummary } from "../types";

interface CaseTableProps {
  cases: CaseSummary[];
  showDatasetColumn: boolean;
}

function formatShape(shape: number[]): string {
  return shape.join(" x ");
}

export function CaseTable({ cases, showDatasetColumn }: CaseTableProps) {
  return (
    <section className="table-panel" aria-label="Case results">
      <table>
        <thead>
          <tr>
            <th>case</th>
            {showDatasetColumn ? <th>dataset</th> : null}
            <th>shape</th>
            <th>dtype</th>
            <th>source</th>
            <th>coverage tag</th>
            <th>available series</th>
          </tr>
        </thead>
        <tbody>
          {cases.map((caseSummary) => (
            <tr key={`${caseSummary.dataset}-${caseSummary.caseId}`}>
              <td>{caseSummary.caseId}</td>
              {showDatasetColumn ? <td>{caseSummary.dataset}</td> : null}
              <td>{formatShape(caseSummary.shape)}</td>
              <td>{caseSummary.dtype}</td>
              <td>{caseSummary.sourceLabel}</td>
              <td>{caseSummary.coverageTag}</td>
              <td>{caseSummary.availableSeries.join(" / ")}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </section>
  );
}
