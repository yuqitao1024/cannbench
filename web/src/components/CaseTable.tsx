import type { CaseSummary } from "../types";
import { shapeTraceKey } from "../data/shapeTraceApi";

interface CaseTableProps {
  operator: string;
  cases: CaseSummary[];
  showDatasetColumn: boolean;
  shapeTraceKeys: ReadonlySet<string>;
}

function formatShape(shape: number[]): string {
  return shape.join(" x ");
}

function shapeTraceHref(operator: string, caseSummary: CaseSummary): string {
  const params = new URLSearchParams({
    operator,
    dataset: caseSummary.dataset,
    case: caseSummary.caseId
  });
  return `/shape-explorer?${params.toString()}`;
}

export function CaseTable({ operator, cases, showDatasetColumn, shapeTraceKeys }: CaseTableProps) {
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
          {cases.map((caseSummary) => {
            const shape = formatShape(caseSummary.shape);
            const hasShapeTrace = shapeTraceKeys.has(
              shapeTraceKey({
                operator,
                dataset: caseSummary.dataset,
                case_id: caseSummary.caseId
              })
            );
            return (
              <tr key={`${caseSummary.dataset}-${caseSummary.caseId}`}>
                <td>{caseSummary.caseId}</td>
                {showDatasetColumn ? <td>{caseSummary.dataset}</td> : null}
                <td>
                  {hasShapeTrace ? (
                    <a
                      className="shape-link"
                      href={shapeTraceHref(operator, caseSummary)}
                      target="_blank"
                      rel="noreferrer"
                    >
                      {shape}
                    </a>
                  ) : (
                    shape
                  )}
                </td>
                <td>{caseSummary.dtype}</td>
                <td>{caseSummary.sourceLabel}</td>
                <td>{caseSummary.coverageTag}</td>
                <td>{caseSummary.availableSeries.join(" / ")}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </section>
  );
}
