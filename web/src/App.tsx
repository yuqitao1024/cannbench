import { useEffect, useState } from "react";
import rawRecords from "../public/data/benchmark-results.json";
import { BenchmarkChart } from "./components/BenchmarkChart";
import { CaseTable } from "./components/CaseTable";
import { KernelTraceRail } from "./components/KernelTraceRail";
import { OperatorRail } from "./components/OperatorRail";
import { RunFilters } from "./components/RunFilters";
import { buildBenchmarkViewModel } from "./data/benchmarkData";
import type { BenchmarkRecord } from "./types";

const benchmarkRecords = rawRecords as BenchmarkRecord[];
const viewModel = buildBenchmarkViewModel(benchmarkRecords);
const defaultOperator = viewModel.operators.some((operator) => operator.name === "softmax")
  ? "softmax"
  : (viewModel.operators[0]?.name ?? "");
const defaultDataset = (operator: string) => {
  const datasets = viewModel.datasetsFor(operator);
  return datasets.includes("realistic") ? "realistic" : (datasets[0] ?? "");
};

export function App() {
  const [operatorSearch, setOperatorSearch] = useState("");
  const [selectedOperator, setSelectedOperator] = useState(defaultOperator);
  const [selectedDataset, setSelectedDataset] = useState(defaultDataset(defaultOperator));
  const cases = viewModel.casesFor(selectedOperator, selectedDataset);
  const [selectedCaseId, setSelectedCaseId] = useState(cases[0]?.caseId ?? "");
  const selectedCaseRecords = selectedCaseId
    ? viewModel.recordsForCase(selectedOperator, selectedDataset, selectedCaseId)
    : [];
  const dtypes = [...new Set(cases.map((item) => item.dtype))];
  const customVersions = [
    ...new Set(
      cases.flatMap((item) =>
        item.records
          .filter((record) => record.implementation === "custom")
          .map((record) => record.implementation_version)
      )
    )
  ].sort();
  const series = viewModel.seriesFor(selectedOperator, selectedDataset);

  useEffect(() => {
    const datasets = viewModel.datasetsFor(selectedOperator);
    if (!datasets.includes(selectedDataset)) {
      setSelectedDataset(defaultDataset(selectedOperator));
      return;
    }
    const nextCases = viewModel.casesFor(selectedOperator, selectedDataset);
    if (!nextCases.some((item) => item.caseId === selectedCaseId)) {
      setSelectedCaseId(nextCases[0]?.caseId ?? "");
    }
  }, [selectedCaseId, selectedDataset, selectedOperator]);

  const datasets = viewModel.datasetsFor(selectedOperator);

  return (
    <main className="app-shell">
      <header className="console-header">
        <div>
          <p className="eyebrow">CannBench / Operator Trace</p>
          <h1 id="page-title">Operator Performance Console</h1>
          <p className="hero-copy">
            Compare GPU H800, Ascend NPU library, and custom operator versions across curated benchmark cases.
          </p>
        </div>
        <dl className="shell-metrics" aria-label="Loaded benchmark data">
          <div>
            <dt>records</dt>
            <dd>{viewModel.records.length}</dd>
          </div>
          <div>
            <dt>operators</dt>
            <dd>{viewModel.operators.length}</dd>
          </div>
        </dl>
      </header>

      <div className="console-grid">
        <OperatorRail
          operators={viewModel.operators}
          selectedOperator={selectedOperator}
          search={operatorSearch}
          onSearchChange={setOperatorSearch}
          onSelectOperator={setSelectedOperator}
        />
        <section className="workspace" aria-labelledby="selected-operator-title">
          <div className="workspace-head">
            <div>
              <p className="panel-kicker">Selected operator</p>
              <h2 id="selected-operator-title">{selectedOperator}</h2>
            </div>
            <RunFilters
              datasets={datasets}
              selectedDataset={selectedDataset}
              dtypes={dtypes}
              customVersions={customVersions}
              onSelectDataset={setSelectedDataset}
            />
          </div>
          <KernelTraceRail records={selectedCaseRecords} />
          <BenchmarkChart series={series} caseIds={cases.map((item) => item.caseId)} />
          <CaseTable cases={cases} selectedCaseId={selectedCaseId} onSelectCase={setSelectedCaseId} />
        </section>
      </div>
    </main>
  );
}
