import { useEffect, useState } from "react";
import { BenchmarkChart } from "./components/BenchmarkChart";
import { CaseTable } from "./components/CaseTable";
import { CodeDiffPanel } from "./components/CodeDiffPanel";
import { GpuBenchmarkImport } from "./components/GpuBenchmarkImport";
import { KernelTraceRail } from "./components/KernelTraceRail";
import { OperatorRail } from "./components/OperatorRail";
import { RunFilters } from "./components/RunFilters";
import { buildBenchmarkViewModel } from "./data/benchmarkData";
import rawRecords from "./data/sample/benchmark-results.json";
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

function themeForCurrentHour(): "light" | "dark" {
  const hour = new Date().getHours();
  return hour >= 7 && hour < 19 ? "light" : "dark";
}

export function App() {
  const [operatorSearch, setOperatorSearch] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [titleClickCount, setTitleClickCount] = useState(0);
  const [theme, setTheme] = useState<"light" | "dark">(themeForCurrentHour);
  const [selectedOperator, setSelectedOperator] = useState(defaultOperator);
  const [selectedDataset, setSelectedDataset] = useState(defaultDataset(defaultOperator));
  const cases = viewModel.casesFor(selectedOperator, selectedDataset);
  const [selectedCaseId, setSelectedCaseId] = useState(cases[0]?.caseId ?? "");
  const selectedCaseRecords = selectedCaseId
    ? viewModel.recordsForCase(selectedOperator, selectedDataset, selectedCaseId)
    : [];
  const selectedDiffRef =
    selectedCaseRecords.find((record) => record.implementation === "custom" && record.diff_ref)?.diff_ref ?? null;
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
    document.documentElement.dataset.theme = theme;
    document.body.dataset.theme = theme;

    return () => {
      delete document.documentElement.dataset.theme;
      delete document.body.dataset.theme;
    };
  }, [theme]);

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
  const openImportFromTitle = () => {
    setTitleClickCount((count) => {
      const next = count + 1;
      if (next >= 3) {
        setImportOpen(true);
        return 0;
      }
      return next;
    });
  };

  return (
    <main className="app-shell" data-theme={theme}>
      <div className="formula-field" aria-hidden="true">
        <span>σ(x)</span>
        <span>BW=t⁻¹B</span>
        <span>∂L / ∂x</span>
        <span>Δt</span>
        <span>E = mc²</span>
        <span>O(n log n)</span>
        <span>η = a / p</span>
        <span>F ≤ BW×AI</span>
        <span>Σ exp(x)</span>
        <span>p95(t)</span>
        <span>Δ = c - b</span>
        <span>H = -Σ p log p</span>
        <span>σ(x)</span>
        <span>∇f(x)</span>
        <span>Q = KᵀV</span>
        <span>λ = c / f</span>
        <span>μs/op</span>
        <span>GB/s</span>
      </div>
      <header className="console-header">
        <div>
          <button
            type="button"
            className="eyebrow eyebrow-trigger"
            aria-label="CannBench operator trace"
            onClick={openImportFromTitle}
          >
            CannBench / Operator Trace
          </button>
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
        <button
          type="button"
          className="theme-toggle"
          aria-label="Toggle light and dark theme"
          onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
        >
          {theme === "dark" ? "moon" : "sun"}
        </button>
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
          <CodeDiffPanel diffRef={selectedDiffRef} />
        </section>
      </div>
      <GpuBenchmarkImport uploadEnabled={false} open={importOpen} onClose={() => setImportOpen(false)} />
    </main>
  );
}
