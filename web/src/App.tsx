import { useEffect, useState } from "react";
import { BenchmarkChart } from "./components/BenchmarkChart";
import { CaseTable } from "./components/CaseTable";
import { CodeDiffPanel } from "./components/CodeDiffPanel";
import { CudaTreasureMapModal } from "./components/CudaTreasureMapModal";
import { GpuBenchmarkImport } from "./components/GpuBenchmarkImport";
import { KernelTraceRail } from "./components/KernelTraceRail";
import { OperatorRail } from "./components/OperatorRail";
import { RunFilters } from "./components/RunFilters";
import logoDarkUrl from "./assets/brand/cannbench-logo-dark.png";
import logoLightUrl from "./assets/brand/cannbench-logo-light.png";
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
  const [treasureMapOpen, setTreasureMapOpen] = useState(false);
  const [titleClickCount, setTitleClickCount] = useState(0);
  const [theme, setTheme] = useState<"light" | "dark">(themeForCurrentHour);
  const [selectedOperator, setSelectedOperator] = useState(defaultOperator);
  const [selectedDataset, setSelectedDataset] = useState(defaultDataset(defaultOperator));
  const cases = viewModel.casesFor(selectedOperator, selectedDataset);
  const [selectedCaseId, setSelectedCaseId] = useState(cases[0]?.caseId ?? "");
  const selectedCaseRecords = selectedCaseId
    ? viewModel.recordsForCase(selectedOperator, selectedDataset, selectedCaseId)
    : [];
  const dtypes = [...new Set(cases.map((item) => item.dtype))];
  const simtVersions = [
    ...new Set(
      cases.flatMap((item) =>
        item.records
          .filter((record) => record.implementation === "simt")
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
  const openHiddenModalFromTitle = () => {
    setTitleClickCount((count) => {
      const next = count + 1;
      if (next >= 3) {
        if (theme === "dark") {
          setImportOpen(false);
          setTreasureMapOpen(true);
        } else {
          setTreasureMapOpen(false);
          setImportOpen(true);
        }
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
        <h1 id="page-title" className="brand-title" aria-label="CANNBench">
          <button
            type="button"
            className="brand-trigger"
            aria-label="CANNBench"
            onClick={openHiddenModalFromTitle}
          >
            <img
              className="brand-logo"
              src={theme === "dark" ? logoDarkUrl : logoLightUrl}
              alt=""
              aria-hidden="true"
            />
          </button>
        </h1>
        <button
          type="button"
          className="theme-toggle"
          aria-label="Toggle light and dark theme"
          onClick={() => {
            setTitleClickCount(0);
            setTheme((current) => (current === "dark" ? "light" : "dark"));
          }}
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
              simtVersions={simtVersions}
              onSelectDataset={setSelectedDataset}
            />
          </div>
          <KernelTraceRail records={selectedCaseRecords} />
          <BenchmarkChart series={series} caseIds={cases.map((item) => item.caseId)} />
          <CaseTable cases={cases} selectedCaseId={selectedCaseId} onSelectCase={setSelectedCaseId} />
          <CodeDiffPanel operator={selectedOperator} />
        </section>
      </div>
      <GpuBenchmarkImport uploadEnabled={false} open={importOpen} onClose={() => setImportOpen(false)} />
      <CudaTreasureMapModal open={treasureMapOpen} onClose={() => setTreasureMapOpen(false)} />
    </main>
  );
}
