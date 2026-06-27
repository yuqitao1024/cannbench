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
import { loadBenchmarkRecords } from "./data/benchmarkRecordsApi";
import type { BenchmarkRecord } from "./types";

function themeForCurrentHour(): "light" | "dark" {
  const hour = new Date().getHours();
  return hour >= 7 && hour < 19 ? "light" : "dark";
}

function defaultOperatorName(records: BenchmarkRecord[]): string {
  const viewModel = buildBenchmarkViewModel(records);
  return viewModel.operators.some((operator) => operator.name === "softmax")
    ? "softmax"
    : (viewModel.operators[0]?.name ?? "");
}

function defaultDatasetName(records: BenchmarkRecord[], operator: string): string {
  const viewModel = buildBenchmarkViewModel(records);
  const datasets = viewModel.datasetsFor(operator);
  return datasets.includes("realistic") ? "realistic" : (datasets[0] ?? "");
}

export function App() {
  const [benchmarkRecords, setBenchmarkRecords] = useState<BenchmarkRecord[]>([]);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [operatorSearch, setOperatorSearch] = useState("");
  const [importOpen, setImportOpen] = useState(false);
  const [treasureMapOpen, setTreasureMapOpen] = useState(false);
  const [titleClickCount, setTitleClickCount] = useState(0);
  const [theme, setTheme] = useState<"light" | "dark">(themeForCurrentHour);
  const [selectedOperator, setSelectedOperator] = useState("");
  const [selectedDataset, setSelectedDataset] = useState("");
  const viewModel = buildBenchmarkViewModel(benchmarkRecords);
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
  const showDiffPanel = simtVersions.length > 1;
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
    const controller = new AbortController();
    loadBenchmarkRecords(controller.signal)
      .then((records) => {
        setBenchmarkRecords(records);
        setLoadError(null);
        const operator = defaultOperatorName(records);
        const dataset = defaultDatasetName(records, operator);
        setSelectedOperator(operator);
        setSelectedDataset(dataset);
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        setBenchmarkRecords([]);
        setLoadError(error instanceof Error ? error.message : "failed to load benchmark records");
      });
    return () => controller.abort();
  }, []);

  useEffect(() => {
    const datasets = viewModel.datasetsFor(selectedOperator);
    if (!datasets.includes(selectedDataset)) {
      setSelectedDataset(defaultDatasetName(benchmarkRecords, selectedOperator));
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
          {loadError ? (
            <div className="workspace-empty" role="status">
              Failed to load benchmark records.
            </div>
          ) : benchmarkRecords.length === 0 ? (
            <div className="workspace-empty" role="status">
              Loading benchmark records...
            </div>
          ) : (
            <>
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
              {showDiffPanel ? <CodeDiffPanel operator={selectedOperator} /> : null}
            </>
          )}
        </section>
      </div>
      <GpuBenchmarkImport uploadEnabled={false} open={importOpen} onClose={() => setImportOpen(false)} />
      <CudaTreasureMapModal open={treasureMapOpen} onClose={() => setTreasureMapOpen(false)} />
    </main>
  );
}
