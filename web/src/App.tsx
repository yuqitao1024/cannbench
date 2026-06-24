import rawRecords from "../public/data/benchmark-results.json";
import { buildBenchmarkViewModel } from "./data/benchmarkData";
import type { BenchmarkRecord } from "./types";

const benchmarkRecords = rawRecords as BenchmarkRecord[];
const viewModel = buildBenchmarkViewModel(benchmarkRecords);
const firstOperator = viewModel.operators[0];

export function App() {
  return (
    <main className="app-shell">
      <section className="hero-card" aria-labelledby="page-title">
        <p className="eyebrow">CannBench</p>
        <h1 id="page-title">Operator Performance Console</h1>
        <p className="hero-copy">
          GPU H800, NPU library, and Ascend custom-op latency comparison.
        </p>
        <dl className="shell-metrics" aria-label="Loaded benchmark data">
          <div>
            <dt>records</dt>
            <dd>{viewModel.records.length}</dd>
          </div>
          <div>
            <dt>first operator</dt>
            <dd>{firstOperator?.name ?? "none"}</dd>
          </div>
        </dl>
      </section>
    </main>
  );
}
