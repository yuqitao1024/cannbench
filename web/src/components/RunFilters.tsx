import type { ReactNode } from "react";
import { Lock } from "lucide-react";
import type { MetricOption, SeriesOption } from "../types";

interface RunFiltersProps {
  metrics: MetricOption[];
  selectedMetric: MetricOption["key"];
  datasets: string[];
  selectedDataset: string;
  seriesOptions: SeriesOption[];
  selectedSeries: string[];
  lockedSeries: string | null;
  onSelectMetric: (metric: MetricOption["key"]) => void;
  onSelectDataset: (dataset: string) => void;
  onToggleSeries: (series: string) => void;
}

interface FilterRowProps {
  label: string;
  children: ReactNode;
}

function FilterRow({ label, children }: FilterRowProps) {
  return (
    <div className="filter-row">
      <span className="filter-row-label">{label}</span>
      <div className="filter-chip-row">{children}</div>
    </div>
  );
}

export function RunFilters({
  metrics,
  selectedMetric,
  datasets,
  selectedDataset,
  seriesOptions,
  selectedSeries,
  lockedSeries,
  onSelectMetric,
  onSelectDataset,
  onToggleSeries
}: RunFiltersProps) {
  return (
    <section className="run-filters" aria-label="Benchmark filters">
      <FilterRow label="Metric">
        {metrics.map((metric) => (
          <button
            key={metric.key}
            type="button"
            className="filter-chip"
            aria-pressed={metric.key === selectedMetric}
            onClick={() => onSelectMetric(metric.key)}
          >
            {metric.name}
          </button>
        ))}
      </FilterRow>
      <FilterRow label="Dataset split">
        {datasets.map((dataset) => (
          <button
            key={dataset}
            type="button"
            className="filter-chip"
            aria-pressed={dataset === selectedDataset}
            onClick={() => onSelectDataset(dataset)}
          >
            {dataset}
          </button>
        ))}
      </FilterRow>
      <FilterRow label="Series">
        {seriesOptions.map((series) => {
          const locked = series.key === lockedSeries;
          return (
            <button
              key={series.key}
              type="button"
              className={`filter-chip${locked ? " is-locked" : ""}`}
              aria-label={locked ? `${series.name}, baseline locked` : series.name}
              aria-pressed={selectedSeries.includes(series.key)}
              disabled={!series.available || locked}
              onClick={() => onToggleSeries(series.key)}
            >
              <span>{series.name}</span>
              {locked ? <Lock size={13} aria-hidden="true" /> : null}
            </button>
          );
        })}
      </FilterRow>
    </section>
  );
}
