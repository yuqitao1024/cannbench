import type { ReactNode } from "react";
import { Lock } from "lucide-react";
import type { MetricOption, SeriesOption } from "../types";

interface RunFiltersProps {
  metric: MetricOption;
  datasets: string[];
  selectedDataset: string;
  seriesOptions: SeriesOption[];
  selectedSeries: string[];
  lockedSeries: string | null;
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
  metric,
  datasets,
  selectedDataset,
  seriesOptions,
  selectedSeries,
  lockedSeries,
  onSelectDataset,
  onToggleSeries
}: RunFiltersProps) {
  return (
    <section className="run-filters" aria-label="Benchmark filters">
      <FilterRow label="Metric">
        <span className="filter-chip is-selected" role="status" aria-label={`${metric.name}, selected`}>
          {metric.name}
        </span>
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
