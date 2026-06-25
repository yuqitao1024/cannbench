interface RunFiltersProps {
  datasets: string[];
  selectedDataset: string;
  dtypes: string[];
  simtVersions: string[];
  onSelectDataset: (dataset: string) => void;
}

export function RunFilters({
  datasets,
  selectedDataset,
  dtypes,
  simtVersions,
  onSelectDataset
}: RunFiltersProps) {
  return (
    <section className="run-filters" aria-label="Benchmark filters">
      <div className="dataset-tabs" role="tablist" aria-label="Datasets">
        {datasets.map((dataset) => (
          <button
            key={dataset}
            type="button"
            role="tab"
            aria-selected={dataset === selectedDataset}
            className="dataset-tab"
            onClick={() => onSelectDataset(dataset)}
          >
            {dataset}
          </button>
        ))}
      </div>
      <div className="filter-chips" aria-label="Run metadata">
        <span>dtype: {dtypes.join(", ") || "none"}</span>
        <span>SIMT op: {simtVersions.join(", ") || "none"}</span>
      </div>
    </section>
  );
}
