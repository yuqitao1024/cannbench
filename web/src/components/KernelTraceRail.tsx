import type { BenchmarkRecord } from "../types";

interface KernelTraceRailProps {
  records: BenchmarkRecord[];
}

function labelFor(record: BenchmarkRecord): string {
  if (record.backend === "nvidia" || record.backend === "gpu") {
    return `GPU ${record.device_class}`;
  }
  if (record.implementation === "cann_ops_library") {
    return "CANN ops library";
  }
  return `SIMT op ${record.implementation_version}`;
}

export function KernelTraceRail({ records }: KernelTraceRailProps) {
  return (
    <section className="trace-rail" aria-label="Selected case kernel trace">
      {records.map((record) => (
        <div
          key={`${record.backend}-${record.implementation}-${record.implementation_version}`}
          className={`trace-chip trace-chip--${record.backend === "nvidia" || record.backend === "gpu" ? "gpu" : record.implementation}`}
        >
          <span>{labelFor(record)}</span>
          <strong>{record.metrics.latency_ms_avg.toFixed(4)} ms</strong>
        </div>
      ))}
    </section>
  );
}
