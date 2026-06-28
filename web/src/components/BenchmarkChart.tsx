import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { GridComponent, LegendComponent, MarkLineComponent, TooltipComponent } from "echarts/components";
import { LineChart } from "echarts/charts";
import { SVGRenderer } from "echarts/renderers";
import type { ChartSegment, ChartSeries } from "../types";

echarts.use([GridComponent, LegendComponent, TooltipComponent, MarkLineComponent, LineChart, SVGRenderer]);

const SERIES_COLORS = ["#b8bb26", "#fe8019", "#83a598", "#689d6a", "#d3869b"];

interface BenchmarkChartProps {
  series: ChartSeries[];
  segments: ChartSegment[];
}

function tooltipHtml(params: Array<{ seriesName: string; value: number | null; dataIndex: number }>, series: ChartSeries[]) {
  const lines = params
    .map((param) => {
      const matchedSeries = series.find((item) => item.name === param.seriesName);
      const point = matchedSeries?.points[param.dataIndex];
      const record = point?.record;
      if (!record || point?.latencyMs === null) {
        return null;
      }
      return [
        `<strong>${param.seriesName}</strong>`,
        `case: ${record.case_id}`,
        `latency: ${point.latencyMs.toFixed(4)} ms`,
        `shape: ${record.shape.join(" x ")}`,
        `dtype: ${record.dtype}`
      ].join("<br/>");
    })
    .filter(Boolean);
  return lines.join("<hr/>");
}

export function BenchmarkChart({ series, segments }: BenchmarkChartProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!chartRef.current) {
      return;
    }

    const chart = echarts.init(chartRef.current, undefined, { renderer: "svg" });
    const caseIds = series[0]?.points.map((point) => point.caseId) ?? [];
    chart.setOption({
      color: SERIES_COLORS,
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        backgroundColor: "rgba(29, 32, 33, 0.92)",
        borderColor: "rgba(235, 219, 178, 0.12)",
        textStyle: { color: "#ebdbb2", fontFamily: "JetBrains Mono, monospace" },
        formatter: (params: unknown) => tooltipHtml(params as Array<{ seriesName: string; value: number | null; dataIndex: number }>, series)
      },
      legend: {
        top: 0,
        textStyle: { color: "#a89984", fontFamily: "JetBrains Mono, monospace" }
      },
      grid: { top: 62, right: 20, bottom: 72, left: 62 },
      xAxis: {
        type: "category",
        data: caseIds,
        axisLabel: { color: "#a89984", rotate: 22, interval: 0 },
        axisLine: { lineStyle: { color: "rgba(168, 153, 132, 0.24)" } }
      },
      yAxis: {
        type: "value",
        name: "latency ms",
        nameTextStyle: { color: "#a89984", fontFamily: "JetBrains Mono, monospace" },
        axisLabel: { color: "#a89984" },
        splitLine: { lineStyle: { color: "rgba(168, 153, 132, 0.12)" } }
      },
      series: series.map((item) => ({
        name: item.name,
        type: "line",
        connectNulls: false,
        smooth: false,
        showSymbol: true,
        symbolSize: 7,
        data: item.points.map((point) => point.latencyMs),
        markLine:
          segments.length > 1
            ? {
                symbol: "none",
                label: { show: false },
                lineStyle: { color: "rgba(250, 189, 47, 0.18)", type: "dashed" },
                data: segments.slice(0, -1).map((segment) => ({ xAxis: segment.end + 0.5 }))
              }
            : undefined
      }))
    });

    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [segments, series]);

  return (
    <section className="chart-panel" aria-label="Latency comparison chart">
      {segments.length > 1 ? (
        <div className="chart-segment-bar" aria-hidden="true">
          {segments.map((segment) => (
            <span key={segment.key}>{segment.label}</span>
          ))}
        </div>
      ) : null}
      <div ref={chartRef} className="chart-canvas" />
    </section>
  );
}
