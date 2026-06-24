import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { GridComponent, LegendComponent, TooltipComponent } from "echarts/components";
import { LineChart } from "echarts/charts";
import { SVGRenderer } from "echarts/renderers";
import type { ChartSeries } from "../types";

echarts.use([GridComponent, LegendComponent, TooltipComponent, LineChart, SVGRenderer]);

const SERIES_COLORS = ["#75f94c", "#ff7a3d", "#2df1ff", "#4da3ff", "#8cc8ff"];

interface BenchmarkChartProps {
  series: ChartSeries[];
  caseIds: string[];
}

export function BenchmarkChart({ series, caseIds }: BenchmarkChartProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    if (!chartRef.current) {
      return;
    }

    const chart = echarts.init(chartRef.current, undefined, { renderer: "svg" });
    chart.setOption({
      color: SERIES_COLORS,
      backgroundColor: "transparent",
      tooltip: { trigger: "axis" },
      legend: {
        top: 0,
        textStyle: { color: "#8fa6ad", fontFamily: "JetBrains Mono, monospace" }
      },
      grid: { top: 56, right: 18, bottom: 58, left: 58 },
      xAxis: {
        type: "category",
        data: caseIds,
        axisLabel: { color: "#8fa6ad", rotate: 18 },
        axisLine: { lineStyle: { color: "rgba(143, 166, 173, 0.22)" } }
      },
      yAxis: {
        type: "value",
        name: "latency ms",
        nameTextStyle: { color: "#8fa6ad" },
        axisLabel: { color: "#8fa6ad" },
        splitLine: { lineStyle: { color: "rgba(143, 166, 173, 0.12)" } }
      },
      series: series.map((item) => ({
        name: item.name,
        type: "line",
        smooth: true,
        symbolSize: 8,
        data: item.points.map((point) => point.latencyMs)
      }))
    });

    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      chart.dispose();
    };
  }, [caseIds, series]);

  return (
    <section className="chart-panel" aria-label="Latency comparison chart">
      <div ref={chartRef} className="chart-canvas" />
    </section>
  );
}
