import { useEffect, useRef } from "react";
import * as echarts from "echarts/core";
import { GridComponent, LegendComponent, MarkLineComponent, TooltipComponent } from "echarts/components";
import { LineChart } from "echarts/charts";
import { SVGRenderer } from "echarts/renderers";
import { relativePerformanceValue } from "../data/performanceBaseline";
import type { ChartSegment, ChartSeries, PerformanceBaseline } from "../types";

echarts.use([GridComponent, LegendComponent, TooltipComponent, MarkLineComponent, LineChart, SVGRenderer]);

const SERIES_COLORS = ["#b8bb26", "#fe8019", "#83a598", "#689d6a", "#d3869b"];
const MS_TO_US = 1000;
const TOOLTIP_OFFSET = 12;

interface BenchmarkChartProps {
  series: ChartSeries[];
  segments: ChartSegment[];
  baseline: PerformanceBaseline | null;
}

interface TooltipPositionSize {
  contentSize?: [number, number];
  viewSize?: [number, number];
}

interface BaselineSummaryItem {
  key: string;
  name: string;
  ratio: number;
}

function baselineLabel(baseline: PerformanceBaseline): string {
  return baseline.kind === "cuda" ? "CUDA" : "CANN Ops";
}

function pointMap(series: ChartSeries): Map<string, number | null> {
  return new Map(series.points.map((point) => [point.caseId, point.latencyMs]));
}

function chartValues(item: ChartSeries, baselineSeries: ChartSeries | null): Array<number | null> {
  if (!baselineSeries) {
    return item.points.map((point) => (point.latencyMs === null ? null : point.latencyMs * MS_TO_US));
  }
  const baselineByCase = pointMap(baselineSeries);
  return item.points.map((point) => relativePerformanceValue(baselineByCase.get(point.caseId) ?? null, point.latencyMs));
}

function formatPerformance(ratio: number) {
  if (ratio > 1) {
    return `${ratio.toFixed(2)}x faster`;
  }
  if (ratio < 1) {
    return `${(1 / ratio).toFixed(2)}x slower`;
  }
  return "equal performance";
}

function performanceClassName(ratio: number) {
  if (ratio > 1) return "is-faster";
  if (ratio < 1) return "is-slower";
  return "is-equal";
}

function geometricMean(values: number[]) {
  if (values.length === 0) {
    return null;
  }
  if (values.some((value) => value <= 0)) {
    return null;
  }
  const logSum = values.reduce((sum, value) => sum + Math.log(value), 0);
  return Math.exp(logSum / values.length);
}

function baselineSummary(
  series: ChartSeries[],
  baseline: PerformanceBaseline | null,
  baselineSeries: ChartSeries | null
): BaselineSummaryItem[] {
  if (!baseline || !baselineSeries) {
    return [];
  }
  const baselineByCase = pointMap(baselineSeries);

  return series
    .filter((item) => item.key !== baseline.seriesKey)
    .map((item) => {
      const ratios = item.points
        .map((point) => relativePerformanceValue(baselineByCase.get(point.caseId) ?? null, point.latencyMs))
        .filter((value): value is number => value !== null);
      const ratio = geometricMean(ratios);
      return ratio === null ? null : { key: item.key, name: item.name, ratio };
    })
    .filter((item): item is BaselineSummaryItem => item !== null);
}

function tooltipPosition(anchor: [number, number], size?: TooltipPositionSize): [number, number] {
  const [contentWidth = 0, contentHeight = 0] = size?.contentSize ?? [];
  const [viewWidth = 0, viewHeight = 0] = size?.viewSize ?? [];
  let x = anchor[0] + TOOLTIP_OFFSET;
  let y = anchor[1] + TOOLTIP_OFFSET;

  if (viewWidth > 0 && contentWidth > 0 && x + contentWidth + TOOLTIP_OFFSET > viewWidth) {
    x = anchor[0] - contentWidth - TOOLTIP_OFFSET;
  }
  if (viewHeight > 0 && contentHeight > 0 && y + contentHeight + TOOLTIP_OFFSET > viewHeight) {
    y = anchor[1] - contentHeight - TOOLTIP_OFFSET;
  }

  return [Math.max(TOOLTIP_OFFSET, x), Math.max(TOOLTIP_OFFSET, y)];
}

function tooltipHtml(
  params: Array<{ seriesName: string; value: number | null; dataIndex: number }>,
  series: ChartSeries[],
  baseline: PerformanceBaseline | null
) {
  const baselineSeries = baseline ? series.find((item) => item.key === baseline.seriesKey) ?? null : null;
  const lines = params
    .map((param) => {
      const matchedSeries = series.find((item) => item.name === param.seriesName);
      const point = matchedSeries?.points[param.dataIndex];
      const record = point?.record;
      if (!record || point?.latencyMs === null) {
        return null;
      }
      const baselinePoint = baselineSeries?.points.find((candidate) => candidate.caseId === point.caseId) ?? null;
      const performanceLine = baseline
        ? `vs ${baselineLabel(baseline)}: ${formatTooltipPerformance(
            relativePerformanceValue(baselinePoint?.latencyMs ?? null, point.latencyMs),
            matchedSeries?.key === baseline.seriesKey
          )}`
        : null;
      return [
        `<strong>${param.seriesName}</strong>`,
        `case: ${record.case_id}`,
        `latency: ${(point.latencyMs * MS_TO_US).toFixed(2)} us`,
        performanceLine,
        `shape: ${record.shape.join(" x ")}`,
        `dtype: ${record.dtype}`
      ].filter((line): line is string => line !== null).join("<br/>");
    })
    .filter(Boolean);
  return lines.join("<hr/>");
}

function formatTooltipPerformance(ratio: number | null, isBaseline: boolean): string {
  if (isBaseline) {
    return "baseline";
  }
  if (ratio === null) {
    return "n/a";
  }
  return `<span class="tooltip-ratio ${performanceClassName(ratio)}">${formatPerformance(ratio)}</span>`;
}

export function BenchmarkChart({ series, segments, baseline }: BenchmarkChartProps) {
  const chartRef = useRef<HTMLDivElement | null>(null);
  const baselineSeries = baseline ? series.find((item) => item.key === baseline.seriesKey) ?? null : null;
  const summary = baselineSummary(series, baseline, baselineSeries);

  useEffect(() => {
    if (!chartRef.current) {
      return;
    }

    const chart = echarts.init(chartRef.current, undefined, { renderer: "svg" });
    let pinnedTooltipAnchor: [number, number] | null = null;
    const hidePinnedTooltip = () => {
      pinnedTooltipAnchor = null;
      chart.setOption({ tooltip: { alwaysShowContent: false } });
      chart.dispatchAction({ type: "hideTip" });
    };
    const pinTooltip = (params: unknown) => {
      const point = params as {
        componentType?: string;
        seriesIndex?: number;
        dataIndex?: number;
        event?: { offsetX?: number; offsetY?: number };
      };
      if (
        point.componentType !== "series" ||
        typeof point.seriesIndex !== "number" ||
        typeof point.dataIndex !== "number" ||
        typeof point.event?.offsetX !== "number" ||
        typeof point.event?.offsetY !== "number"
      ) {
        return;
      }
      pinnedTooltipAnchor = [point.event.offsetX, point.event.offsetY];
      chart.setOption({ tooltip: { alwaysShowContent: true } });
      chart.dispatchAction({
        type: "showTip",
        seriesIndex: point.seriesIndex,
        dataIndex: point.dataIndex
      });
    };
    const clearPinnedTooltip = (event?: { target?: unknown }) => {
      if (event?.target) {
        return;
      }
      hidePinnedTooltip();
    };
    const clearPinnedTooltipOnDocumentClick = (event: PointerEvent) => {
      if (chartRef.current?.contains(event.target as Node | null)) {
        return;
      }
      hidePinnedTooltip();
    };
    const caseMarkers = series[0]?.points.map((_, index) => index + 1) ?? [];
    chart.setOption({
      color: SERIES_COLORS,
      backgroundColor: "transparent",
      tooltip: {
        trigger: "axis",
        triggerOn: "mousemove|click",
        enterable: true,
        alwaysShowContent: false,
        backgroundColor: "rgba(29, 32, 33, 0.92)",
        borderColor: "rgba(235, 219, 178, 0.12)",
        textStyle: { color: "#ebdbb2", fontFamily: "JetBrains Mono, monospace" },
        extraCssText: "max-height: 260px; overflow-y:auto; max-width: 420px;",
        confine: true,
        position: (point: number[], _params: unknown, _dom: unknown, _rect: unknown, size?: TooltipPositionSize) =>
          tooltipPosition(pinnedTooltipAnchor ?? [point[0], point[1]], size),
        formatter: (params: unknown) =>
          tooltipHtml(params as Array<{ seriesName: string; value: number | null; dataIndex: number }>, series, baseline)
      },
      legend: {
        top: 0,
        textStyle: { color: "#a89984", fontFamily: "JetBrains Mono, monospace" }
      },
      grid: { top: 62, right: 20, bottom: 72, left: 62 },
      xAxis: {
        type: "category",
        data: caseMarkers,
        axisLabel: { show: false },
        axisTick: { show: false },
        axisLine: { lineStyle: { color: "rgba(168, 153, 132, 0.24)" } }
      },
      yAxis: {
        type: "value",
        name: baseline ? `Performance vs ${baselineLabel(baseline)} (x)` : "latency us",
        nameTextStyle: { color: "#a89984", fontFamily: "JetBrains Mono, monospace" },
        axisLabel: { color: "#a89984" },
        splitLine: { lineStyle: { color: "rgba(168, 153, 132, 0.12)" } }
      },
      series: series.map((item) => {
        const segmentLines =
          segments.length > 1
            ? segments.slice(0, -1).map((segment) => ({ xAxis: segment.end + 0.5 }))
            : [];
        const referenceLines =
          baseline && item.key === baseline.seriesKey
            ? [
                {
                  yAxis: 1,
                  lineStyle: { color: "rgba(184, 187, 38, 0.55)", type: "dashed" },
                  label: { show: true, formatter: "1x", color: "#a89984" }
                }
              ]
            : [];
        const markLineData = [...segmentLines, ...referenceLines];
        return {
          name: item.name,
          type: "line",
          connectNulls: false,
          smooth: false,
          showSymbol: true,
          symbolSize: 7,
          data: chartValues(item, baselineSeries),
          markLine:
            markLineData.length > 0
              ? {
                  symbol: "none",
                  label: { show: false },
                  lineStyle: { color: "rgba(250, 189, 47, 0.18)", type: "dashed" },
                  data: markLineData
                }
              : undefined
        };
      })
    });
    chart.on("click", pinTooltip);
    chart.getZr().on("click", clearPinnedTooltip);
    document.addEventListener("pointerdown", clearPinnedTooltipOnDocumentClick);

    const resize = () => chart.resize();
    window.addEventListener("resize", resize);
    return () => {
      window.removeEventListener("resize", resize);
      document.removeEventListener("pointerdown", clearPinnedTooltipOnDocumentClick);
      chart.off("click", pinTooltip);
      chart.getZr().off("click", clearPinnedTooltip);
      chart.dispose();
    };
  }, [baseline, baselineSeries, segments, series]);

  return (
    <section
      className="chart-panel"
      aria-label={baseline ? `Performance comparison chart vs ${baselineLabel(baseline)} baseline` : "Latency comparison chart"}
    >
      {summary.length > 0 ? (
        <div className="chart-baseline-card" aria-label={`${baselineLabel(baseline!)} baseline comparison`}>
          <span className="chart-baseline-title">vs {baselineLabel(baseline!)} baseline</span>
          <div className="chart-baseline-items">
            {summary.map((item) => (
              <span key={item.key} className="chart-baseline-item">
                <span>{item.name}</span>
                <strong className={performanceClassName(item.ratio)}>{formatPerformance(item.ratio)}</strong>
              </span>
            ))}
          </div>
        </div>
      ) : null}
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
