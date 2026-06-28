import { describe, expect, it, vi } from "vitest";
import { render, waitFor } from "@testing-library/react";
import { BenchmarkChart } from "./BenchmarkChart";
import type { ChartSegment, ChartSeries } from "../types";

const setOption = vi.fn();
const resize = vi.fn();
const dispose = vi.fn();

vi.mock("echarts/core", () => ({
  use: vi.fn(),
  init: vi.fn(() => ({
    setOption,
    resize,
    dispose
  }))
}));

vi.mock("echarts/components", () => ({
  GridComponent: {},
  LegendComponent: {},
  MarkLineComponent: {},
  TooltipComponent: {}
}));

vi.mock("echarts/charts", () => ({
  LineChart: {}
}));

vi.mock("echarts/renderers", () => ({
  SVGRenderer: {}
}));

describe("BenchmarkChart", () => {
  it("hides raw case labels on the x-axis", async () => {
    const series: ChartSeries[] = [
      {
        key: "ascend-950pr-cannops",
        name: "Ascend 950PR CANN Ops",
        records: [],
        points: [
          { caseId: "bert_pytorch_attention", latencyMs: 0.01, record: null },
          { caseId: "camembert_logits", latencyMs: 0.71, record: null }
        ]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 1 }];

    render(<BenchmarkChart series={series} segments={segments} />);

    await waitFor(() => {
      expect(setOption).toHaveBeenCalled();
    });

    const option = setOption.mock.calls.at(-1)?.[0];
    expect(option.xAxis.data).toEqual([1, 2]);
    expect(option.xAxis.axisLabel.show).toBe(false);
  });
});
