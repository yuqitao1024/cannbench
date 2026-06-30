import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, waitFor } from "@testing-library/react";
import { BenchmarkChart } from "./BenchmarkChart";
import type { ChartSegment, ChartSeries } from "../types";

const setOption = vi.fn();
const on = vi.fn();
const off = vi.fn();
const zrOn = vi.fn();
const zrOff = vi.fn();
const dispatchAction = vi.fn();
const resize = vi.fn();
const dispose = vi.fn();

vi.mock("echarts/core", () => ({
  use: vi.fn(),
  init: vi.fn(() => ({
    setOption,
    on,
    off,
    getZr: () => ({
      on: zrOn,
      off: zrOff
    }),
    dispatchAction,
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
  beforeEach(() => {
    setOption.mockClear();
    on.mockClear();
    off.mockClear();
    zrOn.mockClear();
    zrOff.mockClear();
    dispatchAction.mockClear();
    resize.mockClear();
    dispose.mockClear();
  });

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
    expect(option.yAxis.name).toBe("latency us");
    expect(option.series[0].data).toEqual([10, 710]);
  });

  it("keeps dense multi-series tooltips enterable and scrollable", async () => {
    const series: ChartSeries[] = [
      {
        key: "nvidia-h800-cuda-pytorch",
        name: "NVIDIA H800 PyTorch",
        records: [],
        points: [{ caseId: "bert_pytorch_attention", latencyMs: 18.5, record: null }]
      },
      {
        key: "ascend-950pr-cannops",
        name: "Ascend 950PR CANN Ops",
        records: [],
        points: [{ caseId: "bert_pytorch_attention", latencyMs: 0.01, record: null }]
      },
      {
        key: "ascend-950pr-simt-v1",
        name: "Ascend 950PR SIMT v1",
        records: [],
        points: [{ caseId: "bert_pytorch_attention", latencyMs: 0.01, record: null }]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 0 }];

    render(<BenchmarkChart series={series} segments={segments} />);

    await waitFor(() => {
      expect(setOption).toHaveBeenCalled();
    });

    const option = setOption.mock.calls.at(-1)?.[0];
    expect(option.tooltip.enterable).toBe(true);
    expect(option.tooltip.triggerOn).toBe("mousemove|click");
    expect(option.tooltip.alwaysShowContent).toBe(true);
    expect(option.tooltip.extraCssText).toContain("max-height");
    expect(option.tooltip.extraCssText).toContain("overflow-y:auto");
  });

  it("pins tooltip position after clicking a chart point", async () => {
    const series: ChartSeries[] = [
      {
        key: "nvidia-h800-cuda-pytorch",
        name: "NVIDIA H800 PyTorch",
        records: [],
        points: [{ caseId: "bert_pytorch_attention", latencyMs: 0.01, record: null }]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 0 }];

    render(<BenchmarkChart series={series} segments={segments} />);

    await waitFor(() => {
      expect(setOption).toHaveBeenCalled();
    });

    const option = setOption.mock.calls.at(-1)?.[0];
    const clickHandler = on.mock.calls.find(([eventName]) => eventName === "click")?.[1];

    expect(option.tooltip.position([40, 50], null, null, null, { contentSize: [180, 100], viewSize: [800, 400] })).toEqual([52, 62]);
    clickHandler({
      componentType: "series",
      seriesIndex: 0,
      dataIndex: 0,
      event: { offsetX: 100, offsetY: 120 }
    });

    expect(option.tooltip.position([40, 50], null, null, null, { contentSize: [180, 100], viewSize: [800, 400] })).toEqual([112, 132]);
    expect(dispatchAction).toHaveBeenCalledWith({
      type: "showTip",
      seriesIndex: 0,
      dataIndex: 0
    });
  });

  it("keeps pinned tooltips inside the chart viewport", async () => {
    const series: ChartSeries[] = [
      {
        key: "nvidia-h800-cuda-pytorch",
        name: "NVIDIA H800 PyTorch",
        records: [],
        points: [{ caseId: "right_edge_case", latencyMs: 0.01, record: null }]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 0 }];

    render(<BenchmarkChart series={series} segments={segments} />);

    await waitFor(() => {
      expect(setOption).toHaveBeenCalled();
    });

    const option = setOption.mock.calls.at(-1)?.[0];
    const clickHandler = on.mock.calls.find(([eventName]) => eventName === "click")?.[1];

    clickHandler({
      componentType: "series",
      seriesIndex: 0,
      dataIndex: 0,
      event: { offsetX: 760, offsetY: 350 }
    });

    expect(option.tooltip.position([760, 350], null, null, null, { contentSize: [180, 100], viewSize: [800, 400] })).toEqual([568, 238]);
  });

  it("hides a pinned tooltip when clicking outside the chart", async () => {
    const series: ChartSeries[] = [
      {
        key: "nvidia-h800-cuda-pytorch",
        name: "NVIDIA H800 PyTorch",
        records: [],
        points: [{ caseId: "bert_pytorch_attention", latencyMs: 0.01, record: null }]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 0 }];

    render(<BenchmarkChart series={series} segments={segments} />);

    await waitFor(() => {
      expect(setOption).toHaveBeenCalled();
    });

    const clickHandler = on.mock.calls.find(([eventName]) => eventName === "click")?.[1];
    clickHandler({
      componentType: "series",
      seriesIndex: 0,
      dataIndex: 0,
      event: { offsetX: 100, offsetY: 120 }
    });
    dispatchAction.mockClear();

    fireEvent.pointerDown(document.body);

    expect(dispatchAction).toHaveBeenCalledWith({ type: "hideTip" });
  });
});
