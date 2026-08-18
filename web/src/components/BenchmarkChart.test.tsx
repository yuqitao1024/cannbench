import "@testing-library/jest-dom/vitest";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
  afterEach(() => {
    cleanup();
  });

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

  it("labels raw and relative charts according to the active baseline", () => {
    const series: ChartSeries[] = [
      {
        key: "baseline",
        name: "Baseline",
        records: [],
        points: [{ caseId: "case", latencyMs: 0.01, record: null }]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 0 }];
    const { rerender } = render(<BenchmarkChart series={series} segments={segments} baseline={null} />);

    expect(screen.getByLabelText("Latency comparison chart")).toBeInTheDocument();

    rerender(
      <BenchmarkChart
        series={series}
        segments={segments}
        baseline={{ seriesKey: "baseline", seriesName: "Baseline", kind: "cuda" }}
      />
    );
    expect(screen.getByLabelText("Performance comparison chart vs CUDA baseline")).toBeInTheDocument();

    rerender(
      <BenchmarkChart
        series={series}
        segments={segments}
        baseline={{ seriesKey: "baseline", seriesName: "Baseline", kind: "cann_ops" }}
      />
    );
    expect(screen.getByLabelText("Performance comparison chart vs CANN Ops baseline")).toBeInTheDocument();
  });

  it("plots performance relative to CUDA", async () => {
    const series: ChartSeries[] = [
      {
        key: "nvidia-h800-cuda-pytorch",
        name: "NVIDIA H800 PyTorch",
        records: [],
        points: [
          { caseId: "bert_pytorch_attention", latencyMs: 0.02, record: null },
          { caseId: "camembert_logits", latencyMs: 0.02, record: null }
        ]
      },
      {
        key: "ascend-950pr-simt-v1",
        name: "Ascend 950PR SIMT v1",
        records: [],
        points: [
          { caseId: "bert_pytorch_attention", latencyMs: 0.01, record: null },
          { caseId: "camembert_logits", latencyMs: 0.04, record: null }
        ]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 1 }];
    const baseline = {
      seriesKey: "nvidia-h800-cuda-pytorch",
      seriesName: "NVIDIA H800 PyTorch",
      kind: "cuda"
    } as const;

    render(<BenchmarkChart series={series} segments={segments} baseline={baseline} />);

    await waitFor(() => {
      expect(setOption).toHaveBeenCalled();
    });

    const option = setOption.mock.calls.at(-1)?.[0];
    expect(option.xAxis.data).toEqual([1, 2]);
    expect(option.xAxis.axisLabel.show).toBe(false);
    expect(option.yAxis.name).toBe("Performance vs CUDA (x)");
    expect(option.series[0].data).toEqual([1, 1]);
    expect(option.series[1].data).toEqual([2, 0.5]);
    expect(option.series[0].markLine.data).toContainEqual(expect.objectContaining({ yAxis: 1 }));
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

    render(<BenchmarkChart series={series} segments={segments} baseline={null} />);

    await waitFor(() => {
      expect(setOption).toHaveBeenCalled();
    });

    const option = setOption.mock.calls.at(-1)?.[0];
    expect(option.tooltip.enterable).toBe(true);
    expect(option.tooltip.triggerOn).toBe("mousemove|click");
    expect(option.tooltip.alwaysShowContent).toBe(false);
    expect(option.tooltip.extraCssText).toContain("max-height");
    expect(option.tooltip.extraCssText).toContain("overflow-y:auto");
  });

  it("summarizes selected series against the CUDA baseline with geometric mean", async () => {
    const series: ChartSeries[] = [
      {
        key: "nvidia-h800-cuda-pytorch",
        name: "NVIDIA H800 PyTorch",
        records: [],
        points: [
          { caseId: "a", latencyMs: 0.01, record: null },
          { caseId: "b", latencyMs: 0.02, record: null },
          { caseId: "c", latencyMs: 0.04, record: null }
        ]
      },
      {
        key: "ascend-950pr-simt-v2",
        name: "Ascend 950PR SIMT v2",
        records: [],
        points: [
          { caseId: "a", latencyMs: 0.01, record: null },
          { caseId: "b", latencyMs: 0.02, record: null },
          { caseId: "c", latencyMs: 0.16, record: null }
        ]
      },
      {
        key: "ascend-950pr-cannops",
        name: "Ascend 950PR CANN Ops",
        records: [],
        points: [
          { caseId: "a", latencyMs: 0.005, record: null },
          { caseId: "b", latencyMs: 0.01, record: null },
          { caseId: "c", latencyMs: 0.02, record: null }
        ]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 2 }];

    render(
      <BenchmarkChart
        series={series}
        segments={segments}
        baseline={{
          seriesKey: "nvidia-h800-cuda-pytorch",
          seriesName: "NVIDIA H800 PyTorch",
          kind: "cuda"
        }}
      />
    );

    expect(screen.getByText(/vs cuda baseline/i)).toBeInTheDocument();
    expect(screen.getByText(/Ascend 950PR SIMT v2/i)).toBeInTheDocument();
    expect(screen.getByText(/1.59x slower/i)).toHaveClass("is-slower");
    expect(screen.getByText(/Ascend 950PR CANN Ops/i)).toBeInTheDocument();
    expect(screen.getByText(/2.00x faster/i)).toHaveClass("is-faster");
  });

  it("adds per-case CUDA comparison to the tooltip", async () => {
    const cudaRecord = {
      case_id: "bert_pytorch_attention",
      shape: [1, 16, 128, 128],
      dtype: "float16"
    };
    const series: ChartSeries[] = [
      {
        key: "nvidia-h800-cuda-pytorch",
        name: "NVIDIA H800 PyTorch",
        records: [],
        points: [{ caseId: "bert_pytorch_attention", latencyMs: 0.01, record: cudaRecord as never }]
      },
      {
        key: "ascend-950pr-simt-v2",
        name: "Ascend 950PR SIMT v2",
        records: [],
        points: [{ caseId: "bert_pytorch_attention", latencyMs: 0.03, record: cudaRecord as never }]
      }
    ];
    const segments: ChartSegment[] = [{ key: "realistic", label: "realistic", start: 0, end: 0 }];

    render(
      <BenchmarkChart
        series={series}
        segments={segments}
        baseline={{
          seriesKey: "nvidia-h800-cuda-pytorch",
          seriesName: "NVIDIA H800 PyTorch",
          kind: "cuda"
        }}
      />
    );

    await waitFor(() => {
      expect(setOption).toHaveBeenCalled();
    });

    const option = setOption.mock.calls.at(-1)?.[0];
    const html = option.tooltip.formatter([
      { seriesName: "NVIDIA H800 PyTorch", value: 10, dataIndex: 0 },
      { seriesName: "Ascend 950PR SIMT v2", value: 30, dataIndex: 0 }
    ]);

    expect(html).toContain("latency: 30.00 us");
    expect(html).toContain("vs CUDA: baseline");
    expect(html).toContain("vs CUDA: <span class=\"tooltip-ratio is-slower\">3.00x slower</span>");
  });

  it("leaves a case blank when its selected baseline point is missing", async () => {
    const series: ChartSeries[] = [
      {
        key: "cuda",
        name: "NVIDIA H800 PyTorch",
        records: [],
        points: [
          { caseId: "a", latencyMs: 0.02, record: null },
          { caseId: "b", latencyMs: null, record: null }
        ]
      },
      {
        key: "simt",
        name: "Ascend 950PR SIMT v1",
        records: [],
        points: [
          { caseId: "a", latencyMs: 0.01, record: null },
          { caseId: "b", latencyMs: 0.01, record: null }
        ]
      }
    ];

    render(
      <BenchmarkChart
        series={series}
        segments={[{ key: "realistic", label: "realistic", start: 0, end: 1 }]}
        baseline={{ seriesKey: "cuda", seriesName: "NVIDIA H800 PyTorch", kind: "cuda" }}
      />
    );

    await waitFor(() => expect(setOption).toHaveBeenCalled());
    const option = setOption.mock.calls.at(-1)?.[0];
    expect(option.series[1].data).toEqual([2, null]);
    expect(option.series[1].connectNulls).toBe(false);
  });

  it("labels CANN Ops as the relative baseline", async () => {
    const series: ChartSeries[] = [
      {
        key: "cann",
        name: "Ascend 950PR CANN Ops",
        records: [],
        points: [{ caseId: "a", latencyMs: 0.02, record: null }]
      },
      {
        key: "simt",
        name: "Ascend 950PR SIMT v1",
        records: [],
        points: [{ caseId: "a", latencyMs: 0.01, record: null }]
      }
    ];

    render(
      <BenchmarkChart
        series={series}
        segments={[{ key: "realistic", label: "realistic", start: 0, end: 0 }]}
        baseline={{ seriesKey: "cann", seriesName: "Ascend 950PR CANN Ops", kind: "cann_ops" }}
      />
    );

    await waitFor(() => expect(setOption).toHaveBeenCalled());
    const option = setOption.mock.calls.at(-1)?.[0];
    expect(option.yAxis.name).toBe("Performance vs CANN Ops (x)");
    expect(option.series[0].data).toEqual([1]);
  });

  it("retains raw latency when there is no baseline", async () => {
    const record = {
      case_id: "a",
      shape: [1, 16],
      dtype: "float16"
    };
    const series: ChartSeries[] = [
      {
        key: "simt",
        name: "Ascend 950PR SIMT v1",
        records: [],
        points: [{ caseId: "a", latencyMs: 0.015, record: record as never }]
      }
    ];

    render(
      <BenchmarkChart
        series={series}
        segments={[{ key: "realistic", label: "realistic", start: 0, end: 0 }]}
        baseline={null}
      />
    );

    await waitFor(() => expect(setOption).toHaveBeenCalled());
    const option = setOption.mock.calls.at(-1)?.[0];
    expect(option.yAxis.name).toBe("latency us");
    expect(option.series[0].data).toEqual([15]);
    expect(option.series[0].markLine?.data ?? []).not.toContainEqual(expect.objectContaining({ yAxis: 1 }));

    const html = option.tooltip.formatter([{ seriesName: "Ascend 950PR SIMT v1", value: 15, dataIndex: 0 }]);
    expect(html).toContain("latency:");
    expect(html).not.toContain("vs CUDA");
    expect(html).not.toContain("vs CANN Ops");
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

    render(<BenchmarkChart series={series} segments={segments} baseline={null} />);

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
    expect(setOption).toHaveBeenCalledWith({ tooltip: { alwaysShowContent: true } });
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

    render(<BenchmarkChart series={series} segments={segments} baseline={null} />);

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

    render(<BenchmarkChart series={series} segments={segments} baseline={null} />);

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

    expect(setOption).toHaveBeenCalledWith({ tooltip: { alwaysShowContent: false } });
    expect(dispatchAction).toHaveBeenCalledWith({ type: "hideTip" });
  });
});
