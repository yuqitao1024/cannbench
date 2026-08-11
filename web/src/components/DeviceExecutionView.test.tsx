import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, expect, it } from "vitest";
import type { ShapeTrace } from "../shape-trace/types";
import { DeviceExecutionView } from "./DeviceExecutionView";

const availableExecution: ShapeTrace["device_execution"] = {
  status: "available",
  implementation: "simt",
  version: "v2",
  message: null,
  kernels: [
    {
      id: "indexer",
      title: "Indexer score + radix TopK",
      summary: "32 mixed tasks",
      task_count: 32,
      used_core_count: 32,
      task_formula: "2 x 1 x 16",
      task_axes: [
        { symbol: "B", value: 2, meaning: "batch requests", role: "preserved" },
        { symbol: "Cs", value: 16, meaning: "context shards", role: "produced" }
      ],
      tile_tensors: [
        {
          id: "query-atom",
          label: "query atom",
          logical_only: false,
          axes: [
            { symbol: "Q", value: 2, meaning: "query tokens", role: "preserved" },
            { symbol: "Hi", value: 64, meaning: "index heads", role: "preserved" },
            { symbol: "Di", value: 128, meaning: "index feature", role: "contracted" }
          ]
        }
      ],
      steps: ["score", "TopK"]
    }
  ]
};

const prefillUnavailableExecution: ShapeTrace["device_execution"] = {
  status: "unavailable",
  implementation: "simt",
  version: null,
  message: "No device trace for prefill",
  kernels: []
};

afterEach(cleanup);

it("renders generic task and tile facts", () => {
  render(<DeviceExecutionView execution={availableExecution} />);

  expect(screen.getByText("SIMT v2")).toBeInTheDocument();
  expect(screen.getByText("32 mixed tasks")).toBeInTheDocument();
  expect(screen.getByText("[2,64,128]")).toBeInTheDocument();
  expect(screen.getByText("score")).toBeInTheDocument();
  expect(screen.getByText("TopK")).toBeInTheDocument();
  expect(screen.getByText("B0 / Cs0")).toBeInTheDocument();
  expect(screen.getByText("B0 / Cs7")).toBeInTheDocument();
  expect(screen.getByText("+24 tasks")).toBeInTheDocument();
});

it("renders prefill unavailable without decode facts", () => {
  render(<DeviceExecutionView execution={prefillUnavailableExecution} />);

  expect(screen.getByText("No device trace for prefill")).toBeInTheDocument();
  expect(screen.queryByText("Device execution unavailable")).not.toBeInTheDocument();
  expect(screen.queryByText("32 mixed tasks")).not.toBeInTheDocument();
  expect(screen.queryByText("Head64")).not.toBeInTheDocument();
});

it("does not invent unavailable copy when the payload message is absent", () => {
  const { container } = render(
    <DeviceExecutionView execution={{ ...prefillUnavailableExecution, message: null }} />
  );

  expect(container).toBeEmptyDOMElement();
  expect(screen.queryByText("No device trace is available for this phase.")).not.toBeInTheDocument();
});
