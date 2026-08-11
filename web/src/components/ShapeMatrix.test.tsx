import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { ShapeStage, ShapeTensor } from "../shape-trace/types";
import { ShapeMatrix, ShapeMatrixEquation } from "./ShapeMatrix";

const wideTensor: ShapeTensor = {
  id: "k",
  label: "K^T",
  logical_only: false,
  axes: [
    { symbol: "Di", value: 128, meaning: "feature", role: "contracted" },
    { symbol: "C", value: 32768, meaning: "context", role: "produced" }
  ]
};

const tallTensor: ShapeTensor = {
  id: "kv",
  label: "shared_kv",
  logical_only: false,
  axes: [
    { symbol: "C", value: 32768, meaning: "context", role: "preserved" },
    { symbol: "Dqk", value: 576, meaning: "QK feature", role: "preserved" }
  ]
};

const vectorTensor: ShapeTensor = {
  id: "indices",
  label: "indices",
  logical_only: false,
  axes: [{ symbol: "S", value: 2048, meaning: "selected tokens", role: "produced" }]
};

afterEach(cleanup);

describe("ShapeMatrix", () => {
  it("renders a wide matrix as wide with exact axis annotations", () => {
    render(<ShapeMatrix tensor={wideTensor} contractedAxes={new Set(["Di"])} />);

    const matrix = screen.getByTestId("matrix-k");
    expect(Number(matrix.style.width.replace("px", ""))).toBeGreaterThan(
      Number(matrix.style.height.replace("px", ""))
    );
    expect(screen.getByText("C=32,768")).toBeInTheDocument();
    expect(screen.getByText("Di=128")).toHaveClass("contracted-axis");
  });

  it("renders a context-major tensor as tall", () => {
    render(<ShapeMatrix tensor={tallTensor} contractedAxes={new Set()} />);

    const matrix = screen.getByTestId("matrix-kv");
    expect(Number(matrix.style.height.replace("px", ""))).toBeGreaterThan(
      Number(matrix.style.width.replace("px", ""))
    );
  });

  it("renders vectors as strips", () => {
    render(<ShapeMatrix tensor={vectorTensor} contractedAxes={new Set()} />);

    const matrix = screen.getByTestId("matrix-indices");
    expect(matrix).toHaveAttribute("data-rank", "1");
    expect(matrix).toHaveStyle({ height: "36px" });
  });

  it("uses a leading aggregate axis as a scope badge", () => {
    const aggregateTensor: ShapeTensor = {
      id: "scores",
      label: "scores",
      logical_only: true,
      axes: [
        { symbol: "R", value: 4096, meaning: "workflow rows", role: "preserved" },
        { symbol: "H", value: 128, meaning: "query heads", role: "preserved" },
        { symbol: "S", value: 2048, meaning: "selected tokens", role: "produced" }
      ]
    };

    render(<ShapeMatrix tensor={aggregateTensor} contractedAxes={new Set()} />);

    expect(screen.getByText("R=4,096")).toHaveClass("shape-matrix-scope");
    expect(screen.getByText("logical view")).toBeInTheDocument();
  });
});

describe("ShapeMatrixEquation", () => {
  it("orders stage tensors and annotates a matmul equation", () => {
    const queryTensor: ShapeTensor = {
      id: "q",
      label: "Q",
      logical_only: false,
      axes: [
        { symbol: "H", value: 128, meaning: "heads", role: "preserved" },
        { symbol: "Dqk", value: 576, meaning: "feature", role: "contracted" }
      ]
    };
    const outputTensor: ShapeTensor = {
      id: "scores",
      label: "scores",
      logical_only: true,
      axes: [
        { symbol: "H", value: 128, meaning: "heads", role: "produced" },
        { symbol: "S", value: 2048, meaning: "selected tokens", role: "produced" }
      ]
    };
    const stage: ShapeStage = {
      id: "qk",
      component: "attention",
      title: "Sparse attention score matrix",
      operation: "matmul",
      formula: "[H,Dqk] x [Dqk,S] -> [H,S]",
      scope: "one query row",
      tensors: [outputTensor, wideTensor, queryTensor],
      input_ids: ["q", "k"],
      output_ids: ["scores"],
      contracted_axes: ["Dqk"],
      insight: "The shared feature axis contracts."
    };

    const { container } = render(<ShapeMatrixEquation stage={stage} />);
    const matrices = Array.from(
      container.querySelectorAll<HTMLElement>("[data-testid^='matrix-']")
    );

    expect(matrices.map((matrix) => matrix.dataset.testid)).toEqual([
      "matrix-q",
      "matrix-k",
      "matrix-scores"
    ]);
    expect(screen.getByText("x")).toHaveClass("shape-equation-operator");
    expect(screen.getByText("->")).toHaveClass("shape-equation-operator");
    expect(screen.getByText(stage.formula)).toBeInTheDocument();
    expect(screen.getByText(stage.insight)).toBeInTheDocument();
    expect(screen.getByTestId("matrix-scores").closest("figure")).toHaveClass(
      "shape-matrix-output"
    );
  });
});
