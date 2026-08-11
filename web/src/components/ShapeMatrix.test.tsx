import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, within } from "@testing-library/react";
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
    expect(screen.getByText("C=32,768")).toHaveAccessibleName("C=32,768: context");
    expect(screen.getByText("Di=128")).toHaveClass("contracted-axis");
    expect(screen.getByText("C / Di = 256x")).toHaveClass("shape-matrix-ratio");
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

  it("computes a rank-three ratio from the two geometry axes", () => {
    const aggregateTensor: ShapeTensor = {
      id: "index-query-all",
      label: "All index queries",
      logical_only: false,
      axes: [
        { symbol: "R", value: 4096, meaning: "flattened query rows", role: "preserved" },
        { symbol: "Hi", value: 64, meaning: "index heads", role: "preserved" },
        { symbol: "Di", value: 128, meaning: "index feature", role: "preserved" }
      ]
    };

    render(<ShapeMatrix tensor={aggregateTensor} contractedAxes={new Set()} />);

    expect(screen.getByText("R=4,096")).toHaveClass("shape-matrix-scope");
    expect(screen.getByText("Di / Hi = 2x")).toHaveClass("shape-matrix-ratio");
    expect(screen.queryByText("R / Hi = 64x")).not.toBeInTheDocument();
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
    expect(screen.getByLabelText(`${stage.title} tensor equation`)).toHaveAttribute(
      "tabindex",
      "0"
    );
  });

  it("renders unreferenced tensors in declared order after the main equation", () => {
    const aggregateFirst = { ...tallTensor, id: "aggregate-first", label: "All queries" };
    const aggregateSecond = { ...vectorTensor, id: "aggregate-second", label: "All indices" };
    const stage: ShapeStage = {
      id: "aggregate-stage",
      component: "generic-component",
      title: "Aggregate stage",
      operation: "transform",
      formula: "K -> indices",
      scope: "one row and all rows",
      tensors: [aggregateFirst, vectorTensor, wideTensor, aggregateSecond],
      input_ids: ["k"],
      output_ids: ["indices"],
      contracted_axes: [],
      insight: "Shows row and aggregate shapes."
    };

    const { container } = render(<ShapeMatrixEquation stage={stage} />);
    const matrices = Array.from(
      container.querySelectorAll<HTMLElement>("[data-testid^='matrix-']")
    );
    expect(matrices.map((matrix) => matrix.dataset.testid)).toEqual([
      "matrix-k",
      "matrix-indices",
      "matrix-aggregate-first",
      "matrix-aggregate-second"
    ]);

    const supplemental = screen.getByRole("group", {
      name: "Aggregate and supplemental tensors"
    });
    expect(within(supplemental).getByText("All queries")).toBeInTheDocument();
    expect(within(supplemental).getByText("All indices")).toBeInTheDocument();
  });

  it("renders canonical prefill QK aggregate tensors as supplemental matrices", () => {
    const axis = (
      symbol: string,
      value: number,
      meaning: string,
      role: "preserved" | "contracted" | "produced"
    ) => ({ symbol, value, meaning, role });
    const h = axis("H", 128, "query attention heads", "preserved");
    const dqk = axis("Dqk", 576, "QK feature dimension", "contracted");
    const s = axis("S", 2048, "selected context tokens", "produced");
    const r = axis("R", 4096, "flattened query rows", "produced");
    const stage: ShapeStage = {
      id: "qk",
      component: "sparse_attention",
      title: "Sparse attention score matrix",
      operation: "matmul",
      formula: "[H,Dqk] x [Dqk,S] -> [H,S]",
      scope: "one flattened query row; aggregate shapes cover all R rows",
      tensors: [
        { id: "query", label: "Query", axes: [h, dqk], logical_only: false },
        {
          id: "selected_k_t",
          label: "Selected K transposed",
          axes: [dqk, s],
          logical_only: true
        },
        { id: "scores", label: "Attention scores", axes: [h, s], logical_only: true },
        {
          id: "query_all",
          label: "All attention queries",
          axes: [r, h, dqk],
          logical_only: false
        },
        {
          id: "scores_all",
          label: "All attention scores / probabilities",
          axes: [r, h, s],
          logical_only: true
        }
      ],
      input_ids: ["query", "selected_k_t"],
      output_ids: ["scores"],
      contracted_axes: ["Dqk"],
      insight: "[H,Dqk] x [Dqk,S] -> [H,S]"
    };

    const { container } = render(<ShapeMatrixEquation stage={stage} />);
    const matrices = Array.from(
      container.querySelectorAll<HTMLElement>("[data-testid^='matrix-']")
    );
    expect(matrices.map((matrix) => matrix.dataset.testid)).toEqual([
      "matrix-query",
      "matrix-selected_k_t",
      "matrix-scores",
      "matrix-query_all",
      "matrix-scores_all"
    ]);

    const supplemental = screen.getByRole("group", {
      name: "Aggregate and supplemental tensors"
    });
    expect(within(supplemental).getByText("All attention queries")).toBeInTheDocument();
    expect(
      within(supplemental).getByText("All attention scores / probabilities")
    ).toBeInTheDocument();
    expect(within(supplemental).getAllByText("R=4,096")).toHaveLength(2);
  });

  it("groups non-matmul inputs and shows one transition before fan-out outputs", () => {
    const inputB = { ...vectorTensor, id: "mask", label: "Mask" };
    const outputB = { ...vectorTensor, id: "counts", label: "Counts" };
    const stage: ShapeStage = {
      id: "fan-out",
      component: "generic-component",
      title: "Generic fan-out",
      operation: "gather",
      formula: "values, mask -> indices, counts",
      scope: "one row",
      tensors: [wideTensor, inputB, vectorTensor, outputB],
      input_ids: ["k", "mask"],
      output_ids: ["indices", "counts"],
      contracted_axes: [],
      insight: "One operation produces two results."
    };

    render(<ShapeMatrixEquation stage={stage} />);

    expect(
      screen.getAllByTestId("shape-equation-operator").map((operator) => operator.textContent)
    ).toEqual(["and", "->", "and"]);
  });
});
