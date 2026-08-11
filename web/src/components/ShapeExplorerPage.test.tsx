import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, expect, it, vi } from "vitest";
import { jsonResponse, makeShapeTrace } from "../shape-trace/testFixtures";
import type { ShapeTraceIndexEntry } from "../shape-trace/types";
import { ShapeExplorerPage } from "./ShapeExplorerPage";

function makeExplorerTrace(phase: "decode" | "prefill") {
  const trace = makeShapeTrace(phase);
  return {
    ...trace,
    stages: trace.stages.map((stage, index) =>
      index === 5 ? { ...stage, id: "qk" } : stage
    )
  };
}

const decodeTrace = makeExplorerTrace("decode");
const prefillTrace = makeExplorerTrace("prefill");
const traceIndex: ShapeTraceIndexEntry[] = [decodeTrace, prefillTrace].map(
  ({ operator, dataset, case_id, phase, group }) => ({
    operator,
    dataset,
    case_id,
    phase,
    group
  })
);

function shapeTraceFetchStub(): typeof fetch {
  return vi.fn(async (input: string | URL | Request) => {
    const url = String(input);
    if (url === "/api/shape-traces") return jsonResponse({ traces: traceIndex });
    if (url.includes("case=decode-case")) return jsonResponse(decodeTrace);
    if (url.includes("case=prefill-case")) return jsonResponse(prefillTrace);
    return jsonResponse({ error: "missing" }, 404);
  }) as typeof fetch;
}

beforeEach(() => {
  window.history.replaceState({}, "", "/shape-explorer");
  vi.stubGlobal("fetch", shapeTraceFetchStub());
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
});

it("loads a deep-linked trace and renders the first stage", async () => {
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=dsa_decode&dataset=realistic&case=decode-case"
  );
  render(<ShapeExplorerPage />);

  expect(
    await screen.findByRole("heading", {
      name: "Matrix flow from index selection to sparse attention"
    })
  ).toBeInTheDocument();
  expect(screen.getByText("Indexer projection")).toBeInTheDocument();
});

it("switches phase within the same generic group and resets the stage", async () => {
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=dsa_decode&dataset=realistic&case=decode-case"
  );
  render(<ShapeExplorerPage />);

  await userEvent.click(await screen.findByRole("button", { name: "QK" }));
  expect(screen.getByText("Step 6 of 8")).toBeInTheDocument();
  await userEvent.click(screen.getByRole("button", { name: "Prefill" }));

  expect(await screen.findByText("Indexer projection")).toBeInTheDocument();
  expect(screen.getByText("Step 1 of 8")).toBeInTheDocument();
  expect(window.location.search).toContain("case=prefill-case");
});

it("supports roving keyboard focus across view tabs", async () => {
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=dsa_decode&dataset=realistic&case=decode-case"
  );
  render(<ShapeExplorerPage />);

  const algorithmTab = await screen.findByRole("tab", { name: "Algorithm" });
  const deviceTab = screen.getByRole("tab", { name: "Device" });
  algorithmTab.focus();
  fireEvent.keyDown(screen.getByRole("tablist", { name: "Explorer view" }), {
    key: "ArrowRight"
  });
  expect(deviceTab).toHaveFocus();
  expect(deviceTab).toHaveAttribute("aria-selected", "true");
  expect(algorithmTab).toHaveAttribute("tabindex", "-1");

  fireEvent.keyDown(screen.getByRole("tablist", { name: "Explorer view" }), {
    key: "Home"
  });
  expect(algorithmTab).toHaveFocus();
  expect(algorithmTab).toHaveAttribute("aria-selected", "true");
});

it("renders an invalid-link state", () => {
  render(<ShapeExplorerPage />);

  expect(screen.getByRole("status")).toHaveTextContent("Invalid shape trace link");
});

it("renders a not-found state", async () => {
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=x&dataset=y&case=z"
  );
  vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ error: "missing" }, 404));
  render(<ShapeExplorerPage />);

  expect(await screen.findByRole("status")).toHaveTextContent("Shape trace not found");
});
