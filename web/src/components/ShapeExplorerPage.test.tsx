import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
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
const foreignGroupEntry: ShapeTraceIndexEntry = {
  operator: "foreign_operator",
  dataset: "realistic",
  case_id: "foreign-case",
  phase: "training",
  group: "another-group"
};

function deferred<T>() {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((nextResolve) => {
    resolve = nextResolve;
  });
  return { promise, resolve };
}

function shapeTraceFetchStub(): typeof fetch {
  return vi.fn(async (input: string | URL | Request) => {
    const url = String(input);
    if (url === "/api/shape-traces") {
      return jsonResponse({ traces: [...traceIndex, foreignGroupEntry] });
    }
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

it("excludes phases from foreign trace groups", async () => {
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=dsa_decode&dataset=realistic&case=decode-case"
  );
  render(<ShapeExplorerPage />);

  expect(await screen.findByRole("button", { name: "Prefill" })).toBeInTheDocument();
  expect(screen.queryByRole("button", { name: "Training" })).not.toBeInTheDocument();
});

it("supports roving keyboard focus across view tabs", async () => {
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=dsa_decode&dataset=realistic&case=decode-case"
  );
  render(<ShapeExplorerPage />);

  const algorithmTab = await screen.findByRole("tab", { name: "Algorithm flow" });
  const deviceTab = screen.getByRole("tab", { name: "Device execution" });
  expect(algorithmTab).toHaveAccessibleName("Algorithm flow");
  expect(deviceTab).toHaveAccessibleName("Device execution");
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

it("shows device status only in the Device execution view", async () => {
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=dsa_decode&dataset=realistic&case=decode-case"
  );
  render(<ShapeExplorerPage />);

  expect(await screen.findByRole("tab", { name: "Algorithm flow" })).toBeInTheDocument();
  expect(screen.queryByText("Device trace unavailable")).not.toBeInTheDocument();
  expect(screen.queryByText("No device trace.")).not.toBeInTheDocument();

  await userEvent.click(screen.getByRole("tab", { name: "Device execution" }));
  expect(screen.getByText("No device trace.")).toBeInTheDocument();
});

it.each([404, 500])(
  "preserves a valid trace when the index request fails with %s",
  async (indexStatus) => {
    window.history.pushState(
      {},
      "",
      "/shape-explorer?operator=dsa_decode&dataset=realistic&case=decode-case"
    );
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        if (url === "/api/shape-traces") {
          return jsonResponse({ error: "index unavailable" }, indexStatus);
        }
        if (url.includes("case=decode-case")) return jsonResponse(decodeTrace);
        return jsonResponse({ error: "missing" }, 404);
      })
    );

    render(<ShapeExplorerPage />);

    expect(
      await screen.findByRole("heading", {
        name: "Matrix flow from index selection to sparse attention"
      })
    ).toBeInTheDocument();
    expect(screen.queryByText("Shape trace not found")).not.toBeInTheDocument();
    expect(screen.queryByLabelText("Case phase")).not.toBeInTheDocument();
  }
);

it("ignores stale responses from an aborted navigation", async () => {
  const staleTrace = deferred<Response>();
  const staleIndex = deferred<Response>();
  let indexRequestCount = 0;
  const requestSignals: AbortSignal[] = [];
  vi.stubGlobal(
    "fetch",
    vi.fn((input: string | URL | Request, init?: RequestInit) => {
      const url = String(input);
      if (init?.signal) requestSignals.push(init.signal);
      if (url === "/api/shape-traces") {
        indexRequestCount += 1;
        return indexRequestCount === 1
          ? staleIndex.promise
          : Promise.resolve(jsonResponse({ traces: traceIndex }));
      }
      if (url.includes("case=decode-case")) return staleTrace.promise;
      if (url.includes("case=prefill-case")) return Promise.resolve(jsonResponse(prefillTrace));
      return Promise.resolve(jsonResponse({ error: "missing" }, 404));
    })
  );
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=dsa_decode&dataset=realistic&case=decode-case"
  );
  const { rerender } = render(<ShapeExplorerPage />);
  await waitFor(() => expect(vi.mocked(fetch)).toHaveBeenCalledTimes(2));

  window.history.replaceState(
    {},
    "",
    "/shape-explorer?operator=dsa_prefill&dataset=realistic&case=prefill-case"
  );
  rerender(<ShapeExplorerPage />);

  expect(await screen.findByText("prefill-case")).toBeInTheDocument();
  expect(requestSignals.slice(0, 2).every((signal) => signal.aborted)).toBe(true);

  staleTrace.resolve(jsonResponse(decodeTrace));
  staleIndex.resolve(jsonResponse({ traces: traceIndex }));
  await waitFor(() => expect(screen.getByText("prefill-case")).toBeInTheDocument());
  expect(screen.queryByText("decode-case")).not.toBeInTheDocument();
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

it("renders an invalid-link state for a malformed route component response", async () => {
  window.history.pushState(
    {},
    "",
    "/shape-explorer?operator=..%2Funsafe&dataset=realistic&case=case"
  );
  vi.mocked(fetch).mockResolvedValueOnce(jsonResponse({ error: "unsafe" }, 400));
  render(<ShapeExplorerPage />);

  expect(await screen.findByRole("status")).toHaveTextContent("Invalid shape trace link");
});
