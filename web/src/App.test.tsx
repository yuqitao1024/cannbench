import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeAll, beforeEach, describe, expect, it, vi } from "vitest";
import { App } from "./App";
import { DEFAULT_PUBLISHED_RUN } from "./data/benchmarkRecordsApi";

const benchmarkPayload = {
  records: [
    {
      schema_version: 1,
      run_id: "softmax-realistic-h800-20260628",
      operator: "softmax",
      dataset: "realistic",
      case_id: "gptj_attention",
      shape: [1, 16, 128, 128],
      dtype: "float16",
      backend: "nvidia",
      device_class: "H800",
      implementation: "cuda_event",
      implementation_version: "cuda-event",
      metrics: { latency_ms_avg: 0.011, latency_ms_p50: 0.011, latency_ms_p95: 0.012, sample_count: 1 },
      accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
      diff_ref: null
    },
    {
      schema_version: 1,
      run_id: "embedding-realistic-h800-20260628",
      operator: "embedding",
      dataset: "realistic",
      case_id: "bert_token_embedding",
      shape: [16, 128, 768],
      dtype: "float16",
      backend: "nvidia",
      device_class: "H800",
      implementation: "cuda_event",
      implementation_version: "cuda-event",
      metrics: { latency_ms_avg: 0.021, latency_ms_p50: 0.021, latency_ms_p95: 0.023, sample_count: 1 },
      accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
      diff_ref: null
    }
  ]
};

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    measureText: (text: string) => ({ width: text.length * 8 })
  } as CanvasRenderingContext2D);
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      if (url === `/published/${DEFAULT_PUBLISHED_RUN}/meta/benchmark-records.json`) {
        return {
          ok: true,
          json: async () => benchmarkPayload
        };
      }
      if (url.includes("/api/simt-versions")) {
        return {
          ok: true,
          json: async () => ({
            operator: "softmax",
            versions: ["v1"]
          })
        };
      }
      throw new Error(`unexpected fetch: ${url}`);
    })
  );
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
});

beforeEach(() => {
  vi.clearAllMocks();
  vi.useFakeTimers({ toFake: ["Date"] });
  vi.setSystemTime(new Date(2024, 0, 1, 12, 0, 0));
});

describe("App", () => {
  it("switches operators and datasets without leaving the page", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByRole("heading", { name: /^CANNBench$/i })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /softmax/i })).toHaveAttribute("aria-pressed", "true");
    });
    expect(screen.getByRole("button", { name: /gptj_attention/i })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^embedding\s+1 cases/i }));

    expect(screen.getByRole("button", { name: /^embedding\s+1 cases/i })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /bert_token_embedding/i })).toBeInTheDocument();
  });

  it("opens the GPU JSON import dialog after three title clicks in light theme", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /CUDA operator treasure route/i })).not.toBeInTheDocument();

    const titleTrigger = screen.getByRole("button", { name: /^CANNBench$/i });
    await user.click(titleTrigger);
    await user.click(titleTrigger);
    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /CUDA operator treasure route/i })).not.toBeInTheDocument();

    await user.click(titleTrigger);
    expect(screen.getByRole("dialog", { name: /GPU benchmark import/i })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /CUDA operator treasure route/i })).not.toBeInTheDocument();
    expect(screen.getByText(/Import GPU benchmark/i)).toBeInTheDocument();
    expect(screen.queryByText(/^Local validation required$/i)).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      /GPU benchmark data only\. Never upload code, documents, environment details, employee IDs, or any sensitive content\. Violations are your responsibility\./i
    );
    expect(screen.getByRole("button", { name: /^Submit$/i })).toBeDisabled();
  });

  it("opens the CUDA treasure map dialog after three title clicks in dark theme", async () => {
    vi.setSystemTime(new Date(2024, 0, 1, 22, 0, 0));
    const user = userEvent.setup();
    render(<App />);

    expect(screen.queryByRole("dialog", { name: /CUDA operator treasure route/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();

    const titleTrigger = screen.getByRole("button", { name: /^CANNBench$/i });
    await user.click(titleTrigger);
    await user.click(titleTrigger);
    expect(screen.queryByRole("dialog", { name: /CUDA operator treasure route/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();

    await user.click(titleTrigger);
    expect(screen.getByRole("dialog", { name: /CUDA operator treasure route/i })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();
  });

  it("syncs the selected theme to document body for portaled dialogs", async () => {
    const user = userEvent.setup();
    render(<App />);

    const appShell = screen.getByRole("main");
    expect(document.body.dataset.theme).toBe(appShell.dataset.theme);

    await user.click(screen.getByRole("button", { name: /Toggle light and dark theme/i }));

    expect(document.body.dataset.theme).toBe(appShell.dataset.theme);
  });

  it("does not render the diff card when the selected operator has only one simt version", async () => {
    const fetchSpy = vi.mocked(fetch);
    render(<App />);

    await waitFor(() => {
      expect(screen.queryByLabelText(/simt operator diff/i)).not.toBeInTheDocument();
    });
    expect(fetchSpy).toHaveBeenCalledWith(
      `/published/${DEFAULT_PUBLISHED_RUN}/meta/benchmark-records.json`,
      expect.any(Object)
    );
    expect(
      fetchSpy.mock.calls.some(([url]) => String(url).includes("/api/simt-versions"))
    ).toBe(false);
  });

  it("resets the hidden click streak when the theme changes", async () => {
    const user = userEvent.setup();
    render(<App />);

    const titleTrigger = screen.getByRole("button", { name: /^CANNBench$/i });
    await user.click(titleTrigger);
    await user.click(titleTrigger);

    await user.click(screen.getByRole("button", { name: /Toggle light and dark theme/i }));
    await user.click(titleTrigger);

    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /CUDA operator treasure route/i })).not.toBeInTheDocument();

    await user.click(titleTrigger);
    await user.click(titleTrigger);

    expect(screen.getByRole("dialog", { name: /CUDA operator treasure route/i })).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();
  });

  it("shows a load failure state when published benchmark records cannot be fetched", async () => {
    vi.mocked(fetch).mockImplementationOnce(async () => ({
      ok: false,
      status: 404
    } as Response));

    render(<App />);

    await waitFor(() => {
      expect(screen.getByRole("status")).toHaveTextContent(/Failed to load benchmark records\./i);
    });
  });
});
