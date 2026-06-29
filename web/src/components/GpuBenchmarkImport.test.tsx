import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { GpuBenchmarkImport } from "./GpuBenchmarkImport";

const validUploadText = JSON.stringify({
  records: [
    {
      schema_version: 1,
      run_id: "opbench-nvidia-h800-cuda-pytorch-softmax-realistic-float16",
      operator: "softmax",
      dataset: "realistic",
      case_id: "gptj_attention",
      family: "attention",
      shape: [1, 16, 128, 128],
      dtype: "float16",
      backend: "nvidia",
      device_class: "H800",
      implementation: "cuda-pytorch",
      implementation_version: "cuda-pytorch",
      source_kind: "real_model",
      source_project: "TritonBench",
      source_model: "GPTJForCausalLM",
      source_file: "hf_train/GPTJForCausalLM_train.json",
      source_op: "aten._softmax.default",
      metrics: { latency_ms_avg: 0.1, latency_ms_p50: 0.1, latency_ms_p95: 0.1, sample_count: 1 },
      accuracy: { passed: true, max_abs_error: 0, max_rel_error: 0 },
      diff_ref: null
    }
  ]
});

describe("GpuBenchmarkImport", () => {
  afterEach(() => {
    cleanup();
    vi.unstubAllGlobals();
  });

  it("shows local validation entry and disabled submit button", () => {
    render(<GpuBenchmarkImport uploadEnabled={false} open={true} onClose={() => undefined} />);

    expect(screen.getByText(/Import GPU benchmark/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Paste GPU benchmark JSON/i)).toBeInTheDocument();
    expect(screen.queryByText(/^Local validation required$/i)).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      /GPU benchmark data only\. Never upload code, documents, environment details, employee IDs, or any sensitive content\. Violations are your responsibility\./i
    );
    expect(screen.getByRole("button", { name: /^Submit$/i })).toBeDisabled();
  });

  it("submits validated payload when upload is enabled", async () => {
    const user = userEvent.setup();
    const fetchMock = vi.fn(async () => ({
      ok: true,
      json: async () => ({ ok: true, path: "uploads/gpu-results-20260629T000000Z.json" })
    }));
    vi.stubGlobal("fetch", fetchMock);

    render(<GpuBenchmarkImport uploadEnabled={true} open={true} onClose={() => undefined} />);

    fireEvent.change(screen.getByLabelText(/Paste GPU benchmark JSON/i), {
      target: {
        value: validUploadText
      }
    });

    const submitButton = screen.getByRole("button", { name: /^Submit$/i });
    expect(submitButton).toBeEnabled();
    await user.click(submitButton);

    await waitFor(() => {
      expect(fetchMock).toHaveBeenCalledWith(
        "/api/gpu-results",
        expect.objectContaining({
          method: "POST"
        })
      );
    });
    expect(screen.getByText(/saved to uploads\/gpu-results-20260629T000000Z\.json/i)).toBeInTheDocument();
  });
});
