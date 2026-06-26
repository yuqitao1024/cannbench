import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { App } from "./App";

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    measureText: (text: string) => ({ width: text.length * 8 })
  } as CanvasRenderingContext2D);
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => ({
        operator: "softmax",
        base_version: "dynamic-ubuf",
        compare_version: "tiled-v2",
        patch: `diff --git a/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc b/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc
--- a/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc
+++ b/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc
@@ -1,2 +1,2 @@
-alpha
+beta
 gamma
`
      })
    }))
  );
});

afterEach(() => {
  cleanup();
});

describe("App", () => {
  it("switches operators and datasets without leaving the page", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByRole("heading", { name: /^CANNBench$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /softmax/i })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /gptj_attention/i })).toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /^embedding\s+1 cases/i }));

    expect(screen.getByRole("button", { name: /^embedding\s+1 cases/i })).toHaveAttribute("aria-pressed", "true");
    expect(screen.getByRole("button", { name: /bert_token_embedding/i })).toBeInTheDocument();
  });

  it("opens the GPU JSON import dialog after three title clicks", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();

    const titleTrigger = screen.getByRole("button", { name: /^CANNBench$/i });
    await user.click(titleTrigger);
    await user.click(titleTrigger);
    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();

    await user.click(titleTrigger);
    expect(screen.getByRole("dialog", { name: /GPU benchmark import/i })).toBeInTheDocument();
    expect(screen.getByText(/Import GPU benchmark/i)).toBeInTheDocument();
    expect(screen.queryByText(/^Local validation required$/i)).not.toBeInTheDocument();
    expect(screen.getByRole("alert")).toHaveTextContent(
      /GPU benchmark data only\. Never upload code, documents, environment details, employee IDs, or any sensitive content\. Violations are your responsibility\./i
    );
    expect(screen.getByRole("button", { name: /^Submit$/i })).toBeDisabled();
  });

  it("syncs the selected theme to document body for portaled dialogs", async () => {
    const user = userEvent.setup();
    render(<App />);

    const appShell = screen.getByRole("main");
    expect(document.body.dataset.theme).toBe(appShell.dataset.theme);

    await user.click(screen.getByRole("button", { name: /Toggle light and dark theme/i }));

    expect(document.body.dataset.theme).toBe(appShell.dataset.theme);
  });
});
