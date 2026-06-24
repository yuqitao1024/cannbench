import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeAll, describe, expect, it, vi } from "vitest";
import { App } from "./App";

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    measureText: (text: string) => ({ width: text.length * 8 })
  } as CanvasRenderingContext2D);
});

afterEach(() => {
  cleanup();
});

describe("App", () => {
  it("switches operators and datasets without leaving the page", async () => {
    const user = userEvent.setup();
    render(<App />);

    expect(screen.getByRole("heading", { name: /Operator Performance Console/i })).toBeInTheDocument();
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

    const titleTrigger = screen.getByRole("button", { name: /CannBench operator trace/i });
    await user.click(titleTrigger);
    await user.click(titleTrigger);
    expect(screen.queryByRole("dialog", { name: /GPU benchmark import/i })).not.toBeInTheDocument();

    await user.click(titleTrigger);
    expect(screen.getByRole("dialog", { name: /GPU benchmark import/i })).toBeInTheDocument();
    expect(screen.getByText(/Upload disabled by server policy/i)).toBeInTheDocument();
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
