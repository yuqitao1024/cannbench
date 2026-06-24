import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { beforeAll, describe, expect, it, vi } from "vitest";
import { App } from "./App";

beforeAll(() => {
  vi.spyOn(HTMLCanvasElement.prototype, "getContext").mockReturnValue({
    measureText: (text: string) => ({ width: text.length * 8 })
  } as CanvasRenderingContext2D);
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
});
