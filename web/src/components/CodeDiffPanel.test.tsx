import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { CodeDiffPanel } from "./CodeDiffPanel";

describe("CodeDiffPanel", () => {
  it("renders repository-owned diff content and switches view mode", async () => {
    const user = userEvent.setup();

    render(<CodeDiffPanel diffRef="softmax/custom/dynamic-ubuf" />);

    expect(screen.getByRole("heading", { name: /aten_softmax dynamic UB reduction/i })).toBeInTheDocument();
    expect(screen.getByText(/repository diff/i)).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /repository diff workspace/i })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /open diff/i }));
    expect(screen.getByRole("dialog", { name: /repository diff workspace/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /unified/i }));

    expect(screen.getByText(/unified diff/i)).toBeInTheDocument();
  });
});
