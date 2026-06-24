import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GpuBenchmarkImport } from "./GpuBenchmarkImport";

describe("GpuBenchmarkImport", () => {
  it("shows upload disabled policy and local validation entry", () => {
    render(<GpuBenchmarkImport uploadEnabled={false} />);

    expect(screen.getByText(/Upload disabled by server policy/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/Select GPU benchmark JSON/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /Upload/i })).toBeDisabled();
  });
});
