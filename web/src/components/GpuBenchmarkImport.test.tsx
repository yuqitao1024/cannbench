import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { GpuBenchmarkImport } from "./GpuBenchmarkImport";

describe("GpuBenchmarkImport", () => {
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
});
