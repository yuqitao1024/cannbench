import "@testing-library/jest-dom/vitest";
import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { RunFilters } from "./RunFilters";

describe("RunFilters", () => {
  it("shows the active baseline as selected and locked", async () => {
    const user = userEvent.setup();
    const onToggleSeries = vi.fn();

    render(
      <RunFilters
        metrics={[{ key: "relative_performance", name: "Relative performance" }]}
        selectedMetric="relative_performance"
        datasets={["realistic"]}
        selectedDataset="realistic"
        seriesOptions={[
          { key: "cuda", name: "NVIDIA H800 PyTorch", available: true },
          { key: "simt", name: "Ascend 950PR SIMT v1", available: true }
        ]}
        selectedSeries={["cuda", "simt"]}
        lockedSeries="cuda"
        onSelectMetric={vi.fn()}
        onSelectDataset={vi.fn()}
        onToggleSeries={onToggleSeries}
      />
    );

    const baseline = screen.getByRole("button", { name: /NVIDIA H800 PyTorch.*baseline locked/i });
    expect(baseline).toBeDisabled();
    expect(baseline).toHaveAttribute("aria-pressed", "true");
    await user.click(baseline);
    expect(onToggleSeries).not.toHaveBeenCalled();
  });
});
