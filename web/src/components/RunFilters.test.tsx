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
        metric={{ key: "relative_performance", name: "Relative performance" }}
        datasets={["realistic"]}
        selectedDataset="realistic"
        seriesOptions={[
          { key: "cuda", name: "NVIDIA H800 PyTorch", available: true },
          { key: "simt", name: "Ascend 950PR SIMT v1", available: true }
        ]}
        selectedSeries={["cuda", "simt"]}
        lockedSeries="cuda"
        onSelectDataset={vi.fn()}
        onToggleSeries={onToggleSeries}
      />
    );

    const baseline = screen.getByRole("button", { name: /NVIDIA H800 PyTorch.*baseline locked/i });
    expect(baseline).toBeDisabled();
    expect(baseline).toHaveAttribute("aria-pressed", "true");

    const metric = screen.getByRole("status", { name: "Relative performance, selected" });
    expect(metric).toBeVisible();
    expect(metric).toHaveClass("is-selected");
    expect(screen.queryByRole("button", { name: /^Relative performance$/i })).not.toBeInTheDocument();
    await user.click(metric);
    expect(screen.getByRole("status", { name: "Relative performance, selected" })).toBeInTheDocument();

    await user.click(baseline);
    expect(onToggleSeries).not.toHaveBeenCalled();
  });
});
