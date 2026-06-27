import "@testing-library/jest-dom/vitest";
import { useState } from "react";
import { cleanup, render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { CudaTreasureMap } from "./CudaTreasureMap";
import { CudaTreasureMapModal } from "./CudaTreasureMapModal";

afterEach(() => {
  cleanup();
});

describe("CudaTreasureMapModal", () => {
  function FocusHarness() {
    const [open, setOpen] = useState(false);

    return (
      <>
        <button type="button" onClick={() => setOpen(true)}>
          Open treasure route
        </button>
        <CudaTreasureMapModal open={open} onClose={() => setOpen(false)} />
      </>
    );
  }

  it("renders the main CUDA optimization route labels", () => {
    render(<CudaTreasureMapModal open={true} onClose={() => undefined} />);

    expect(screen.getByRole("dialog", { name: /CUDA operator treasure route/i })).toBeInTheDocument();
    expect(screen.getByText(/Profile the Truth/i)).toBeInTheDocument();
    expect(screen.getByText(/Fix Global Access/i)).toBeInTheDocument();
    expect(screen.getByText(/Polish Instructions/i)).toBeInTheDocument();
  });

  it("renders the fixed map background with frontend anchor rings", () => {
    render(<CudaTreasureMapModal open={true} onClose={() => undefined} />);

    const background = document.body.querySelector(".cuda-treasure-map__background");
    const mainAnchors = Array.from(document.body.querySelectorAll(".cuda-treasure-map__anchor--main"));
    const branchAnchors = Array.from(document.body.querySelectorAll(".cuda-treasure-map__anchor--branch"));

    expect(background?.tagName.toLowerCase()).toBe("img");
    expect(background).toHaveAttribute("src");
    expect(background?.getAttribute("src")).toMatch(/cuda-treasure-map-dark/i);
    expect(mainAnchors).toHaveLength(8);
    expect(branchAnchors).toHaveLength(6);
  });

  it("reveals the Fix Global Access field note on hover and focus with tooltip wiring", async () => {
    const user = userEvent.setup();

    render(<CudaTreasureMapModal open={true} onClose={() => undefined} />);

    const fixGlobalAccessNode = screen.getByRole("button", { name: /Fix Global Access/i });

    expect(screen.queryByText(/^Guide:\s*10\.2\.1$/i)).not.toBeInTheDocument();

    await user.hover(fixGlobalAccessNode);
    const hoveredTooltip = screen.getByRole("tooltip");
    expect(fixGlobalAccessNode).toHaveAttribute("aria-describedby", hoveredTooltip.id);
    expect(within(hoveredTooltip).getByRole("heading", { name: /Fix Global Access/i })).toBeInTheDocument();
    expect(within(hoveredTooltip).getByText(/^Guide:\s*10\.2\.1$/i)).toBeInTheDocument();
    expect(
      within(hoveredTooltip).getByText(/Repair coalescing, stride, and alignment before instruction micro-tuning\./i)
    ).toBeInTheDocument();
    expect(within(hoveredTooltip).getByText(/Make adjacent threads touch adjacent memory\./i)).toBeInTheDocument();
    expect(within(hoveredTooltip).getByText(/^Route IDs:\s*O9, O10$/i)).toBeInTheDocument();

    await user.unhover(fixGlobalAccessNode);
    expect(screen.queryByText(/^Guide:\s*10\.2\.1$/i)).not.toBeInTheDocument();
    expect(fixGlobalAccessNode).not.toHaveAttribute("aria-describedby");

    fixGlobalAccessNode.focus();
    expect(fixGlobalAccessNode).toHaveFocus();
    const focusedTooltip = await screen.findByRole("tooltip");
    expect(fixGlobalAccessNode).toHaveAttribute("aria-describedby", focusedTooltip.id);
    expect(within(focusedTooltip).getByText(/^Guide:\s*10\.2\.1$/i)).toBeInTheDocument();
  });

  it("lets hover override a previously clicked node focus", async () => {
    const user = userEvent.setup();

    render(<CudaTreasureMapModal open={true} onClose={() => undefined} />);

    const fixGlobalAccessNode = screen.getByRole("button", { name: /Fix Global Access/i });
    const targetGpuBuildNode = screen.getByRole("button", { name: /Target GPU build/i });

    await user.click(fixGlobalAccessNode);
    expect(screen.getByRole("tooltip")).toHaveTextContent(/Fix Global Access/i);

    await user.hover(targetGpuBuildNode);

    expect(screen.getByRole("tooltip")).toHaveTextContent(/Target GPU build/i);
    expect(targetGpuBuildNode).toHaveAttribute("aria-describedby", screen.getByRole("tooltip").id);
  });

  it("closes on Escape and backdrop click but not on dialog clicks", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<CudaTreasureMapModal open={true} onClose={onClose} />);

    const dialog = screen.getByRole("dialog", { name: /CUDA operator treasure route/i });
    const backdrop = dialog.parentElement;

    expect(backdrop).not.toBeNull();

    await user.click(dialog);
    expect(onClose).not.toHaveBeenCalled();

    await user.click(backdrop!);
    expect(onClose).toHaveBeenCalledTimes(1);

    await user.keyboard("{Escape}");
    expect(onClose).toHaveBeenCalledTimes(2);
  });

  it("closes when the close button is clicked", async () => {
    const user = userEvent.setup();
    const onClose = vi.fn();

    render(<CudaTreasureMapModal open={true} onClose={onClose} />);

    await user.click(screen.getByRole("button", { name: /Close CUDA operator treasure route/i }));

    expect(onClose).toHaveBeenCalledTimes(1);
  });

  it("moves focus into the dialog, traps it, and restores focus on close", async () => {
    const user = userEvent.setup();

    render(<FocusHarness />);

    const openButton = screen.getByRole("button", { name: /Open treasure route/i });
    openButton.focus();

    await user.click(openButton);

    const closeButton = screen.getByRole("button", { name: /Close CUDA operator treasure route/i });
    expect(closeButton).toHaveFocus();

    await user.tab({ shift: true });
    expect(screen.getByRole("button", { name: /Target GPU build/i })).toHaveFocus();

    await user.tab();
    expect(closeButton).toHaveFocus();

    await user.keyboard("{Escape}");
    expect(screen.queryByRole("dialog", { name: /CUDA operator treasure route/i })).not.toBeInTheDocument();
    expect(openButton).toHaveFocus();
  });

  it("throws when a branch node points to a missing parent", () => {
    expect(() =>
      render(
        <CudaTreasureMap
          route={[
            {
              id: "branch-only",
              label: "Broken branch",
              kind: "branch",
              x: 20,
              y: 20,
              summary: "Invalid route shape",
              details: ["Missing branch parent should fail loudly."],
              guideSections: ["0"],
              relatedOptimizationIds: ["OX"],
              branchFrom: "missing-parent"
            }
          ]}
          mainRouteOrder={[]}
        />
      )
    ).toThrow(/missing branch parent/i);
  });

  it("throws when duplicate node ids are provided", () => {
    expect(() =>
      render(
        <CudaTreasureMap
          route={[
            {
              id: "duplicate-node",
              label: "First node",
              kind: "main",
              x: 10,
              y: 20,
              summary: "First entry",
              details: ["First detail."],
              guideSections: ["1"],
              relatedOptimizationIds: ["O1"]
            },
            {
              id: "duplicate-node",
              label: "Second node",
              kind: "main",
              x: 20,
              y: 30,
              summary: "Second entry",
              details: ["Second detail."],
              guideSections: ["2"],
              relatedOptimizationIds: ["O2"]
            }
          ]}
          mainRouteOrder={["duplicate-node"]}
        />
      )
    ).toThrow(/duplicate node id/i);
  });
});
