import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { useState } from "react";
import { afterEach, expect, it, vi } from "vitest";
import { makeShapeTrace } from "../shape-trace/testFixtures";
import { ShapeTimeline } from "./ShapeTimeline";

const stages = makeShapeTrace().stages.map((stage, index) =>
  index === 5 ? { ...stage, id: "qk" } : stage
);

function ControlledTimeline({
  initialIndex,
  onChange = vi.fn()
}: {
  initialIndex: number;
  onChange?: (index: number) => void;
}) {
  const [activeIndex, setActiveIndex] = useState(initialIndex);
  return (
    <ShapeTimeline
      stages={stages}
      activeIndex={activeIndex}
      onActiveIndexChange={(index) => {
        onChange(index);
        setActiveIndex(index);
      }}
    />
  );
}

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.unstubAllGlobals();
});

it("plays forward and stops at Output", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("matchMedia", () => ({
    matches: false,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  }));
  const setIndex = vi.fn();
  render(<ControlledTimeline initialIndex={0} onChange={setIndex} />);

  fireEvent.click(screen.getByRole("button", { name: "Play shape trace" }));
  for (let index = 1; index < stages.length; index += 1) {
    await act(async () => {
      await vi.advanceTimersByTimeAsync(1_100);
    });
  }

  expect(setIndex).toHaveBeenLastCalledWith(stages.length - 1);
  expect(screen.getByRole("button", { name: "Play shape trace" })).toBeInTheDocument();
});

it("restarts at the first stage and plays when Play is clicked at Output", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
  const setIndex = vi.fn();
  render(<ControlledTimeline initialIndex={stages.length - 1} onChange={setIndex} />);

  fireEvent.click(screen.getByRole("button", { name: "Play shape trace" }));

  expect(setIndex).toHaveBeenCalledWith(0);
  expect(screen.getByText(`Step 1 of ${stages.length}`)).toBeInTheDocument();
  expect(screen.getByRole("button", { name: "Pause shape trace" })).toBeInTheDocument();

  await act(async () => {
    await vi.advanceTimersByTimeAsync(1_100);
  });
  expect(setIndex).toHaveBeenLastCalledWith(1);
});

it("supports previous next and direct stage selection", async () => {
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
  const user = userEvent.setup();
  render(<ControlledTimeline initialIndex={2} />);

  await user.click(screen.getByRole("button", { name: "Previous stage" }));
  expect(screen.getByText("Step 2 of 8")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "QK" }));
  expect(screen.getByText("Step 6 of 8")).toBeInTheDocument();
  await user.click(screen.getByRole("button", { name: "Next stage" }));
  expect(screen.getByText("Step 7 of 8")).toBeInTheDocument();
});

it("does not auto-play when reduced motion is requested", () => {
  vi.stubGlobal("matchMedia", () => ({
    matches: true,
    addEventListener: vi.fn(),
    removeEventListener: vi.fn()
  }));
  render(<ControlledTimeline initialIndex={0} />);

  expect(screen.getByRole("button", { name: "Play shape trace" })).toBeDisabled();
});

it("uses the selected playback speed", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
  const setIndex = vi.fn();
  render(<ControlledTimeline initialIndex={0} onChange={setIndex} />);

  fireEvent.change(screen.getByRole("combobox", { name: "Playback speed" }), {
    target: { value: "650" }
  });
  fireEvent.click(screen.getByRole("button", { name: "Play shape trace" }));
  await act(async () => {
    await vi.advanceTimersByTimeAsync(649);
  });
  expect(setIndex).not.toHaveBeenCalled();
  await act(async () => {
    await vi.advanceTimersByTimeAsync(1);
  });
  expect(setIndex).toHaveBeenLastCalledWith(1);
});

it("clears playback on unmount", async () => {
  vi.useFakeTimers();
  vi.stubGlobal("matchMedia", () => ({ matches: false }));
  const setIndex = vi.fn();
  const { unmount } = render(<ControlledTimeline initialIndex={0} onChange={setIndex} />);

  fireEvent.click(screen.getByRole("button", { name: "Play shape trace" }));
  unmount();
  await vi.advanceTimersByTimeAsync(1_100);
  expect(setIndex).not.toHaveBeenCalled();
});
