import { describe, expect, it } from "vitest";
import type { ShapeTensor } from "./types";
import { axisRatioLabel, dimensionPixels } from "./shapeScale";

describe("dimensionPixels", () => {
  it("scales dimensions monotonically with bounded nonlinear growth", () => {
    const values = [1, 64, 128, 512, 2048, 32768].map(dimensionPixels);

    expect(values).toEqual([...values].sort((a, b) => a - b));
    expect(values[0]).toBe(36);
    expect(values.at(-1)).toBeLessThanOrEqual(320);
    expect(values.at(-1)).toBeGreaterThan(values[2]);
  });

  it("keeps the same dimension at the same pixel length", () => {
    expect(dimensionPixels(576)).toBe(dimensionPixels(576));
  });

  it.each([0, -1, 1.5, Number.NaN])("rejects invalid dimension %s", (value) => {
    expect(() => dimensionPixels(value)).toThrow("dimension must be a positive integer");
  });
});

describe("axisRatioLabel", () => {
  it("reports the exact dominant axis ratio", () => {
    const tensor: ShapeTensor = {
      id: "k",
      label: "K",
      logical_only: false,
      axes: [
        { symbol: "Di", value: 128, meaning: "feature", role: "contracted" },
        { symbol: "C", value: 32768, meaning: "context", role: "produced" }
      ]
    };

    expect(axisRatioLabel(tensor)).toBe("C / Di = 256x");
  });

  it("reports a vector dimension with English number formatting", () => {
    const tensor: ShapeTensor = {
      id: "indices",
      label: "indices",
      logical_only: false,
      axes: [{ symbol: "S", value: 2048, meaning: "selected tokens", role: "produced" }]
    };

    expect(axisRatioLabel(tensor)).toBe("S = 2,048");
  });
});
