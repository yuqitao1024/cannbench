import type { ShapeTensor } from "./types";

const MIN_AXIS_PX = 36;
const MAX_AXIS_PX = 320;
const SCALE = 14;
const EXPONENT = 0.3;

export function dimensionPixels(value: number): number {
  if (!Number.isInteger(value) || value <= 0) {
    throw new Error("dimension must be a positive integer");
  }

  return Math.round(
    Math.min(MAX_AXIS_PX, Math.max(MIN_AXIS_PX, SCALE * value ** EXPONENT))
  );
}

export function axisRatioLabel(tensor: ShapeTensor): string {
  if (tensor.axes.length < 2) {
    const axis = tensor.axes[0];
    return `${axis.symbol} = ${axis.value.toLocaleString("en-US")}`;
  }

  const sorted = [...tensor.axes].sort((a, b) => b.value - a.value);
  const largest = sorted[0];
  const smallest = sorted.at(-1)!;
  const ratio = largest.value / smallest.value;
  const rendered = Number.isInteger(ratio) ? String(ratio) : ratio.toFixed(1);

  return `${largest.symbol} / ${smallest.symbol} = ${rendered}x`;
}
