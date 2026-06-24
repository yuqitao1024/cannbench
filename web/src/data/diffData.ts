export interface DiffLine {
  type: "context" | "add" | "delete";
  oldNumber: number | null;
  newNumber: number | null;
  text: string;
}

export interface DiffFile {
  path: string;
  hunks: DiffLine[];
}

export interface RepositoryDiff {
  ref: string;
  title: string;
  baselineLabel: string;
  customLabel: string;
  files: DiffFile[];
}

export const repositoryDiffs: Record<string, RepositoryDiff> = {
  "softmax/custom/dynamic-ubuf": {
    ref: "softmax/custom/dynamic-ubuf",
    title: "aten_softmax dynamic UB reduction",
    baselineLabel: "Ascend operator library",
    customLabel: "custom dynamic-ubuf",
    files: [
      {
        path: "src/cannbench/datasets/data/softmax/custom_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc",
        hunks: [
          { type: "context", oldNumber: 73, newNumber: 73, text: "for (int64_t row = block_idx; row < rows; row += block_num) {" },
          { type: "delete", oldNumber: 74, newNumber: null, text: "  float max_value = input[row * cols];" },
          { type: "delete", oldNumber: 75, newNumber: null, text: "  for (int64_t col = 1; col < cols; ++col) {" },
          { type: "add", oldNumber: null, newNumber: 74, text: "  float max_value = ReduceMax(input + row * cols, cols, ubuf);" },
          { type: "add", oldNumber: null, newNumber: 75, text: "  for (int64_t col = 0; col < cols; ++col) {" },
          { type: "context", oldNumber: 76, newNumber: 76, text: "    float shifted = input[row * cols + col] - max_value;" },
          { type: "add", oldNumber: null, newNumber: 77, text: "    ubuf[col] = exp(shifted);" },
          { type: "context", oldNumber: 77, newNumber: 78, text: "  }" }
        ]
      }
    ]
  },
  "softmax/custom/tiled-v2": {
    ref: "softmax/custom/tiled-v2",
    title: "aten_softmax tiled v2 sketch",
    baselineLabel: "custom dynamic-ubuf",
    customLabel: "custom tiled-v2",
    files: [
      {
        path: "src/cannbench/datasets/data/softmax/custom_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc",
        hunks: [
          { type: "context", oldNumber: 91, newNumber: 91, text: "for (int64_t col = 0; col < cols; col += tile_cols) {" },
          { type: "delete", oldNumber: 92, newNumber: null, text: "  int64_t tile_cols = cols;" },
          { type: "add", oldNumber: null, newNumber: 92, text: "  int64_t tile_cols = Min(cols - col, kSoftmaxTile);" },
          { type: "context", oldNumber: 93, newNumber: 93, text: "  ComputeTile(input, output, row, col, tile_cols);" }
        ]
      }
    ]
  }
};

export function getRepositoryDiff(diffRef: string | null): RepositoryDiff | null {
  if (!diffRef) {
    return null;
  }
  return repositoryDiffs[diffRef] ?? null;
}
