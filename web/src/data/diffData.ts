export interface RepositoryDiff {
  ref: string;
  title: string;
  baselineLabel: string;
  simtLabel: string;
  patch: string;
}

export const repositoryDiffs: Record<string, RepositoryDiff> = {
  "softmax/simt/dynamic-ubuf": {
    ref: "softmax/simt/dynamic-ubuf",
    title: "aten_softmax dynamic UB reduction",
    baselineLabel: "CANN ops library",
    simtLabel: "SIMT op dynamic-ubuf",
    patch: `diff --git a/src/cannbench/datasets/data/softmax/simt_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc b/src/cannbench/datasets/data/softmax/simt_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc
index 91b0f00..f6c1a74 100644
--- a/src/cannbench/datasets/data/softmax/simt_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc
+++ b/src/cannbench/datasets/data/softmax/simt_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc
@@ -73,8 +73,9 @@ for (int64_t row = block_idx; row < rows; row += block_num) {
-  float max_value = input[row * cols];
-  for (int64_t col = 1; col < cols; ++col) {
+  float max_value = ReduceMax(input + row * cols, cols, ubuf);
+  for (int64_t col = 0; col < cols; ++col) {
     float shifted = input[row * cols + col] - max_value;
+    ubuf[col] = exp(shifted);
   }
`
  },
  "softmax/simt/tiled-v2": {
    ref: "softmax/simt/tiled-v2",
    title: "aten_softmax tiled v2 sketch",
    baselineLabel: "SIMT op dynamic-ubuf",
    simtLabel: "SIMT op tiled-v2",
    patch: `diff --git a/src/cannbench/datasets/data/softmax/simt_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc b/src/cannbench/datasets/data/softmax/simt_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc
index f6c1a74..a18b742 100644
--- a/src/cannbench/datasets/data/softmax/simt_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc
+++ b/src/cannbench/datasets/data/softmax/simt_ops/ascend/default/aten_softmax/csrc/simt/spatial_softmax.asc
@@ -91,3 +91,3 @@ for (int64_t col = 0; col < cols; col += tile_cols) {
-  int64_t tile_cols = cols;
+  int64_t tile_cols = Min(cols - col, kSoftmaxTile);
   ComputeTile(input, output, row, col, tile_cols);
`
  }
};

export function getRepositoryDiff(diffRef: string | null): RepositoryDiff | null {
  if (!diffRef) {
    return null;
  }
  return repositoryDiffs[diffRef] ?? null;
}
