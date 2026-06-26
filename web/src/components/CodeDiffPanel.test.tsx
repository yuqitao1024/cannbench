import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import type { BenchmarkRecord } from "../types";
import { CodeDiffPanel } from "./CodeDiffPanel";

afterEach(() => {
  cleanup();
});

beforeEach(() => {
  vi.restoreAllMocks();
  vi.stubGlobal(
    "fetch",
    vi.fn(async () => ({
      ok: true,
      json: async () => ({
        operator: "softmax",
        base_version: "dynamic-ubuf",
        compare_version: "tiled-v2",
        patch: `diff --git a/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc b/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc
--- a/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc
+++ b/src/cannbench/datasets/data/softmax/custom_ops/ascend/aten_softmax/csrc/simt/spatial_softmax.asc
@@ -1,2 +1,2 @@
-alpha
+beta
 gamma
`
      })
    }))
  );
});

const dynamicUbufRecord: BenchmarkRecord = {
  schema_version: 1,
  run_id: "softmax-realistic-ascend-simt-20260624",
  operator: "softmax",
  dataset: "realistic",
  case_id: "gptj_attention",
  shape: [32, 16, 2048],
  dtype: "float16",
  backend: "ascend",
  device_class: "910b",
  implementation: "simt",
  implementation_version: "dynamic-ubuf",
  metrics: {
    latency_ms_avg: 0.12,
    latency_ms_p50: 0.11,
    latency_ms_p95: 0.14,
    sample_count: 1
  },
  accuracy: {
    passed: true,
    max_abs_error: 0,
    max_rel_error: 0
  },
  diff_ref: "softmax/simt/dynamic-ubuf"
};

const tiledV2Record: BenchmarkRecord = {
  ...dynamicUbufRecord,
  implementation_version: "tiled-v2",
  diff_ref: "softmax/simt/tiled-v2"
};

describe("CodeDiffPanel", () => {
  it("renders version selectors and opens the diff workspace", async () => {
    const user = userEvent.setup();

    render(<CodeDiffPanel operator="softmax" simtRecords={[dynamicUbufRecord, tiledV2Record]} />);

    expect(screen.getByText(/simt operator diff/i)).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: /dynamic-ubuf/i })).toHaveLength(2);
    expect(screen.getAllByRole("button", { name: /tiled-v2/i })).toHaveLength(2);
    expect(await screen.findByRole("button", { name: /details/i })).toBeEnabled();
    expect(await screen.findByText(/1 files changed/i)).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /simt operator diff workspace/i })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /details/i }));
    expect(screen.getByRole("dialog", { name: /simt operator diff workspace/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /unified/i }));

    expect(screen.getByText(/unified diff/i)).toBeInTheDocument();
  });

  it("shows the empty state when only one simt version exists", () => {
    render(<CodeDiffPanel operator="softmax" simtRecords={[dynamicUbufRecord]} />);

    expect(screen.getByText(/no diff available/i)).toBeInTheDocument();
    expect(screen.getByText(/need at least two simt operator versions to compare/i)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /details/i })).toBeDisabled();
  });
});
