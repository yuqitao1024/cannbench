import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { CodeDiffPanel } from "./CodeDiffPanel";

afterEach(() => {
  cleanup();
});

beforeEach(() => {
  vi.restoreAllMocks();
  vi.stubGlobal(
    "fetch",
    vi.fn(async (input: string | URL | Request) => {
      const url = String(input);
      if (url.includes("/api/simt-versions")) {
        return {
          ok: true,
          json: async () => ({
            operator: "softmax",
            versions: ["v1", "v2"]
          })
        };
      }
      return {
        ok: true,
        json: async () => ({
          operator: "softmax",
          base_version: "v1",
          compare_version: "v2",
          patch: `diff --git a/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/spatial_softmax.asc b/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/spatial_softmax.asc
--- a/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/spatial_softmax.asc
+++ b/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/spatial_softmax.asc
@@ -1,2 +1,2 @@
-alpha
+beta
 gamma
diff --git a/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/occupancy_common.h b/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/occupancy_common.h
--- a/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/occupancy_common.h
+++ b/src/cannbench/datasets/data/softmax/simt/softmax/csrc/simt/occupancy_common.h
@@ -1 +1,2 @@
 old
+new
`
        })
      };
    })
  );
});

describe("CodeDiffPanel", () => {
  it("renders version selectors and opens the diff workspace", async () => {
    const user = userEvent.setup();

    render(<CodeDiffPanel operator="softmax" />);

    expect(screen.getByText(/simt operator diff/i)).toBeInTheDocument();
    expect(await screen.findByRole("button", { name: /^v1$/i })).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /^v2$/i })).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.getByRole("button", { name: /base/i })).toHaveTextContent(/v1/i);
      expect(screen.getByRole("button", { name: /compare/i })).toHaveTextContent(/v2/i);
    });
    expect(await screen.findByRole("button", { name: /details/i })).toBeEnabled();
    expect(await screen.findByText(/2 files changed/i)).toBeInTheDocument();
    expect(screen.queryByRole("dialog", { name: /simt operator diff workspace/i })).not.toBeInTheDocument();

    await user.click(screen.getByRole("button", { name: /details/i }));
    expect(screen.getByRole("dialog", { name: /simt operator diff workspace/i })).toBeInTheDocument();
    expect(screen.getByRole("tree", { name: /changed files/i })).toBeInTheDocument();
    expect(screen.getByText("csrc")).toBeInTheDocument();
    expect(screen.getByText("simt")).toBeInTheDocument();
    expect(screen.getByRole("treeitem", { name: "spatial_softmax.asc" })).toBeInTheDocument();
    expect(screen.getByRole("treeitem", { name: "occupancy_common.h" })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /spatial_softmax\.asc\s+1\+\+\s+1--/i })).toBeInTheDocument();
    expect(screen.getByRole("heading", { name: /occupancy_common\.h\s+1\+\+\s+0--/i })).toBeInTheDocument();
    await user.click(screen.getByRole("button", { name: /unified/i }));

    expect(screen.getByText(/unified diff/i)).toBeInTheDocument();
  });

  it("does not render the diff panel when only one simt version exists", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        if (url.includes("/api/simt-versions")) {
          return {
            ok: true,
            json: async () => ({
              operator: "softmax",
              versions: ["v1"]
            })
          };
        }
        throw new Error("unexpected diff request");
      })
    );

    render(<CodeDiffPanel operator="softmax" />);

    await waitFor(() => {
      expect(screen.queryByLabelText(/simt operator diff/i)).not.toBeInTheDocument();
    });
  });

  it("shows a clean service error when the api returns html instead of json", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async (input: string | URL | Request) => {
        const url = String(input);
        if (url.includes("/api/simt-versions")) {
          return {
            ok: true,
            headers: {
              get: () => "text/html"
            },
            text: async () => "<!doctype html><html><body>dev index</body></html>"
          };
        }
        throw new Error("unexpected diff request");
      })
    );

    render(<CodeDiffPanel operator="softmax" />);

    expect(await screen.findByText(/no diff available/i)).toBeInTheDocument();
    expect(await screen.findByText(/simt diff service is unavailable/i)).toBeInTheDocument();
    expect(screen.queryByText(/unexpected token/i)).not.toBeInTheDocument();
  });
});
