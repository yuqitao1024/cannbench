import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import type { CaseSummary } from "../types";
import { CaseTable } from "./CaseTable";

const caseSummary: CaseSummary = {
  caseId: "case-1",
  dataset: "realistic",
  family: "decode_workflow",
  shape: [2, 128, 576],
  dtype: "bfloat16",
  records: [],
  sourceLabel: "DeepSeek-V3.2",
  coverageTag: "real-model coverage",
  availableSeries: ["Ascend 950PR SIMT v2"]
};

afterEach(cleanup);

describe("CaseTable", () => {
  it("opens an available shape trace from the shape cell", () => {
    render(
      <CaseTable
        operator="dsa_decode"
        cases={[caseSummary]}
        showDatasetColumn={false}
        shapeTraceKeys={new Set(["dsa_decode\u0000realistic\u0000case-1"])}
      />
    );

    const link = screen.getByRole("link", { name: "2 x 128 x 576" });
    expect(link).toHaveAttribute(
      "href",
      "/shape-explorer?operator=dsa_decode&dataset=realistic&case=case-1"
    );
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noreferrer");
  });

  it("keeps unsupported shapes as plain text", () => {
    render(
      <CaseTable
        operator="softmax"
        cases={[caseSummary]}
        showDatasetColumn={false}
        shapeTraceKeys={new Set()}
      />
    );

    expect(screen.queryByRole("link", { name: "2 x 128 x 576" })).not.toBeInTheDocument();
    expect(screen.getByText("2 x 128 x 576")).toBeInTheDocument();
  });
});
