import "@testing-library/jest-dom/vitest";
import { cleanup, render, screen } from "@testing-library/react";
import { afterEach, describe, expect, it } from "vitest";
import { shapeTraceKey } from "../data/shapeTraceApi";
import type { CaseSummary } from "../types";
import { CaseTable } from "./CaseTable";

const caseSummary: CaseSummary = {
  caseId: "case/1 space",
  dataset: "realistic / 32k",
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
  it("opens an encoded deep link for an exact shape trace index entry", () => {
    render(
      <CaseTable
        operator="dsa decode/v3"
        cases={[caseSummary]}
        showDatasetColumn={false}
        shapeTraceKeys={
          new Set([
            shapeTraceKey({
              operator: "dsa decode/v3",
              dataset: "realistic / 32k",
              case_id: "case/1 space"
            })
          ])
        }
      />
    );

    const link = screen.getByRole("link", { name: "2 x 128 x 576" });
    expect(link).toHaveAttribute(
      "href",
      "/shape-explorer?operator=dsa+decode%2Fv3&dataset=realistic+%2F+32k&case=case%2F1+space"
    );
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noreferrer");
  });

  it.each([
    [
      "operator",
      { operator: "other/operator", dataset: caseSummary.dataset, case_id: caseSummary.caseId }
    ],
    [
      "dataset",
      { operator: "dsa decode/v3", dataset: "other/dataset", case_id: caseSummary.caseId }
    ],
    ["case ID", { operator: "dsa decode/v3", dataset: caseSummary.dataset, case_id: "other/case" }]
  ])("keeps the shape plain when the nonempty index mismatches %s", (_field, indexedTrace) => {
    render(
      <CaseTable
        operator="dsa decode/v3"
        cases={[caseSummary]}
        showDatasetColumn={false}
        shapeTraceKeys={new Set([shapeTraceKey(indexedTrace)])}
      />
    );

    expect(screen.queryByRole("link", { name: "2 x 128 x 576" })).not.toBeInTheDocument();
    expect(screen.getByText("2 x 128 x 576")).toBeInTheDocument();
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
