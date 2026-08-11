import type { ShapeStage, ShapeTrace } from "./types";

export function makeShapeTrace(phase: "decode" | "prefill" = "decode"): ShapeTrace {
  const stages = Array.from({ length: 8 }, (_, index): ShapeStage => ({
    id: `stage-${index}`,
    component: index < 4 ? "lightning_indexer" : "sparse_attention",
    title: index === 0 ? "Indexer projection" : index === 7 ? "Output" : `Stage ${index + 1}`,
    operation: index === 7 ? "matmul" : "inputs",
    formula: index === 7 ? "[H,S] x [S,Dv] -> [H,Dv]" : "input",
    scope: "one query row",
    tensors: [],
    input_ids: [],
    output_ids: [],
    contracted_axes: [],
    insight: "Fixture stage."
  }));
  return {
    schema_version: 1,
    operator: phase === "decode" ? "dsa_decode" : "dsa_prefill",
    dataset: "realistic",
    case_id: `${phase}-case`,
    phase,
    group: "deepseek-v32",
    symbols: [],
    stages,
    device_execution: {
      status: "unavailable",
      implementation: "simt",
      version: null,
      message: phase === "prefill" ? "No device trace for prefill" : "No device trace.",
      kernels: []
    }
  };
}

export function jsonResponse(payload: unknown, status = 200): Response {
  return {
    ok: status >= 200 && status < 300,
    status,
    headers: new Headers({ "content-type": "application/json" }),
    json: async () => payload,
    text: async () => JSON.stringify(payload)
  } as Response;
}
