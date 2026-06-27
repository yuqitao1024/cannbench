import { describe, expect, it } from "vitest";
import { cudaTreasureRoute } from "./cudaOptimizationRoute";

function getNode(nodeId: string) {
  const node = cudaTreasureRoute.find((entry) => entry.id === nodeId);

  expect(node, `Missing CUDA treasure route node "${nodeId}"`).toBeDefined();
  return node!;
}

describe("cudaTreasureRoute", () => {
  it("anchors the lower-half main nodes along the generated main route", () => {
    const stageThroughShared = getNode("stage-through-shared");
    const tuneLaunchGeometry = getNode("tune-launch-geometry");
    const polishInstructions = getNode("polish-instructions");

    expect(stageThroughShared.kind).toBe("main");
    expect(stageThroughShared.x).toBeGreaterThan(70);

    expect(tuneLaunchGeometry.kind).toBe("main");
    expect(tuneLaunchGeometry.y).toBeGreaterThan(80);

    expect(polishInstructions.kind).toBe("main");
    expect(polishInstructions.y).toBeGreaterThan(65);
  });

  it("keeps branch nodes on the image-backed dashed side routes", () => {
    const branchStreams = getNode("branch-streams");
    const branchL2 = getNode("branch-l2");
    const branchBankConflicts = getNode("branch-bank-conflicts");
    const branchAsyncG2S = getNode("branch-async-g2s");
    const branchConcurrency = getNode("branch-concurrency");
    const branchTargetBuild = getNode("branch-target-build");

    expect(branchStreams.kind).toBe("branch");
    expect(branchStreams.x).toBeGreaterThan(68);
    expect(branchStreams.y).toBeGreaterThan(24);
    expect(branchStreams.y).toBeLessThan(27);

    expect(branchL2.kind).toBe("branch");
    expect(branchL2.x).toBeGreaterThan(82);
    expect(branchL2.y).toBeGreaterThan(36);
    expect(branchL2.y).toBeLessThan(39);

    expect(branchBankConflicts.kind).toBe("branch");
    expect(branchBankConflicts.x).toBeGreaterThan(32);
    expect(branchBankConflicts.x).toBeLessThan(35);
    expect(branchBankConflicts.y).toBeGreaterThan(66);
    expect(branchBankConflicts.y).toBeLessThan(68);

    expect(branchAsyncG2S.kind).toBe("branch");
    expect(branchAsyncG2S.x).toBeGreaterThan(54);
    expect(branchAsyncG2S.x).toBeLessThan(57);
    expect(branchAsyncG2S.y).toBeGreaterThan(72);
    expect(branchAsyncG2S.y).toBeLessThan(74);

    expect(branchConcurrency.kind).toBe("branch");
    expect(branchConcurrency.x).toBeGreaterThan(20);
    expect(branchConcurrency.x).toBeLessThan(22);
    expect(branchConcurrency.y).toBeGreaterThan(47);
    expect(branchConcurrency.y).toBeLessThan(50);

    expect(branchTargetBuild.kind).toBe("branch");
    expect(branchTargetBuild.x).toBeGreaterThan(26);
    expect(branchTargetBuild.x).toBeLessThan(28);
    expect(branchTargetBuild.y).toBeGreaterThan(13);
    expect(branchTargetBuild.y).toBeLessThan(15);
  });
});
