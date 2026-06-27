import { useId, useState } from "react";
import cudaTreasureMapDark from "../assets/cuda-treasure-map-dark.png";
import type { TreasureNode } from "../data/cudaOptimizationRoute";

interface CudaTreasureMapProps {
  route: readonly TreasureNode[];
  mainRouteOrder: readonly string[];
}

export function CudaTreasureMap({ route, mainRouteOrder }: CudaTreasureMapProps) {
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const tooltipIdPrefix = useId();

  const nodesById = new Map(route.map((node) => [node.id, node]));
  if (nodesById.size !== route.length) {
    throw new Error("CUDA treasure route contains duplicate node ids.");
  }

  const mainRouteNodeIds = new Set(mainRouteOrder);
  mainRouteOrder.forEach((nodeId) => {
    const node = nodesById.get(nodeId);

    if (!node || node.kind !== "main") {
      throw new Error(`CUDA treasure main route is missing node "${nodeId}".`);
    }
  });
  route
    .filter((node) => node.kind === "branch")
    .forEach((node) => {
      const parentNode = nodesById.get(node.branchFrom);

      if (!parentNode || parentNode.kind !== "main") {
        throw new Error(`CUDA treasure route node "${node.id}" references missing branch parent "${node.branchFrom}".`);
      }
    });
  const activeNodeId = hoveredNodeId ?? focusedNodeId;
  const activeNode = activeNodeId ? nodesById.get(activeNodeId) ?? null : null;

  return (
    <div className="cuda-treasure-map">
      <div
        className="cuda-treasure-map__canvas"
        style={{ position: "relative", minHeight: "40rem", width: "100%", overflow: "hidden" }}
      >
        <img
          className="cuda-treasure-map__background"
          src={cudaTreasureMapDark}
          alt=""
          aria-hidden="true"
        />

        {route.map((node) => {
          const tooltipId = `${tooltipIdPrefix}-${node.id}`;
          const isMainRouteNode = mainRouteNodeIds.has(node.id);
          const labelDx = node.labelDx ?? 0;
          const labelDy = node.labelDy ?? 0;

          return (
            <div
              key={node.id}
              className="cuda-treasure-map__node-slot"
              style={{
                position: "absolute",
                left: `${node.x}%`,
                top: `${node.y}%`,
                transform: "translate(-50%, -50%)",
                zIndex: activeNode?.id === node.id ? 2 : 1
              }}
            >
              <span
                aria-hidden="true"
                className={[
                  "cuda-treasure-map__anchor",
                  `cuda-treasure-map__anchor--${node.kind}`,
                  isMainRouteNode ? "is-on-main-route" : ""
                ]
                  .filter(Boolean)
                  .join(" ")}
              />
              <button
                type="button"
                className={[
                  "cuda-treasure-map__node",
                  `cuda-treasure-map__node--${node.kind}`,
                  node.importance === "high" ? "is-important" : ""
                ]
                  .filter(Boolean)
                  .join(" ")}
                style={{ transform: `translate(${labelDx}rem, ${labelDy}rem)` }}
                aria-describedby={activeNode?.id === node.id ? tooltipId : undefined}
                onClick={() => setFocusedNodeId(node.id)}
                onMouseEnter={() => setHoveredNodeId(node.id)}
                onMouseLeave={() => setHoveredNodeId((current) => (current === node.id ? null : current))}
                onFocus={() => setFocusedNodeId(node.id)}
                onBlur={() => setFocusedNodeId((current) => (current === node.id ? null : current))}
              >
                {node.label}
              </button>
            </div>
          );
        })}

        {activeNode ? (
          <aside
            id={`${tooltipIdPrefix}-${activeNode.id}`}
            role="tooltip"
            className="cuda-treasure-map__field-note"
            style={{
              zIndex: 3,
              left: activeNode.x > 68 ? "auto" : `calc(${activeNode.x}% + 1rem)`,
              right: activeNode.x > 68 ? `calc(${100 - activeNode.x}% + 1rem)` : "auto",
              top: activeNode.y > 72 ? "auto" : `calc(${activeNode.y}% + 1rem)`,
              bottom: activeNode.y > 72 ? `calc(${100 - activeNode.y}% + 1rem)` : "auto"
            }}
          >
            <p className="cuda-treasure-map__field-note-kicker">Field note</p>
            <h3 className="cuda-treasure-map__field-note-title">{activeNode.label}</h3>
            <p className="cuda-treasure-map__field-note-summary">{activeNode.summary}</p>
            <ul className="cuda-treasure-map__field-note-details">
              {activeNode.details.map((detail) => (
                <li key={detail}>{detail}</li>
              ))}
            </ul>
            <p className="cuda-treasure-map__field-note-guide">Guide: {activeNode.guideSections.join(", ")}</p>
            <p className="cuda-treasure-map__field-note-route-ids">
              Route IDs: {activeNode.relatedOptimizationIds.join(", ")}
            </p>
          </aside>
        ) : null}
      </div>
    </div>
  );
}
