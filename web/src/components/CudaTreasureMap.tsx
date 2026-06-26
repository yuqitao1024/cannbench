import { useId, useState } from "react";
import type { TreasureNode } from "../data/cudaOptimizationRoute";

interface CudaTreasureMapProps {
  route: readonly TreasureNode[];
}

export function CudaTreasureMap({ route }: CudaTreasureMapProps) {
  const [hoveredNodeId, setHoveredNodeId] = useState<string | null>(null);
  const [focusedNodeId, setFocusedNodeId] = useState<string | null>(null);
  const tooltipIdPrefix = useId();

  const nodesById = new Map(route.map((node) => [node.id, node]));
  const mainRouteNodes = route.filter((node) => node.kind === "main");
  const branchNodes = route.filter((node) => node.kind === "branch");
  const branchConnectors = branchNodes.map((node) => {
    const parentNode = nodesById.get(node.branchFrom);

    if (!parentNode || parentNode.kind !== "main") {
      throw new Error(`CUDA treasure route node "${node.id}" references missing branch parent "${node.branchFrom}".`);
    }

    return { node, parentNode };
  });
  const activeNodeId = focusedNodeId ?? hoveredNodeId;
  const activeNode = activeNodeId ? nodesById.get(activeNodeId) ?? null : null;

  return (
    <div className="cuda-treasure-map">
      <div
        className="cuda-treasure-map__canvas"
        style={{ position: "relative", minHeight: "40rem", width: "100%", overflow: "hidden" }}
      >
        <svg
          className="cuda-treasure-map__paths"
          viewBox="0 0 100 100"
          preserveAspectRatio="none"
          aria-hidden="true"
          style={{ position: "absolute", inset: 0, pointerEvents: "none", overflow: "visible" }}
        >
          <polyline
            className="cuda-treasure-map__path cuda-treasure-map__path--main"
            points={mainRouteNodes.map((node) => `${node.x},${node.y}`).join(" ")}
            fill="none"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinejoin="round"
            strokeLinecap="round"
          />
          {branchConnectors.map(({ node, parentNode }) => (
            <line
              key={node.id}
              className="cuda-treasure-map__path cuda-treasure-map__path--branch"
              x1={parentNode.x}
              y1={parentNode.y}
              x2={node.x}
              y2={node.y}
              stroke="currentColor"
              strokeWidth="1"
              strokeDasharray="2 2"
            />
          ))}
        </svg>

        {route.map((node) => {
          const tooltipId = `${tooltipIdPrefix}-${node.id}`;

          return (
            <button
              key={node.id}
              type="button"
              className={[
                "cuda-treasure-map__node",
                `cuda-treasure-map__node--${node.kind}`,
                node.importance === "high" ? "is-important" : ""
              ]
                .filter(Boolean)
                .join(" ")}
              style={{
                position: "absolute",
                left: `${node.x}%`,
                top: `${node.y}%`,
                transform: "translate(-50%, -50%)",
                zIndex: activeNode?.id === node.id ? 2 : 1
              }}
              aria-describedby={activeNode?.id === node.id ? tooltipId : undefined}
              onMouseEnter={() => setHoveredNodeId(node.id)}
              onMouseLeave={() => setHoveredNodeId((current) => (current === node.id ? null : current))}
              onFocus={() => setFocusedNodeId(node.id)}
              onBlur={() => setFocusedNodeId((current) => (current === node.id ? null : current))}
            >
              {node.label}
            </button>
          );
        })}

        {activeNode ? (
          <aside
            id={`${tooltipIdPrefix}-${activeNode.id}`}
            role="tooltip"
            className="cuda-treasure-map__field-note"
            style={{
              position: "absolute",
              left: `min(calc(${activeNode.x}% + 1rem), calc(100% - 18rem))`,
              top: `min(calc(${activeNode.y}% + 1rem), calc(100% - 12rem))`,
              maxWidth: "16rem",
              zIndex: 3
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
