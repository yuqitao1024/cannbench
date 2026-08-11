import { Moon, Sun } from "lucide-react";
import { useEffect, useRef, useState, type KeyboardEvent } from "react";
import logoDarkUrl from "../assets/brand/cannbench-logo-dark.png";
import logoLightUrl from "../assets/brand/cannbench-logo-light.png";
import { fetchShapeTrace, fetchShapeTraceIndex } from "../data/shapeTraceApi";
import type { ShapeStage, ShapeTrace, ShapeTraceIndexEntry } from "../shape-trace/types";
import { DeviceExecutionView } from "./DeviceExecutionView";
import { ShapeMatrixEquation } from "./ShapeMatrix";
import { ShapeTimeline } from "./ShapeTimeline";
import "../shape-explorer.css";

type ExplorerStatus = "invalid" | "loading" | "ready" | "not-found" | "error";
type ExplorerView = "algorithm" | "device";

function themeForCurrentHour(): "light" | "dark" {
  const hour = new Date().getHours();
  return hour >= 7 && hour < 19 ? "light" : "dark";
}

function titleCase(value: string): string {
  return value
    .split(/[-_\s]+/)
    .filter(Boolean)
    .map((token) => token.charAt(0).toUpperCase() + token.slice(1))
    .join(" ");
}

function formatSymbolValue(value: number): string {
  return value.toLocaleString("en-US");
}

function implementationLabel(execution: ShapeTrace["device_execution"]): string {
  if (execution.status === "unavailable") return "Device trace unavailable";
  return `${execution.implementation.toUpperCase()} ${execution.version ?? ""}`.trim();
}

function Inspector({ stage }: { stage: ShapeStage }) {
  const contracted = stage.contracted_axes.length > 0 ? stage.contracted_axes.join(", ") : "None";
  return (
    <aside className="shape-inspector" aria-label="Stage inspector">
      <section>
        <h3>Selected operation</h3>
        <dl className="shape-inspector-facts">
          <div>
            <dt>Operation</dt>
            <dd>{stage.operation}</dd>
          </div>
          <div>
            <dt>Scope</dt>
            <dd>{stage.scope}</dd>
          </div>
          <div>
            <dt>Contracted axes</dt>
            <dd>{contracted}</dd>
          </div>
        </dl>
      </section>
      <section>
        <h3>Axis annotation</h3>
        <div className="shape-axis-legend">
          <span className="shape-legend-swatch is-input" aria-hidden="true" />
          <span>Tensor dimensions</span>
          <span className="shape-legend-swatch is-output" aria-hidden="true" />
          <span>Current output</span>
          <span className="shape-legend-swatch is-contracted" aria-hidden="true" />
          <span>Contracted dimension</span>
        </div>
      </section>
      <section>
        <h3>Why this shape changes</h3>
        <p className="shape-insight">{stage.insight}</p>
      </section>
    </aside>
  );
}

function StatusView({
  status,
  theme
}: {
  status: ExplorerStatus;
  theme: "light" | "dark";
}) {
  const message =
    status === "invalid"
      ? "Invalid shape trace link"
      : status === "not-found"
        ? "Shape trace not found"
        : status === "error"
          ? "Unable to load shape trace"
          : "Loading shape trace...";
  return (
    <main
      className="app-shell shape-explorer-shell shape-explorer-status-shell"
      data-theme={theme}
    >
      <div className="shape-explorer-status" role="status">
        {message}
      </div>
    </main>
  );
}

export function ShapeExplorerPage() {
  const [navigationRevision, setNavigationRevision] = useState(0);
  const [trace, setTrace] = useState<ShapeTrace | null>(null);
  const [traceIndex, setTraceIndex] = useState<ShapeTraceIndexEntry[]>([]);
  const [status, setStatus] = useState<ExplorerStatus>("loading");
  const [activeStageIndex, setActiveStageIndex] = useState(0);
  const [activeView, setActiveView] = useState<ExplorerView>("algorithm");
  const [theme, setTheme] = useState<"light" | "dark">(themeForCurrentHour);
  const algorithmTabRef = useRef<HTMLButtonElement>(null);
  const deviceTabRef = useRef<HTMLButtonElement>(null);
  const params = new URLSearchParams(window.location.search);
  const operator = params.get("operator")?.trim() ?? "";
  const dataset = params.get("dataset")?.trim() ?? "";
  const caseId = params.get("case")?.trim() ?? "";
  const hasValidIdentity = operator.length > 0 && dataset.length > 0 && caseId.length > 0;

  useEffect(() => {
    document.documentElement.dataset.theme = theme;
    document.body.dataset.theme = theme;
    return () => {
      delete document.documentElement.dataset.theme;
      delete document.body.dataset.theme;
    };
  }, [theme]);

  useEffect(() => {
    if (!hasValidIdentity) {
      setTrace(null);
      setStatus("invalid");
      return;
    }

    const controller = new AbortController();
    setTrace(null);
    setStatus("loading");
    Promise.all([
      fetchShapeTrace(operator, dataset, caseId, controller.signal),
      fetchShapeTraceIndex(controller.signal)
    ])
      .then(([nextTrace, nextIndex]) => {
        if (controller.signal.aborted) return;
        setTrace(nextTrace);
        setTraceIndex(nextIndex);
        setActiveStageIndex(0);
        setStatus("ready");
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setTrace(null);
        const message = error instanceof Error ? error.message : "";
        setStatus(message.includes(": 404") ? "not-found" : "error");
      });
    return () => controller.abort();
  }, [caseId, dataset, hasValidIdentity, navigationRevision, operator]);

  if (status !== "ready" || trace === null || trace.stages.length === 0) {
    return (
      <StatusView
        status={status === "ready" ? "not-found" : status}
        theme={theme}
      />
    );
  }

  const stage = trace.stages[Math.min(activeStageIndex, trace.stages.length - 1)];
  const phaseOptions = traceIndex.filter((entry) => entry.group === trace.group);
  const selectView = (view: ExplorerView, moveFocus = false) => {
    setActiveView(view);
    if (moveFocus) {
      const target = view === "algorithm" ? algorithmTabRef : deviceTabRef;
      target.current?.focus();
    }
  };
  const handleViewTabKeyDown = (event: KeyboardEvent<HTMLDivElement>) => {
    let nextView: ExplorerView | null = null;
    if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      nextView = activeView === "algorithm" ? "device" : "algorithm";
    } else if (event.key === "Home") {
      nextView = "algorithm";
    } else if (event.key === "End") {
      nextView = "device";
    }
    if (nextView === null) return;
    event.preventDefault();
    selectView(nextView, true);
  };
  const selectPhase = (entry: ShapeTraceIndexEntry) => {
    if (
      entry.operator === trace.operator &&
      entry.dataset === trace.dataset &&
      entry.case_id === trace.case_id
    ) {
      return;
    }
    const nextParams = new URLSearchParams({
      operator: entry.operator,
      dataset: entry.dataset,
      case: entry.case_id
    });
    window.history.replaceState({}, "", `/shape-explorer?${nextParams.toString()}`);
    setActiveStageIndex(0);
    setTrace(null);
    setStatus("loading");
    setNavigationRevision((current) => current + 1);
  };

  return (
    <main className="app-shell shape-explorer-shell" data-theme={theme}>
      <header className="shape-topbar">
        <div className="shape-brand">
          <img
            src={theme === "dark" ? logoDarkUrl : logoLightUrl}
            alt="CANNBench"
          />
          <span>/ Shape Explorer</span>
        </div>
        <div className="shape-topbar-meta">
          <span>{titleCase(trace.group)} / {trace.dataset}</span>
          <button
            type="button"
            className="shape-theme-button"
            aria-label="Toggle light and dark theme"
            title="Toggle light and dark theme"
            onClick={() => setTheme((current) => (current === "dark" ? "light" : "dark"))}
          >
            {theme === "dark" ? <Moon aria-hidden="true" /> : <Sun aria-hidden="true" />}
          </button>
        </div>
      </header>

      <section className="shape-casebar" aria-labelledby="shape-explorer-title">
        <div>
          <p className="shape-eyebrow">{titleCase(trace.phase)} workflow</p>
          <h1 id="shape-explorer-title">Matrix flow from index selection to sparse attention</h1>
          <p className="shape-case-id">{trace.case_id}</p>
        </div>
        <div className="shape-phase-segment" aria-label="Case phase">
          {phaseOptions.map((entry) => (
            <button
              type="button"
              aria-pressed={
                entry.operator === trace.operator &&
                entry.dataset === trace.dataset &&
                entry.case_id === trace.case_id
              }
              onClick={() => selectPhase(entry)}
              key={`${entry.operator}\u0000${entry.dataset}\u0000${entry.case_id}`}
            >
              {titleCase(entry.phase)}
            </button>
          ))}
        </div>
      </section>

      <section className="shape-symbol-strip" aria-label="Shape symbols">
        {trace.symbols.map((symbol) => (
          <div className="shape-symbol" key={symbol.symbol} title={symbol.meaning}>
            <span>{symbol.symbol}</span>
            <strong>{formatSymbolValue(symbol.value)}</strong>
          </div>
        ))}
      </section>

      <div className="shape-modebar">
        <div
          className="shape-view-tabs"
          role="tablist"
          aria-label="Explorer view"
          onKeyDown={handleViewTabKeyDown}
        >
          <button
            type="button"
            id="shape-algorithm-tab"
            ref={algorithmTabRef}
            role="tab"
            aria-label="Algorithm"
            aria-controls="shape-algorithm-panel"
            aria-selected={activeView === "algorithm"}
            tabIndex={activeView === "algorithm" ? 0 : -1}
            onClick={() => selectView("algorithm")}
          >
            Algorithm flow
          </button>
          <button
            type="button"
            id="shape-device-tab"
            ref={deviceTabRef}
            role="tab"
            aria-label="Device"
            aria-controls="shape-device-panel"
            aria-selected={activeView === "device"}
            tabIndex={activeView === "device" ? 0 : -1}
            onClick={() => selectView("device")}
          >
            Device execution
          </button>
        </div>
        <span className="shape-version-label">{implementationLabel(trace.device_execution)}</span>
      </div>

      {activeView === "algorithm" ? (
        <div
          id="shape-algorithm-panel"
          role="tabpanel"
          aria-labelledby="shape-algorithm-tab"
        >
          <ShapeTimeline
            stages={trace.stages}
            activeIndex={activeStageIndex}
            onActiveIndexChange={setActiveStageIndex}
          />
          <div className="shape-workspace">
            <section className="shape-stage-surface" aria-labelledby="shape-stage-title">
              <header className="shape-stage-header">
                <h2 id="shape-stage-title">{stage.title}</h2>
                <code>{stage.component}</code>
              </header>
              <ShapeMatrixEquation stage={stage} />
            </section>
            <Inspector stage={stage} />
          </div>
        </div>
      ) : (
        <div
          className="shape-device-workspace"
          id="shape-device-panel"
          role="tabpanel"
          aria-labelledby="shape-device-tab"
        >
          <DeviceExecutionView execution={trace.device_execution} />
        </div>
      )}
    </main>
  );
}
