import { ChevronLeft, ChevronRight, Pause, Play } from "lucide-react";
import { useEffect, useState } from "react";
import type { ShapeStage } from "../shape-trace/types";

interface ShapeTimelineProps {
  stages: ShapeStage[];
  activeIndex: number;
  onActiveIndexChange: (index: number) => void;
}

const PLAYBACK_SPEEDS = [
  { label: "0.75x", intervalMs: 1600 },
  { label: "1x", intervalMs: 1100 },
  { label: "1.5x", intervalMs: 650 }
];

function capitalizeToken(token: string): string {
  if (/^(qk|pv|kv)$/i.test(token)) return token.toUpperCase();
  if (/^topk$/i.test(token)) return "TopK";
  return token.charAt(0).toUpperCase() + token.slice(1);
}

function stageLabel(stage: ShapeStage): string {
  const [first, ...rest] = stage.id.split("-");
  return [capitalizeToken(first), ...rest.map((token) => {
    const formatted = capitalizeToken(token);
    return formatted === token.toUpperCase() || formatted === "TopK"
      ? formatted
      : token.toLowerCase();
  })].join(" ");
}

export function ShapeTimeline({
  stages,
  activeIndex,
  onActiveIndexChange
}: ShapeTimelineProps) {
  const [playing, setPlaying] = useState(false);
  const [intervalMs, setIntervalMs] = useState(1100);
  const prefersReducedMotion =
    typeof window.matchMedia === "function" &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const finalIndex = Math.max(0, stages.length - 1);

  useEffect(() => {
    if (!playing) return;
    if (activeIndex >= stages.length - 1) {
      setPlaying(false);
      return;
    }
    const timer = window.setTimeout(
      () => onActiveIndexChange(Math.min(activeIndex + 1, stages.length - 1)),
      intervalMs
    );
    return () => window.clearTimeout(timer);
  }, [activeIndex, intervalMs, onActiveIndexChange, playing, stages.length]);

  useEffect(() => {
    if (prefersReducedMotion) setPlaying(false);
  }, [prefersReducedMotion]);

  const selectStage = (index: number) => {
    setPlaying(false);
    onActiveIndexChange(index);
  };

  return (
    <section className="shape-timeline-wrap" aria-label="Shape trace timeline">
      <div className="shape-timeline-head">
        <div className="shape-playback-controls">
          <button
            type="button"
            className="shape-icon-button"
            disabled={activeIndex <= 0}
            title="Previous stage"
            aria-label="Previous stage"
            onClick={() => selectStage(Math.max(0, activeIndex - 1))}
          >
            <ChevronLeft aria-hidden="true" />
          </button>
          <button
            type="button"
            className="shape-icon-button shape-play-button"
            disabled={prefersReducedMotion}
            title={
              prefersReducedMotion
                ? "Playback disabled by reduced-motion preference"
                : playing
                  ? "Pause shape trace"
                  : "Play shape trace"
            }
            aria-label={playing ? "Pause shape trace" : "Play shape trace"}
            onClick={() => setPlaying((current) => !current)}
          >
            {playing ? <Pause aria-hidden="true" /> : <Play aria-hidden="true" />}
          </button>
          <button
            type="button"
            className="shape-icon-button"
            disabled={activeIndex >= finalIndex}
            title="Next stage"
            aria-label="Next stage"
            onClick={() => selectStage(Math.min(finalIndex, activeIndex + 1))}
          >
            <ChevronRight aria-hidden="true" />
          </button>
          <select
            className="shape-playback-speed"
            aria-label="Playback speed"
            value={intervalMs}
            onChange={(event) => setIntervalMs(Number(event.target.value))}
          >
            {PLAYBACK_SPEEDS.map((speed) => (
              <option value={speed.intervalMs} key={speed.intervalMs}>
                {speed.label}
              </option>
            ))}
          </select>
        </div>
        <span className="shape-step-count">
          Step {Math.min(activeIndex + 1, stages.length)} of {stages.length}
        </span>
      </div>
      <div className="shape-timeline">
        {stages.map((stage, index) => {
          const label = stageLabel(stage);
          const state = index === activeIndex ? "active" : index < activeIndex ? "done" : "pending";
          return (
            <button
              type="button"
              className={`shape-timeline-step is-${state}`}
              aria-current={index === activeIndex ? "step" : undefined}
              aria-label={label}
              onClick={() => selectStage(index)}
              key={stage.id}
            >
              <strong>
                {String(index + 1).padStart(2, "0")} {label}
              </strong>
              <small>{stage.component}</small>
            </button>
          );
        })}
      </div>
    </section>
  );
}
