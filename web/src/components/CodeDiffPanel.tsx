import { useState } from "react";
import { getRepositoryDiff, type DiffLine } from "../data/diffData";

interface CodeDiffPanelProps {
  diffRef: string | null;
}

function lineClass(line: DiffLine): string {
  return `diff-line diff-line--${line.type}`;
}

function prefix(line: DiffLine): string {
  if (line.type === "add") {
    return "+";
  }
  if (line.type === "delete") {
    return "-";
  }
  return " ";
}

export function CodeDiffPanel({ diffRef }: CodeDiffPanelProps) {
  const [mode, setMode] = useState<"split" | "unified">("split");
  const diff = getRepositoryDiff(diffRef);

  if (!diff) {
    return (
      <section className="diff-panel diff-panel--empty" aria-label="Repository diff">
        <p className="panel-kicker">Repository diff</p>
        <h3>No custom operator diff</h3>
        <p>Select an NPU custom-operator result to inspect repository-owned code changes.</p>
      </section>
    );
  }

  return (
    <section className="diff-panel" aria-label="Repository diff">
      <div className="diff-head">
        <div>
          <p className="panel-kicker">Repository diff</p>
          <h3>{diff.title}</h3>
          <p>
            {diff.baselineLabel} vs {diff.customLabel}
          </p>
        </div>
        <div className="diff-toggle" aria-label="Diff view mode">
          <button type="button" aria-pressed={mode === "split"} onClick={() => setMode("split")}>
            split
          </button>
          <button type="button" aria-pressed={mode === "unified"} onClick={() => setMode("unified")}>
            unified
          </button>
        </div>
      </div>
      <div className="diff-files" aria-label="Changed files">
        {diff.files.map((file) => (
          <span key={file.path}>{file.path.split("/").slice(-3).join("/")}</span>
        ))}
      </div>
      <p className="diff-mode-label">{mode === "split" ? "split diff" : "unified diff"}</p>
      {diff.files.map((file) => (
        <article key={file.path} className={`diff-file diff-file--${mode}`}>
          <header>{file.path}</header>
          <div className="diff-code">
            {file.hunks.map((line, index) => (
              <div key={`${line.oldNumber}-${line.newNumber}-${index}`} className={lineClass(line)}>
                <span className="diff-number">{line.oldNumber ?? ""}</span>
                {mode === "split" && <span className="diff-number">{line.newNumber ?? ""}</span>}
                <span className="diff-prefix">{prefix(line)}</span>
                <code>{line.text}</code>
              </div>
            ))}
          </div>
        </article>
      ))}
    </section>
  );
}
