import { useState } from "react";
import { createPortal } from "react-dom";
import { Diff, Hunk, parseDiff } from "react-diff-view";
import "react-diff-view/style/index.css";
import { getRepositoryDiff } from "../data/diffData";

interface CodeDiffPanelProps {
  diffRef: string | null;
}

export function CodeDiffPanel({ diffRef }: CodeDiffPanelProps) {
  const [mode, setMode] = useState<"split" | "unified">("split");
  const [open, setOpen] = useState(false);
  const diff = getRepositoryDiff(diffRef);
  const files = diff ? parseDiff(diff.patch, { nearbySequences: "zip" }) : [];
  const workspace =
    open && diff
      ? createPortal(
          <div className="modal-backdrop diff-workspace-backdrop" role="presentation">
            <section className="diff-workspace" role="dialog" aria-modal="true" aria-label="Repository diff workspace">
              <header className="diff-workspace-toolbar">
                <div>
                  <p className="panel-kicker">Repository diff</p>
                  <h3>{diff.title}</h3>
                  <p>
                    {diff.baselineLabel} vs {diff.customLabel}
                  </p>
                </div>
                <div className="diff-workspace-actions">
                  <div className="diff-toggle" aria-label="Diff view mode">
                    <button type="button" aria-pressed={mode === "split"} onClick={() => setMode("split")}>
                      split
                    </button>
                    <button type="button" aria-pressed={mode === "unified"} onClick={() => setMode("unified")}>
                      unified
                    </button>
                  </div>
                  <button type="button" className="modal-close" onClick={() => setOpen(false)}>
                    close
                  </button>
                </div>
              </header>
              <div className="diff-workspace-body">
                <aside className="diff-file-rail" aria-label="Changed files">
                  {files.map((file) => (
                    <span key={file.newPath ?? file.oldPath}>{(file.newPath ?? file.oldPath).split("/").slice(-3).join("/")}</span>
                  ))}
                </aside>
                <div className="diff-workspace-content">
                  <p className="diff-mode-label">{mode === "split" ? "split diff" : "unified diff"}</p>
                  {files.map((file) => (
                    <article key={file.newPath ?? file.oldPath} className="diff-file">
                      <header>{file.newPath ?? file.oldPath}</header>
                      <Diff viewType={mode} diffType={file.type} hunks={file.hunks} gutterType="default">
                        {(hunks) => hunks.map((hunk) => <Hunk key={hunk.content} hunk={hunk} />)}
                      </Diff>
                    </article>
                  ))}
                </div>
              </div>
            </section>
          </div>,
          document.body
        )
      : null;

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
      </div>
      <div className="diff-summary-grid">
        <div>
          <span className="diff-summary-label">changed files</span>
          <strong>{files.length}</strong>
        </div>
        <div>
          <span className="diff-summary-label">default view</span>
          <strong>split</strong>
        </div>
        <button type="button" className="diff-open-button" onClick={() => setOpen(true)}>
          Open diff
        </button>
      </div>
      <div className="diff-files" aria-label="Changed files">
        {files.map((file) => (
          <span key={file.newPath ?? file.oldPath}>{(file.newPath ?? file.oldPath).split("/").slice(-3).join("/")}</span>
        ))}
      </div>
      {workspace}
    </section>
  );
}
