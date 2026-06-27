import { useEffect, useState } from "react";
import { createPortal } from "react-dom";
import { Diff, Hunk, parseDiff } from "react-diff-view";
import "react-diff-view/style/index.css";
import { fetchSimtOperatorDiff, fetchSimtOperatorVersions } from "../data/simtDiffApi";
import type { SimtOperatorDiff } from "../types";

interface CodeDiffPanelProps {
  operator: string;
}

type SelectionSlot = "base" | "compare";

interface SimtVersionOption {
  version: string;
}

function countPatchLines(patch: string) {
  let additions = 0;
  let deletions = 0;
  for (const line of patch.split("\n")) {
    if (line.startsWith("+++")) {
      continue;
    }
    if (line.startsWith("---")) {
      continue;
    }
    if (line.startsWith("+")) {
      additions += 1;
      continue;
    }
    if (line.startsWith("-")) {
      deletions += 1;
    }
  }
  return { additions, deletions };
}

export function CodeDiffPanel({ operator }: CodeDiffPanelProps) {
  const [mode, setMode] = useState<"split" | "unified">("split");
  const [open, setOpen] = useState(false);
  const [activeSlot, setActiveSlot] = useState<SelectionSlot>("compare");
  const [versionOptions, setVersionOptions] = useState<SimtVersionOption[]>([]);
  const [isLoadingVersions, setIsLoadingVersions] = useState(false);
  const [versionError, setVersionError] = useState<string | null>(null);
  const versionSignature = versionOptions.map((option) => option.version).join("|");
  const [baseVersion, setBaseVersion] = useState<string | null>(null);
  const [compareVersion, setCompareVersion] = useState<string | null>(null);
  const [diff, setDiff] = useState<SimtOperatorDiff | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [diffError, setDiffError] = useState<string | null>(null);

  useEffect(() => {
    const controller = new AbortController();
    setIsLoadingVersions(true);
    setVersionError(null);
    setVersionOptions([]);
    setBaseVersion(null);
    setCompareVersion(null);
    setDiff(null);
    setDiffError(null);
    setOpen(false);

    fetchSimtOperatorVersions(operator, controller.signal)
      .then((result) => {
        setVersionOptions(result.versions.map((version) => ({ version })));
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Failed to load SIMT operator versions.";
        setVersionError(message);
        setVersionOptions([]);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoadingVersions(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [operator]);

  useEffect(() => {
    if (versionOptions.length >= 2) {
      setBaseVersion(versionOptions[0].version);
      setCompareVersion(versionOptions[1].version);
      setActiveSlot("base");
      return;
    }
    if (versionOptions.length === 1) {
      setBaseVersion(versionOptions[0].version);
      setCompareVersion(null);
      setActiveSlot("compare");
      return;
    }
    setBaseVersion(null);
    setCompareVersion(null);
    setActiveSlot("compare");
  }, [versionSignature]);

  useEffect(() => {
    if (isLoadingVersions || versionOptions.length < 2 || !baseVersion || !compareVersion) {
      setDiff(null);
      setDiffError(null);
      setIsLoading(false);
      return;
    }

    const controller = new AbortController();
    setIsLoading(true);
    setDiffError(null);
    fetchSimtOperatorDiff(operator, baseVersion, compareVersion, controller.signal)
      .then((result) => {
        setDiff(result.patch ? result : null);
        setDiffError(result.patch ? null : "No code changes detected between the selected SIMT operator versions.");
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) {
          return;
        }
        const message = error instanceof Error ? error.message : "Failed to load SIMT operator diff.";
        setDiff(null);
        setDiffError(message);
      })
      .finally(() => {
        if (!controller.signal.aborted) {
          setIsLoading(false);
        }
      });

    return () => {
      controller.abort();
    };
  }, [baseVersion, compareVersion, operator, versionSignature]);

  const files = diff ? parseDiff(diff.patch, { nearbySequences: "zip" }) : [];
  const patchCounts = diff ? countPatchLines(diff.patch) : { additions: 0, deletions: 0 };

  const assignVersion = (version: string) => {
    const alternativeVersion = versionOptions.find((option) => option.version !== version)?.version ?? null;

    if (activeSlot === "base") {
      setBaseVersion(version);
      setCompareVersion((current) => {
        if (current === version) {
          return alternativeVersion;
        }
        return current;
      });
      setActiveSlot("compare");
      return;
    }

    setCompareVersion(version);
    setBaseVersion((current) => {
      if (current === version) {
        return alternativeVersion;
      }
      return current;
    });
    setActiveSlot("base");
  };

  const emptyState =
    versionOptions.length < 2
      ? {
          title: "No diff available",
          description: versionError ?? "Need at least two SIMT operator versions to compare."
        }
      : isLoadingVersions || isLoading
        ? {
            title: "Loading diff",
            description: "Comparing repository-owned SIMT operator directories."
          }
        : {
            title: "No diff available",
            description: diffError ?? "No recorded diff matches the selected SIMT operator versions."
          };

  const workspace =
    open && diff
      ? createPortal(
          <div className="modal-backdrop diff-workspace-backdrop" role="presentation">
            <section className="diff-workspace" role="dialog" aria-modal="true" aria-label="SIMT operator diff workspace">
              <header className="diff-workspace-toolbar">
                <div>
                  <p className="panel-kicker">SIMT operator diff</p>
                  <h3>
                    {baseVersion} {"->"} {compareVersion}
                  </h3>
                  <p className="diff-summary-inline">
                    {files.length} files changed{" "}
                    <span className="diff-summary-plus">{patchCounts.additions} ++</span>{" "}
                    <span className="diff-summary-minus">{patchCounts.deletions} --</span>
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
                    <span key={file.newPath ?? file.oldPath}>{(file.newPath ?? file.oldPath).split("/").slice(-4).join("/")}</span>
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

  return (
    <section className={`diff-panel${diff ? "" : " diff-panel--empty"}`} aria-label="SIMT operator diff">
      <div className="diff-head">
        <p className="panel-kicker">SIMT operator diff</p>
      </div>
      <div className="diff-selector">
        <div className="diff-slots" role="group" aria-label="Selected SIMT versions">
          <button
            type="button"
            className={`diff-slot${activeSlot === "base" ? " is-active" : ""}`}
            onClick={() => setActiveSlot("base")}
          >
            <span className="diff-slot-label">base</span>
            <span className={`diff-slot-pill${baseVersion ? " is-filled is-base" : ""}`}>
              {baseVersion ?? "select version"}
            </span>
            {activeSlot === "base" ? <span className="diff-slot-hint">Select a version below</span> : null}
          </button>
          <button
            type="button"
            className={`diff-slot${activeSlot === "compare" ? " is-active" : ""}`}
            onClick={() => setActiveSlot("compare")}
          >
            <span className="diff-slot-label">compare</span>
            <span className={`diff-slot-pill${compareVersion ? " is-filled is-compare" : ""}`}>
              {compareVersion ?? "select version"}
            </span>
            {activeSlot === "compare" ? <span className="diff-slot-hint">Select a version below</span> : null}
          </button>
        </div>
        <div className="diff-version-row" aria-label="SIMT operator versions">
          {versionOptions.map((option) => (
            <button
              key={option.version}
              type="button"
              className={[
                "diff-version-pill",
                option.version === baseVersion ? "is-base" : "",
                option.version === compareVersion ? "is-compare" : "",
                activeSlot === "base" && option.version === baseVersion ? "is-active-target" : "",
                activeSlot === "compare" && option.version === compareVersion ? "is-active-target" : ""
              ]
                .filter(Boolean)
                .join(" ")}
              onClick={() => assignVersion(option.version)}
            >
              {option.version}
            </button>
          ))}
        </div>
      </div>
      {diff ? (
        <>
          <div className="diff-summary-row">
            <p className="diff-summary-inline">
              {files.length} files changed <span className="diff-summary-plus">{patchCounts.additions} ++</span>{" "}
              <span className="diff-summary-minus">{patchCounts.deletions} --</span>
            </p>
            <button type="button" className="diff-open-button" onClick={() => setOpen(true)} disabled={!diff}>
              Details
            </button>
          </div>
          <div className="diff-files" aria-label="Changed files">
            {files.map((file) => (
              <span key={file.newPath ?? file.oldPath}>{(file.newPath ?? file.oldPath).split("/").slice(-4).join("/")}</span>
            ))}
          </div>
        </>
      ) : (
        <>
          <div className="diff-summary-row">
            <p className="diff-summary-inline">{isLoadingVersions || isLoading ? "Loading diff summary" : emptyState.title}</p>
            <button type="button" className="diff-open-button" disabled>
              Details
            </button>
          </div>
          <div className="diff-empty-copy">
            <p>{emptyState.description}</p>
          </div>
        </>
      )}
      {workspace}
    </section>
  );
}
