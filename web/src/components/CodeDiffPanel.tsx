import { useEffect, useState, type CSSProperties } from "react";
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

interface DiffFileView {
  id: string;
  path: string;
  shortPath: string;
  fileName: string;
  additions: number;
  deletions: number;
  file: ReturnType<typeof parseDiff>[number];
}

interface DiffTreeNode {
  name: string;
  children: Map<string, DiffTreeNode>;
  file?: DiffFileView;
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

function trimDiffPath(path: string, operator: string) {
  const marker = `/simt/${operator}/`;
  const markerIndex = path.indexOf(marker);
  if (markerIndex >= 0) {
    return path.slice(markerIndex + marker.length);
  }
  return path.split("/").slice(-4).join("/");
}

function countFileChanges(file: ReturnType<typeof parseDiff>[number]) {
  let additions = 0;
  let deletions = 0;
  for (const hunk of file.hunks) {
    for (const change of hunk.changes) {
      if (change.type === "insert") {
        additions += 1;
      }
      if (change.type === "delete") {
        deletions += 1;
      }
    }
  }
  return { additions, deletions };
}

function buildFileTree(files: DiffFileView[]) {
  const root: DiffTreeNode = { name: "", children: new Map() };
  for (const file of files) {
    const segments = file.shortPath.split("/");
    let current = root;
    for (const segment of segments) {
      const existing = current.children.get(segment);
      if (existing) {
        current = existing;
        continue;
      }
      const next: DiffTreeNode = { name: segment, children: new Map() };
      current.children.set(segment, next);
      current = next;
    }
    current.file = file;
  }
  return root;
}

function renderFileTree(node: DiffTreeNode, level = 0) {
  return Array.from(node.children.values()).map((child) => {
    if (child.file) {
      return (
        <button
          key={child.file.id}
          type="button"
          role="treeitem"
          className="diff-tree-file"
          style={{ "--tree-depth": level } as CSSProperties}
          onClick={() => document.getElementById(child.file?.id ?? "")?.scrollIntoView({ behavior: "smooth", block: "start" })}
        >
          {child.name}
        </button>
      );
    }

    return (
      <div key={`${level}-${child.name}`} className="diff-tree-branch">
        <span className="diff-tree-folder" style={{ "--tree-depth": level } as CSSProperties}>
          {child.name}
        </span>
        {renderFileTree(child, level + 1)}
      </div>
    );
  });
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
  const fileViews: DiffFileView[] = files.map((file, index) => {
    const path = file.newPath ?? file.oldPath;
    const shortPath = trimDiffPath(path, operator);
    const changes = countFileChanges(file);
    return {
      id: `diff-file-${index}`,
      path,
      shortPath,
      fileName: shortPath.split("/").at(-1) ?? shortPath,
      additions: changes.additions,
      deletions: changes.deletions,
      file
    };
  });
  const fileTree = buildFileTree(fileViews);

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
                  <div role="tree" aria-label="Changed files" className="diff-file-tree">
                    {renderFileTree(fileTree)}
                  </div>
                </aside>
                <div className="diff-workspace-content">
                  <p className="diff-mode-label">{mode === "split" ? "split diff" : "unified diff"}</p>
                  {fileViews.map((view) => (
                    <article key={view.path} id={view.id} className="diff-file">
                      <header>
                        <div>
                          <h4>
                            {view.fileName}{" "}
                            <span className="diff-file-plus">{view.additions}++</span>{" "}
                            <span className="diff-file-minus">{view.deletions}--</span>
                          </h4>
                          <p>{view.shortPath}</p>
                        </div>
                      </header>
                      <Diff viewType={mode} diffType={view.file.type} hunks={view.file.hunks} gutterType="default">
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

  if (!isLoadingVersions && !versionError && versionOptions.length < 2) {
    return null;
  }

  return (
    <section className={`diff-panel${diff ? "" : " diff-panel--empty"}`} aria-label="SIMT operator diff">
      <div className="diff-head">
        <p className="panel-kicker">SIMT operator diff</p>
      </div>
      <div className="diff-selector">
        <div className="diff-compare-command" role="group" aria-label="Selected SIMT versions">
          <span className="diff-command-label">compare</span>
          <button
            type="button"
            aria-label={`base version ${baseVersion ?? "not selected"}`}
            className={`diff-token diff-token--base${activeSlot === "base" ? " is-active" : ""}`}
            onClick={() => setActiveSlot("base")}
          >
            {baseVersion ?? "select"}
          </button>
          <span className="diff-command-arrow">-&gt;</span>
          <button
            type="button"
            aria-label={`compare version ${compareVersion ?? "not selected"}`}
            className={`diff-token diff-token--compare${activeSlot === "compare" ? " is-active" : ""}`}
            onClick={() => setActiveSlot("compare")}
          >
            {compareVersion ?? "select"}
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
