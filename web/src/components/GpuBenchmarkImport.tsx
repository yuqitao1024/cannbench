import { useState } from "react";
import { createPortal } from "react-dom";
import { validateGpuBenchmarkUpload, type ValidationResult } from "../data/validation";

interface GpuBenchmarkImportProps {
  uploadEnabled: boolean;
  open: boolean;
  onClose: () => void;
}

const initialResult: ValidationResult = {
  ok: false,
  acceptedCount: 0,
  errors: [],
  warnings: []
};

export function GpuBenchmarkImport({ uploadEnabled, open, onClose }: GpuBenchmarkImportProps) {
  const [result, setResult] = useState<ValidationResult>(initialResult);
  const [jsonText, setJsonText] = useState("");
  const [submitting, setSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [savedPath, setSavedPath] = useState<string | null>(null);

  function validateText(value: string): void {
    setJsonText(value);
    setSubmitError(null);
    setSavedPath(null);
    if (!value.trim()) {
      setResult(initialResult);
      return;
    }
    if (value.length > 4 * 1024 * 1024) {
      setResult({
        ok: false,
        acceptedCount: 0,
        errors: ["JSON text must not exceed 4 MiB"],
        warnings: []
      });
      return;
    }

    try {
      setResult(validateGpuBenchmarkUpload(JSON.parse(value)));
    } catch {
      setResult({
        ok: false,
        acceptedCount: 0,
        errors: ["text must be valid JSON"],
        warnings: []
      });
    }
  }

  async function submitUpload(): Promise<void> {
    setSubmitting(true);
    setSubmitError(null);
    setSavedPath(null);
    try {
      const response = await fetch("/api/gpu-results", {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: jsonText
      });
      if (!response.ok) {
        const message = await response.text();
        throw new Error(message || `upload failed with HTTP ${response.status}`);
      }
      const payload = (await response.json()) as { path?: string };
      setSavedPath(payload.path ?? "server storage");
    } catch (error) {
      setSubmitError(error instanceof Error ? error.message : "upload failed");
    } finally {
      setSubmitting(false);
    }
  }

  if (!open) {
    return null;
  }

  return createPortal(
    <div className="modal-backdrop upload-backdrop" role="presentation">
      <section className="upload-panel" role="dialog" aria-modal="true" aria-label="GPU benchmark import">
        <header className="upload-header">
          <div>
            <p className="panel-kicker">Import GPU benchmark</p>
          </div>
          <button type="button" className="modal-close" aria-label="Close GPU benchmark import" onClick={onClose}>
            close
          </button>
        </header>
        <label className="json-input">
          <span>Paste GPU benchmark JSON</span>
          <textarea
            aria-label="Paste GPU benchmark JSON"
            value={jsonText}
            placeholder='{"records":[...]}'
            onChange={(event) => validateText(event.target.value)}
          />
        </label>
        {result.ok ? (
          <div className="validation-summary validation-summary--ok">
            <strong>{result.acceptedCount} records accepted locally</strong>
            <ul>
              {result.warnings.map((warning) => (
                <li key={warning}>{warning}</li>
              ))}
              <li>schema passed; sensitive fields not detected</li>
            </ul>
          </div>
        ) : result.errors.length > 0 ? (
          <div className="validation-summary validation-summary--blocked">
            <strong>Local validation failed</strong>
            <ul>
              {result.errors.map((error) => (
                <li key={error}>{error}</li>
              ))}
            </ul>
          </div>
        ) : (
          <div className="upload-warning validation-summary validation-summary--warning" role="alert">
            <strong>Warning</strong>
            <p>
              GPU benchmark data only. Never upload code, documents, environment details, employee IDs, or any
              sensitive content. Violations are your responsibility.
            </p>
          </div>
        )}
        {savedPath ? (
          <div className="validation-summary validation-summary--ok" role="status">
            saved to {savedPath}
          </div>
        ) : null}
        {submitError ? (
          <div className="validation-summary validation-summary--blocked" role="alert">
            {submitError}
          </div>
        ) : null}
        <button
          type="button"
          className="upload-button"
          disabled={!uploadEnabled || !result.ok || submitting}
          onClick={() => {
            void submitUpload();
          }}
        >
          {submitting ? "Submitting..." : "Submit"}
        </button>
      </section>
    </div>,
    document.body
  );
}
