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
  warnings: ["Paste normalized GPU benchmark JSON to validate locally."]
};

export function GpuBenchmarkImport({ uploadEnabled, open, onClose }: GpuBenchmarkImportProps) {
  const [result, setResult] = useState<ValidationResult>(initialResult);
  const [jsonText, setJsonText] = useState("");

  function validateText(value: string): void {
    setJsonText(value);
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

  if (!open) {
    return null;
  }

  return createPortal(
    <div className="modal-backdrop upload-backdrop" role="presentation">
      <section className="upload-panel" role="dialog" aria-modal="true" aria-label="GPU benchmark import">
        <header className="upload-header">
          <div>
            <p className="panel-kicker">Import GPU benchmark</p>
            <h3>GPU Benchmark Import</h3>
            <p className="upload-policy">
              {uploadEnabled
                ? "Upload enabled by server policy. Backend validation is still required."
                : "Upload disabled by server policy."}
            </p>
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
        <div className={`validation-summary validation-summary--${result.ok ? "ok" : "blocked"}`}>
          <strong>{result.ok ? `${result.acceptedCount} records accepted locally` : "Local validation required"}</strong>
          <ul>
            {result.errors.map((error) => (
              <li key={error}>{error}</li>
            ))}
            {result.warnings.map((warning) => (
              <li key={warning}>{warning}</li>
            ))}
            {result.ok && <li>schema passed; sensitive fields not detected</li>}
          </ul>
        </div>
        <button type="button" className="upload-button" disabled={!uploadEnabled || !result.ok}>
          Submit JSON
        </button>
      </section>
    </div>,
    document.body
  );
}
