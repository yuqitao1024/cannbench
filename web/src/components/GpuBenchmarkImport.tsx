import { useState } from "react";
import { validateGpuBenchmarkUpload, type ValidationResult } from "../data/validation";

interface GpuBenchmarkImportProps {
  uploadEnabled: boolean;
}

const initialResult: ValidationResult = {
  ok: false,
  acceptedCount: 0,
  errors: [],
  warnings: ["Select a normalized GPU benchmark JSON file to validate locally."]
};

export function GpuBenchmarkImport({ uploadEnabled }: GpuBenchmarkImportProps) {
  const [result, setResult] = useState<ValidationResult>(initialResult);

  async function handleFile(file: File | undefined): Promise<void> {
    if (!file) {
      setResult(initialResult);
      return;
    }
    if (file.size > 4 * 1024 * 1024) {
      setResult({
        ok: false,
        acceptedCount: 0,
        errors: ["file size must not exceed 4 MiB"],
        warnings: []
      });
      return;
    }

    try {
      setResult(validateGpuBenchmarkUpload(JSON.parse(await file.text())));
    } catch {
      setResult({
        ok: false,
        acceptedCount: 0,
        errors: ["file must be valid JSON"],
        warnings: []
      });
    }
  }

  return (
    <section className="upload-panel" aria-label="GPU benchmark import">
      <div>
        <p className="panel-kicker">Import GPU benchmark</p>
        <h3>GPU Benchmark Import</h3>
        <p className="upload-policy">
          {uploadEnabled
            ? "Upload enabled by server policy. Backend validation is still required."
            : "Upload disabled by server policy."}
        </p>
      </div>
      <label className="file-input">
        <span>Select GPU benchmark JSON</span>
        <input
          type="file"
          accept="application/json,.json"
          aria-label="Select GPU benchmark JSON"
          onChange={(event) => void handleFile(event.target.files?.[0])}
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
        Upload
      </button>
    </section>
  );
}
