# Frontend and GPU Upload

CannBench includes a static frontend for inspecting published benchmark results. The frontend compares available runs by operator, dataset, implementation, version, and case.

## Data Loading

The frontend loads:

```text
published/index.json
published/<run-name>/meta/benchmark-records.json
```

It does not scan arbitrary directories. A run is visible only when its name is listed in `published/index.json`.

Published data must follow the contract in [../contracts/published-data-contract.md](../contracts/published-data-contract.md).

## Local Development

```bash
cd web
npm install
npm run dev
```

For production-style serving, build the frontend and use CannBench:

```bash
cd web
npm run build
cd ..
cannbench serve \
  --frontend-dir web/dist \
  --published-dir published
```

## Upload Flow

GPU benchmark upload exists for environments where the frontend service cannot directly SSH to the GPU benchmark machine.

Policy:

- Upload is disabled by default.
- Upload accepts only normalized GPU benchmark JSON.
- Upload must not accept code, diffs, logs, command output, environment dumps, hostnames, paths, tokens, employee IDs, or profiler raw files.
- Frontend validation blocks common accidental sensitive uploads.
- Backend validation is still required and is the final enforcement point.

Enable upload only when needed:

```bash
cannbench serve \
  --frontend-dir web/dist \
  --published-dir published \
  --enable-gpu-upload
```

Disable upload after import by restarting the service without `--enable-gpu-upload`.

## Accepted Uploaded Content

The uploaded payload must represent published benchmark records, not arbitrary run directories.

Required intent:

- `backend` must represent NVIDIA GPU data.
- Records must contain normalized performance metrics.
- Records must use the published benchmark record schema.
- Extra fields outside the allowed schema should be rejected.

The upload endpoint stores accepted payloads under the server-side published area. Operators should review and promote imported files into canonical `published/<run-name>/meta/benchmark-records.json` layout as needed.

## SIMT Diff View

The frontend can show diffs between Ascend SIMT operator versions. Diff content is read from this repository's operator source tree, not uploaded by users.

Expected SIMT source layout:

```text
src/cannbench/operators/builtin/<operator>/simt/<version>/
```

If an operator has fewer than two SIMT versions, the diff card is hidden.
