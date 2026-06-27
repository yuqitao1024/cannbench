import type { SimtOperatorDiff, SimtOperatorVersionIndex } from "../types";

const DEV_API_HINT =
  "SIMT diff service is unavailable. Start `cannbench serve` or configure the frontend /api proxy.";

async function readJsonResponse<T>(response: Response, fallbackMessage: string): Promise<T> {
  const contentType = response.headers?.get?.("content-type") ?? "";
  const body = typeof response.text === "function" ? await response.text() : null;

  if (!response.ok) {
    throw new Error(body || fallbackMessage);
  }

  if (!body && typeof response.json === "function") {
    return (await response.json()) as T;
  }

  if (contentType.includes("application/json") && body !== null) {
    return JSON.parse(body) as T;
  }

  const trimmed = body?.trimStart().toLowerCase() ?? "";
  if (trimmed.startsWith("<!doctype") || trimmed.startsWith("<html")) {
    throw new Error(DEV_API_HINT);
  }

  try {
    if (body !== null) {
      return JSON.parse(body) as T;
    }
    if (typeof response.json === "function") {
      return (await response.json()) as T;
    }
  } catch {
    // fall through
  }

  throw new Error(fallbackMessage);
}

export async function fetchSimtOperatorVersions(
  operator: string,
  signal?: AbortSignal
): Promise<SimtOperatorVersionIndex> {
  const params = new URLSearchParams({ operator });
  const response = await fetch(`/api/simt-versions?${params.toString()}`, { signal });
  return await readJsonResponse<SimtOperatorVersionIndex>(
    response,
    `SIMT version request failed with status ${response.status}`
  );
}

export async function fetchSimtOperatorDiff(
  operator: string,
  baseVersion: string,
  compareVersion: string,
  signal?: AbortSignal
): Promise<SimtOperatorDiff> {
  const params = new URLSearchParams({
    operator,
    base_version: baseVersion,
    compare_version: compareVersion
  });
  const response = await fetch(`/api/simt-diff?${params.toString()}`, { signal });
  return await readJsonResponse<SimtOperatorDiff>(response, `SIMT diff request failed with status ${response.status}`);
}
