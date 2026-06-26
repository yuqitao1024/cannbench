import type { SimtOperatorDiff } from "../types";

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
  if (!response.ok) {
    const message = await response.text();
    throw new Error(message || `SIMT diff request failed with status ${response.status}`);
  }
  return (await response.json()) as SimtOperatorDiff;
}
