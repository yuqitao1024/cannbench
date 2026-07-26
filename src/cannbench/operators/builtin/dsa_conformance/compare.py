from __future__ import annotations

import argparse
import json
from pathlib import Path

from .artifact import compare_artifacts, conformance_passed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--atol", type=float, default=0.05)
    parser.add_argument("--rtol", type=float, default=0.05)
    parser.add_argument("--min-indexer-recall", type=float, default=0.95)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    result = compare_artifacts(
        args.reference,
        args.candidate,
        atol=args.atol,
        rtol=args.rtol,
    )
    result["passed"] = conformance_passed(
        result, min_indexer_recall=args.min_indexer_recall
    )
    rendered = json.dumps(result, indent=2, sort_keys=True)
    print(rendered, flush=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
