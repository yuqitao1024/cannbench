#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path
import sys


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--expected-launches", required=True, type=int)
    parser.add_argument("--expected-block-dim", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def collect(raw: Path, kernel: str, expected_block_dim: int):
    paths = sorted(raw.rglob("OpBasicInfo*.csv"))
    if not paths:
        raise ValueError(f"no OpBasicInfo CSV found under {raw}")
    selected = []
    for path in paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "Op Name" not in fields:
                continue
            if "Task Duration(us)" not in fields:
                if any(field.startswith("Task Duration(") for field in fields):
                    raise ValueError(f"required Task Duration(us) column missing in {path}")
                continue
            block_key = next((key for key in ("Block Dim", "BlockDim") if key in fields), None)
            if block_key is None:
                raise ValueError(f"required Block Dim column missing in {path}")
            current_key = next((key for key in (
                "Current Freq", "Execution Time Current Frequency(MHz)") if key in fields), None)
            rated_key = next((key for key in (
                "Rated Freq", "Execution Time Rated Frequency(MHz)") if key in fields), None)
            if current_key is None:
                raise ValueError(f"required current frequency column missing in {path}")
            if rated_key is None:
                raise ValueError(f"required rated frequency column missing in {path}")
            for row in reader:
                op_name = row["Op Name"].strip()
                if op_name != kernel:
                    continue
                duration = float(row["Task Duration(us)"])
                block_dim = int(row[block_key])
                frequency = float(row[current_key])
                rated = float(row[rated_key])
                if not all(math.isfinite(value) and value > 0 for value in (duration, frequency, rated)):
                    raise ValueError(f"invalid numeric target row in {path}")
                if block_dim != expected_block_dim:
                    raise ValueError(
                        f"expected block dimension {expected_block_dim}, observed {block_dim}")
                selected.append({
                    "op_name": op_name,
                    "task_duration_us": duration,
                    "block_dim": block_dim,
                    "frequency_mhz": frequency,
                    "rated_frequency_mhz": rated,
                    "source_csv": str(path),
                })
    return selected


def main():
    args = arguments()
    rows = collect(args.raw, args.kernel, args.expected_block_dim)
    if len(rows) != args.expected_launches:
        raise ValueError(
            f"expected {args.expected_launches} exact target row, observed {len(rows)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(rows[0], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(rows[0], sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error) as error:
        print(f"profile parse failed: {error}", file=sys.stderr)
        raise SystemExit(1)
