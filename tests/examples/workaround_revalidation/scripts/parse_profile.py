#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
import sys


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--expected-launches", type=int, required=True)
    parser.add_argument("--rows-output", type=Path, required=True)
    parser.add_argument("--aggregate-output", type=Path, required=True)
    return parser.parse_args()


def collect_rows(raw_root, kernel):
    csv_paths = sorted(raw_root.rglob("OpBasicInfo*.csv"))
    if not csv_paths:
        raise ValueError(f"no OpBasicInfo CSV found under {raw_root}")

    selected = []
    for path in csv_paths:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames or []
            if "Op Name" not in fields:
                continue
            duration_key = "Task Duration(us)"
            if duration_key not in fields:
                if any(field.startswith("Task Duration(") for field in fields):
                    raise ValueError(
                        f"required column Task Duration(us) not found in {path}"
                    )
                continue
            for row in reader:
                if kernel not in row["Op Name"]:
                    continue
                duration = float(row[duration_key])
                if not math.isfinite(duration) or duration < 0.0:
                    raise ValueError(f"invalid Task Duration in {path}: {row[duration_key]}")
                selected.append((row["Op Name"], duration, str(path)))
    return selected


def write_results(selected, kernel, rows_output, aggregate_output):
    rows_output.parent.mkdir(parents=True, exist_ok=True)
    aggregate_output.parent.mkdir(parents=True, exist_ok=True)
    with rows_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("op_name", "task_duration_us", "source_csv"))
        for op_name, duration, source in selected:
            writer.writerow((op_name, f"{duration:.6f}", source))
    with aggregate_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("kernel", "observed_launches", "task_duration_sum_us"))
        writer.writerow((kernel, len(selected), f"{sum(row[1] for row in selected):.6f}"))


def main():
    args = parse_args()
    if args.expected_launches <= 0:
        raise ValueError("expected launches must be positive")
    selected = collect_rows(args.raw, args.kernel)
    if len(selected) != args.expected_launches:
        raise ValueError(
            f"expected {args.expected_launches} selected rows, observed {len(selected)}"
        )
    write_results(
        selected, args.kernel, args.rows_output, args.aggregate_output
    )
    print(
        f"kernel={args.kernel} observed_launches={len(selected)} "
        f"task_duration_sum_us={sum(row[1] for row in selected):.6f}"
    )


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error) as error:
        print(f"profile parse failed: {error}", file=sys.stderr)
        raise SystemExit(1)
