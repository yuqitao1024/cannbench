#!/usr/bin/env python3

import argparse
import csv
import math
from pathlib import Path
import sys


def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--kernel", required=True)
    parser.add_argument("--sample", required=True, type=int)
    parser.add_argument("--expected-block-dim", required=True, type=int)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def find_column(fields, exact, prefix):
    if exact in fields:
        return exact
    matches = [field for field in fields if field.lower().startswith(prefix.lower())]
    if len(matches) != 1:
        raise ValueError(f"required column {exact} not found")
    return matches[0]


def collect(raw, kernel):
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
            duration_key = find_column(fields, "Task Duration(us)", "Task Duration(")
            block_key = find_column(fields, "Block Dim", "Block Dim")
            frequency_key = find_column(fields, "Current Freq", "Current Freq")
            try:
                rated_frequency_key = find_column(fields, "Rated Freq", "Rated Freq")
            except ValueError as error:
                raise ValueError(
                    f"required rated frequency column missing in {path}"
                ) from error
            for row in reader:
                if row["Op Name"] == kernel:
                    selected.append((
                        row, path, duration_key, block_key, frequency_key,
                        rated_frequency_key,
                    ))
    if len(selected) != 1:
        raise ValueError(f"expected exactly 1 target row, observed {len(selected)}")
    return selected[0]


def main():
    args = arguments()
    row, source, duration_key, block_key, frequency_key, rated_frequency_key = collect(
        args.raw, args.kernel
    )
    duration = float(row[duration_key])
    if not math.isfinite(duration) or duration <= 0.0:
        raise ValueError(f"invalid Task Duration: {row[duration_key]}")
    block_dim = int(row[block_key])
    if block_dim != args.expected_block_dim:
        raise ValueError(f"expected block dimension {args.expected_block_dim}, observed {block_dim}")
    frequency = row[frequency_key].strip()
    if not frequency or not math.isfinite(float(frequency)) or float(frequency) <= 0.0:
        raise ValueError(f"invalid measured frequency: {frequency!r}")
    rated_frequency = row[rated_frequency_key].strip()
    if not rated_frequency or not math.isfinite(float(rated_frequency)) or float(rated_frequency) <= 0.0:
        raise ValueError(f"invalid rated frequency: {rated_frequency!r}")
    if float(frequency) != float(rated_frequency):
        raise ValueError(
            f"current/rated frequency mismatch: {frequency}/{rated_frequency}"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow((
            "sample", "kernel", "task_duration_us", "block_dim",
            "frequency_mhz", "rated_frequency_mhz", "source_csv",
        ))
        writer.writerow((
            args.sample, args.kernel, f"{duration:.6f}", block_dim,
            frequency, rated_frequency, source,
        ))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, csv.Error) as error:
        print(f"profile parse failed: {error}", file=sys.stderr)
        raise SystemExit(1)
