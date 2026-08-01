#!/usr/bin/env python3

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path


def eligible(record, iterations):
    return (
        record.get("validation_passed") is True
        and not record.get("error")
        and len(record.get("samples_us", [])) == iterations
    )


def mark_best_candidates(results):
    groups = defaultdict(list)
    contracts = {}
    for result in results:
        benchmark = result["benchmark"]
        environment_id = result.get("environment_id", "")
        group = (benchmark, environment_id)
        contract = (result["work_items"], result["warmup"], result["iterations"])
        if group in contracts and contracts[group] != contract:
            raise ValueError(
                f"inconsistent comparison contract for {benchmark!r} in {environment_id!r}"
            )
        contracts[group] = contract
        for record in result["records"]:
            record["is_best_candidate"] = False
            if eligible(record, result["iterations"]):
                groups[group].append(record)
    for records in groups.values():
        best = min(records, key=lambda record: record["median_us"])
        for record in records:
            record["is_best_candidate"] = (
                max(record["min_us"], best["min_us"])
                <= min(record["max_us"], best["max_us"])
            )


def write_csv(path, results):
    fieldnames = [
        "benchmark",
        "variant",
        "environment_id",
        "profile_path",
        "launch_bounds",
        "used_registers_per_thread",
        "stack_size_bytes",
        "grid_blocks",
        "block_threads",
        "work_items",
        "warmup",
        "iterations",
        "samples_us",
        "validation_passed",
        "error",
        "median_us",
        "min_us",
        "max_us",
        "is_best_candidate",
    ]
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            common = {name: result.get(name, "") for name in fieldnames}
            for record in result["records"]:
                row = dict(common)
                row.update(record)
                row["samples_us"] = ";".join(str(sample) for sample in record["samples_us"])
                writer.writerow({name: row.get(name, "") for name in fieldnames})


def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate Ascend occupancy benchmark results")
    parser.add_argument("--json-out", required=True)
    parser.add_argument("--csv-out", required=True)
    parser.add_argument("inputs", nargs="+")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        results = []
        for input_path in args.inputs:
            loaded = json.loads(Path(input_path).read_text(encoding="utf-8"))
            if not isinstance(loaded, dict) or not isinstance(loaded.get("records"), list):
                raise ValueError(f"invalid benchmark result: {input_path}")
            results.append(loaded)
        mark_best_candidates(results)
        Path(args.json_out).write_text(
            json.dumps({"schema_version": 1, "results": results}, indent=2) + "\n",
            encoding="utf-8",
        )
        write_csv(Path(args.csv_out), results)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        print(f"failed to aggregate occupancy results: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
