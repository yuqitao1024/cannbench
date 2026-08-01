import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def write_result(
    path: Path,
    variant: str,
    median: float,
    minimum: float,
    maximum: float,
    work_items: int = 393216,
) -> None:
    path.write_text(
        json.dumps(
            {
                "benchmark": "register_spill",
                "variant": variant,
                "environment_id": "tools2/cann-9.1",
                "profile_path": "/tmp/profiles",
                "launch_bounds": 1024 if variant == "lb1024" else 512,
                "used_registers_per_thread": 32 if variant == "lb1024" else 50,
                "stack_size_bytes": 40 if variant == "lb1024" else 0,
                "work_items": work_items,
                "warmup": 10,
                "iterations": 3,
                "records": [
                    {
                        "grid_blocks": 48,
                        "block_threads": 512,
                        "samples_us": [minimum, median, maximum],
                        "validation_passed": True,
                        "error": "",
                        "median_us": median,
                        "min_us": minimum,
                        "max_us": maximum,
                        "is_best_candidate": True,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )


def main() -> int:
    aggregator = Path(sys.argv[1])
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        spill = root / "spill.json"
        no_spill = root / "no-spill.json"
        summary_json = root / "summary.json"
        summary_csv = root / "summary.csv"
        write_result(spill, "lb1024", 9.0, 8.8, 9.2)
        write_result(no_spill, "lb512", 10.0, 9.8, 10.2)
        completed = subprocess.run(
            [
                sys.executable,
                str(aggregator),
                "--json-out",
                str(summary_json),
                "--csv-out",
                str(summary_csv),
                str(spill),
                str(no_spill),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            print(completed.stdout, completed.stderr)
            return 1
        summary = json.loads(summary_json.read_text(encoding="utf-8"))
        records = {
            result["variant"]: result["records"][0]
            for result in summary["results"]
        }
        if not records["lb1024"]["is_best_candidate"]:
            return 2
        if records["lb512"]["is_best_candidate"]:
            return 3
        with summary_csv.open(newline="", encoding="utf-8") as csv_file:
            rows = list(csv.DictReader(csv_file))
        if len(rows) != 2 or {row["variant"] for row in rows} != {"lb1024", "lb512"}:
            return 4

        write_result(no_spill, "lb512", 10.0, 9.8, 10.2, work_items=1)
        mismatch = subprocess.run(
            [
                sys.executable,
                str(aggregator),
                "--json-out",
                str(summary_json),
                "--csv-out",
                str(summary_csv),
                str(spill),
                str(no_spill),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if mismatch.returncode == 0 or "inconsistent comparison contract" not in mismatch.stderr:
            return 5
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
