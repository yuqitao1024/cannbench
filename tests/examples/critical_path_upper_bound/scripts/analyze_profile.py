#!/usr/bin/env python3

import argparse
import csv
import json
from pathlib import Path
import re


IDEAL_SPEEDUP = 4.0
RETENTION_GATE_PERCENT = 5.0
LANE_A_KERNELS = (
    "lane_a_stage0_kernel",
    "lane_a_stage1_kernel",
    "lane_a_stage2_kernel",
)
LANE_B_KERNELS = {
    0: "lane_b_candidate_baseline_kernel",
    1: "lane_b_candidate_counterfactual_kernel",
}
TIMING_PATTERN = re.compile(
    r"TIMING lane_a_event_ms=(?P<lane_a>[0-9.]+) "
    r"lane_b_event_ms=(?P<lane_b>[0-9.]+) "
    r"join_wall_ms=(?P<join>[0-9.]+)"
)


def load_profile_rows(raw_root: Path) -> list[dict[str, str]]:
    files = sorted(raw_root.rglob("OpBasicInfo*.csv"))
    if not files:
        raise SystemExit(f"no OpBasicInfo CSV files found under {raw_root}")
    rows = []
    for path in files:
        with path.open(newline="", encoding="utf-8-sig") as handle:
            rows.extend(csv.DictReader(handle))
    return rows


def matching_durations(rows: list[dict[str, str]], kernel: str) -> list[float]:
    durations = []
    for row in rows:
        if kernel not in row.get("Op Name", ""):
            continue
        try:
            durations.append(float(row["Task Duration(us)"]))
        except (KeyError, ValueError) as error:
            raise SystemExit(f"invalid Task Duration(us) for {kernel}: {error}") from error
    return durations


def load_timing(log_path: Path) -> dict[str, float]:
    match = TIMING_PATTERN.search(log_path.read_text(encoding="utf-8"))
    if match is None:
        raise SystemExit(f"missing TIMING line in {log_path}")
    return {
        "lane_a_event_us": float(match.group("lane_a")) * 1000.0,
        "lane_b_event_us": float(match.group("lane_b")) * 1000.0,
        "join_wall_us": float(match.group("join")) * 1000.0,
    }


def profile_command(args: argparse.Namespace) -> None:
    scenario = int(args.scenario)
    rows = load_profile_rows(Path(args.raw))
    lane_a_durations = []
    for kernel in LANE_A_KERNELS:
        matched = matching_durations(rows, kernel)
        if len(matched) != 1:
            raise SystemExit(f"expected one launch for {kernel}, found {len(matched)}")
        lane_a_durations.extend(matched)
    lane_b_kernel = LANE_B_KERNELS[scenario]
    lane_b_durations = matching_durations(rows, lane_b_kernel)
    if len(lane_b_durations) != 1:
        raise SystemExit(f"expected one launch for {lane_b_kernel}, found {len(lane_b_durations)}")

    lane_a_sum = sum(lane_a_durations)
    lane_b_sum = sum(lane_b_durations)
    critical_path = max(lane_a_sum, lane_b_sum)
    ideal_lane_b = lane_b_sum / IDEAL_SPEEDUP
    ideal_critical = max(lane_a_sum, ideal_lane_b)
    upper_bound = (
        (critical_path - ideal_critical) * 100.0 / critical_path
        if critical_path > 0.0
        else 0.0
    )
    result = {
        "scenario": scenario,
        "lane_a_launch_count": len(lane_a_durations),
        "lane_b_launch_count": len(lane_b_durations),
        "lane_a_sum_us": lane_a_sum,
        "lane_b_sum_us": lane_b_sum,
        "critical_path_us": critical_path,
        "additive_lane_sum_us_forbidden": lane_a_sum + lane_b_sum,
        "declared_ideal_speedup": IDEAL_SPEEDUP,
        "idealized_lane_b_us": ideal_lane_b,
        "idealized_critical_path_us": ideal_critical,
        "theoretical_upper_bound_percent": upper_bound,
        "retention_gate_percent": RETENTION_GATE_PERCENT,
        "retention_gate_passed": upper_bound >= RETENTION_GATE_PERCENT,
        **load_timing(Path(args.log)),
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def compare_command(args: argparse.Namespace) -> None:
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
    counterfactual = json.loads(Path(args.counterfactual).read_text(encoding="utf-8"))
    if baseline["scenario"] != 0 or counterfactual["scenario"] != 1:
        raise SystemExit("compare requires scenario 0 baseline and scenario 1 counterfactual")
    baseline_wall = baseline["join_wall_us"]
    counterfactual_wall = counterfactual["join_wall_us"]
    wall_change = (
        (baseline_wall - counterfactual_wall) * 100.0 / baseline_wall
        if baseline_wall > 0.0
        else 0.0
    )
    result = {
        "baseline_critical_path_us": baseline["critical_path_us"],
        "counterfactual_critical_path_us": counterfactual["critical_path_us"],
        "baseline_join_wall_us": baseline_wall,
        "counterfactual_join_wall_us": counterfactual_wall,
        "single_run_join_wall_change_percent": wall_change,
        "theoretical_upper_bound_percent": baseline["theoretical_upper_bound_percent"],
        "retention_gate_percent": baseline["retention_gate_percent"],
        "retention_gate_passed": baseline["retention_gate_passed"],
        "interpretation": "single-run observation; repeat before retention",
    }
    output = Path(args.output)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--raw", required=True)
    profile.add_argument("--log", required=True)
    profile.add_argument("--scenario", choices=("0", "1"), required=True)
    profile.add_argument("--output", required=True)
    profile.set_defaults(function=profile_command)
    compare = subparsers.add_parser("compare")
    compare.add_argument("--baseline", required=True)
    compare.add_argument("--counterfactual", required=True)
    compare.add_argument("--output", required=True)
    compare.set_defaults(function=compare_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
