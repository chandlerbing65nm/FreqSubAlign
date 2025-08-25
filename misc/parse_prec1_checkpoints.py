#!/usr/bin/env python3
"""
Compare ViTTA logs side by side and print Prec@1_avg at specified sample checkpoints.

Default logs:
  vitta: /users/doloriel/work/Repo/ViTTA/logs/august_24_2025/output_12517305.txt
  adwt : /users/doloriel/work/Repo/ViTTA/logs/august_24_2025/output_12517308.txt

Usage:
  python misc/parse_prec1_checkpoints.py
  python misc/parse_prec1_checkpoints.py --logs log1.txt log2.txt ... [--names name1 name2 ...]

It extracts for lines like:
  2025-08-24 21:09:01,910 - 10 - test_time_adaptation.py - tta_standard - TTA Epoch1: [0/45392] ... Prec@1 0.000 (0.000) ...

For each corruption checkpoint (0-based counts):
  gauss 3782, pepper 7565, salt 11348, shot 15132, zoom 18193,
  impulse 22696, defocus 26479, motion 30260, jpeg 34043, contrast 37826,
  rain 41608, h265_abr 45391

The script prints only Prec@1_avg (the value in parentheses) at those steps, in a side-by-side
table across all provided logs. If the exact step isn't found, it falls back to the nearest previous
step silently. Finally, it prints the mean Prec@1_avg across all listed corruptions for each log.
"""

import argparse
import os
import re
import sys
from bisect import bisect_right
from typing import Dict, List, Tuple

# Default log paths (can be overridden via CLI)
DEFAULT_VITTA = "/users/doloriel/work/Repo/ViTTA/logs/august_24_2025/output_12517305.txt"
DEFAULT_ADWT = "/users/doloriel/work/Repo/ViTTA/logs/august_24_2025/output_12517308.txt"

# Checkpoints (0-based counts as they appear in: TTA Epoch1: [N/TOTAL])
CHECKPOINTS = {
    "gauss": 3782,
    "pepper": 7565,
    "salt": 11348,
    "shot": 15132,
    "zoom": 18193,
    "impulse": 22696,
    "defocus": 26479,
    "motion": 30260,
    "jpeg": 34043,
    "contrast": 37826,
    "rain": 41608,
    "h265_abr": 45391,
}

LINE_RE = re.compile(
    r"TTA\s+Epoch1:\s*\[(?P<step>\d+)/(?:\d+)\].*?Prec@1\s+(?P<inst>[0-9]*\.?[0-9]+)\s*\((?P<avg>[0-9]*\.?[0-9]+)\)",
)


def parse_log(path: str) -> Tuple[Dict[int, Tuple[float, float]], List[int]]:
    """Parse log file and return mapping: step -> (prec1_inst, prec1_avg) and sorted steps list."""
    if not os.path.isfile(path):
        print(f"Error: log file not found: {path}", file=sys.stderr)
        sys.exit(1)

    step_to_vals: Dict[int, Tuple[float, float]] = {}
    steps: List[int] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_RE.search(line)
            if not m:
                continue
            step = int(m.group("step"))
            inst = float(m.group("inst"))
            avg = float(m.group("avg"))
            # Keep the latest occurrence for a given step
            if step not in step_to_vals:
                steps.append(step)
            step_to_vals[step] = (inst, avg)

    steps.sort()
    if not steps:
        print("Error: No matching 'TTA Epoch1: [...] Prec@1 a (b)' lines found.", file=sys.stderr)
        sys.exit(2)

    return step_to_vals, steps


def find_at_or_before(target: int, sorted_steps: List[int]) -> int:
    """Return the step value at or before target using sorted list, or -1 if none."""
    idx = bisect_right(sorted_steps, target) - 1
    return sorted_steps[idx] if idx >= 0 else -1


def extract_prec1_avg_for_checkpoints(log_path: str) -> Dict[str, float]:
    """Return mapping corruption -> Prec@1_avg for the desired checkpoints (fallback to prior step)."""
    step_to_vals, steps_sorted = parse_log(log_path)
    result: Dict[str, float] = {}
    for name, target in CHECKPOINTS.items():
        matched = target if target in step_to_vals else find_at_or_before(target, steps_sorted)
        if matched == -1:
            continue  # no data before this target
        _inst, avg = step_to_vals[matched]
        result[name] = avg
    return result


def main():
    ap = argparse.ArgumentParser(description="Compare Prec@1_avg at specified checkpoints across multiple logs")
    ap.add_argument(
        "--logs",
        type=str,
        nargs="+",
        default=[DEFAULT_VITTA, DEFAULT_ADWT],
        help="Paths to one or more log files",
    )
    ap.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Optional column names; must match --logs count if provided",
    )
    args = ap.parse_args()

    logs: List[str] = args.logs
    if args.names is not None and len(args.names) != len(logs):
        print("Error: --names count must match --logs count", file=sys.stderr)
        sys.exit(3)

    names: List[str] = (
        args.names
        if args.names is not None
        else [os.path.basename(p) or p for p in logs]
    )

    # Extract per-log values
    perlog_values: List[Dict[str, float]] = []
    for p in logs:
        try:
            perlog_values.append(extract_prec1_avg_for_checkpoints(p))
        except SystemExit:
            # propagate existing error codes from parse_log
            raise
        except Exception as e:
            print(f"Error parsing {p}: {e}", file=sys.stderr)
            sys.exit(4)

    # Print side-by-side table of Prec@1_avg only
    widths = [12] + [max(8, len(n)) for n in names]
    header_line = f"{'corruption':<{widths[0]}}" + " " + " ".join(
        f"{name:>{widths[i+1]}}" for i, name in enumerate(names)
    )
    print(header_line)
    print("-" * len(header_line))

    # Rows per corruption in defined order
    for corr in CHECKPOINTS.keys():
        cells = []
        for vals in perlog_values:
            v = vals.get(corr, float("nan"))
            cells.append(f"{v:.4f}" if v == v else "NA")
        line = f"{corr:<{widths[0]}}" + " " + " ".join(
            f"{cells[i]:>{widths[i+1]}}" for i in range(len(cells))
        )
        print(line)

    # Means per log (over available corruptions)
    means = []
    for vals in perlog_values:
        arr = [x for x in vals.values() if x == x]
        means.append(sum(arr) / len(arr) if arr else float("nan"))

    mean_cells = [f"{m:.4f}" if m == m else "NA" for m in means]
    mean_line = f"{'mean':<{widths[0]}}" + " " + " ".join(
        f"{mean_cells[i]:>{widths[i+1]}}" for i in range(len(mean_cells))
    )
    print("-" * len(mean_line))
    print(mean_line)


if __name__ == "__main__":
    main()
