#!/usr/bin/env python3
"""
Compare ViTTA logs side by side and print Prec@1_avg at specified sample checkpoints.

Default logs:
  vitta: /users/doloriel/work/Repo/SWaveletA/logs/august_24_2025/output_12517305.txt
  adwt : /users/doloriel/work/Repo/SWaveletA/logs/august_24_2025/output_12517308.txt

Usage:
  python misc/parse_prec1_checkpoints.py
  python misc/parse_prec1_checkpoints.py --logs log1.txt log2.txt ... [--names name1 name2 ...]
  python misc/parse_prec1_checkpoints.py --dataset ucf   # use UCF-101 checkpoints (default)
  python misc/parse_prec1_checkpoints.py --dataset ssv2  # use Something-Something V2 checkpoints
  python misc/parse_prec1_checkpoints.py --continual /path/to/continual.txt  # derive checkpoints from file

It extracts for lines like:
  2025-08-24 21:09:01,910 - 10 - test_time_adaptation.py - tta_standard - TTA Epoch1: [0/45392] ... Prec@1 0.000 (0.000) ...

The script uses 1-based step indices internally (adds +1 to the parsed N). It prints only Prec@1_avg
(the value in parentheses) at those steps, in a side-by-side
table across all provided logs. If the exact step isn't found, it falls back to the nearest previous
step silently. Finally, it prints the mean Prec@1_avg across all listed corruptions for each log.
"""

import argparse
import os
import re
import sys
from bisect import bisect_right
from typing import Dict, List, Tuple

# # tanet - ucf
# DEFAULT_VITTA = "/users/doloriel/work/Repo/SWaveletA/logs/august_25_2025/output_12528532.txt"
# DEFAULT_SWA = "/users/doloriel/work/Repo/SWaveletA/logs/august_25_2025/output_12528553.txt"

# # tanet - ssv2
# DEFAULT_VITTA = "/users/doloriel/work/Repo/SWaveletA/logs/output_12537841.txt"
# DEFAULT_SWA = "/users/doloriel/work/Repo/SWaveletA/logs/output_12537879.txt"

# # videoswin - ucf
# DEFAULT_VITTA = "/users/doloriel/work/Repo/SWaveletA/logs/output_12540046.txt"
# DEFAULT_SWA = "/users/doloriel/work/Repo/SWaveletA/logs/output_12540133.txt"

# videoswin - ss2
DEFAULT_VITTA = ""
DEFAULT_SWA = ""

# Auto-selection map for default logs based on (arch, dataset)
# Keys are (arch, dataset) -> [vitta_log, swa_log]

# LL+LH+HL+HH
# --names noadapt vitta swa_adapt swa_noll swa_nolh swa_nohl swa_nohh swa_l1 swa_l2 swa_3d
DEFAULT_LOGS_BY_COMBO = {
    ("tanet", "ucf"): [
        '/scratch/project_465001897/datasets/ucf/results/source/tanet_ucf101/source_continual/20250827_213352_baseline=source_corruption=continual_bs1',
        "/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250825_203523_adaptepoch=1_views2_corruption=continual_bs1",
        '/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250828_202059_adaptepoch=1_views2_dwtAlign2D-L1_adaptive_LL1.0+LH1.0+HL1.0+HH1.0_corruption=continual_bs1',
        '/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250827_145310_adaptepoch=1_views2_dwtAlign-L1_LH1.0+HL1.0+HH1.0_corruption=continual_bs1',
        '/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250828_181605_adaptepoch=1_views2_dwtAlign2D-L1_LL1.0+HL1.0+HH1.0_corruption=continual_bs1',
        '/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250828_181748_adaptepoch=1_views2_dwtAlign2D-L1_LL1.0+LH1.0+HH1.0_corruption=continual_bs1',
        '/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250828_182309_adaptepoch=1_views2_dwtAlign2D-L1_LL1.0+LH1.0+HL1.0_corruption=continual_bs1',
        "/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250825_204203_adaptepoch=1_views2_dwtAlign-L1_LL1.0+LH1.0+HL1.0+HH1.0_corruption=continual_bs1",
        # '/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250828_174354_adaptepoch=1_views2_dwtAlign2D-L2_LL1.0+LH1.0+HL1.0+HH1.0_corruption=continual_bs1',
        # '/scratch/project_465001897/datasets/ucf/results/corruptions/tanet_ucf101/tta_continual/20250828_173254_adaptepoch=1_views2_dwtAlign3D-L1_LL1.0+LH1.0+HL1.0+HH1.0_corruption=continual_bs1',
    ],
    ("tanet", "ssv2"): [
        '/scratch/project_465001897/datasets/ss2/results/source/tanet_somethingv2/source_continual/20250827_214606_baseline=source_corruption=continual_bs1',
        "/scratch/project_465001897/datasets/ss2/results/corruptions/tanet_somethingv2/tta_continual/20250826_175113_adaptepoch=1_views2_corruption=continual_bs1",
        "/scratch/project_465001897/datasets/ss2/results/corruptions/tanet_somethingv2/tta_continual/20250827_132717_adaptepoch=1_views2_dwtAlign-L1_LH1.0+HL1.0+HH1.0_corruption=continual_bs1",
        "/scratch/project_465001897/datasets/ss2/results/corruptions/tanet_somethingv2/tta_continual/20250826_175226_adaptepoch=1_views2_dwtAlign-L1_LL1.0+LH1.0+HL1.0+HH1.0_corruption=continual_bs1",
    ],
    ("videoswin", "ucf"): [
        '/scratch/project_465001897/datasets/ucf/results/source/videoswintransformer_ucf101/source_continual/20250827_213933_baseline=source_corruption=continual_bs1',
        "/scratch/project_465001897/datasets/ucf/results/corruptions/videoswintransformer_ucf101/tta_continual/20250826_192000_adaptepoch=1_views2_corruption=continual_bs1",
        "/scratch/project_465001897/datasets/ucf/results/corruptions/videoswintransformer_ucf101/tta_continual/20250827_145802_adaptepoch=1_views2_dwtAlign-L1_LH1.0+HL1.0+HH1.0_corruption=continual_bs1",
        "/scratch/project_465001897/datasets/ucf/results/corruptions/videoswintransformer_ucf101/tta_continual/20250826_192226_adaptepoch=1_views2_dwtAlign-L1_LL1.0+LH1.0+HL1.0+HH1.0_corruption=continual_bs1",
    ],
    ("videoswin", "ssv2"): [
        '/scratch/project_465001897/datasets/ss2/results/source/videoswintransformer_somethingv2/source_continual/20250827_214451_baseline=source_corruption=continual_bs1',
        "/scratch/project_465001897/datasets/ss2/results/corruptions/videoswintransformer_somethingv2/tta_continual/20250826_194947_adaptepoch=1_views2_corruption=continual_bs1",
        "/scratch/project_465001897/datasets/ss2/results/corruptions/videoswintransformer_somethingv2/tta_continual/20250827_114943_adaptepoch=1_views2_dwtAlign-L1_LH1.0+HL1.0+HH1.0_corruption=continual_bs1",
        "/scratch/project_465001897/datasets/ss2/results/corruptions/videoswintransformer_somethingv2/tta_continual/20250826_234019_adaptepoch=1_views2_dwtAlign-L1_LL1.0+LH1.0+HL1.0+HH1.0_corruption=continual_bs1",
    ],
}

DEFAULT_CONTINUAL_UCF = "/scratch/project_465001897/datasets/ucf/list_video_perturbations/continual.txt"
DEFAULT_CONTINUAL_SSV2 = "/scratch/project_465001897/datasets/ss2/list_video_perturbations/continual.txt"

# Corruption order used throughout
CORR_NAMES: List[str] = [
    "gauss",
    "pepper",
    "salt",
    "shot",
    "zoom",
    "impulse",
    "defocus",
    "motion",
    "jpeg",
    "contrast",
    "rain",
    "h265_abr",
]

# Checkpoints (originally noted as 0-based counts in logs; parser now uses 1-based internally)
# Top block (smaller totals) corresponds to UCF-101.
CHECKPOINTS_UCF: Dict[str, int] = {
    'gauss': 316,
    'pepper': 632,
    'salt': 948,
    'shot': 1263,
    'zoom': 1578,
    'impulse': 1893,
    'defocus': 2208,
    'motion': 2523,
    'jpeg': 2838,
    'contrast': 3153,
    'rain': 3468,
    'h265_abr': 3783,
}

# Bottom block (larger totals) corresponds to Something-Something V2 (SSv2).
CHECKPOINTS_SSV2: Dict[str, int] = {
    'gauss': 2065,
    'pepper': 4130,
    'salt': 6195,
    'shot': 8260,
    'zoom': 10325,
    'impulse': 12390,
    'defocus': 14455,
    'motion': 16520,
    'jpeg': 18584,
    'contrast': 20648,
    'rain': 22712,
    'h265_abr': 24776,
}

LINE_RE = re.compile(
    r"TTA\s+Epoch1:\s*\[(?P<step>\d+)/(?:\d+)\].*?Prec@1\s+(?P<inst>[0-9]*\.?[0-9]+)\s*\((?P<avg>[0-9]*\.?[0-9]+)\)",
)

# Combined regex to capture both Prec@1 and Prec@5 (instantaneous and average) on the same line
LINE_BOTH_RE = re.compile(
    r"TTA\s+Epoch1:\s*\[(?P<step>\d+)/(?:\d+)\].*?"
    r"Prec@1\s+(?P<inst1>[0-9]*\.?[0-9]+)\s*\((?P<avg1>[0-9]*\.?[0-9]+)\).*?"
    r"Prec@5\s+(?P<inst5>[0-9]*\.?[0-9]+)\s*\((?P<avg5>[0-9]*\.?[0-9]+)\)"
)

# Source-only baseline logs use a different prefix (no "TTA Epoch1"), e.g.:
#   "... training.py - validate - Test: [i/N] ... Prec@1 a (b) ... Prec@5 c (d)"
LINE_SRC_RE = re.compile(
    r"validate\s*-\s*Test:\s*\[(?P<step>\d+)/(?:\d+)\].*?Prec@1\s+(?P<inst>[0-9]*\.?[0-9]+)\s*\((?P<avg>[0-9]*\.?[0-9]+)\)",
)
LINE_BOTH_SRC_RE = re.compile(
    r"validate\s*-\s*Test:\s*\[(?P<step>\d+)/(?:\d+)\].*?"
    r"Prec@1\s+(?P<inst1>[0-9]*\.?[0-9]+)\s*\((?P<avg1>[0-9]*\.?[0-9]+)\).*?"
    r"Prec@5\s+(?P<inst5>[0-9]*\.?[0-9]+)\s*\((?P<avg5>[0-9]*\.?[0-9]+)\)"
)

# New regex for the provided log format
LINE_VALIDATE_BRIEF_RE = re.compile(
    r"validate_brief\s*-\s*Test Epoch \d+:\s*\[(?P<step>\d+)/(?:\d+)\].*?Prec@1\s+(?P<inst>[0-9]*\.?[0-9]+)\s*\((?P<avg>[0-9]*\.?[0-9]+)\)",
)
LINE_BOTH_VALIDATE_BRIEF_RE = re.compile(
    r"validate_brief\s*-\s*Test Epoch \d+:\s*\[(?P<step>\d+)/(?:\d+)\].*?"
    r"Prec@1\s+(?P<inst1>[0-9]*\.?[0-9]+)\s*\((?P<avg1>[0-9]*\.?[0-9]+)\).*?"
    r"Prec@5\s+(?P<inst5>[0-9]*\.?[0-9]+)\s*\((?P<avg5>[0-9]*\.?[0-9]+)\)"
)

# Regex for test log format like: "Test: [0/3783]\tTime 7.950 (7.950)\tPrec@1 100.000 (100.000)\tPrec@5 100.000 (100.000)"
TEST_LOG_RE = re.compile(
    r"Test:\s*\[(?P<step>\d+)/(?:\d+)\].*?Prec@1\s+(?P<inst>[0-9]*\.?[0-9]+)\s*\((?P<avg>[0-9]*\.?[0-9]+)\)"
)
TEST_LOG_BOTH_RE = re.compile(
    r"Test:\s*\[(?P<step>\d+)/(?:\d+)\].*?"
    r"Prec@1\s+(?P<inst1>[0-9]*\.?[0-9]+)\s*\((?P<avg1>[0-9]*\.?[0-9]+)\).*?"
    r"Prec@5\s+(?P<inst5>[0-9]*\.?[0-9]+)\s*\((?P<avg5>[0-9]*\.?[0-9]+)\)"
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
                m = LINE_SRC_RE.search(line)
            if not m:
                m = LINE_VALIDATE_BRIEF_RE.search(line)
            if not m:
                m = TEST_LOG_RE.search(line)  # Add test log format check
            if not m:
                continue
            # Use 1-based steps internally so the first record is step=1
            step = int(m.group("step")) + 1
            inst = float(m.group("inst"))
            avg = float(m.group("avg"))
            # Keep the latest occurrence for a given step
            if step not in step_to_vals:
                steps.append(step)
            step_to_vals[step] = (inst, avg)

    steps.sort()
    if not steps:
        print("Error: No matching lines found for either TTA ('TTA Epoch1: [...]') or source-only ('validate - Test: [...]').",
              file=sys.stderr)
        sys.exit(2)

    return step_to_vals, steps


def parse_log_both(path: str) -> Tuple[
    Dict[int, Tuple[float, float]],
    Dict[int, Tuple[float, float]],
    List[int],
]:
    """Parse log file and return two mappings:
    - step_to_p1: step -> (prec1_inst, prec1_avg)
    - step_to_p5: step -> (prec5_inst, prec5_avg)
    and the sorted list of steps.
    """
    if not os.path.isfile(path):
        print(f"Error: log file not found: {path}", file=sys.stderr)
        sys.exit(1)

    step_to_p1: Dict[int, Tuple[float, float]] = {}
    step_to_p5: Dict[int, Tuple[float, float]] = {}
    steps: List[int] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = LINE_BOTH_RE.search(line)
            if not m:
                m = LINE_BOTH_SRC_RE.search(line)
            if not m:
                m = LINE_BOTH_VALIDATE_BRIEF_RE.search(line)
            if not m:
                m = TEST_LOG_BOTH_RE.search(line)  # Add test log format check
            if not m:
                continue
            step = int(m.group("step")) + 1  # 1-based
            inst1 = float(m.group("inst1"))
            avg1 = float(m.group("avg1"))
            inst5 = float(m.group("inst5"))
            avg5 = float(m.group("avg5"))
            if step not in step_to_p1:
                steps.append(step)
            step_to_p1[step] = (inst1, avg1)
            step_to_p5[step] = (inst5, avg5)

    steps.sort()
    if not steps:
        print("Error: No matching lines found for either TTA ('TTA Epoch1: [...]') or source-only ('validate - Test: [...]') with Prec@1 and Prec@5.",
              file=sys.stderr)
        sys.exit(2)

    return step_to_p1, step_to_p5, steps


def infer_checkpoints_from_continual(path: str) -> Dict[str, int]:
    """Infer cumulative checkpoints from a continual.txt file.

    Expected line format examples:
      ucf:  gauss/HorseRiding/v_HorseRiding_g04_c01.mp4 201 41
      ssv2: gauss/127/116154 52 127
    We only need the prefix before the first '/', which is the corruption name.
    Returns a dict mapping corruption -> cumulative count (1-based compatible).
    """
    counts = {k: 0 for k in CORR_NAMES}
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                first = line.split()[0]
                head = first.split("/", 1)[0].lower()
                if head in counts:
                    counts[head] += 1
    except FileNotFoundError:
        raise
    # Turn per-corruption counts into cumulative checkpoints in fixed order
    cum = 0
    checkpoints: Dict[str, int] = {}
    for name in CORR_NAMES:
        cum += counts[name]
        checkpoints[name] = cum
    return checkpoints


def find_at_or_before(target: int, sorted_steps: List[int]) -> int:
    """Return the step value at or before target using sorted list, or -1 if none."""
    idx = bisect_right(sorted_steps, target) - 1
    return sorted_steps[idx] if idx >= 0 else -1


def extract_prec1_avg_for_checkpoints(log_path: str, checkpoints: Dict[str, int]) -> Dict[str, float]:
    """Return mapping corruption -> Prec@1_avg for the desired checkpoints (fallback to prior step)."""
    step_to_vals, steps_sorted = parse_log(log_path)
    result: Dict[str, float] = {}
    for name, target in checkpoints.items():
        # Prefer exact 1-based match; if missing, try target+1 (tolerate 0-based maps), then fallback
        if target in step_to_vals:
            matched = target
        elif (target + 1) in step_to_vals:
            matched = target + 1
        else:
            matched = find_at_or_before(target, steps_sorted)
        if matched == -1:
            continue  # no data before this target
        _inst, avg = step_to_vals[matched]
        result[name] = avg
    return result


def extract_prec5_avg_for_checkpoints(log_path: str, checkpoints: Dict[str, int]) -> Dict[str, float]:
    """Return mapping corruption -> Prec@5_avg for the desired checkpoints (fallback to prior step)."""
    _p1_map, p5_map, steps_sorted = parse_log_both(log_path)
    result: Dict[str, float] = {}
    for name, target in checkpoints.items():
        if target in p5_map:
            matched = target
        elif (target + 1) in p5_map:
            matched = target + 1
        else:
            matched = find_at_or_before(target, steps_sorted)
        if matched == -1:
            continue
        _inst, avg = p5_map[matched]
        result[name] = avg
    return result


def main():
    ap = argparse.ArgumentParser(description="Compare Prec@1_avg at specified checkpoints across multiple logs")
    ap.add_argument(
        "--logs",
        type=str,
        nargs="+",
        default=None,
        help="Paths to one or more log files. If omitted and --arch is provided, logs are auto-selected by (arch,dataset).",
    )
    ap.add_argument(
        "--names",
        type=str,
        nargs="+",
        default=None,
        help="Optional column names; must match --logs count if provided",
    )
    # Allow selecting which dataset checkpoint schedule to use. Keep --daset as alias.
    ap.add_argument(
        "--dataset",
        "--daset",
        dest="dataset",
        type=str,
        choices=["ucf", "ssv2"],
        default="ucf",
        help="Select checkpoint set: 'ucf' (default) or 'ssv2'",
    )
    # Optional architecture selector used for auto-choosing default logs when --logs is omitted
    ap.add_argument(
        "--arch",
        type=str,
        choices=["tanet", "videoswin"],
        default=None,
        help="Model architecture. When set and --logs is omitted, picks default logs for (arch,dataset)",
    )
    # Optional continual file to derive checkpoints from; if omitted, auto-detect per dataset
    ap.add_argument(
        "--continual",
        type=str,
        default=None,
        help="Path to continual.txt to derive checkpoints. If omitted, tries dataset-specific default.",
    )
    ap.add_argument(
        "--print-checkpoints",
        action="store_true",
        help="Print the resolved checkpoint map and exit",
    )
    args = ap.parse_args()

    # Determine which logs to parse
    logs: List[str]
    if args.logs is not None:
        logs = args.logs
    else:
        if args.arch is not None:
            key = (args.arch, args.dataset)
            if key in DEFAULT_LOGS_BY_COMBO:
                logs = DEFAULT_LOGS_BY_COMBO[key]
                print(f"Auto-selected logs for arch={args.arch}, dataset={args.dataset}", file=sys.stderr)
            else:
                print(
                    f"Error: no default logs configured for arch={args.arch}, dataset={args.dataset}. Please provide --logs.",
                    file=sys.stderr,
                )
                sys.exit(5)
        else:
            # Legacy fallback to the module-level defaults
            logs = [DEFAULT_VITTA, DEFAULT_SWA]
            print(
                "Using legacy default logs (no --logs and no --arch provided); consider passing --arch for auto-selection.",
                file=sys.stderr,
            )
    if args.names is not None and len(args.names) != len(logs):
        print("Error: --names count must match --logs count", file=sys.stderr)
        sys.exit(3)

    names: List[str] = (
        args.names
        if args.names is not None
        else [os.path.basename(p) or p for p in logs]
    )

    # Choose checkpoint map: prefer continual-derived if available, else fallback to hardcoded by dataset
    checkpoints: Dict[str, int]
    continual_path = args.continual
    if continual_path is None:
        continual_path = DEFAULT_CONTINUAL_UCF if args.dataset == "ucf" else DEFAULT_CONTINUAL_SSV2
    if continual_path and os.path.isfile(continual_path):
        try:
            checkpoints = infer_checkpoints_from_continual(continual_path)
            print(f"Using checkpoints inferred from: {continual_path}", file=sys.stderr)
        except Exception as e:
            print(f"Warning: failed to infer checkpoints from {continual_path}: {e}. Falling back to hardcoded.", file=sys.stderr)
            checkpoints = CHECKPOINTS_UCF if args.dataset == "ucf" else CHECKPOINTS_SSV2
    else:
        checkpoints = CHECKPOINTS_UCF if args.dataset == "ucf" else CHECKPOINTS_SSV2

    if args.print_checkpoints:
        # Print in canonical order
        for name in CORR_NAMES:
            if name in checkpoints:
                print(f"{name}: {checkpoints[name]}")
        return

    # Extract per-log values (Top-1 and Top-5)
    perlog_p1: List[Dict[str, float]] = []
    perlog_p5: List[Dict[str, float]] = []
    for p in logs:
        try:
            perlog_p1.append(extract_prec1_avg_for_checkpoints(p, checkpoints))
            perlog_p5.append(extract_prec5_avg_for_checkpoints(p, checkpoints))
        except SystemExit:
            raise
        except Exception as e:
            print(f"Error parsing {p}: {e}", file=sys.stderr)
            sys.exit(4)

    # Common formatting
    max_corr_len = max(len(c) for c in checkpoints.keys())
    col_width = 15

    # --- Top-1 (Prec@1_avg) ---
    header_line = f"{'corruption':<{max_corr_len}}"
    for name in names:
        header_line += f" {name:>{col_width}}"
    print(header_line)
    print("-" * len(header_line))
    for corr in checkpoints.keys():
        line = f"{corr:<{max_corr_len}}"
        for vals in perlog_p1:
            v = vals.get(corr, float("nan"))
            val_str = f"{v:.4f}" if v == v else "NA"
            line += f" {val_str:>{col_width}}"
        print(line)
    means = []
    for vals in perlog_p1:
        arr = [x for x in vals.values() if x == x]
        means.append(sum(arr) / len(arr) if arr else float("nan"))
    mean_line = f"{'mean':<{max_corr_len}}"
    for m in means:
        mean_str = f"{m:.4f}" if m == m else "NA"
        mean_line += f" {mean_str:>{col_width}}"
    print("-" * len(header_line))
    print(mean_line)

    # Blank line between tables
    print()

    # --- Top-5 (Prec@5_avg) ---
    header_line5 = f"{'corruption':<{max_corr_len}}"
    for name in names:
        header_line5 += f" {name:>{col_width}}"
    print(header_line5)
    print("-" * len(header_line5))
    for corr in checkpoints.keys():
        line = f"{corr:<{max_corr_len}}"
        for vals in perlog_p5:
            v = vals.get(corr, float("nan"))
            val_str = f"{v:.4f}" if v == v else "NA"
            line += f" {val_str:>{col_width}}"
        print(line)
    means5 = []
    for vals in perlog_p5:
        arr = [x for x in vals.values() if x == x]
        means5.append(sum(arr) / len(arr) if arr else float("nan"))
    mean_line5 = f"{'mean':<{max_corr_len}}"
    for m in means5:
        mean_str = f"{m:.4f}" if m == m else "NA"
        mean_line5 += f" {mean_str:>{col_width}}"
    print("-" * len(header_line5))
    print(mean_line5)


if __name__ == "__main__":
    main()
