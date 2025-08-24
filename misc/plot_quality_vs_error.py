#!/usr/bin/env python3
import argparse
import os
import re
import sys
from collections import defaultdict, OrderedDict
from datetime import datetime
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

# Patterns
CORRUPTION_HEADER_RE = re.compile(r"^####\s*Starting Evaluation for :::\s*(.+?)\s*corruption\s*####\s*$")
PREC_RE = re.compile(r"Prec@1\s+([0-9]*\.?[0-9]+)\s*\([0-9]*\.?[0-9]+\)")
SS_SLOPE_RE = re.compile(r"SS-slope\s+([+-]?(?:\d+\.\d+|\d+|\.\d+)|nan|NaN|NAN)")
DEVA2_RE = re.compile(r"Dev\|a-2\|\s+([+-]?(?:\d+\.\d+|\d+|\.\d+)|nan|NaN|NAN)")


def parse_args():
    p = argparse.ArgumentParser(description="Plot SS-slope/Dev|a-2| vs Top-1 error from ViTTA logs")
    p.add_argument("--logs_dir", type=str, default="./logs", help="Directory containing log files")
    p.add_argument("--file", type=str, default=None, help="Specific log file to parse (overrides --logs_dir)")
    p.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output prefix (PNG). Two files will be created: <prefix>_ss_vs_error.png and "
            "<prefix>_deva2_vs_error.png. If omitted, defaults to <misc_dir>/<log_basename>"
        ),
    )
    p.add_argument("--dpi", type=int, default=200, help="Output figure DPI")
    p.add_argument("--bins", type=int, default=20, help="Number of quantile bins for the binned-mean line")
    p.add_argument("--no_scatter", action="store_true", help="Hide scatter points; show only the binned-mean line")
    p.add_argument("--show_trend", action="store_true", help="Overlay a linear trend line (polyfit deg=1)")
    return p.parse_args()


def pick_latest_txt(logs_dir: str) -> str:
    if not os.path.isdir(logs_dir):
        raise FileNotFoundError(f"Logs directory not found: {logs_dir}")
    candidates = [
        os.path.join(logs_dir, f)
        for f in os.listdir(logs_dir)
        if f.lower().endswith(".txt") and os.path.isfile(os.path.join(logs_dir, f))
    ]
    if not candidates:
        raise FileNotFoundError(f"No .txt log files found in {logs_dir}")
    candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return candidates[0]


def safe_float(s: str) -> float:
    try:
        if s.lower() == "nan":
            return float("nan")
    except AttributeError:
        pass
    try:
        return float(s)
    except Exception:
        return float("nan")


def parse_log(path: str) -> Tuple[OrderedDict, List[Tuple[str, float, float, float]]]:
    """
    Returns:
      groups: OrderedDict[str, List[Tuple[prec1, ss_slope, dev_a2]]]
      flat:   List[(group, prec1, ss_slope, dev_a2)]
    """
    groups: "OrderedDict[str, List[Tuple[float, float, float]]]" = OrderedDict()
    current_group = None

    def ensure_group(name: str):
        nonlocal groups
        if name not in groups:
            groups[name] = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.rstrip("\n")

            # Group header
            m = CORRUPTION_HEADER_RE.match(line)
            if m:
                current_group = m.group(1).strip()
                ensure_group(current_group)
                continue

            # Parse metrics on iteration lines
            m_prec = PREC_RE.search(line)
            if not m_prec:
                continue
            m_ss = SS_SLOPE_RE.search(line)
            m_deva2 = DEVA2_RE.search(line)

            prec1 = safe_float(m_prec.group(1)) if m_prec else float("nan")
            ss = safe_float(m_ss.group(1)) if m_ss else float("nan")
            deva2 = safe_float(m_deva2.group(1)) if m_deva2 else float("nan")

            if current_group is None:
                current_group = "default"
                ensure_group(current_group)

            groups[current_group].append((prec1, ss, deva2))

    # Flatten for convenience
    flat = []
    for g, rows in groups.items():
        for (prec1, ss, deva2) in rows:
            flat.append((g, prec1, ss, deva2))

    return groups, flat


def compute_error_percent(values: List[float]) -> List[float]:
    """Convert instantaneous Prec@1 series to Top-1 error in percent.
    Auto-detect if Prec@1 is in [0, 1] or in [0, 100]."""
    arr = np.array(values, dtype=float)
    # Heuristic: if any value > 1.5, treat as percent already
    if np.nanmax(arr) > 1.5:
        error = 100.0 - arr
    else:
        error = (1.0 - arr) * 100.0
    return error.tolist()


def make_figure_for_metric(groups: OrderedDict, metric: str, out_path: str, dpi: int = 200, *, bins: int = 20, show_trend: bool = False, no_scatter: bool = False):
    """
    metric: 'ss' or 'deva2'
    Creates a single-column figure with one subplot per corruption set.
    """
    n_groups = max(1, len(groups))
    fig, axes = plt.subplots(n_groups, 1, figsize=(6.5, max(3, 2.6 * n_groups)), squeeze=False)

    for row_idx, (gname, rows) in enumerate(groups.items() if groups else [("default", [])]):
        ax = axes[row_idx, 0]
        if not rows:
            ax.set_visible(False)
            continue

        prec1 = [r[0] for r in rows]
        ss = [r[1] for r in rows]
        deva2 = [r[2] for r in rows]

        # Convert to error percent
        err = compute_error_percent(prec1)

        # Choose x and labels
        if metric == "ss":
            x = ss
            xlab = "SS-slope"
            title = f"{gname} — SS-slope vs error"
        else:
            x = deva2
            xlab = "Dev|a-2|"
            title = f"{gname} — Dev|a-2| vs error"

        # Scatter with NaN filtering
        x = np.array(x, dtype=float)
        y = np.array(err, dtype=float)
        m = np.isfinite(x) & np.isfinite(y)
        if not no_scatter:
            ax.scatter(x[m], y[m], s=10, alpha=0.45, edgecolors='none', label=None)
        ax.set_xlabel(xlab)
        ax.set_ylabel("Top-1 error (%)")
        ax.grid(True, alpha=0.3, linestyle=":")
        ax.set_title(title)

        # Binned-mean line (quantile bins)
        if m.sum() >= 2 and bins > 0:
            try:
                # Quantile edges; handle duplicates by unique
                qs = np.linspace(0, 1, bins + 1)
                edges = np.quantile(x[m], qs)
                edges = np.unique(edges)
                if edges.size >= 2:
                    idx = np.digitize(x[m], edges, right=True)
                    bin_x = []
                    bin_y = []
                    for b in range(1, len(edges)):
                        sel = idx == b
                        if np.any(sel):
                            bin_x.append(np.nanmean(x[m][sel]))
                            bin_y.append(np.nanmean(y[m][sel]))
                    if len(bin_x) >= 2:
                        ax.plot(bin_x, bin_y, "-o", color="tab:blue", linewidth=2.0, markersize=3.0, alpha=0.95, label=f"binned mean (n={m.sum()})")
            except Exception:
                pass

        # Optional trend line
        if show_trend and m.sum() >= 3:
            try:
                z = np.polyfit(x[m], y[m], 1)
                xp = np.linspace(np.nanmin(x[m]), np.nanmax(x[m]), 50)
                yp = z[0] * xp + z[1]
                ax.plot(xp, yp, color="tab:red", linewidth=1.0, alpha=0.8, label=f"trend: y={z[0]:.2f}x+{z[1]:.2f}")
            except Exception:
                pass

        # Add Pearson r in legend if available
        try:
            if m.sum() >= 2:
                r = float(np.corrcoef(x[m], y[m])[0, 1])
                if np.isfinite(r):
                    ax.legend(loc="best", fontsize=8, title=f"r={r:.2f}")
        except Exception:
            pass

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved: {out_path}")


def main():
    args = parse_args()

    if args.file is None:
        log_path = pick_latest_txt(args.logs_dir)
    else:
        log_path = args.file
    if not os.path.isfile(log_path):
        print(f"Log file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    groups, flat = parse_log(log_path)
    if not groups:
        print("No parseable data found.", file=sys.stderr)
        sys.exit(2)

    # Determine output prefix
    if args.output is None:
        base = os.path.splitext(os.path.basename(log_path))[0]
        # Default to script directory (misc folder)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        prefix = os.path.join(script_dir, base)
    else:
        prefix = args.output
        if prefix.lower().endswith(".png"):
            prefix = prefix[:-4]

    out_ss = f"{prefix}_ss_vs_error.png"
    out_deva2 = f"{prefix}_deva2_vs_error.png"

    make_figure_for_metric(groups, "ss", out_ss, dpi=args.dpi, bins=args.bins, show_trend=args.show_trend, no_scatter=args.no_scatter)
    make_figure_for_metric(groups, "deva2", out_deva2, dpi=args.dpi, bins=args.bins, show_trend=args.show_trend, no_scatter=args.no_scatter)


if __name__ == "__main__":
    main()
