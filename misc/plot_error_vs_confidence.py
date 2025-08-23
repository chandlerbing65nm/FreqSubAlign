#!/usr/bin/env python3
"""
Plot classification error vs. confidence from a ViTTA log file, with optional pHF plots.

Reads a .txt log, extracts Top-1 accuracy, confidence metrics (e.g., Conf[max_softmax]),
and pHF (proportion of high-frequency) if present. Aggregates by bins to produce
discrete plots:
  - Error vs Confidence (default)
  - Error vs pHF (optional)
  - Confidence vs pHF (optional)

Examples:
  # Default: error vs confidence (instantaneous), 20 bins in [0,1]
  python misc/plot_error_vs_confidence.py --file path/to/log.txt --out misc/error_vs_conf.png

  # Running average series, custom bins, plus both pHF plots
  python misc/plot_error_vs_confidence.py --file path/to/log.txt \
      --metric avg --bins 15 --plot-phf-error --plot-phf-conf \
      --out misc/results.png

Options (subset):
  --conf-key           Confidence name in logs, e.g., 'max_softmax' or 'entropy' (default: max_softmax)
  --metric             Use 'inst' (instantaneous) or 'avg' (running average) series (default: inst)
  --bins               Number of equal-width bins for confidence (default: 20)
  --range              Confidence range to consider, e.g. 0.0 1.0 (default: 0 to 1)
  --plot-phf-error     Also plot error vs pHF (binned)
  --plot-phf-conf      Also plot confidence vs pHF (binned)
  --phf-bins           Number of bins for pHF plots (default: 20)
  --phf-range          pHF range, e.g. 0.0 1.0 (default: 0 to 1)
  --title              Optional base plot title
  --out                Output image path. If multiple plots, auto-suffixed files are written.

Log line examples this script understands:
  ... TTA Epoch1: [4/3783]\t...\tPrec@1 100.000 (20.000) ... Conf[max_softmax] 0.876 (0.651) pHF 0.2691

Requires: matplotlib, numpy
"""
from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# -----------------
# Parsing utilities
# -----------------
# Anchor for step presence
RE_STEP = re.compile(r"TTA\s+Epoch(\d+):\s*\[(\d+)/(\d+)\]")
# Flexible float
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RE_P1 = re.compile(rf"Prec@1\s+({_FLOAT})\s*\((({_FLOAT}))\)")
RE_CONF = re.compile(rf"Conf\[(.*?)\]\s+({_FLOAT})\s*\((({_FLOAT}))\)")
RE_PHF = re.compile(rf"\bpHF\s+({_FLOAT})\b")


def parse_log_series(file_path: Path, conf_key: str, metric: str) -> Dict[str, np.ndarray]:
    """Parse a log file and return per-step series for top1, chosen confidence, and pHF.

    Returns dict with keys: 'top1', 'conf', 'phf' (each numpy arrays). Steps lacking
    Top-1 are skipped. For missing 'conf' or 'phf' on a step, NaN is inserted.

    metric: 'inst' uses instantaneous values; 'avg' uses running averages.
    conf_key: which Conf[...] series to pick (e.g., 'max_softmax').
    """
    top1_list: List[float] = []
    conf_list: List[float] = []
    phf_list: List[float] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not RE_STEP.search(line):
                continue

            # Accuracy Top-1
            m_p1 = RE_P1.search(line)
            if not m_p1:
                continue
            top1_inst = float(m_p1.group(1))
            top1_avg = float(m_p1.group(2))
            top1 = top1_inst if metric == "inst" else top1_avg

            # Confidence (selected key)
            chosen_conf: float | float('nan')
            chosen_conf = float("nan")
            for m in RE_CONF.finditer(line):
                name = m.group(1)
                ci = float(m.group(2))
                ca = float(m.group(3))
                val = ci if metric == "inst" else ca
                if name == conf_key:
                    chosen_conf = val
                    break

            # pHF (single value, no running avg)
            m_phf = RE_PHF.search(line)
            phf_val = float(m_phf.group(1)) if m_phf else float("nan")

            top1_list.append(top1)
            conf_list.append(chosen_conf)
            phf_list.append(phf_val)

    return {
        "top1": np.asarray(top1_list, dtype=float),
        "conf": np.asarray(conf_list, dtype=float),
        "phf": np.asarray(phf_list, dtype=float),
    }


# --------------
# Plotting logic
# --------------

def _bin_indices(x: np.ndarray, bins: int, x_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = x_range
    edges = np.linspace(lo, hi, bins + 1)
    inds = np.digitize(x, edges, right=False) - 1
    inds = np.clip(inds, 0, bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return inds, centers


def bin_error_by_x(x: np.ndarray, errors: np.ndarray, bins: int, x_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin samples by x and compute error rate per bin (mean of errors)."""
    lo, hi = x_range
    mask = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(errors)
    if mask.sum() == 0:
        return np.array([]), np.array([]), np.array([])
    x = x[mask]
    errors = errors[mask]
    inds, centers = _bin_indices(x, bins, (lo, hi))
    err_rate = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        idx = (inds == b)
        n = int(idx.sum())
        counts[b] = n
        if n > 0:
            err_rate[b] = float(errors[idx].mean())
    return centers, err_rate, counts


def bin_mean_y_by_x(x: np.ndarray, y: np.ndarray, bins: int, x_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bin samples by x and compute mean(y) per bin."""
    lo, hi = x_range
    mask = (x >= lo) & (x <= hi) & np.isfinite(x) & np.isfinite(y)
    if mask.sum() == 0:
        return np.array([]), np.array([]), np.array([])
    x = x[mask]
    y = y[mask]
    inds, centers = _bin_indices(x, bins, (lo, hi))
    means = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        idx = (inds == b)
        n = int(idx.sum())
        counts[b] = n
        if n > 0:
            means[b] = float(y[idx].mean())
    return centers, means, counts


def make_error_plot(centers: np.ndarray, err_rate: np.ndarray, counts: np.ndarray, x_label: str, title: str | None, out_path: Path | None) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Bar plot for error rate
    ax1.bar(centers, err_rate * 100.0, width=(centers[1] - centers[0]) if centers.size > 1 else 0.04,
            color="#1f77b4", alpha=0.8, label="Error rate")
    ax1.set_ylabel("Error Rate (%)")
    ax1.set_xlabel(x_label)
    ax1.set_ylim(0, 100)
    ax1.grid(True, axis="y", alpha=0.3)

    # Secondary axis for counts
    ax2 = ax1.twinx()
    ax2.plot(centers, counts, color="#ff7f0e", marker="o", linewidth=1.5, label="Count")
    ax2.set_ylabel("Samples per bin")

    # Build a combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    if title:
        ax1.set_title(title)

    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def make_mean_plot(centers: np.ndarray, means: np.ndarray, counts: np.ndarray, x_label: str, y_label: str, title: str | None, out_path: Path | None) -> None:
    fig, ax1 = plt.subplots(figsize=(9, 5))

    # Line plot for mean value
    ax1.plot(centers, means, color="#2ca02c", marker="o", label=y_label)
    ax1.set_ylabel(y_label)
    ax1.set_xlabel(x_label)
    ax1.grid(True, axis="both", alpha=0.3)

    # Secondary axis for counts
    ax2 = ax1.twinx()
    ax2.bar(centers, counts, width=(centers[1] - centers[0]) if centers.size > 1 else 0.04,
            color="#ff7f0e", alpha=0.3, label="Count")
    ax2.set_ylabel("Samples per bin")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper right")

    if title:
        ax1.set_title(title)

    fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


# -----
# Main
# -----

def main():
    parser = argparse.ArgumentParser(description="Plot error vs confidence from a ViTTA log file")
    parser.add_argument("--file", type=str, required=True, help="Path to log .txt file")
    parser.add_argument("--conf-key", type=str, default="max_softmax", help="Confidence key to use (e.g., max_softmax, entropy)")
    parser.add_argument("--metric", choices=["inst", "avg"], default="inst", help="Use instantaneous or running average series")
    parser.add_argument("--bins", type=int, default=20, help="Number of equal-width confidence bins")
    parser.add_argument("--range", dest="range_", type=float, nargs=2, default=(0.0, 1.0), metavar=("LO", "HI"),
                        help="Confidence range to include [LO, HI]")
    parser.add_argument("--plot-phf-error", action="store_true", help="Also plot error vs pHF (binned)")
    parser.add_argument("--plot-phf-conf", action="store_true", help="Also plot confidence vs pHF (binned)")
    parser.add_argument("--phf-bins", type=int, default=20, help="Number of pHF bins")
    parser.add_argument("--phf-range", dest="phf_range", type=float, nargs=2, default=(0.0, 1.0), metavar=("LO", "HI"),
                        help="pHF range to include [LO, HI]")
    parser.add_argument("--title", type=str, default=None, help="Plot title")
    parser.add_argument("--out", type=str, default=None, help="Output image path")

    args = parser.parse_args()

    log_path = Path(args.file)
    if not log_path.exists():
        print(f"Error: file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    series = parse_log_series(log_path, conf_key=args.conf_key, metric=args.metric)
    top1 = series["top1"]
    conf = series["conf"]
    phf = series["phf"]

    # Compute per-sample error from Top-1
    with np.errstate(invalid="ignore"):
        errors = 1.0 - (top1 / 100.0)

    made_any_plot = False

    # ---- Plot 1: Error vs Confidence (default) ----
    mask_ec = np.isfinite(errors) & np.isfinite(conf)
    if mask_ec.sum() > 0:
        centers, err_rate, counts = bin_error_by_x(conf[mask_ec], errors[mask_ec], bins=args.bins, x_range=tuple(args.range_))
        if centers.size > 0:
            out_path = Path(args.out) if (args.out and not (args.plot_phf_error or args.plot_phf_conf)) else (
                Path(args.out).with_name(Path(args.out).stem + "_error_vs_conf" + Path(args.out).suffix)
                if args.out else None
            )
            title = args.title or f"Error vs Confidence (key={args.conf_key}, metric={args.metric})"
            make_error_plot(centers, err_rate, counts, x_label="Confidence", title=title, out_path=out_path)
            made_any_plot = True
    else:
        print("Warning: No pairs for Error vs Confidence. Check --conf-key or log content.", file=sys.stderr)

    # ---- Plot 2: Error vs pHF (optional) ----
    if args.plot_phf_error:
        mask_pe = np.isfinite(errors) & np.isfinite(phf)
        if mask_pe.sum() > 0:
            centers, err_rate, counts = bin_error_by_x(phf[mask_pe], errors[mask_pe], bins=args.phf_bins, x_range=tuple(args.phf_range))
            if centers.size > 0:
                out_path = Path(args.out).with_name(Path(args.out).stem + "_error_vs_phf" + Path(args.out).suffix) if args.out else None
                title = (args.title + " — ") if args.title else ""
                title += f"Error vs pHF (metric={args.metric})"
                make_error_plot(centers, err_rate, counts, x_label="pHF", title=title, out_path=out_path)
                made_any_plot = True
        else:
            print("Warning: No pairs for Error vs pHF. Ensure 'pHF' appears in the log.", file=sys.stderr)

    # ---- Plot 3: Confidence vs pHF (optional) ----
    if args.plot_phf_conf:
        mask_pc = np.isfinite(conf) & np.isfinite(phf)
        if mask_pc.sum() > 0:
            centers, mean_conf, counts = bin_mean_y_by_x(phf[mask_pc], conf[mask_pc], bins=args.phf_bins, x_range=tuple(args.phf_range))
            if centers.size > 0:
                out_path = Path(args.out).with_name(Path(args.out).stem + "_conf_vs_phf" + Path(args.out).suffix) if args.out else None
                title = (args.title + " — ") if args.title else ""
                title += f"Confidence vs pHF (key={args.conf_key}, metric={args.metric})"
                make_mean_plot(centers, mean_conf, counts, x_label="pHF", y_label="Mean Confidence", title=title, out_path=out_path)
                made_any_plot = True
        else:
            print("Warning: No pairs for Confidence vs pHF. Ensure both 'Conf[...]' and 'pHF' appear in the log.", file=sys.stderr)

    if not made_any_plot:
        print("No plots were produced. Check the log format and selected options.", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
