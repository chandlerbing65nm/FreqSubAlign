#!/usr/bin/env python3
"""
Plot LF/HF vs Entropy from a ViTTA log file.

- LF is defined as DWT LL.
- HF is defined as 1 - DWT LL.
- Supports plotting either instantaneous (inst) or running average (avg) values.

Example:
  python misc/plot_lf_hf_vs_entropy.py \
    --file Repo/ViTTA/logs/output_12503457.txt \
    --metric inst \
    --plot-type line \
    --bins 20 --range 0.0 1.0 \
    --out Repo/ViTTA/misc/lf_hf_vs_entropy.png

Expected log line pattern (single line):
  ... TTA Epoch1: [4/500] ... DWT LL 0.7524 (0.8085) LH 0.1009 (0.0777) HL 0.0981 (0.0768) HH 0.0485 (0.0370) Entropy 0.4311 (1.5466)

Requires: matplotlib, numpy
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
import math
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Regex utilities
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RE_STEP = re.compile(r"TTA\s+Epoch(\d+):\s*\[(\d+)/(\d+)\]")
RE_DWT = re.compile(
    rf"DWT\s+(?:EntropyProp|Energy)?\s*LL\s+({_FLOAT})\s*\(\s*({_FLOAT})\s*\)\s+"
    rf"LH\s+({_FLOAT})\s*\(\s*({_FLOAT})\s*\)\s+"
    rf"HL\s+({_FLOAT})\s*\(\s*({_FLOAT})\s*\)\s+"
    rf"HH\s+({_FLOAT})\s*\(\s*({_FLOAT})\s*\)"
)
RE_ENT = re.compile(rf"Entropy\s+({_FLOAT})\s*\(\s*({_FLOAT})\s*\)")
RE_CORR = re.compile(r"=+\s*Arguments for\s+([^\s]+)\s+corruption\s*=+")


def parse_lf_hf_entropy(file_path: Path) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse log and return arrays: ll_inst, ll_avg, ent_inst, ent_avg.

    Only steps that have both DWT and Entropy on the same line are included.
    """
    ll_inst: List[float] = []
    ll_avg: List[float] = []
    ent_inst: List[float] = []
    ent_avg: List[float] = []

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Optional: ensure it's a TTA step line to reduce false positives
            if not RE_STEP.search(line):
                continue

            m_dwt = RE_DWT.search(line)
            m_ent = RE_ENT.search(line)
            if not (m_dwt and m_ent):
                continue

            try:
                ll_i = float(m_dwt.group(1))
                ll_a = float(m_dwt.group(2))
                # lh_i = float(m_dwt.group(3)); lh_a = float(m_dwt.group(4))  # parsed but unused
                # hl_i = float(m_dwt.group(5)); hl_a = float(m_dwt.group(6))  # parsed but unused
                # hh_i = float(m_dwt.group(7)); hh_a = float(m_dwt.group(8))  # parsed but unused
                ent_i = float(m_ent.group(1))
                ent_a = float(m_ent.group(2))
            except Exception:
                continue

            ll_inst.append(ll_i)
            ll_avg.append(ll_a)
            ent_inst.append(ent_i)
            ent_avg.append(ent_a)

    return (
        np.asarray(ll_inst, dtype=float),
        np.asarray(ll_avg, dtype=float),
        np.asarray(ent_inst, dtype=float),
        np.asarray(ent_avg, dtype=float),
    )


def parse_lf_hf_entropy_by_corruption(file_path: Path) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Parse log grouped by corruption set.

    Returns a dict: name -> (ll_inst, ll_avg, ent_inst, ent_avg).
    Only counts lines that appear after a marker like:
      === Arguments for {corruption_set} corruption ===
    """
    data: Dict[str, Dict[str, List[float]]] = {}
    current: str | None = None

    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Detect corruption boundary
            m_corr = RE_CORR.search(line)
            if m_corr:
                current = m_corr.group(1)
                if current not in data:
                    data[current] = {"ll_i": [], "ll_a": [], "ent_i": [], "ent_a": []}
                continue

            if not RE_STEP.search(line):
                continue
            if current is None:
                # Skip steps until a corruption is specified
                continue

            m_dwt = RE_DWT.search(line)
            m_ent = RE_ENT.search(line)
            if not (m_dwt and m_ent):
                continue

            try:
                ll_i = float(m_dwt.group(1)); ll_a = float(m_dwt.group(2))
                ent_i = float(m_ent.group(1)); ent_a = float(m_ent.group(2))
            except Exception:
                continue

            data[current]["ll_i"].append(ll_i)
            data[current]["ll_a"].append(ll_a)
            data[current]["ent_i"].append(ent_i)
            data[current]["ent_a"].append(ent_a)

    out: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for k, v in data.items():
        out[k] = (
            np.asarray(v["ll_i"], dtype=float),
            np.asarray(v["ll_a"], dtype=float),
            np.asarray(v["ent_i"], dtype=float),
            np.asarray(v["ent_a"], dtype=float),
        )
    return out


def _bin_indices(x: np.ndarray, bins: int, x_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    lo, hi = x_range
    edges = np.linspace(lo, hi, bins + 1)
    inds = np.digitize(x, edges, right=False) - 1
    inds = np.clip(inds, 0, bins - 1)
    centers = (edges[:-1] + edges[1:]) / 2.0
    return inds, centers


def _bin_mean_y_by_x(x: np.ndarray, y: np.ndarray, bins: int, x_range: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    lo, hi = x_range
    mask = np.isfinite(x) & np.isfinite(y) & (x >= lo) & (x <= hi)
    if mask.sum() == 0:
        return np.array([]), np.array([]), np.array([])
    x = x[mask]
    y = y[mask]
    inds, centers = _bin_indices(x, bins, (lo, hi))
    means = np.full(bins, np.nan)
    counts = np.zeros(bins, dtype=int)
    for b in range(bins):
        sel = inds == b
        n = int(sel.sum())
        counts[b] = n
        if n > 0:
            means[b] = float(y[sel].mean())
    return centers, means, counts


def plot_lf_hf_vs_entropy(
    ll: np.ndarray,
    ent: np.ndarray,
    metric_label: str,
    out_path: Path | None,
    plot_type: str,
    bins: int,
    x_range: Tuple[float, float],
) -> None:
    """Binned line/bar plots: mean Entropy vs LF(=LL) and vs HF(=1-LL)."""
    if ll.size == 0 or ent.size == 0:
        print("No data to plot (parsed arrays are empty).", file=sys.stderr)
        sys.exit(2)

    lf = ll
    hf = 1.0 - ll

    # Bin and compute mean entropy per bin
    centers_lf, mean_ent_lf, counts_lf = _bin_mean_y_by_x(lf, ent, bins=bins, x_range=x_range)
    centers_hf, mean_ent_hf, counts_hf = _bin_mean_y_by_x(hf, ent, bins=bins, x_range=x_range)

    if centers_lf.size == 0 and centers_hf.size == 0:
        print("No data in specified range/bins to plot.", file=sys.stderr)
        sys.exit(2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

    def _draw(ax, centers, means, counts, label, color):
        if centers.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            return
        width = (centers[1] - centers[0]) if centers.size > 1 else 0.04
        # Plot mean entropy (primary y-axis)
        if plot_type == "bar":
            ax.bar(centers, means, width=width, color=color, alpha=0.8, label=label, zorder=2)
        else:
            ax.plot(centers, means, marker="o", color=color, linewidth=1.8, label=label, zorder=3)
        # Overlay counts on secondary y-axis as translucent bars behind
        ax2 = ax.twinx()
        ax2.bar(centers, counts, width=width, color="#999999", alpha=0.25, edgecolor="none", zorder=1)
        ax2.set_ylabel("Count")
        ax2.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="best")

    # LF vs Entropy
    _draw(axes[0], centers_lf, mean_ent_lf, counts_lf, label=f"LF ({metric_label})", color="#1f77b4")
    axes[0].set_xlabel("LF (LL)")
    axes[0].set_ylabel("Mean Entropy")
    axes[0].set_title(f"Entropy vs LF ({metric_label})")

    # HF vs Entropy
    _draw(axes[1], centers_hf, mean_ent_hf, counts_hf, label=f"HF ({metric_label})", color="#ff7f0e")
    axes[1].set_xlabel("HF (1 - LL)")
    axes[1].set_title(f"Entropy vs HF ({metric_label})")

    fig.tight_layout()

    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Ensure a default suffix
        if out_path.suffix == "":
            out_path = out_path.with_suffix(".png")
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def plot_combined_per_corruption(
    series_map: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    metric: str,
    plot_type: str,
    bins: int,
    x_range: Tuple[float, float],
    out_base: Path | None,
) -> None:
    """Create two figures: Entropy vs LF and Entropy vs HF. Each figure contains
    one subplot per corruption set. Saves JPEGs.

    If out_base is a file path, save alongside it using stems with _LF and _HF.
    If out_base is a directory, save inside it with default names.
    If out_base is None, save next to the log base (handled by caller via out_base=None,
    we will default to current working directory).
    """
    names = sorted(series_map.keys())
    if not names:
        print("No corruption data to plot.", file=sys.stderr)
        sys.exit(2)

    def _sanitize(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    # Prepare grid
    n = len(names)
    ncols = int(math.ceil(math.sqrt(n)))
    nrows = int(math.ceil(n / ncols))

    # Helper to draw a single subplot
    def _draw(ax, centers, means, counts, title, color):
        if centers.size == 0:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(title)
            return
        width = (centers[1] - centers[0]) if centers.size > 1 else 0.04
        if plot_type == "bar":
            ax.bar(centers, means, width=width, color=color, alpha=0.85, zorder=2)
        else:
            ax.plot(centers, means, marker="o", color=color, linewidth=1.8, zorder=3)
        ax2 = ax.twinx()
        ax2.bar(centers, counts, width=width, color="#999999", alpha=0.25, edgecolor="none", zorder=1)
        ax2.set_ylabel("Count")
        ax2.set_ylim(bottom=0)
        ax.grid(True, alpha=0.3)
        ax.set_title(title)

    # Figure 1: Entropy vs LF
    fig_lf, axes_lf = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False, sharex=True, sharey=True)
    # Figure 2: Entropy vs HF
    fig_hf, axes_hf = plt.subplots(nrows, ncols, figsize=(6 * ncols, 4.2 * nrows), squeeze=False, sharex=True, sharey=True)

    for idx, name in enumerate(names):
        r, c = divmod(idx, ncols)
        ll_i, ll_a, ent_i, ent_a = series_map[name]
        ll = ll_i if metric == "inst" else ll_a
        ent = ent_i if metric == "inst" else ent_a
        if ll.size == 0 or ent.size == 0:
            continue
        # LF
        centers_lf, mean_ent_lf, counts_lf = _bin_mean_y_by_x(ll, ent, bins=bins, x_range=x_range)
        # HF
        hf = 1.0 - ll
        centers_hf, mean_ent_hf, counts_hf = _bin_mean_y_by_x(hf, ent, bins=bins, x_range=x_range)

        _draw(axes_lf[r][c], centers_lf, mean_ent_lf, counts_lf, title=_sanitize(name), color="#1f77b4")
        _draw(axes_hf[r][c], centers_hf, mean_ent_hf, counts_hf, title=_sanitize(name), color="#ff7f0e")

    for ax in axes_lf[-1]:
        ax.set_xlabel("LF (LL)")
    for row in axes_lf:
        row[0].set_ylabel("Mean Entropy")
    fig_lf.suptitle(f"Entropy vs LF ({metric})", y=0.995)
    fig_lf.tight_layout(rect=[0, 0.00, 1, 0.96])

    for ax in axes_hf[-1]:
        ax.set_xlabel("HF (1 - LL)")
    for row in axes_hf:
        row[0].set_ylabel("Mean Entropy")
    fig_hf.suptitle(f"Entropy vs HF ({metric})", y=0.995)
    fig_hf.tight_layout(rect=[0, 0.00, 1, 0.96])

    # Determine output paths
    if out_base is None:
        out_lf = Path(f"lf_vs_entropy_{metric}_combined.jpg")
        out_hf = Path(f"hf_vs_entropy_{metric}_combined.jpg")
    else:
        if out_base.suffix:
            out_lf = out_base.with_name(f"{out_base.stem}_LF.jpg")
            out_hf = out_base.with_name(f"{out_base.stem}_HF.jpg")
        else:
            out_lf = out_base / f"lf_vs_entropy_{metric}_combined.jpg"
            out_hf = out_base / f"hf_vs_entropy_{metric}_combined.jpg"

    out_lf.parent.mkdir(parents=True, exist_ok=True)
    out_hf.parent.mkdir(parents=True, exist_ok=True)
    fig_lf.savefig(out_lf, dpi=150)
    fig_hf.savefig(out_hf, dpi=150)
    print(f"Saved plot to {out_lf}")
    print(f"Saved plot to {out_hf}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot LF/HF vs Entropy from ViTTA log")
    parser.add_argument("--file", type=str, required=True, help="Path to log .txt file")
    parser.add_argument(
        "--metric",
        choices=["inst", "avg"],
        default="inst",
        help="Use instantaneous or running average values",
    )
    parser.add_argument("--plot-type", choices=["line", "bar"], default="line", help="Plot type for binned mean entropy")
    parser.add_argument("--bins", type=int, default=20, help="Number of equal-width bins for LF/HF (x-axis)")
    parser.add_argument("--range", dest="range_", type=float, nargs=2, default=(0.0, 1.0), metavar=("LO", "HI"), help="X range to include [LO, HI]")
    parser.add_argument("--per-corruption", action="store_true", help="Split the log by corruption set markers and save a separate JPEG for each")
    parser.add_argument("--per-corruption-combined", action="store_true", help="Split by corruption set and save only two JPEGs (LF and HF), each with one subplot per corruption")
    parser.add_argument("--out", type=str, default=None, help="Output image path (PNG). If omitted, shows interactively.")

    args = parser.parse_args()

    log_path = Path(args.file)
    if not log_path.exists():
        print(f"Error: file not found: {log_path}", file=sys.stderr)
        sys.exit(1)

    # Validate bins
    if args.bins <= 0:
        print("--bins must be > 0", file=sys.stderr)
        sys.exit(2)
    x_range = (float(args.range_[0]), float(args.range_[1]))

    def _sanitize(name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

    if args.per_corruption or getattr(args, "per_corruption_combined", False):
        series_map = parse_lf_hf_entropy_by_corruption(log_path)
        if not series_map:
            print("No corruption sections found in log (looking for '=== Arguments for <name> corruption ===').", file=sys.stderr)
            sys.exit(2)
        base = Path(args.out) if args.out else None
        label = args.metric
        if getattr(args, "per_corruption_combined", False):
            plot_combined_per_corruption(series_map, label, args.plot_type, args.bins, x_range, base)
        else:
            for name, (ll_i, ll_a, ent_i, ent_a) in series_map.items():
                ll = ll_i if args.metric == "inst" else ll_a
                ent = ent_i if args.metric == "inst" else ent_a
                if ll.size == 0 or ent.size == 0:
                    continue
                safe = _sanitize(name)
                if base is None:
                    # Default next to log file
                    out_path = log_path.with_name(f"{log_path.stem}_{label}_{safe}.jpg")
                else:
                    if base.suffix:
                        # Treat as a file path; create sibling with corruption suffix and .jpg
                        out_path = base.with_name(f"{base.stem}_{safe}.jpg")
                    else:
                        # Treat as a directory
                        out_path = base / f"lf_hf_vs_entropy_{label}_{safe}.jpg"
                plot_lf_hf_vs_entropy(ll, ent, label, out_path, args.plot_type, args.bins, x_range)
    else:
        ll_inst, ll_avg, ent_inst, ent_avg = parse_lf_hf_entropy(log_path)
        if args.metric == "inst":
            ll = ll_inst
            ent = ent_inst
            label = "inst"
        else:
            ll = ll_avg
            ent = ent_avg
            label = "avg"

        out_path = Path(args.out) if args.out else None
        plot_lf_hf_vs_entropy(ll, ent, label, out_path, args.plot_type, args.bins, x_range)


if __name__ == "__main__":
    main()
