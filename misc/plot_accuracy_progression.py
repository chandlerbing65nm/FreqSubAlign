#!/usr/bin/env python3
"""
Plot metrics from ViTTA log .txt files and compare two runs.

It parses lines like:
2025-08-20 19:00:21,994 - 10 - test_time_adaptation.py - tta_standard - TTA Epoch1: [4/3783]\tTime 0.606 (1.405)\tLoss reg 8.4538 (9.1583)\tLoss wpa 6.2539 (6.3792)\tPrec@1 100.000 (20.000)\tPrec@5 100.000 (40.000)

- The first number after Prec@k is instantaneous accuracy for the current sample.
- The number in parentheses is the running average up to that sample.

Usage examples:
1) Compare two files explicitly (defaults: first is "ours", second is "base")
   python misc/plot_accuracy_progression.py --file1 logs/ours.txt --file2 logs/base.txt --out compare.png

2) Provide a directory containing exactly two .txt files
   python misc/plot_accuracy_progression.py --dir logs/ --out compare.png

Optional:
  --labels "Run A" "Run B"    Custom legend labels (override default "ours"/"base")
  --metric avg|inst             Plot running averages (default: avg) or instantaneous values
  --title "My Plot Title"       Custom figure title

Requires: matplotlib
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

# Regex to capture epoch/index (optional) and metrics
RE_STEP = re.compile(r"TTA\s+Epoch(\d+):\s*\[(\d+)/(\d+)\]")
# Flexible float (supports integers and scientific notation)
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RE_P1 = re.compile(rf"Prec@1\s+({_FLOAT})\s*\((({_FLOAT}))\)")
RE_P5 = re.compile(rf"Prec@5\s+({_FLOAT})\s*\((({_FLOAT}))\)")
# Generic regex for losses and confidence metrics
# Loss names: allow non-space tokens (letters, digits, underscore, hyphen, etc.)
RE_LOSS = re.compile(rf"Loss\s+([^\s]+)\s+({_FLOAT})\s*\((({_FLOAT}))\)")
RE_CONF = re.compile(rf"Conf\[(.*?)\]\s+({_FLOAT})\s*\((({_FLOAT}))\)")


def parse_log(file_path: Path) -> Dict[str, Any]:
    """Parse a ViTTA log file to extract metrics per step.

    Returns a dict with keys:
      - step: global step index (0..N-1)
      - epoch: epoch indices (if found), else -1
      - idx: per-epoch sample index (if found), else -1
      - top1_inst, top1_avg, top5_inst, top5_avg
      - losses_inst: dict[name] -> list
      - losses_avg: dict[name] -> list
      - conf_inst: dict[name] -> list
      - conf_avg: dict[name] -> list
    """
    step, epoch, idx = [], [], []
    top1_inst, top1_avg = [], []
    top5_inst, top5_avg = [], []

    losses_inst: Dict[str, List[float]] = {}
    losses_avg: Dict[str, List[float]] = {}
    conf_inst: Dict[str, List[float]] = {}
    conf_avg: Dict[str, List[float]] = {}

    loss_names: List[str] = []
    conf_names: List[str] = []

    gstep = 0
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            # Anchor on step marker. If absent, skip line entirely.
            m_step = RE_STEP.search(line)
            if not m_step:
                continue

            ep = int(m_step.group(1)) if m_step else -1
            i = int(m_step.group(2)) if m_step else -1

            # Accuracies: parse independently, allow missing
            m_p1 = RE_P1.search(line)
            m_p5 = RE_P5.search(line)
            if m_p1:
                t1_i = float(m_p1.group(1))
                t1_a = float(m_p1.group(2))
            else:
                t1_i = float("nan")
                t1_a = float("nan")
            if m_p5:
                t5_i = float(m_p5.group(1))
                t5_a = float(m_p5.group(2))
            else:
                t5_i = float("nan")
                t5_a = float("nan")

            # Parse losses present on this line
            present_loss: set[str] = set()
            for m in RE_LOSS.finditer(line):
                name = m.group(1)
                li = float(m.group(2))
                la = float(m.group(3))
                if name not in losses_inst:
                    losses_inst[name] = [float("nan")] * gstep
                    losses_avg[name] = [float("nan")] * gstep
                    loss_names.append(name)
                losses_inst[name].append(li)
                losses_avg[name].append(la)
                present_loss.add(name)
            # For any known loss not present in this line, append NaN to keep lengths aligned
            for name in loss_names:
                if name not in present_loss:
                    losses_inst[name].append(float("nan"))
                    losses_avg[name].append(float("nan"))

            # Parse confidences present on this line
            present_conf: set[str] = set()
            for m in RE_CONF.finditer(line):
                name = m.group(1)
                ci = float(m.group(2))
                ca = float(m.group(3))
                if name not in conf_inst:
                    conf_inst[name] = [float("nan")] * gstep
                    conf_avg[name] = [float("nan")] * gstep
                    conf_names.append(name)
                conf_inst[name].append(ci)
                conf_avg[name].append(ca)
                present_conf.add(name)
            for name in conf_names:
                if name not in present_conf:
                    conf_inst[name].append(float("nan"))
                    conf_avg[name].append(float("nan"))

            # Standard metrics
            step.append(gstep)
            epoch.append(ep)
            idx.append(i)
            top1_inst.append(t1_i)
            top1_avg.append(t1_a)
            top5_inst.append(t5_i)
            top5_avg.append(t5_a)

            gstep += 1

    return {
        "step": step,
        "epoch": epoch,
        "idx": idx,
        "top1_inst": top1_inst,
        "top1_avg": top1_avg,
        "top5_inst": top5_inst,
        "top5_avg": top5_avg,
        "losses_inst": losses_inst,
        "losses_avg": losses_avg,
        "conf_inst": conf_inst,
        "conf_avg": conf_avg,
    }


def pick_two_txt_from_dir(dir_path: Path) -> Tuple[Path, Path]:
    txts = sorted([p for p in dir_path.glob("*.txt") if p.is_file()])
    if len(txts) != 2:
        raise ValueError(f"--dir must contain exactly two .txt files, found {len(txts)} in {dir_path}")
    return txts[0], txts[1]


def make_plot(
    data1: Dict[str, List[float]],
    data2: Dict[str, List[float]],
    label1: str,
    label2: str,
    metric: str = "avg",
    title: str | None = None,
    out_path: Path | None = None,
) -> None:
    # Choose which series to plot
    if metric == "avg":
        y1_top1, y1_top5 = data1["top1_avg"], data1["top5_avg"]
        y2_top1, y2_top5 = data2["top1_avg"], data2["top5_avg"]
        metric_name = "Running Average"
    elif metric == "inst":
        y1_top1, y1_top5 = data1["top1_inst"], data1["top5_inst"]
        y2_top1, y2_top5 = data2["top1_inst"], data2["top5_inst"]
        metric_name = "Instantaneous"
    else:
        raise ValueError("metric must be one of: avg, inst")

    x1 = data1["step"]
    x2 = data2["step"]

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top-1 subplot
    ax = axes[0]
    ax.plot(x1, y1_top1, label=f"{label1}")
    ax.plot(x2, y2_top1, label=f"{label2}")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Top-1 Accuracy Progression ({metric_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Top-5 subplot
    ax = axes[1]
    ax.plot(x1, y1_top5, label=f"{label1}")
    ax.plot(x2, y2_top5, label=f"{label2}")
    ax.set_ylabel("Top-5 Accuracy (%)")
    ax.set_xlabel("Step")
    ax.set_title(f"Top-5 Accuracy Progression ({metric_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    else:
        fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def make_plot_single(
    data: Dict[str, Any],
    metric: str = "avg",
    title: str | None = None,
    out_path: Path | None = None,
) -> None:
    # Choose series
    if metric == "avg":
        y_top1, y_top5 = data["top1_avg"], data["top5_avg"]
        losses = data.get("losses_avg", {})
        confs = data.get("conf_avg", {})
        metric_name = "Running Average"
    elif metric == "inst":
        y_top1, y_top5 = data["top1_inst"], data["top5_inst"]
        losses = data.get("losses_inst", {})
        confs = data.get("conf_inst", {})
        metric_name = "Instantaneous"
    else:
        raise ValueError("metric must be one of: avg, inst")

    x = data["step"]

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

    # Top-1/Top-5
    ax = axes[0]
    ax.plot(x, y_top1, label="Top-1")
    ax.plot(x, y_top5, label="Top-5")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"Top-1/Top-5 ({metric_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Losses
    ax = axes[1]
    if losses:
        for name in sorted(losses.keys()):
            ax.plot(x, losses[name], label=f"Loss {name}")
        ax.set_ylabel("Loss")
        ax.set_title("Loss Metrics")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No loss metrics found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # Confidence
    ax = axes[2]
    if confs:
        for name in sorted(confs.keys()):
            ax.plot(x, confs[name], label=f"Conf[{name}]")
        ax.set_ylabel("Confidence")
        ax.set_xlabel("Step")
        ax.set_title("Confidence Metrics")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No confidence metrics found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    else:
        fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def make_plot_compare_full(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    label1: str,
    label2: str,
    metric: str = "avg",
    title: str | None = None,
    out_path: Path | None = None,
) -> None:
    """Compare two runs in one figure: accuracies, losses, confidences.

    Conventions:
      - {label1} is plotted as solid lines (ours)
      - {label2} is plotted as dashed lines (base)
    """
    # Select series according to metric
    if metric == "avg":
        y1_top1, y1_top5 = data1["top1_avg"], data1["top5_avg"]
        y2_top1, y2_top5 = data2["top1_avg"], data2["top5_avg"]
        losses1 = data1.get("losses_avg", {})
        losses2 = data2.get("losses_avg", {})
        confs1 = data1.get("conf_avg", {})
        confs2 = data2.get("conf_avg", {})
        metric_name = "Running Average"
    elif metric == "inst":
        y1_top1, y1_top5 = data1["top1_inst"], data1["top5_inst"]
        y2_top1, y2_top5 = data2["top1_inst"], data2["top5_inst"]
        losses1 = data1.get("losses_inst", {})
        losses2 = data2.get("losses_inst", {})
        confs1 = data1.get("conf_inst", {})
        confs2 = data2.get("conf_inst", {})
        metric_name = "Instantaneous"
    else:
        raise ValueError("metric must be one of: avg, inst")

    x1 = data1["step"]
    x2 = data2["step"]

    # Prepare unions of loss/conf names
    loss_names = sorted(set(losses1.keys()) | set(losses2.keys()))
    conf_names = sorted(set(confs1.keys()) | set(confs2.keys()))

    # Build figure
    n_rows = 4
    fig, axes = plt.subplots(n_rows, 1, figsize=(11, 12), sharex=True)

    # Top-1
    ax = axes[0]
    ax.plot(x1, y1_top1, label=f"Top-1 {label1}", linestyle="-")
    ax.plot(x2, y2_top1, label=f"Top-1 {label2}", linestyle="--")
    ax.set_ylabel("Top-1 (%)")
    ax.set_title(f"Top-1 Accuracy ({metric_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Top-5
    ax = axes[1]
    ax.plot(x1, y1_top5, label=f"Top-5 {label1}", linestyle="-")
    ax.plot(x2, y2_top5, label=f"Top-5 {label2}", linestyle="--")
    ax.set_ylabel("Top-5 (%)")
    ax.set_title(f"Top-5 Accuracy ({metric_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Losses
    ax = axes[2]
    if loss_names:
        for idx, name in enumerate(loss_names):
            y1 = losses1.get(name, [float("nan")] * len(x1))
            y2 = losses2.get(name, [float("nan")] * len(x2))
            # Use a consistent color per loss name via color cycle
            # We don't manually set color to keep compatibility with current mpl rcParams
            line1, = ax.plot(x1, y1, label=f"{name}", linestyle="-")
            ax.plot(x2, y2, linestyle="--", color=line1.get_color())
        ax.set_ylabel("Loss")
        ax.set_title("Loss Metrics (ours=solid, base=dashed)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No loss metrics found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    # Confidence
    ax = axes[3]
    if conf_names:
        for idx, name in enumerate(conf_names):
            y1 = confs1.get(name, [float("nan")] * len(x1))
            y2 = confs2.get(name, [float("nan")] * len(x2))
            line1, = ax.plot(x1, y1, label=f"Conf[{name}]", linestyle="-")
            ax.plot(x2, y2, linestyle="--", color=line1.get_color())
        ax.set_ylabel("Confidence")
        ax.set_xlabel("Step")
        ax.set_title("Confidence Metrics (ours=solid, base=dashed)")
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=3, fontsize=9)
    else:
        ax.text(0.5, 0.5, "No confidence metrics found", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    if title:
        fig.suptitle(title)
        fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    else:
        fig.tight_layout()

    if out_path:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description="Plot Top-1/Top-5 accuracy progression from two ViTTA logs")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=str, help="Directory containing exactly two .txt log files")
    src.add_argument("--file1", type=str, help="Path to first log file (single-file mode if --file2 omitted)")
    parser.add_argument("--file2", type=str, help="Path to second log file (optional; if provided with --file1, plots comparison)")
    parser.add_argument("--labels", nargs=2, type=str, default=None, help="Two legend labels: label1 label2")
    parser.add_argument("--metric", choices=["avg", "inst"], default="avg", help="Plot running averages or instantaneous values")
    parser.add_argument("--title", type=str, default=None, help="Optional overall plot title")
    parser.add_argument("--out", type=str, default=None, help="Output image path (e.g., misc/accuracy_compare.png). If omitted, shows the plot interactively.")

    args = parser.parse_args()

    f1: Path | None = None
    f2: Path | None = None

    if args.dir:
        f1, f2 = pick_two_txt_from_dir(Path(args.dir))
    else:
        if args.file1 and args.file2:
            f1, f2 = Path(args.file1), Path(args.file2)
        elif args.file1:
            f1 = Path(args.file1)
        else:
            parser.error("Provide --dir, or --file1 (and optionally --file2)")

    if not f1 or not f1.exists():
        print(f"Error: file not found: {f1}", file=sys.stderr)
        sys.exit(1)
    if f2 is not None and not f2.exists():
        print(f"Error: file not found: {f2}", file=sys.stderr)
        sys.exit(1)

    data1 = parse_log(f1)
    data2 = parse_log(f2) if f2 is not None else None

    if len(data1["step"]) == 0:
        print(f"Warning: No accuracy lines parsed from {f1}")
    if data2 is not None and len(data2["step"]) == 0:
        print(f"Warning: No accuracy lines parsed from {f2}")

    if data2 is not None:
        if args.labels:
            label1, label2 = args.labels
        else:
            # Default to requested convention: first is 'ours', second is 'base'
            label1, label2 = "ours", "base"

    out_path = Path(args.out) if args.out else None

    if data2 is not None:
        make_plot_compare_full(
            data1=data1,
            data2=data2,
            label1=label1,
            label2=label2,
            metric=args.metric,
            title=args.title if args.title else f"Comparison: {label1} vs {label2}",
            out_path=out_path,
        )
    else:
        make_plot_single(
            data=data1,
            metric=args.metric,
            title=args.title,
            out_path=out_path,
        )


if __name__ == "__main__":
    main()
