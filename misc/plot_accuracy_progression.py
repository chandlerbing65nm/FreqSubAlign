#!/usr/bin/env python3
"""
Plot metrics from ViTTA log .txt files and compare 1, 2, 3, or more runs.

It parses lines like:
2025-08-20 19:00:21,994 - 10 - test_time_adaptation.py - tta_standard - TTA Epoch1: [4/3783]\tTime 0.606 (1.405)\tLoss reg 8.4538 (9.1583)\tLoss wpa 6.2539 (6.3792)\tPrec@1 100.000 (20.000)\tPrec@5 100.000 (40.000)

- The first number after Prec@k is instantaneous accuracy for the current sample.
- The number in parentheses is the running average up to that sample.

Usage examples:
1) Compare explicitly two files (defaults: first is "swa", second is "vitta")
   python misc/plot_accuracy_progression.py --file1 logs/ours.txt --file2 logs/base.txt --out compare.png

2) Compare N files (N>=2)
   python misc/plot_accuracy_progression.py --files logs/a.txt logs/b.txt logs/c.txt --labels A B C --plots acc1 err1

3) Provide a directory containing >=2 .txt files
   python misc/plot_accuracy_progression.py --dir logs/ --plots acc1

Optional:
  --labels L1 L2 ...            Custom legend labels (must match number of files if given)
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
from cycler import cycler

# Global plotting style: thicker lines and distinctive colors
# Use the tab10 palette (10 high-contrast colors). If more are needed, it will cycle.
PALETTE = list(plt.cm.tab10.colors)
plt.rcParams["lines.linewidth"] = 2.8
plt.rcParams["axes.prop_cycle"] = cycler(color=PALETTE)
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["grid.linewidth"] = 0.9
plt.rcParams["legend.framealpha"] = 0.9

# Regex to capture step markers and metrics from different log styles
# 1) TTA style: "TTA Epoch1: [4/3783] ..."
RE_STEP_TTA = re.compile(r"TTA\s+Epoch(\d+):\s*\[(\d+)/(\d+)\]")
# 2) Baseline validate style with explicit epoch: "Test Epoch 37: [4/3783] ..."
RE_STEP_TEST_EPOCH = re.compile(r"Test\s+Epoch\s+(\d+):\s*\[(\d+)/(\d+)\]")
# 3) Baseline validate style without epoch: "Test: [4/3783] ..."
RE_STEP_TEST = re.compile(r"Test:\s*\[(\d+)/(\d+)\]")
# Flexible float (supports integers and scientific notation)
_FLOAT = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
RE_P1 = re.compile(rf"Prec@1\s+({_FLOAT})\s*\((({_FLOAT}))\)")
RE_P5 = re.compile(rf"Prec@5\s+({_FLOAT})\s*\((({_FLOAT}))\)")
# Generic regex for losses and confidence metrics
# Loss with name: allow non-space tokens (letters, digits, underscore, hyphen, etc.)
RE_LOSS = re.compile(rf"Loss\s+([^\s]+)\s+({_FLOAT})\s*\((({_FLOAT}))\)")
# Bare loss without a name: "Loss 4.7036 (4.7036)" -> will be recorded under key 'total'
RE_LOSS_BARE = re.compile(rf"Loss\s+({_FLOAT})\s*\((({_FLOAT}))\)")
RE_CONF = re.compile(rf"Conf\[(.*?)\]\s+({_FLOAT})\s*\((({_FLOAT}))\)")
# Noise-type section header
RE_NOISE_HDR = re.compile(r"^#+\s*Starting\s+Evaluation\s+for\s+:::\s*([A-Za-z0-9_\-]+)\s+corruption\s*#+\s*$")


def parse_log_lines(lines: List[str]) -> Dict[str, Any]:
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
    for line in lines:
            # Anchor on step marker. Support TTA, Test Epoch, and Test formats.
            m_tta = RE_STEP_TTA.search(line)
            if m_tta:
                ep = int(m_tta.group(1))
                i = int(m_tta.group(2))
            else:
                m_test_epoch = RE_STEP_TEST_EPOCH.search(line)
                if m_test_epoch:
                    ep = int(m_test_epoch.group(1))
                    i = int(m_test_epoch.group(2))
                else:
                    m_test = RE_STEP_TEST.search(line)
                    if not m_test:
                        continue
                    ep = -1
                    i = int(m_test.group(1))

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

            # Parse losses present on this line (named and bare)
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
            # Handle bare total loss (no name)
            if RE_LOSS_BARE.search(line):
                m = RE_LOSS_BARE.search(line)
                if m:
                    name = "total"
                    li = float(m.group(1))
                    la = float(m.group(2))
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


def parse_log(file_path: Path) -> Dict[str, Any]:
    """Parse entire log file (no noise-type splitting)."""
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        return parse_log_lines(f.readlines())


def parse_log_by_noise(file_path: Path) -> Dict[str, Dict[str, Any]]:
    """Parse a log into sections keyed by noise type using RE_NOISE_HDR.

    Returns: { noise_type: parsed_data_dict }
    Only lines between a header and the next header (or EOF) are parsed.
    Steps reset to 0 for each section.
    """
    sections: Dict[str, List[str]] = {}
    current: List[str] = []
    current_noise: str | None = None
    with file_path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            m = RE_NOISE_HDR.match(raw.strip())
            if m:
                # flush previous
                if current_noise is not None:
                    sections[current_noise] = current[:]
                current_noise = m.group(1)
                current = []
            else:
                if current_noise is not None:
                    current.append(raw)
    # flush tail
    if current_noise is not None:
        sections[current_noise] = current[:]

    parsed: Dict[str, Dict[str, Any]] = {}
    for noise, lines in sections.items():
        parsed[noise] = parse_log_lines(lines)
    return parsed


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
    fig_scale: float = 1.0,
) -> None:
    def _scale(sz: tuple[float, float]) -> tuple[float, float]:
        return (sz[0] * fig_scale, sz[1] * fig_scale)
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

    fig, axes = plt.subplots(2, 1, figsize=_scale((10, 8)), sharex=True)

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
    ax.set_xlabel("Number of Videos")
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


def make_plot_selected_multi(
    data_list: List[Dict[str, Any]],
    labels: List[str],
    plots: List[str],
    metric: str = "avg",
    title: str | None = None,
    out_path: Path | None = None,
    fig_scale: float = 1.0,
) -> None:
    """Render only the requested subplots for N runs (N >= 1).

    plots: list of items among
      - acc1, err1, loss_reg, loss_consis, acc5, err5
    """
    assert len(data_list) == len(labels) and len(data_list) >= 1

    if metric == "avg":
        tops1 = [d["top1_avg"] for d in data_list]
        tops5 = [d["top5_avg"] for d in data_list]
        losses_all = [d.get("losses_avg", {}) for d in data_list]
        metric_name = "Running Average"
    elif metric == "inst":
        tops1 = [d["top1_inst"] for d in data_list]
        tops5 = [d["top5_inst"] for d in data_list]
        losses_all = [d.get("losses_inst", {}) for d in data_list]
        metric_name = "Instantaneous"
    else:
        raise ValueError("metric must be one of: avg, inst")

    steps = [d["step"] for d in data_list]

    n_rows = len(plots)
    if n_rows == 0:
        print("No plots selected; nothing to render.")
        return
    base_h = max(3 * n_rows, 5)
    fig, axes = plt.subplots(n_rows, 1, figsize=(11 * fig_scale, base_h * fig_scale), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, which in zip(axes, plots):
        if which == "acc1":
            for i, (x, y, lab) in enumerate(zip(steps, tops1, labels)):
                ax.plot(x, y, label=f"{lab}")
            ax.set_ylabel("Top-1 (%)")
            ax.set_title(f"Top-1 Accuracy ({metric_name})")
        elif which == "err1":
            for i, (x, y, lab) in enumerate(zip(steps, tops1, labels)):
                ax.plot(x, [100 - v for v in y], label=f"{lab}")
            ax.set_ylabel("Top-1 Error (%)")
            ax.set_title(f"Top-1 Error ({metric_name})")
        elif which == "acc5":
            for i, (x, y, lab) in enumerate(zip(steps, tops5, labels)):
                ax.plot(x, y, label=f"{lab}")
            ax.set_ylabel("Top-5 (%)")
            ax.set_title(f"Top-5 Accuracy ({metric_name})")
        elif which == "err5":
            for i, (x, y, lab) in enumerate(zip(steps, tops5, labels)):
                ax.plot(x, [100 - v for v in y], label=f"{lab}")
            ax.set_ylabel("Top-5 Error (%)")
            ax.set_title(f"Top-5 Error ({metric_name})")
        elif which == "loss_reg":
            any_found = False
            for i, (x, losses, lab) in enumerate(zip(steps, losses_all, labels)):
                key = _find_loss_key(losses, "reg") if losses else None
                if key:
                    ax.plot(x, losses.get(key, [float("nan")] * len(x)), label=f"{lab}")
                    any_found = True
            if not any_found:
                ax.text(0.5, 0.5, "No 'reg' loss found", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            ax.set_ylabel("Loss reg")
            ax.set_title(f"Loss reg ({metric_name})")
        elif which == "loss_consis":
            any_found = False
            for i, (x, losses, lab) in enumerate(zip(steps, losses_all, labels)):
                key = _find_loss_key(losses, "consis") if losses else None
                if key:
                    ax.plot(x, losses.get(key, [float("nan")] * len(x)), label=f"{lab}")
                    any_found = True
            if not any_found:
                ax.text(0.5, 0.5, "No 'consis' loss found", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            ax.set_ylabel("Loss consis")
            ax.set_title(f"Loss consis ({metric_name})")
        else:
            ax.text(0.5, 0.5, f"Unknown plot '{which}'", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        if ax.has_data():
            ax.grid(True, alpha=0.3)
            ax.legend()

    axes[-1].set_xlabel("Step")

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


def make_plot_top1_and_error_multi(
    data_list: List[Dict[str, Any]],
    labels: List[str],
    metric: str = "avg",
    title: str | None = None,
    out_path: Path | None = None,
    fig_scale: float = 1.0,
) -> None:
    """Create a 2-row figure comparing Top-1 accuracy and Top-1 error for N runs (N>=2)."""
    assert len(data_list) == len(labels) and len(data_list) >= 2

    if metric == "avg":
        tops1 = [d["top1_avg"] for d in data_list]
        metric_name = "Running Average"
    elif metric == "inst":
        tops1 = [d["top1_inst"] for d in data_list]
        metric_name = "Instantaneous"
    else:
        raise ValueError("metric must be one of: avg, inst")

    steps = [d["step"] for d in data_list]

    fig, axes = plt.subplots(2, 1, figsize=(10 * fig_scale, 7 * fig_scale), sharex=True)

    # Top-1 Accuracy
    ax = axes[0]
    for x, y, lab in zip(steps, tops1, labels):
        ax.plot(x, y, label=f"{lab}")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Top-1 Accuracy ({metric_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Top-1 Error
    ax = axes[1]
    for x, y, lab in zip(steps, tops1, labels):
        ax.plot(x, [100.0 - v for v in y], label=f"{lab}")
    ax.set_ylabel("Top-1 Error (%)")
    ax.set_xlabel("Number of Videos")
    ax.set_title(f"Top-1 Error ({metric_name})")
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


def _find_loss_key(losses: Dict[str, List[float]], keyword: str) -> str | None:
    """Find a loss key containing keyword (case-insensitive). Returns first match or None."""
    kw = keyword.lower()
    for name in sorted(losses.keys()):
        if kw in name.lower():
            return name
    # Fallback exact common names
    if keyword in losses:
        return keyword
    return None


def _apply_max_steps(data: Dict[str, Any], max_steps: int | None) -> Dict[str, Any]:
    """Return a sliced copy of parsed data, keeping only first N (or last N if negative) steps.

    If max_steps is None or 0, returns data unchanged.
    """
    if not max_steps:
        return data

    # Decide slice
    sl = slice(0, max_steps) if max_steps > 0 else slice(max_steps, None)

    def _slice_list(x: List[float] | List[int]) -> List[Any]:
        return x[sl] if isinstance(x, list) else x

    out: Dict[str, Any] = {}
    # Top-level series
    for k in [
        "step",
        "epoch",
        "idx",
        "top1_inst",
        "top1_avg",
        "top5_inst",
        "top5_avg",
    ]:
        if k in data:
            out[k] = _slice_list(data[k])

    # Nested dicts
    for k in ["losses_inst", "losses_avg", "conf_inst", "conf_avg"]:
        if k in data and isinstance(data[k], dict):
            out[k] = {name: _slice_list(vals) for name, vals in data[k].items()}
        else:
            out[k] = data.get(k, {})

    return out


def make_plot_selected_compare(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    label1: str,
    label2: str,
    plots: List[str],
    metric: str = "avg",
    title: str | None = None,
    out_path: Path | None = None,
    fig_scale: float = 1.0,
) -> None:
    """Render only the requested subplots for two runs.

    plots: list of items among
      - acc1, err1, loss_reg, loss_consis, acc5, err5
    """
    if metric == "avg":
        top1_1, top5_1 = data1["top1_avg"], data1["top5_avg"]
        top1_2, top5_2 = data2["top1_avg"], data2["top5_avg"]
        losses1 = data1.get("losses_avg", {})
        losses2 = data2.get("losses_avg", {})
        metric_name = "Running Average"
    elif metric == "inst":
        top1_1, top5_1 = data1["top1_inst"], data1["top5_inst"]
        top1_2, top5_2 = data2["top1_inst"], data2["top5_inst"]
        losses1 = data1.get("losses_inst", {})
        losses2 = data2.get("losses_inst", {})
        metric_name = "Instantaneous"
    else:
        raise ValueError("metric must be one of: avg, inst")

    x1, x2 = data1["step"], data2["step"]

    n_rows = len(plots)
    if n_rows == 0:
        print("No plots selected; nothing to render.")
        return
    base_h = max(3 * n_rows, 5)
    fig, axes = plt.subplots(n_rows, 1, figsize=(10 * fig_scale, base_h * fig_scale), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, which in zip(axes, plots):
        if which == "acc1":
            ax.plot(x1, top1_1, label=f"{label1}", linestyle="-")
            ax.plot(x2, top1_2, label=f"{label2}", linestyle="--")
            ax.set_ylabel("Top-1 (%)")
            ax.set_title(f"Top-1 Accuracy ({metric_name})")
        elif which == "err1":
            ax.plot(x1, [100 - v for v in top1_1], label=f"{label1}", linestyle="-")
            ax.plot(x2, [100 - v for v in top1_2], label=f"{label2}", linestyle="--")
            ax.set_ylabel("Top-1 Error (%)")
            ax.set_title(f"Top-1 Error ({metric_name})")
        elif which == "acc5":
            ax.plot(x1, top5_1, label=f"{label1}", linestyle="-")
            ax.plot(x2, top5_2, label=f"{label2}", linestyle="--")
            ax.set_ylabel("Top-5 (%)")
            ax.set_title(f"Top-5 Accuracy ({metric_name})")
        elif which == "err5":
            ax.plot(x1, [100 - v for v in top5_1], label=f"{label1}", linestyle="-")
            ax.plot(x2, [100 - v for v in top5_2], label=f"{label2}", linestyle="--")
            ax.set_ylabel("Top-5 Error (%)")
            ax.set_title(f"Top-5 Error ({metric_name})")
        elif which == "loss_reg":
            key1 = _find_loss_key(losses1, "reg") if losses1 else None
            key2 = _find_loss_key(losses2, "reg") if losses2 else None
            if key1 or key2:
                y1 = losses1.get(key1, [float("nan")] * len(x1)) if key1 else [float("nan")] * len(x1)
                y2 = losses2.get(key2, [float("nan")] * len(x2)) if key2 else [float("nan")] * len(x2)
                line1, = ax.plot(x1, y1, label=f"{label1}", linestyle="-")
                ax.plot(x2, y2, label=f"{label2}", linestyle="--", color=line1.get_color())
            else:
                ax.text(0.5, 0.5, "No 'reg' loss found", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            ax.set_ylabel("Loss reg")
            ax.set_title(f"Loss reg ({metric_name})")
        elif which == "loss_consis":
            # Search for any loss name containing 'consis'
            key1 = _find_loss_key(losses1, "consis") if losses1 else None
            key2 = _find_loss_key(losses2, "consis") if losses2 else None
            if key1 or key2:
                y1 = losses1.get(key1, [float("nan")] * len(x1)) if key1 else [float("nan")] * len(x1)
                y2 = losses2.get(key2, [float("nan")] * len(x2)) if key2 else [float("nan")] * len(x2)
                line1, = ax.plot(x1, y1, label=f"{label1}", linestyle="-")
                ax.plot(x2, y2, label=f"{label2}", linestyle="--", color=line1.get_color())
            else:
                ax.text(0.5, 0.5, "No 'consis' loss found", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            ax.set_ylabel("Loss consis")
            ax.set_title(f"Loss consis ({metric_name})")
        else:
            ax.text(0.5, 0.5, f"Unknown plot '{which}'", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        if ax.has_data():
            ax.grid(True, alpha=0.3)
            ax.legend()

    axes[-1].set_xlabel("Step")

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


def make_plot_selected_single(
    data: Dict[str, Any],
    plots: List[str],
    metric: str = "avg",
    title: str | None = None,
    out_path: Path | None = None,
    fig_scale: float = 1.0,
) -> None:
    """Render only the requested subplots for a single run."""
    if metric == "avg":
        top1 = data["top1_avg"]
        top5 = data["top5_avg"]
        losses = data.get("losses_avg", {})
        metric_name = "Running Average"
    elif metric == "inst":
        top1 = data["top1_inst"]
        top5 = data["top5_inst"]
        losses = data.get("losses_inst", {})
        metric_name = "Instantaneous"
    else:
        raise ValueError("metric must be one of: avg, inst")

    x = data["step"]

    n_rows = len(plots)
    if n_rows == 0:
        print("No plots selected; nothing to render.")
        return
    fig, axes = plt.subplots(n_rows, 1, figsize=(10, max(3 * n_rows, 5)), sharex=True)
    if n_rows == 1:
        axes = [axes]

    for ax, which in zip(axes, plots):
        if which == "acc1":
            ax.plot(x, top1, label="Top-1", linestyle="-")
            ax.set_ylabel("Top-1 (%)")
            ax.set_title(f"Top-1 Accuracy ({metric_name})")
        elif which == "err1":
            ax.plot(x, [100 - v for v in top1], label="Top-1 Error", linestyle="-")
            ax.set_ylabel("Top-1 Error (%)")
            ax.set_title(f"Top-1 Error ({metric_name})")
        elif which == "acc5":
            ax.plot(x, top5, label="Top-5", linestyle="-")
            ax.set_ylabel("Top-5 (%)")
            ax.set_title(f"Top-5 Accuracy ({metric_name})")
        elif which == "err5":
            ax.plot(x, [100 - v for v in top5], label="Top-5 Error", linestyle="-")
            ax.set_ylabel("Top-5 Error (%)")
            ax.set_title(f"Top-5 Error ({metric_name})")
        elif which == "loss_reg":
            key = _find_loss_key(losses, "reg") if losses else None
            if key:
                ax.plot(x, losses.get(key, [float("nan")] * len(x)), label=f"Loss {key}")
            else:
                ax.text(0.5, 0.5, "No 'reg' loss found", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            ax.set_ylabel("Loss reg")
            ax.set_title(f"Loss reg ({metric_name})")
        elif which == "loss_consis":
            key = _find_loss_key(losses, "consis") if losses else None
            if key:
                ax.plot(x, losses.get(key, [float("nan")] * len(x)), label=f"Loss {key}")
            else:
                ax.text(0.5, 0.5, "No 'consis' loss found", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
            ax.set_ylabel("Loss consis")
            ax.set_title(f"Loss consis ({metric_name})")
        else:
            ax.text(0.5, 0.5, f"Unknown plot '{which}'", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()

        if ax.has_data():
            ax.grid(True, alpha=0.3)
            ax.legend()

    axes[-1].set_xlabel("Step")

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
    fig_scale: float = 1.0,
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

    fig, axes = plt.subplots(3, 1, figsize=(10 * fig_scale, 10 * fig_scale), sharex=True)

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
        ax.set_xlabel("Number of Videos")
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
    fig_scale: float = 1.0,
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
    fig, axes = plt.subplots(n_rows, 1, figsize=(11 * fig_scale, 12 * fig_scale), sharex=True)

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
        ax.set_xlabel("Number of Videos")
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


def make_plot_top1_and_error_compare(
    data1: Dict[str, Any],
    data2: Dict[str, Any],
    label1: str,
    label2: str,
    metric: str = "avg",
    title: str | None = None,
    out_path: Path | None = None,
    fig_scale: float = 1.0,
) -> None:
    """Create a 2-row figure comparing Top-1 accuracy and Top-1 error for two runs.

    - Row 1: Top-1 accuracy (%) vs. step
    - Row 2: Top-1 error (100 - Top-1) (%) vs. step
    label1 is drawn with solid line, label2 with dashed line.
    """
    if metric == "avg":
        y1_top1 = data1["top1_avg"]
        y2_top1 = data2["top1_avg"]
        metric_name = "Running Average"
    elif metric == "inst":
        y1_top1 = data1["top1_inst"]
        y2_top1 = data2["top1_inst"]
        metric_name = "Instantaneous"
    else:
        raise ValueError("metric must be one of: avg, inst")

    x1 = data1["step"]
    x2 = data2["step"]

    # Compute errors as 100 - accuracy (NaNs propagate as NaN)
    err1 = [100.0 - v for v in y1_top1]
    err2 = [100.0 - v for v in y2_top1]

    fig, axes = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

    # Top-1 Accuracy
    ax = axes[0]
    ax.plot(x1, y1_top1, label=f"{label1}", linestyle="-")
    ax.plot(x2, y2_top1, label=f"{label2}", linestyle="--")
    ax.set_ylabel("Top-1 Accuracy (%)")
    ax.set_title(f"Top-1 Accuracy ({metric_name})")
    ax.grid(True, alpha=0.3)
    ax.legend()

    # Top-1 Error
    ax = axes[1]
    ax.plot(x1, err1, label=f"{label1}", linestyle="-")
    ax.plot(x2, err2, label=f"{label2}", linestyle="--")
    ax.set_ylabel("Top-1 Error (%)")
    ax.set_xlabel("Number of Videos")
    ax.set_title(f"Top-1 Error ({metric_name})")
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


def main():
    parser = argparse.ArgumentParser(description="Plot accuracy/loss/conf progression for one or many ViTTA logs, optionally per-noise section.")
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--dir", type=str, help="Directory containing >=2 .txt log files")
    src.add_argument("--file1", type=str, help="Path to first log file (single-file mode if --file2 omitted)")
    src.add_argument("--files", nargs="+", type=str, help="Two or more .txt log files to compare")
    parser.add_argument("--file2", type=str, help="Path to second log file (optional; if provided with --file1, plots comparison)")
    parser.add_argument("--labels", nargs="+", type=str, default=None, help="Legend labels matching number of files (default for 2 files: swa vitta; otherwise stems)")
    parser.add_argument("--metric", choices=["avg", "inst"], default="avg", help="Plot running averages or instantaneous values")
    parser.add_argument("--title", type=str, default=None, help="Optional overall plot title")
    parser.add_argument("--out", type=str, default=None, help="Output image path or directory. If per-noise, a file is created per noise type.")
    parser.add_argument(
        "--plots",
        nargs="+",
        choices=["acc1", "err1", "loss_reg", "loss_consis", "acc5", "err5"],
        help="Select one or more subplots to render: acc1 err1 loss_reg loss_consis acc5 err5. If omitted, default behavior is used.",
    )
    parser.add_argument(
        "--figscale",
        type=float,
        default=1.0,
        help="Scale factor applied to figure size (e.g., 1.5 for 50% larger figures).",
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=None,
        help="Limit plotted steps. If positive, keeps first N steps; if negative, keeps last N steps; default: all steps.",
    )

    args = parser.parse_args()

    # Resolve list of files
    file_paths: List[Path] = []
    if args.files:
        file_paths = [Path(p) for p in args.files]
    elif args.dir:
        dirp = Path(args.dir)
        if not dirp.exists() or not dirp.is_dir():
            print(f"Error: directory not found: {dirp}", file=sys.stderr)
            sys.exit(1)
        # Accept both .txt and .out files
        txts = [p for p in dirp.glob("*.txt") if p.is_file()]
        outs = [p for p in dirp.glob("*.out") if p.is_file()]
        file_paths = sorted(txts + outs)
        if len(file_paths) < 2:
            print(
                f"Error: --dir must contain at least two .txt/.out files (found {len(file_paths)})",
                file=sys.stderr,
            )
            sys.exit(1)
    else:
        # Backward compatibility: --file1 (and optional --file2)
        if args.file1 and args.file2:
            file_paths = [Path(args.file1), Path(args.file2)]
        elif args.file1:
            file_paths = [Path(args.file1)]
        else:
            parser.error("Provide --dir, or --files, or --file1 (and optionally --file2)")

    # Validate files exist
    for p in file_paths:
        if not p.exists():
            print(f"Error: file not found: {p}", file=sys.stderr)
            sys.exit(1)

    # Parse logs
    data_list = [parse_log(p) for p in file_paths]
    if args.max_steps is not None:
        data_list = [_apply_max_steps(d, args.max_steps) for d in data_list]
    for p, d in zip(file_paths, data_list):
        if len(d.get("step", [])) == 0:
            print(f"Warning: No accuracy lines parsed from {p}")

    # Labels
    if len(file_paths) == 2:
        if args.labels and len(args.labels) == 2:
            labels = args.labels
        elif args.labels and len(args.labels) != 2:
            print("Warning: --labels length does not match 2 files; using defaults swa vitta")
            labels = ["swa", "vitta"]
        else:
            labels = ["swa", "vitta"]
    else:
        if args.labels:
            if len(args.labels) != len(file_paths):
                print(f"Error: --labels must have {len(file_paths)} items (got {len(args.labels)})", file=sys.stderr)
                sys.exit(1)
            labels = args.labels
        else:
            labels = [p.stem for p in file_paths]

    # Detect per-noise mode if headers exist in any file
    def has_noise_headers(p: Path | None) -> bool:
        if p is None:
            return False
        try:
            with p.open("r", encoding="utf-8", errors="ignore") as f:
                for _ in range(100000):
                    line = f.readline()
                    if not line:
                        break
                    if RE_NOISE_HDR.match(line.strip()):
                        return True
        except Exception:
            return False
        return False

    per_noise = any(has_noise_headers(p) for p in file_paths)

    if per_noise:
        sections_list = [parse_log_by_noise(p) for p in file_paths]
        noise_sets = [set(s.keys()) for s in sections_list]
        noises = sorted(set().union(*noise_sets))
        if not noises:
            print("Warning: No noise sections found. Falling back to overall plot.")
        produced_any = False
        for noise in noises:
            # Require presence in all files for a fair comparison
            if not all(noise in s for s in sections_list):
                print(f"Warning: Noise '{noise}' missing in one of the logs; skipping.")
                continue
            d_list = [s[noise] for s in sections_list]
            if args.max_steps is not None:
                d_list = [_apply_max_steps(d, args.max_steps) for d in d_list]

            # Determine output path per-noise
            if args.out:
                out_arg = Path(args.out)
                if out_arg.suffix.lower() == ".png":
                    out_path = out_arg.with_name(f"{out_arg.stem}_{noise}{out_arg.suffix}")
                else:
                    out_path = (out_arg if out_arg.suffix == "" else out_arg.parent) / f"{noise}.png"
            else:
                out_path = Path("misc/noise_plots") / f"{noise}.png"

            if args.plots:
                title = args.title if args.title else f"{noise} — " + " vs ".join(labels)
                make_plot_selected_multi(
                    data_list=d_list,
                    labels=labels,
                    plots=args.plots,
                    metric=args.metric,
                    title=title,
                    out_path=out_path,
                    fig_scale=args.figscale,
                )
            else:
                if len(d_list) == 1:
                    title = args.title if args.title else f"{noise}"
                    make_plot_single(
                        data=d_list[0],
                        metric=args.metric,
                        title=title,
                        out_path=out_path,
                        fig_scale=args.figscale,
                    )
                elif len(d_list) == 2:
                    title = args.title if args.title else f"{noise} — {labels[0]} vs {labels[1]}"
                    make_plot_top1_and_error_compare(
                        data1=d_list[0],
                        data2=d_list[1],
                        label1=labels[0],
                        label2=labels[1],
                        metric=args.metric,
                        title=title,
                        out_path=out_path,
                        fig_scale=args.figscale,
                    )
                else:
                    title = args.title if args.title else f"{noise} — " + " vs ".join(labels)
                    make_plot_top1_and_error_multi(
                        data_list=d_list,
                        labels=labels,
                        metric=args.metric,
                        title=title,
                        out_path=out_path,
                        fig_scale=args.figscale,
                    )
            produced_any = True
        if produced_any:
            return

    # Fallback: overall comparison or single-run
    out_path = Path(args.out) if args.out else None
    if args.plots:
        title = args.title if args.title else ("Comparison: " + " vs ".join(labels) if len(labels) >= 2 else None)
        make_plot_selected_multi(
            data_list=data_list,
            labels=labels,
            plots=args.plots,
            metric=args.metric,
            title=title,
            out_path=out_path,
            fig_scale=args.figscale,
        )
    else:
        if len(data_list) == 1:
            make_plot_single(
                data=data_list[0],
                metric=args.metric,
                title=args.title,
                out_path=out_path,
                fig_scale=args.figscale,
            )
        elif len(data_list) == 2:
            make_plot_top1_and_error_compare(
                data1=data_list[0],
                data2=data_list[1],
                label1=labels[0],
                label2=labels[1],
                metric=args.metric,
                title=args.title if args.title else f"Comparison: {labels[0]} vs {labels[1]}",
                out_path=out_path,
                fig_scale=args.figscale,
            )
        else:
            make_plot_top1_and_error_multi(
                data_list=data_list,
                labels=labels,
                metric=args.metric,
                title=args.title if args.title else ("Comparison: " + " vs ".join(labels)),
                out_path=out_path,
                fig_scale=args.figscale,
            )


if __name__ == "__main__":
    main()
