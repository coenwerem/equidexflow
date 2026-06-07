#!/usr/bin/env python3
"""Generate the three training-curve PDFs the paper expects:
  - figures/training_hand_q_nll.pdf    (left)
  - figures/training_geometric.pdf     (centre)
  - figures/training_total_physics.pdf (right)

Reads the `[ Validation ]` lines from a run's logging.log and plots each panel
with consistent styling. Restored from origin/phydex_port and adapted for the
LEAP Full run (jl_20260525-0040), whose validation lines carry the extra
`normal=` and `reach=` fields.

Usage:
  ~/ResearchProjects/MuJoCoDex/.venv/bin/python \
    ~/ResearchProjects/frogger/scripts/equidex/plot_training_curves_3panel.py \
    [--log <logging.log>] [--out-dir <dir>] [--step-max N]
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

# --- CMU Sans Serif (matches MuJoCoDex figure runners) ---
_CMU_BOLD = Path("/usr/share/fonts/truetype/cmu/cmunsx.ttf")
_CMU_REG = Path("/usr/share/fonts/truetype/cmu/cmunss.ttf")
if _CMU_BOLD.exists():
    fm.fontManager.addfont(str(_CMU_BOLD))
if _CMU_REG.exists():
    fm.fontManager.addfont(str(_CMU_REG))

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["CMU Sans Serif", "DejaVu Sans", "Arial"],
    "font.size":          14,
    "axes.labelsize":     16,
    "axes.titlesize":     16,
    "axes.titleweight":   "bold",
    "axes.labelweight":   "normal",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.0,
    "xtick.labelsize":    13,
    "ytick.labelsize":    13,
    "legend.fontsize":    12,
    "figure.dpi":         600,
    "savefig.dpi":        600,
    "savefig.bbox":       "tight",
    "grid.alpha":         0.25,
    "grid.linewidth":     0.4,
    "text.usetex":        False,
    "axes.unicode_minus": False,
})

# --- MuJoCoDex palette ---
C_DEEP_BLUE = "#0d47a1"
C_VERMILION = "#bf360c"
C_FOREST    = "#2e7d32"
C_NEUTRAL   = "#757575"

DEFAULT_PAPER_DIR = Path.home() / ("ResearchPapers/CORL2026_II_EquiDexFlow_"
    "Contact-Grounded-SE-3--Equivariant_Dexterous_Grasp_Generative_Flows/figures")
DEFAULT_LOG = Path("/home/drce/ResearchProjects/MuJoCoDex/third_party/grasp_syn/"
    "EquiDexFlow/outputs/training_results/equidexflow_leap_full_fc_reach_flow/"
    "jl_20260525-0040/logging.log")

# Tolerant named-group regex: normal/reach optional, trailing fields ignored.
VAL_PATTERN = re.compile(
    r"\[ Validation \].*?step=(?P<step>\d+)\s+"
    r"flow=(?P<flow>[\d.]+)\s+hand_q=(?P<hand_q>-?[\d.]+)\s+"
    r"contact=(?P<contact>[\d.]+)\s+"
    r"(?:normal=(?P<normal>[\d.]+)\s+)?"
    r"force=(?P<force>[\d.]+)\s+"
    r"physics_wrench=(?P<wrench>[\d.]+)\s+"
    r"physics_friction=(?P<friction>[\d.]+)\s+"
    r"physics_collision=(?P<collision>[\d.]+)\s+"
    r"(?:reach=(?P<reach>[\d.]+)\s+)?"
    r"total=(?P<total>[\d.]+)"
)

PANEL_W, PANEL_H = 3.2, 2.2


def parse_log(path: Path):
    keys = ["step", "flow", "hand_q", "contact", "force", "wrench",
            "friction", "collision", "total"]
    out = {k: [] for k in keys}
    with open(path) as f:
        for line in f:
            m = VAL_PATTERN.search(line)
            if not m:
                continue
            out["step"].append(int(m.group("step")))
            for k in keys[1:]:
                out[k].append(float(m.group(k)))
    return {k: np.asarray(v) for k, v in out.items()}


def _setup_axis(ax, xlabel, ylabel):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, linestyle="-", alpha=0.15, linewidth=0.4)


def plot_hand_q_nll(d, out_dir):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    ax.plot(d["step"], d["hand_q"], color=C_DEEP_BLUE, linewidth=2.0, label="Hand-q NLL")
    ax.axhline(0, color=C_NEUTRAL, linewidth=0.6, linestyle="--", alpha=0.5)
    _setup_axis(ax, "Training step", "Per-dim NLL")
    ax.legend(loc="center right", framealpha=0.9, edgecolor="none",
              fancybox=False, bbox_to_anchor=(1.0, 0.05))
    fig.tight_layout(pad=0.4)
    out = out_dir / "training_hand_q_nll.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    print(f"  wrote {out}"); plt.close(fig)


def plot_geometric(d, out_dir):
    fig, ax = plt.subplots(figsize=(PANEL_W, PANEL_H))
    ax.plot(d["step"], d["flow"], color=C_FOREST, linewidth=2.0, label="Flow matching")
    _setup_axis(ax, "Training step", "Flow loss")
    ax.legend(loc="center right", framealpha=0.9, edgecolor="none",
              fancybox=False, bbox_to_anchor=(1.0, 0.5))
    fig.tight_layout(pad=0.4)
    out = out_dir / "training_geometric.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    print(f"  wrote {out}"); plt.close(fig)


def plot_total_physics(d, out_dir):
    fig, ax1 = plt.subplots(figsize=(PANEL_W + 0.4, PANEL_H))
    l1 = ax1.plot(d["step"], d["total"], color=C_DEEP_BLUE, linewidth=2.0, label="Total")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Total loss", color=C_DEEP_BLUE)
    ax1.tick_params(axis="y", labelcolor=C_DEEP_BLUE)
    ax1.grid(True, linestyle="-", alpha=0.15, linewidth=0.4)
    ax1.spines["top"].set_visible(False)
    ax2 = ax1.twinx()
    ax2.spines["right"].set_visible(True)
    l2 = ax2.plot(d["step"], d["wrench"], color=C_VERMILION, linewidth=2.0,
                  label="Wrench loss", alpha=0.9)
    ax2.set_ylabel("Wrench loss", color=C_VERMILION)
    ax2.tick_params(axis="y", labelcolor=C_VERMILION)
    ax2.spines["top"].set_visible(False)
    lines = l1 + l2
    ax1.legend(lines, [l.get_label() for l in lines], loc="upper right",
               framealpha=0.9, edgecolor="none", fancybox=False)
    fig.tight_layout(pad=0.5)
    out = out_dir / "training_total_physics.pdf"
    fig.savefig(out); fig.savefig(str(out).replace(".pdf", ".png"))
    print(f"  wrote {out}"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", type=Path, default=DEFAULT_LOG)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_PAPER_DIR)
    ap.add_argument("--step-max", type=int, default=11000)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"=== Parsing {args.log} ===")
    d = parse_log(args.log)
    if len(d["step"]) == 0:
        print("ERROR: no validation points found"); return
    mask = d["step"] <= args.step_max
    d = {k: v[mask] for k, v in d.items()}
    print(f"=== {len(d['step'])} val points; steps {d['step'][0]}..{d['step'][-1]} ===")
    print(f"    hand_q NLL: {d['hand_q'][0]:.3f} -> {d['hand_q'][-1]:.3f}")
    print(f"    flow: {d['flow'][0]:.3f} -> {d['flow'][-1]:.3f}")
    print(f"    total: {d['total'][0]:.3f} -> {d['total'][-1]:.3f}")
    print(f"    wrench: {d['wrench'][0]:.3f} -> {d['wrench'][-1]:.3f}")
    plot_hand_q_nll(d, args.out_dir)
    plot_geometric(d, args.out_dir)
    plot_total_physics(d, args.out_dir)
    print("Done.")


if __name__ == "__main__":
    main()
