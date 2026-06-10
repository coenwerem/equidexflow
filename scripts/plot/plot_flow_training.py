#!/usr/bin/env python3
"""Plot training curves for the EquiDexFlow flow model (hand_q NLL variant).

Shows validation losses vs training step for the Conditional RealNVP run.
Parses logging.log directly - no TensorBoard dependency.

Usage:
    ~/frogger_env/bin/python scripts/equidex/plot_flow_training.py
"""
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import matplotlib.gridspec as gridspec
import numpy as np

_CMU_BOLD = Path("/usr/share/fonts/truetype/cmu/cmunsx.ttf")
_CMU_REG  = Path("/usr/share/fonts/truetype/cmu/cmunss.ttf")
if _CMU_BOLD.exists():
    fm.fontManager.addfont(str(_CMU_BOLD))
if _CMU_REG.exists():
    fm.fontManager.addfont(str(_CMU_REG))

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["CMU Sans Serif", "DejaVu Sans", "Arial"],
    "font.size":          15,
    "axes.labelsize":     18,
    "axes.titlesize":     20,
    "axes.titleweight":   "bold",
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "axes.linewidth":     1,
    "legend.fontsize":    14,
    "xtick.labelsize":    14,
    "ytick.labelsize":    14,
    "figure.dpi":         300,
    "grid.alpha":         0.12,
    "grid.linewidth":     0.3,
    "text.usetex":        False,
    "axes.unicode_minus": False,
})

LOG = Path.home() / "ResearchProjects/MuJoCoDex/third_party/grasp_syn/EquiDexFlow/outputs/training_results/equidexflow_dex_full_flow/20260523-0233/logging.log"

VAL_PATTERN = re.compile(
    r"\[ Validation \].*step=(\d+)\s+"
    r"flow=([\d.]+)\s+hand_q=([\d.e+-]+)\s+contact=([\d.]+)\s+"
    r"(?:normal=[\d.]+\s+)?"
    r"force=([\d.]+)\s+"
    r"physics_wrench=([\d.]+)\s+"
    r"physics_friction=([\d.]+)\s+"
    r"physics_collision=([\d.]+)\s+"
    r"total=([\d.]+)"
)

TRAIN_PATTERN = re.compile(
    r"\[\s+Training\s+\].*step=(\d+)\s+"
    r"flow=([\d.e+-]+)\s+hand_q=([\d.e+-]+)\s+contact=([\d.e+-]+)\s+"
    r"(?:normal=[\d.e+-]+\s+)?"
    r"force=([\d.e+-]+)\s+"
    r"physics_wrench=([\d.e+-]+)\s+"
    r"physics_friction=([\d.e+-]+)\s+"
    r"physics_collision=([\d.e+-]+)\s+"
    r"total=([\d.e+-]+)"
)

X_MAX = 12000


def parse_log(path: Path, pattern) -> dict:
    steps, flow, hand_q, contact, force, wrench, friction, collision, total = (
        [] for _ in range(9)
    )
    with open(path) as f:
        for line in f:
            m = pattern.search(line)
            if m:
                steps.append(int(m.group(1)))
                flow.append(float(m.group(2)))
                hand_q.append(float(m.group(3)))
                contact.append(float(m.group(4)))
                force.append(float(m.group(5)))
                wrench.append(float(m.group(6)))
                friction.append(float(m.group(7)))
                collision.append(float(m.group(8)))
                total.append(float(m.group(9)))
    return {k: np.array(v) for k, v in {
        "step": steps, "flow": flow, "hand_q": hand_q, "contact": contact,
        "force": force, "wrench": wrench, "friction": friction,
        "collision": collision, "total": total,
    }.items()}


def clip(data: dict, x_max: int) -> dict:
    mask = data["step"] <= x_max
    return {k: v[mask] for k, v in data.items()}


def _plot_hand_q_nll(ax, val, train):
    ax.plot(val["step"], val["hand_q"], color="#d62728", linewidth=2.2, label="Val")
    if len(train["step"]) > 0:
        ax.plot(train["step"], train["hand_q"], color="#d62728", linewidth=1.0,
                alpha=0.3, label="Train")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Hand Q NLL")
    ax.set_xlim(0, X_MAX)
    ax.legend(loc="upper right")


def _plot_geometric(ax, val):
    ax.plot(val["step"], val["flow"], color="#1f77b4", linewidth=2.2, label="Flow")
    ax.plot(val["step"], val["contact"] * 100, color="#2ca02c", linewidth=2.2,
            label=r"Contact ($\times$100)")
    ax.plot(val["step"], val["collision"] * 10, color="#ff7f0e", linewidth=2.2,
            label=r"Collision ($\times$10)")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, X_MAX)
    ax.legend(loc="right")


def _plot_total_physics(ax, val):
    ax.plot(val["step"], val["total"], color="#d62728", linewidth=2.2, label="Total")
    ax.plot(val["step"], val["wrench"], color="#9467bd", linewidth=2.2,
            linestyle="--", label="Wrench")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_xlim(0, X_MAX)
    ax.legend(loc="upper right")


def main():
    val_raw = parse_log(LOG, VAL_PATTERN)
    train_raw = parse_log(LOG, TRAIN_PATTERN)
    print(f"Val: {len(val_raw['step'])} points, Train: {len(train_raw['step'])} points")

    val = clip(val_raw, X_MAX)
    train = clip(train_raw, X_MAX)
    print(f"After clipping to x<={X_MAX}: Val {len(val['step'])}, Train {len(train['step'])}")

    out_dir = Path.home() / "ResearchProjects/frogger/outputs/paper_figures_flow"
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Combined 3-panel figure (no subtitles) ---
    fig = plt.figure(figsize=(15, 4.5))
    gs = gridspec.GridSpec(1, 3, figure=fig,
                           left=0.06, right=0.98,
                           bottom=0.18, top=0.96,
                           wspace=0.32)

    _plot_hand_q_nll(fig.add_subplot(gs[0, 0]), val, train)
    _plot_geometric(fig.add_subplot(gs[0, 1]), val)
    _plot_total_physics(fig.add_subplot(gs[0, 2]), val)

    combined = out_dir / "flow_training_curves.png"
    fig.savefig(str(combined), dpi=300, bbox_inches="tight",
                facecolor="white", pad_inches=0.08)
    fig.savefig(str(combined.with_suffix(".pdf")), bbox_inches="tight",
                facecolor="white", pad_inches=0.08)
    plt.close(fig)
    print(f"Combined -> {combined}")

    # --- Individual figures (no subtitles) ---
    for name, plot_fn, extra in [
        ("hand_q_nll", _plot_hand_q_nll, {"train": train}),
        ("geometric_losses", _plot_geometric, {}),
        ("total_physics", _plot_total_physics, {}),
    ]:
        fig_i, ax_i = plt.subplots(figsize=(5.5, 4))
        if "train" in extra:
            plot_fn(ax_i, val, extra["train"])
        else:
            plot_fn(ax_i, val)
        path_i = out_dir / f"flow_{name}.png"
        fig_i.savefig(str(path_i), dpi=300, bbox_inches="tight",
                      facecolor="white", pad_inches=0.08)
        fig_i.savefig(str(path_i.with_suffix(".pdf")), bbox_inches="tight",
                      facecolor="white", pad_inches=0.08)
        plt.close(fig_i)
        print(f"Individual -> {path_i}")


if __name__ == "__main__":
    main()
