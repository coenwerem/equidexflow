#!/usr/bin/env python3
"""Regenerate the Top-1 score distribution violin plot for the paper.

Bumped font sizes, no title, CMU Sans Serif. Outputs directly to the paper
figures directory as leap_score_distributions.pdf.

Usage:
    ~/frogger_env/bin/python scripts/equidex/plot_score_distributions.py
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

_CMU_BOLD = Path("/usr/share/fonts/truetype/cmu/cmunsx.ttf")
_CMU_REG = Path("/usr/share/fonts/truetype/cmu/cmunss.ttf")
if _CMU_BOLD.exists():
    fm.fontManager.addfont(str(_CMU_BOLD))
if _CMU_REG.exists():
    fm.fontManager.addfont(str(_CMU_REG))

plt.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["CMU Sans Serif", "DejaVu Sans", "Arial"],
    "font.size":          13,
    "axes.labelsize":     15,
    "axes.titlesize":     15,
    "axes.titleweight":   "bold",
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     0.8,
    "xtick.labelsize":    13,
    "ytick.labelsize":    12,
    "legend.fontsize":    12,
    "figure.dpi":         300,
    "savefig.dpi":        300,
    "savefig.bbox":       "tight",
    "grid.alpha":         0.2,
    "grid.linewidth":     0.4,
    "text.usetex":        False,
    "axes.unicode_minus": False,
})

C_DEEP_BLUE = "#0d47a1"
C_VERMILION = "#bf360c"
C_FOREST    = "#2e7d32"
C_PURPLE    = "#6a1b9a"
C_NEUTRAL   = "#757575"

RESULTS_DIR = Path.home() / "ResearchProjects/frogger/outputs/paper_results/equidex/equidex_results"
OUT = Path.home() / "ResearchPapers/CORL2026_II_EquiDexFlow_Contact-Grounded-SE-3--Equivariant_Dexterous_Grasp_Generative_Flows/figures/leap_score_distributions.pdf"

VARIANTS = {
    "Full":        "equidexflow_leap_full_fc_reach_flow",
    "GeomOnly":    "equidexflow_leap_geom_only_fc_reach_flow",
    "PoseOnly":    "equidexflow_leap_pose_only_fc_reach_flow",
    "ContactOnly": "equidexflow_leap_contact_only_fc_reach_flow",
}
VARIANT_COLORS = {
    "Full": C_DEEP_BLUE, "GeomOnly": C_FOREST,
    "PoseOnly": C_VERMILION, "ContactOnly": C_PURPLE,
}


def load_rollout(variant_dir):
    path = RESULTS_DIR / variant_dir / "rollout.csv"
    top1 = []
    with open(path) as f:
        for row in csv.DictReader(f):
            top1.append(float(row["top1_score"]))
    return np.array(top1)


def main():
    from scipy.stats import bootstrap

    data = {name: load_rollout(vdir) for name, vdir in VARIANTS.items()}

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    variant_names = list(VARIANTS.keys())
    positions = range(1, len(variant_names) + 1)

    parts = ax.violinplot([data[v] for v in variant_names], positions=positions,
                          showmeans=False, showextrema=False, widths=0.7)
    for pc, vname in zip(parts["bodies"], variant_names):
        pc.set_facecolor(VARIANT_COLORS[vname])
        pc.set_alpha(0.6)
        pc.set_edgecolor("none")

    rng = np.random.default_rng(42)
    for i, vname in enumerate(variant_names):
        jitter = rng.uniform(-0.12, 0.12, size=len(data[vname]))
        ax.scatter(np.full_like(data[vname], i + 1) + jitter, data[vname],
                   c=VARIANT_COLORS[vname], s=10, alpha=0.4, edgecolors="none", zorder=3)

    for i, vname in enumerate(variant_names):
        arr = data[vname]
        m = float(np.mean(arr))
        ci = bootstrap((arr,), np.mean, n_resamples=10000, method="BCa",
                       confidence_level=0.95, random_state=0).confidence_interval
        ax.plot([i + 1 - 0.18, i + 1 + 0.18], [m, m], color="black", lw=2.2, zorder=5)
        ax.errorbar(i + 1, m, yerr=[[m - ci.low], [ci.high - m]], fmt="none",
                    ecolor="black", elinewidth=1.4, capsize=5, capthick=1.4, zorder=5)

    ymax = max(np.max(data[v]) for v in variant_names)
    span = ymax - min(np.min(data[v]) for v in variant_names)
    targets = ["GeomOnly", "ContactOnly", "PoseOnly"]
    for k, tgt in enumerate(targets):
        x1, x2 = 1, variant_names.index(tgt) + 1
        y = ymax + span * (0.06 + 0.085 * k)
        ax.plot([x1, x1, x2, x2], [y - span * 0.015, y, y, y - span * 0.015],
                color=C_NEUTRAL, lw=1.0, zorder=4)
        ax.text((x1 + x2) / 2, y + span * 0.005, "p < 0.001", ha="center",
                va="bottom", fontsize=11, color=C_NEUTRAL)
    ax.set_ylim(top=ymax + span * (0.06 + 0.085 * len(targets)) + span * 0.05)

    ax.set_xticks(list(positions))
    ax.set_xticklabels(variant_names)
    ax.set_ylabel("Top-1 composite score")
    ax.grid(True, axis="y", alpha=0.15)

    fig.tight_layout()
    fig.savefig(OUT)
    print(f"Saved to {OUT}")
    plt.close(fig)


if __name__ == "__main__":
    main()
