"""Generate training curves figure for EquiDexFlow paper (Figure 7).

Two subplots side by side:
  Left:  total loss, contact loss, force loss vs. steps (all variants overlaid)
  Right: physics_wrench and physics_friction vs. steps (Full only - shows
         cone projection makes friction flat at zero while wrench descends)

Uses TensorBoard event files from completed training runs.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 7.5,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 200,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

BASE = Path("/home/drce/ResearchProjects/MuJoCoDex/third_party/grasp_syn/EquiDexFlow/outputs/training_results")

RUNS = {
    "PoseOnly":    BASE / "equidexflow_dex_pose_only/20260521-2014",
    "ContactOnly": BASE / "equidexflow_dex_contact_only/20260521-2043",
    "GeomOnly":    BASE / "equidexflow_dex_geom_only_81/20260521-2304",
    "Full":        BASE / "equidexflow_dex_full/20260521-1625",
}

COLORS = {
    "PoseOnly":    "#1f77b4",
    "ContactOnly": "#ff7f0e",
    "GeomOnly":    "#2ca02c",
    "Full":        "#d62728",
}


def load_scalars(logdir: Path, tag: str) -> tuple[np.ndarray, np.ndarray]:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(logdir))
    ea.Reload()
    events = ea.Scalars(tag)
    steps = np.array([e.step for e in events])
    vals  = np.array([e.value for e in events])
    return steps, vals


def smooth(vals: np.ndarray, window: int = 5) -> np.ndarray:
    if len(vals) < window:
        return vals
    kernel = np.ones(window) / window
    return np.convolve(vals, kernel, mode="same")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=str,
                        default="/home/drce/ResearchPapers/CORL2026_II_EquiDexFlow_Contact-Grounded-SE-3--Equivariant_Dexterous_Grasp_Generative_Flows/figures/training_curves.pdf")
    parser.add_argument("--window", type=int, default=5)
    args = parser.parse_args()

    fig, axes = plt.subplots(1, 3, figsize=(7.5, 2.4))
    ax_flow, ax_force, ax_phys = axes

    # (a) Flow loss - comparable across all variants
    for name, logdir in RUNS.items():
        steps, vals = load_scalars(logdir, "val/flow")
        ax_flow.plot(steps, smooth(vals, args.window), label=name,
                     color=COLORS[name], linewidth=1.2)
    ax_flow.set_xlabel("Training Step")
    ax_flow.set_ylabel("Loss")
    ax_flow.set_title("(a) SE(3) Flow Loss")
    ax_flow.legend(loc="upper right")
    ax_flow.grid(True, alpha=0.3)

    # (b) Force loss - only meaningful for GeomOnly and Full
    for name in ["GeomOnly", "Full"]:
        logdir = RUNS[name]
        steps, vals = load_scalars(logdir, "val/force")
        ax_force.plot(steps, smooth(vals, args.window), label=name,
                      color=COLORS[name], linewidth=1.2)
    ax_force.set_xlabel("Training Step")
    ax_force.set_ylabel("Loss")
    ax_force.set_title("(b) Force Loss")
    ax_force.legend(loc="upper right")
    ax_force.grid(True, alpha=0.3)

    # (c) Physics losses - Full model only
    logdir = RUNS["Full"]
    steps_w, vals_w = load_scalars(logdir, "train/physics_wrench")
    steps_f, vals_f = load_scalars(logdir, "train/physics_friction")
    ax_phys.plot(steps_w, smooth(vals_w, args.window),
                 label="Wrench balance", color="#d62728", linewidth=1.2)
    ax_phys.plot(steps_f, smooth(vals_f, args.window),
                 label="Friction violation", color="#9467bd", linewidth=1.2,
                 linestyle="--")
    ax_phys.set_xlabel("Training Step")
    ax_phys.set_ylabel("Loss")
    ax_phys.set_title("(c) Physics Losses (Full)")
    ax_phys.legend(loc="upper right")
    ax_phys.grid(True, alpha=0.3)

    fig.tight_layout(w_pad=2.0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out)
    print(f"Saved to {args.out}")

    png_out = args.out.replace(".pdf", ".png")
    fig.savefig(png_out)
    print(f"Saved PNG to {png_out}")


if __name__ == "__main__":
    main()
