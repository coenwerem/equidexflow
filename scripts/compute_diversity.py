#!/usr/bin/env python3
"""Compute diversity and coverage metrics for EquiDexFlow variants.

For each variant, generates K=20 grasp candidates per test object and computes:
  - Translation diversity: std of wrist translation (mm)
  - Rotation diversity: std of wrist rotation angle from mean (degrees)
  - Joint diversity: mean std across 16 joint angles (radians)
  - Contact spread: mean pairwise distance between contact sets (mm)
  - Coverage@k: fraction of objects where at least one grasp exceeds threshold

Usage:
    ~/ResearchProjects/MuJoCoDex/.venv/bin/python \
        ~/ResearchProjects/frogger/scripts/equidex/compute_diversity.py \
        --device 0 --num-samples 20
"""
from __future__ import annotations

import os
import sys

_EDF_DIR = os.environ.get("EQUIDEXFLOW_OUTPUTS_DIR", os.getcwd())

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from equidexflow.models import get_dex_model
from equidexflow.loaders import get_dataloader
from equidexflow.physics.scorer import GraspScorer

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

FROGGER_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = FROGGER_ROOT / "outputs" / "paper_results" / "equidex" / "diversity_metrics"

EQUIDEXFLOW_DIR = Path(_EDF_DIR)

VARIANTS = {
    "Full": "equidexflow_leap_full_fc_reach_flow",
    "GeomOnly": "equidexflow_leap_geom_only_fc_reach_flow",
    "PoseOnly": "equidexflow_leap_pose_only_fc_reach_flow",
    "ContactOnly": "equidexflow_leap_contact_only_fc_reach_flow",
}

TEST_CONFIG = {
    "dataset": {
        "name": "dexgrasp",
        "grasp_db_dir": str(Path.home() / "ResearchProjects/frogger/outputs/datasets/dexgraspdb/v3/leap"),
        "object_mesh_dir": str(Path.home() / "ResearchProjects/MuJoCoDex/assets/misc/objects"),
        "n_object_points": 512,
        "max_contacts": 64,
        "mu": 0.5,
        "object_mass": 0.2,
        "augment": False,
        "split": "test",
        "object_names": None,
    },
    "batch_size": 2,
    "num_workers": 4,
    "shuffle": False,
}


def _latest_run(variant_dir: str) -> Path:
    base = EQUIDEXFLOW_DIR / "outputs" / "training_results" / variant_dir
    runs = [p for p in base.iterdir() if p.is_dir() and (p / "checkpoint_best.pt").is_file()]
    return max(runs, key=lambda p: (p / "checkpoint_best.pt").stat().st_mtime)
def _unused_old_latest(base):
    runs = sorted(p for p in base.iterdir() if p.is_dir())
    return runs[-1]


# ---------------------------------------------------------------------------
# Diversity computation
# ---------------------------------------------------------------------------

def rotation_diversity(rotations: np.ndarray) -> float:
    """Std of rotation angles (degrees) from the Frechet mean.

    Args:
        rotations: (K, 3, 3) rotation matrices
    Returns:
        std of geodesic distances from mean rotation in degrees
    """
    K = rotations.shape[0]
    if K < 2:
        return 0.0

    r_scipy = Rotation.from_matrix(rotations)
    mean_rot = r_scipy.mean()
    angles = []
    for i in range(K):
        diff = mean_rot.inv() * r_scipy[i]
        angles.append(diff.magnitude())
    return float(np.std(angles) * 180.0 / np.pi)


def translation_diversity(translations: np.ndarray) -> float:
    """Std of wrist translations (mm)."""
    return float(np.std(np.linalg.norm(translations - translations.mean(axis=0), axis=1)) * 1000.0)


def joint_diversity(hand_qs: np.ndarray) -> float:
    """Mean per-joint std across samples (radians)."""
    return float(np.mean(np.std(hand_qs, axis=0)))


def contact_spread(contacts_list: np.ndarray) -> float:
    """Mean pairwise L2 distance between contact sets (mm).

    Args:
        contacts_list: (K, n_fingers, 3)
    Returns:
        mean distance in mm
    """
    K = contacts_list.shape[0]
    if K < 2:
        return 0.0
    # Flatten each contact set to (n_fingers*3,)
    flat = contacts_list.reshape(K, -1)
    dists = []
    for i in range(K):
        for j in range(i + 1, K):
            dists.append(np.linalg.norm(flat[i] - flat[j]))
    return float(np.mean(dists) * 1000.0)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate_diversity(
    model, loader, device: torch.device, num_samples: int, scorer: GraspScorer,
    quality_threshold: float = -4.0,
) -> dict:
    """Generate K samples per object, compute diversity and coverage."""
    pts_by_object: dict[str, torch.Tensor] = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="  Collecting objects", leave=False):
            names = batch["object_name"]
            obj_pts = batch["object_points"]
            for b in range(obj_pts.shape[0]):
                name = names[b]
                if name not in pts_by_object:
                    pts_by_object[name] = obj_pts[b].cpu()

        objects = sorted(pts_by_object.keys())
        n_objects = len(objects)

        per_object = []
        coverage_counts = {k: 0 for k in [1, 4, 8, 20]}

        for name in tqdm(objects, desc="  Computing diversity", leave=False):
            obj_pts_1 = pts_by_object[name].unsqueeze(0).to(device)
            obj_pts_n3 = obj_pts_1.permute(0, 2, 1)

            preds = model.sample(obj_pts_1, num_samples)
            ranked = scorer.rank_candidates(preds, obj_pts_n3[0])

            # Extract arrays
            wrist_poses = np.array([p["wrist_pose"].cpu().numpy() for p in preds])
            hand_qs = np.array([p["hand_q"].cpu().numpy() for p in preds])
            contacts_arr = np.array([p["contacts"].cpu().numpy() for p in preds])
            scores = [c["score"] for c in ranked]

            K = len(preds)
            rotations = wrist_poses[:, :3, :3]
            translations = wrist_poses[:, :3, 3]

            obj_metrics = {
                "object": name,
                "trans_div_mm": translation_diversity(translations),
                "rot_div_deg": rotation_diversity(rotations),
                "joint_div_rad": joint_diversity(hand_qs),
                "contact_spread_mm": contact_spread(contacts_arr),
                "top1_score": scores[0],
                "mean_score": float(np.mean(scores)),
            }
            per_object.append(obj_metrics)

            # Coverage@k
            for k in coverage_counts:
                k_eff = min(k, len(scores))
                if any(s > quality_threshold for s in scores[:k_eff]):
                    coverage_counts[k] += 1

    # Aggregate
    agg = {
        "trans_div_mm": float(np.mean([o["trans_div_mm"] for o in per_object])),
        "rot_div_deg": float(np.mean([o["rot_div_deg"] for o in per_object])),
        "joint_div_rad": float(np.mean([o["joint_div_rad"] for o in per_object])),
        "contact_spread_mm": float(np.mean([o["contact_spread_mm"] for o in per_object])),
        "coverage": {k: v / n_objects for k, v in coverage_counts.items()},
        "n_objects": n_objects,
        "num_samples": num_samples,
    }

    return {"aggregate": agg, "per_object": per_object}


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", default="0")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS.keys()))
    parser.add_argument("--out", type=Path, default=OUT_DIR)
    parser.add_argument("--quality-threshold", type=float, default=-4.0,
                        help="Score threshold for Coverage@k")
    args = parser.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(device)}")

    test_cfg = OmegaConf.create(TEST_CONFIG)
    loader = get_dataloader("test", test_cfg)
    print(f"Test set: {len(loader.dataset)} grasps")

    scorer = GraspScorer(mu=0.5, object_mass=0.2,
                         beta1=1.0, beta2=2.0, beta3=1.0, beta4=0.5)

    all_results = {}

    for variant_name in args.variants:
        variant_dir = VARIANTS[variant_name]
        print(f"\n{'='*60}")
        print(f"  {variant_name} ({variant_dir})")
        print(f"{'='*60}")

        run_dir = _latest_run(variant_dir)
        cp = run_dir / "checkpoint_best.pt"
        print(f"  Checkpoint: {cp}")

        cfg_files = sorted(run_dir.glob("*.yml"))
        cfg = OmegaConf.load(str(cfg_files[0])) if cfg_files else OmegaConf.create({})
        model = get_dex_model(
            p_uncond=float(cfg.model.get("p_uncond", 0.1)),
            guidance=float(cfg.model.get("guidance", 2.0)),
            num_ode_steps=int(cfg.model.get("num_ode_steps", 10)),
            hand_q_decoder_type=str(cfg.model.get("hand_q_decoder", "flow")),
            surface_proj_tau=float(cfg.model.get("surface_proj_tau", 0.005)),
            hand=str(cfg.model.get("hand", "leap")),
            wrist_frame=str(cfg.model.get("wrist_frame", "grasp_center")),
            cond_norm=bool(cfg.model.get("cond_norm", False)),
            n_coupling_layers=int(cfg.model.get("n_coupling_layers", 8)),
        ).to(device)

        ckpt = torch.load(str(cp), map_location="cpu")
        state = ckpt.get("model", ckpt.get("model_state", ckpt))
        model.load_state_dict(state, strict=False)

        results = evaluate_diversity(
            model, loader, device, args.num_samples, scorer, args.quality_threshold)

        agg = results["aggregate"]
        all_results[variant_name] = agg

        print(f"  Translation div: {agg['trans_div_mm']:.1f} mm")
        print(f"  Rotation div:    {agg['rot_div_deg']:.1f} deg")
        print(f"  Joint div:       {agg['joint_div_rad']:.4f} rad")
        print(f"  Contact spread:  {agg['contact_spread_mm']:.1f} mm")
        print(f"  Coverage@1:      {agg['coverage'][1]*100:.1f}%")
        print(f"  Coverage@4:      {agg['coverage'][4]*100:.1f}%")
        print(f"  Coverage@8:      {agg['coverage'][8]*100:.1f}%")
        print(f"  Coverage@20:     {agg['coverage'][20]*100:.1f}%")

        # Save per-object CSV
        var_out = args.out / variant_name
        var_out.mkdir(parents=True, exist_ok=True)
        csv_path = var_out / "diversity_per_object.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results["per_object"][0].keys())
            writer.writeheader()
            writer.writerows(results["per_object"])

        del model
        torch.cuda.empty_cache()

    # --- Summary ---
    print("\n" + "=" * 60)
    print("DIVERSITY METRICS SUMMARY")
    print("=" * 60)
    print(f"{'Variant':<15} {'Trans(mm)':<10} {'Rot(deg)':<10} "
          f"{'Joint(rad)':<11} {'Spread(mm)':<11} {'Cov@1':<7} {'Cov@8':<7}")
    print("-" * 71)
    for name, agg in all_results.items():
        print(f"{name:<15} {agg['trans_div_mm']:<10.1f} {agg['rot_div_deg']:<10.1f} "
              f"{agg['joint_div_rad']:<11.4f} {agg['contact_spread_mm']:<11.1f} "
              f"{agg['coverage'][1]*100:<7.1f} {agg['coverage'][8]*100:<7.1f}")

    # Save YAML
    args.out.mkdir(parents=True, exist_ok=True)
    summary_path = args.out / "diversity_summary.yaml"
    with open(summary_path, "w") as f:
        yaml.safe_dump({
            "description": "Diversity and coverage metrics across EquiDexFlow variants",
            "num_samples": args.num_samples,
            "quality_threshold": args.quality_threshold,
            "test_set": "81 objects",
            "results": all_results,
        }, f, sort_keys=False)
    print(f"\nSummary: {summary_path}")

    # LaTeX table
    tex_lines = [
        r"\begin{table}[ht]",
        r"\caption{Diversity and coverage metrics (K=20 samples per object, 81 test objects).}",
        r"\label{tab:diversity}",
        r"\centering\small",
        r"\begin{tabular}{@{}lrrrrrr@{}}",
        r"\toprule",
        r"Variant & Trans.\ (mm) & Rot.\ ($^\circ$) & Joint (rad) & Spread (mm) & Cov@1 & Cov@8 \\",
        r"\midrule",
    ]
    for name, agg in all_results.items():
        tex_lines.append(
            f"  {name} & {agg['trans_div_mm']:.1f} & {agg['rot_div_deg']:.1f} & "
            f"{agg['joint_div_rad']:.3f} & {agg['contact_spread_mm']:.1f} & "
            f"{agg['coverage'][1]*100:.0f}\\% & {agg['coverage'][8]*100:.0f}\\% \\\\"
        )
    tex_lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]

    tex_path = args.out / "diversity_latex.tex"
    with open(tex_path, "w") as f:
        f.write("\n".join(tex_lines) + "\n")
    print(f"LaTeX table: {tex_path}")


if __name__ == "__main__":
    main()
