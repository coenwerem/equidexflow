#!/usr/bin/env python3
"""Sweep ODE steps and CFG guidance scale at inference time.

Measures wrist-to-nearest-GT distance across a grid of
(num_ode_steps, guidance) settings. No retraining needed.

Usage (run from EquiDexFlow dir):
    cd ~/ResearchProjects/MuJoCoDex/third_party/grasp_syn/EquiDexFlow
    ~/ResearchProjects/MuJoCoDex/.venv/bin/python \
        ~/ResearchProjects/frogger/scripts/equidex/sweep_ode_guidance.py \
        --checkpoint outputs/training_results/equidexflow_dex_full_flow_film/film_20260523-0908/checkpoint_best.pt \
        --device 0
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import trimesh

_EDF_DIR = os.environ.get("EQUIDEXFLOW_OUTPUTS_DIR", os.getcwd())

import argparse
from omegaconf import OmegaConf


DB_DIR = Path.home() / "ResearchProjects/frogger/outputs/datasets/dexgraspdb/v3/allegro"
MESH_DIR = Path.home() / "ResearchProjects/MuJoCoDex/assets/misc/objects"
EGAD_DIR = Path.home() / "ResearchProjects/Datasets/egad_eval_set"

MESH_STEM = {
    "mustard_bottle": "frogger_ycb/006_mustard_bottle",
    "cube": "graspit/cube",
    "sphere": "graspit/sphere",
    "bleach_cleanser": "frogger_ycb/021_bleach_cleanser",
}

TEST_OBJECTS = ["mustard_bottle", "cube", "sphere", "bleach_cleanser"]
ODE_STEPS = [10, 25, 50, 100]
GUIDANCE_SCALES = [2.0, 3.0, 4.0]
NUM_SAMPLES = 10


def load_gt_wrists_centered(obj_name: str) -> np.ndarray | None:
    db_path = DB_DIR / f"{obj_name}.json"
    if not db_path.exists():
        return None
    db = json.load(open(db_path))

    # Load mesh centroid for centering
    stem = MESH_STEM.get(obj_name)
    centroid = np.zeros(3)
    if stem:
        for ext in (".stl", ".obj", ".ply"):
            path = MESH_DIR / (stem + ext)
            if path.exists():
                mesh = trimesh.load(str(path), force="mesh")
                centroid = mesh.centroid
                break

    wrists = []
    for g in db["grasps"]:
        if "wrist_pose_object" in g:
            wp = np.array(g["wrist_pose_object"])
            wrists.append(wp[:3, 3] - centroid)
    return np.array(wrists) if wrists else None


def nearest_gt_dist(pred_wrist: np.ndarray, gt_wrists: np.ndarray) -> float:
    return float(np.linalg.norm(gt_wrists - pred_wrist[None, :], axis=1).min())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--objects", nargs="+", default=TEST_OBJECTS)
    parser.add_argument("--ode-steps", nargs="+", type=int, default=ODE_STEPS)
    parser.add_argument("--guidance", nargs="+", type=float, default=GUIDANCE_SCALES)
    parser.add_argument("--num-samples", type=int, default=NUM_SAMPLES)
    args = parser.parse_args()

    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")

    # Load GT wrists for each object
    gt_wrists = {}
    for obj in args.objects:
        w = load_gt_wrists_centered(obj)
        if w is not None:
            gt_wrists[obj] = w
            print(f"  GT wrists for {obj}: {len(w)} grasps")

    # Load test point clouds once
    from equidexflow.loaders import get_dataloader
    test_cfg = OmegaConf.create({
        "dataset": {
            "name": "dexgrasp",
            "grasp_db_dir": str(DB_DIR),
            "object_mesh_dir": str(MESH_DIR),
            "n_object_points": 512, "max_contacts": 64,
            "mu": 0.5, "object_mass": 0.2,
            "augment": False, "split": "test", "object_names": None,
        },
        "batch_size": 1, "num_workers": 0, "shuffle": False,
    })
    loader = get_dataloader("test", test_cfg)
    pts_by_obj = {}
    for batch in loader:
        names = batch["object_name"]
        for b in range(batch["object_points"].shape[0]):
            name = names[b]
            if name in args.objects and name not in pts_by_obj:
                pts_by_obj[name] = batch["object_points"][b]
        if all(o in pts_by_obj for o in args.objects):
            break

    # Load checkpoint weights once
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state", ckpt))

    print(f"\n{'ODE':>5s} {'CFG':>5s} | ", end="")
    for obj in args.objects:
        print(f"{obj:>18s}", end="")
    print(f" | {'Mean':>8s}  {'Time':>6s}")
    print("-" * (14 + 18 * len(args.objects) + 20))

    results = []

    for n_steps in args.ode_steps:
        for guidance in args.guidance:
            torch.manual_seed(args.seed)

            from equidexflow.models import get_dex_model
            from equidexflow.utils.ode_solvers import SE3_RK4_MK

            model = get_dex_model(
                p_uncond=0.1, guidance=guidance, num_ode_steps=n_steps,
                hand_q_decoder_type="flow",
                n_coupling_layers=8,
                surface_proj_tau=0.005,
            ).to(device)
            model.load_state_dict(state, strict=False)
            model.eval()

            row = {"ode_steps": n_steps, "guidance": guidance}
            obj_dists = []

            t0 = time.time()
            with torch.no_grad():
                for obj_name in args.objects:
                    if obj_name not in pts_by_obj or obj_name not in gt_wrists:
                        continue

                    obj_pts = pts_by_obj[obj_name].unsqueeze(0).to(device)
                    preds = model.sample(obj_pts, args.num_samples)

                    dists = []
                    for pred in preds:
                        wp = pred["wrist_pose"][:3, 3].cpu().numpy()
                        d = nearest_gt_dist(wp, gt_wrists[obj_name])
                        dists.append(d)

                    avg_d = np.mean(dists) * 1000
                    min_d = np.min(dists) * 1000
                    row[obj_name] = {"mean_mm": avg_d, "min_mm": min_d}
                    obj_dists.append(avg_d)

            elapsed = time.time() - t0
            mean_all = np.mean(obj_dists) if obj_dists else 0

            print(f"{n_steps:5d} {guidance:5.1f} | ", end="")
            for obj in args.objects:
                if obj in row and isinstance(row[obj], dict):
                    print(f"  {row[obj]['mean_mm']:6.1f}({row[obj]['min_mm']:4.0f})", end="")
                else:
                    print(f"{'--':>18s}", end="")
            print(f" | {mean_all:6.1f}mm  {elapsed:5.1f}s")

            row["mean_all_mm"] = mean_all
            row["time_s"] = elapsed
            results.append(row)

            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

    print(f"\nFormat: mean(min) in mm. {args.num_samples} samples per object.")


if __name__ == "__main__":
    main()
