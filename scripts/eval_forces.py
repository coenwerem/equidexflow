#!/usr/bin/env python3
"""Evaluate force prediction quality of a trained EquiDexFlow model.

Metrics reported
----------------
* Mean force error (N) per finger
* Force direction error (degrees) per finger
* Friction cone violation rate (%) — fraction of samples where at least one
  finger violates the Coulomb cone
* Wrench balance residual (N·m) — mean ||G f + w_ext||₂

Usage
-----
    python eval_forces.py \\
        --config configs/equidexflow_dex_full.yml \\
        --checkpoint path/to/checkpoint.pt \\
        [--split test] \\
        [--num_samples 10] \\
        [--output results_forces.csv]
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

from equidexflow.kinematics import friction_cone_violation_rate
from equidexflow.kinematics.grasp_map import wrench_balance_residual
from equidexflow.loaders.schema import N_FINGERS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _to_device(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}


def _aggregate_per_finger(
    data: torch.Tensor,       # (B, M, D)
    finger_ids: torch.Tensor, # (B, M)
    valid_mask: torch.Tensor, # (B, M)
    n_fingers: int = N_FINGERS,
) -> torch.Tensor:
    """Returns (B, n_fingers, D), NaN where no GT contact for that finger."""
    B, M, D = data.shape
    result = torch.full((B, n_fingers, D), float("nan"), dtype=data.dtype, device=data.device)
    for f in range(n_fingers):
        mask = valid_mask & (finger_ids == f)
        count = mask.float().sum(dim=1)
        has = count > 0
        if has.any():
            weighted = (data * mask.unsqueeze(-1).float()).sum(dim=1)
            mean = weighted / count.unsqueeze(-1).clamp(min=1.0)
            result[has, f] = mean[has]
    return result


def evaluate_forces(model, loader, device, num_samples: int,
                    scorer=None, mu: float = 0.5, object_mass: float = 0.2):
    """Top-1 predicted grasp vs nearest GT grasp, per test object.

    Two-pass evaluation matching eval_contacts.py:
      Pass 1 — collect all GT (contacts, forces, normals) per object.
      Pass 2 — for each unique object, generate top-1 prediction, find
               nearest GT grasp by contact position, compare forces
               against that GT.  Friction violation and wrench balance
               are prediction-only (no GT needed).

    Returns
    -------
    finger_mag_errors : (5,) mean force magnitude error (N) per finger
    finger_dir_errors : (5,) mean direction error (degrees) per finger
    fvr               : scalar mean friction violation rate across objects
    mean_wb           : scalar mean wrench balance residual (N·m)
    """
    from collections import defaultdict
    gt_by_object: dict[str, list[dict]] = defaultdict(list)
    pts_by_object: dict[str, torch.Tensor] = {}

    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc="Collecting GT"):
            batch = _to_device(batch, device)
            obj_pts = batch["object_points"]  # (B, 3, N)
            B = obj_pts.shape[0]
            names = batch["object_name"]

            gt_contacts_f = _aggregate_per_finger(
                batch["contacts"], batch["finger_ids"], batch["valid_mask"]
            )
            gt_forces_f = _aggregate_per_finger(
                batch["forces"], batch["finger_ids"], batch["valid_mask"]
            )
            gt_normals_f = _aggregate_per_finger(
                batch["normals"], batch["finger_ids"], batch["valid_mask"]
            )

            for b_idx in range(B):
                name = names[b_idx]
                gt_by_object[name].append({
                    "contacts": gt_contacts_f[b_idx].cpu(),
                    "forces":   gt_forces_f[b_idx].cpu(),
                    "normals":  gt_normals_f[b_idx].cpu(),
                })
                if name not in pts_by_object:
                    pts_by_object[name] = obj_pts[b_idx].cpu()

        finger_mag_errors: list[list[float]] = [[] for _ in range(N_FINGERS)]
        finger_dir_errors: list[list[float]] = [[] for _ in range(N_FINGERS)]
        violation_rates: list[float] = []
        wrench_res: list[float] = []
        objects = sorted(gt_by_object.keys())
        print(f"Evaluating {len(objects)} unique objects, "
              f"{sum(len(v) for v in gt_by_object.values())} total GT grasps")

        for name in tqdm(objects, desc="Evaluating objects"):
            obj_pts_1 = pts_by_object[name].unsqueeze(0).to(device)
            obj_pts_n3 = obj_pts_1.permute(0, 2, 1)

            preds = model.sample(obj_pts_1, num_samples)

            if scorer is not None:
                ranked = scorer.rank_candidates(preds, obj_pts_n3[0])
                best = ranked[0]
            else:
                best = preds[0]

            pred_c = best["contacts"].to(device)  # (5, 3)
            pred_f = best["forces"].to(device)     # (5, 3)

            all_gt = gt_by_object[name]

            best_gt_idx = _find_nearest_gt(pred_c, all_gt, device)
            nearest_gt = all_gt[best_gt_idx]

            for f in range(N_FINGERS):
                gt_f = nearest_gt["forces"][f].to(device)
                if not torch.isfinite(gt_f).all():
                    continue
                mag_err = (pred_f[f].norm() - gt_f.norm()).abs().item()
                finger_mag_errors[f].append(mag_err)
                pf_norm = F.normalize(pred_f[f], dim=0, eps=1e-8)
                gf_norm = F.normalize(gt_f, dim=0, eps=1e-8)
                cos_sim = pf_norm.dot(gf_norm).clamp(-1.0, 1.0)
                dir_err = math.degrees(math.acos(cos_sim.item()))
                finger_dir_errors[f].append(dir_err)

            contacts_1 = pred_c.unsqueeze(0)
            forces_1 = pred_f.unsqueeze(0)
            obj_centroid = obj_pts_1.mean(dim=-1).unsqueeze(0)
            normals_1 = F.normalize(obj_centroid - contacts_1 + 1e-8, dim=-1)
            valid_1 = torch.ones(1, N_FINGERS, dtype=torch.bool, device=device)
            rate = friction_cone_violation_rate(forces_1, normals_1, valid_1, mu=mu)
            violation_rates.append(rate.item())

            gt_n = nearest_gt["normals"].unsqueeze(0).to(device)
            gt_n = torch.nan_to_num(gt_n, nan=0.0)
            gt_n = F.normalize(gt_n, dim=-1, eps=1e-8)
            wb = wrench_balance_residual(
                contacts_1, gt_n, forces_1, valid_1,
                object_mass=object_mass,
            )
            wrench_res.append(wb.item())

    mean_mag = np.array([np.mean(e) if e else float("nan") for e in finger_mag_errors])
    mean_dir = np.array([np.mean(e) if e else float("nan") for e in finger_dir_errors])
    mean_fvr = float(np.mean(violation_rates)) if violation_rates else float("nan")
    mean_wb  = float(np.mean(wrench_res)) if wrench_res else float("nan")

    return mean_mag, mean_dir, mean_fvr, mean_wb


def _find_nearest_gt(pred_contacts: torch.Tensor, all_gt: list[dict],
                     device: torch.device) -> int:
    """Return index of GT grasp with minimum total contact distance."""
    best_idx, best_dist = 0, float("inf")
    for i, gt in enumerate(all_gt):
        gt_c = gt["contacts"].to(device)  # (N_FINGERS, 3)
        valid = torch.isfinite(gt_c).all(dim=-1) & torch.isfinite(pred_contacts).all(dim=-1)
        if not valid.any():
            continue
        dist = (pred_contacts[valid] - gt_c[valid]).norm(dim=-1).sum().item()
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def print_results_table(mean_mag, mean_dir, mean_fvr, mean_wb):
    finger_names = ["Thumb", "Index", "Middle", "Ring"][:N_FINGERS]
    print("\n" + "=" * 56)
    print(f"{'Finger':<10} {'Mag Err (N)':>12} {'Dir Err (°)':>12}")
    print("-" * 56)
    for i, name in enumerate(finger_names):
        print(f"{name:<10} {mean_mag[i]:>12.4f} {mean_dir[i]:>11.1f}°")
    print("-" * 56)
    print(
        f"{'Overall':<10} {np.nanmean(mean_mag):>12.4f} "
        f"{np.nanmean(mean_dir):>11.1f}°"
    )
    print("=" * 56)
    print(f"\nFriction violation rate : {mean_fvr*100:.1f}%")
    print(f"Wrench balance residual : {mean_wb:.4f} N·m\n")


def save_csv(path: str, mean_mag, mean_dir, mean_fvr, mean_wb):
    finger_names = ["Thumb", "Index", "Middle", "Ring"][:N_FINGERS]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["finger", "mag_error_N", "dir_error_deg"])
        for i, name in enumerate(finger_names):
            writer.writerow([name, mean_mag[i], mean_dir[i]])
        writer.writerow(["friction_violation_rate", mean_fvr, ""])
        writer.writerow(["wrench_balance_residual_Nm", mean_wb, ""])
    print(f"Saved results to {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate force prediction quality.")
    parser.add_argument("--config",      required=True)
    parser.add_argument("--checkpoint",  required=True)
    parser.add_argument("--split",       default="test", choices=["train", "val", "test"])
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--device",      default="0")
    parser.add_argument("--output",      default=None)
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    device = torch.device("cpu" if args.device == "cpu" else f"cuda:{args.device}")

    from equidexflow.models import get_dex_model
    model = get_dex_model(
        p_uncond=float(cfg.model.get("p_uncond", 0.1)),
        guidance=float(cfg.model.get("guidance", 2.0)),
        num_ode_steps=int(cfg.model.get("num_ode_steps", 10)),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state = ckpt.get("model", ckpt.get("model_state", ckpt))
    model.load_state_dict(state, strict=False)
    print(f"Loaded checkpoint from {args.checkpoint}")

    from equidexflow.loaders import get_dataloader
    split_cfg = cfg.data[args.split]
    loader = get_dataloader(args.split, split_cfg)

    num_samples = args.num_samples or int(cfg.evaluation.num_samples)
    mu          = float(cfg.data.get("mu", 0.5))
    obj_mass    = float(cfg.data.get("object_mass", 0.2))

    from equidexflow.physics.scorer import GraspScorer
    pw = cfg.evaluation.get("physics_weights", {})
    scorer = GraspScorer(
        mu=mu,
        object_mass=obj_mass,
        beta1=float(pw.get("beta1", 1.0)),
        beta2=float(pw.get("beta2", 2.0)),
        beta3=float(pw.get("beta3", 1.0)),
        beta4=float(pw.get("beta4", 0.5)),
    )

    mean_mag, mean_dir, mean_fvr, mean_wb = evaluate_forces(
        model, loader, device, num_samples, scorer=scorer, mu=mu, object_mass=obj_mass
    )

    print_results_table(mean_mag, mean_dir, mean_fvr, mean_wb)

    if args.output:
        save_csv(args.output, mean_mag, mean_dir, mean_fvr, mean_wb)


if __name__ == "__main__":
    main()
